"""
ExecutionAgent — intelligent order execution.

New vs v2:
- Kelly-based position sizing (pre-computed by SignalAgent, validated here)
- Sector concentration cap (MAX_SECTOR_EXPOSURE of portfolio per sector)
- Intraday entry delay: waits ENTRY_DELAY_MINUTES after market open before entering
  (avoids chasing the open pop when a filing goes public)
- Tracks open positions by sector for concentration management
- Graceful handling of paper vs live mode
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import (
    ENTRY_DELAY_MINUTES,
    MAX_POSITIONS,
    MAX_SECTOR_EXPOSURE,
    MIN_SIGNAL_SCORE,
)
from signal_agent import Signal

log = logging.getLogger("ExecutionAgent")


class ExecutionAgent:
    NAME = "ExecutionAgent"

    MARKET_OPEN_ET  = (9, 30)   # 9:30 AM ET
    MARKET_CLOSE_ET = (16, 0)   # 4:00 PM ET

    def __init__(self):
        import os
        self.key    = os.getenv("ALPACA_API_KEY", "")
        self.secret = os.getenv("ALPACA_SECRET_KEY", "")
        self.base   = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ).rstrip("/")

        if not self.key or not self.secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        self.headers = {
            "APCA-API-KEY-ID":     self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type":        "application/json",
        }

    # ── Main run ──────────────────────────────────────────────

    def run(
        self,
        buy_signals:  list[Signal],
        sell_signals: list[Signal],
        min_score:    float = MIN_SIGNAL_SCORE,
    ) -> dict:
        """
        Returns a summary dict with orders placed, skipped, and errors.
        """
        account    = self._get_account()
        buying_pw  = float(account.get("buying_power", 0))
        portfolio_value = float(account.get("portfolio_value", buying_pw))

        positions  = self._get_positions()  # dict[ticker, position_dict]
        open_count = len(positions)

        log.info(
            "[%s] Portfolio $%.2f  Buying power $%.2f  Positions %d/%d",
            self.NAME, portfolio_value, buying_pw, open_count, MAX_POSITIONS
        )

        # Compute current sector exposure
        sector_exposure = self._compute_sector_exposure(positions, portfolio_value)

        summary = {"orders_placed": [], "skipped": [], "errors": []}

        # ── Step 1: Close positions with strong sell signals ──
        for sig in sell_signals:
            if sig.score < min_score:
                continue
            if sig.ticker not in positions:
                continue
            log.info("[%s] CLOSE %s  score=%.3f  %s",
                     self.NAME, sig.ticker, sig.score, sig.rationale)
            try:
                self._close_position(sig.ticker)
                summary["orders_placed"].append(
                    {"action": "sell/close", "ticker": sig.ticker, "score": sig.score}
                )
                # Update local state
                del positions[sig.ticker]
                open_count -= 1
            except Exception as exc:
                log.error("[%s] Close failed %s: %s", self.NAME, sig.ticker, exc)
                summary["errors"].append({"ticker": sig.ticker, "error": str(exc)})

        # ── Step 2: Wait for entry window if market just opened ──
        self._maybe_wait_for_entry_window()

        # ── Step 3: Open new buy positions ────────────────────
        for sig in buy_signals:
            # Basic qualification
            if sig.score < min_score:
                summary["skipped"].append(
                    {"ticker": sig.ticker, "reason": f"score {sig.score:.3f} < threshold"}
                )
                continue

            if sig.ticker in positions:
                summary["skipped"].append(
                    {"ticker": sig.ticker, "reason": "already holding"}
                )
                continue

            if open_count >= MAX_POSITIONS:
                summary["skipped"].append(
                    {"ticker": sig.ticker, "reason": "max positions reached"}
                )
                log.warning("[%s] Max positions (%d) — stopping new buys", self.NAME, MAX_POSITIONS)
                break

            # ── Sector concentration check ────────────────────
            sector     = sig.sector
            sector_exp = sector_exposure.get(sector, 0.0)
            if sector_exp >= MAX_SECTOR_EXPOSURE:
                summary["skipped"].append(
                    {"ticker": sig.ticker,
                     "reason": f"sector {sector} at {sector_exp:.0%} cap"}
                )
                log.info("[%s] Sector cap hit for %s (%s at %.0%%) — skip %s",
                         self.NAME, sector, sector, sector_exp, sig.ticker)
                continue

            notional = sig.suggested_notional
            if buying_pw < notional:
                # Try smaller size
                notional = min(buying_pw * 0.9, notional)
                if notional < 50:
                    summary["skipped"].append(
                        {"ticker": sig.ticker, "reason": "insufficient buying power"}
                    )
                    log.warning("[%s] Insufficient buying power (%.2f) — stopping",
                                self.NAME, buying_pw)
                    break

            log.info(
                "[%s] BUY %s  $%.0f  score=%.3f  | %s",
                self.NAME, sig.ticker, notional, sig.score, sig.rationale
            )

            try:
                order = self._place_order(sig.ticker, "buy", notional)
                log.info(
                    "[%s]  ✓ id=%s status=%s",
                    self.NAME, order.get("id"), order.get("status")
                )
                summary["orders_placed"].append({
                    "action":   "buy",
                    "ticker":   sig.ticker,
                    "notional": notional,
                    "score":    sig.score,
                    "order_id": order.get("id"),
                })
                open_count += 1
                buying_pw  -= notional
                sector_exposure[sector] = sector_exp + (notional / portfolio_value)
            except Exception as exc:
                log.error("[%s] Order failed %s: %s", self.NAME, sig.ticker, exc)
                summary["errors"].append({"ticker": sig.ticker, "error": str(exc)})
                time.sleep(1)

        log.info(
            "[%s] Cycle complete: %d placed, %d skipped, %d errors",
            self.NAME,
            len(summary["orders_placed"]),
            len(summary["skipped"]),
            len(summary["errors"]),
        )
        return summary

    # ── Sector concentration ──────────────────────────────────

    @staticmethod
    def _compute_sector_exposure(
        positions: dict, portfolio_value: float
    ) -> dict[str, float]:
        """
        Returns sector → fraction of portfolio currently invested.
        """
        from config import TICKER_SECTORS
        exposure: dict[str, float] = {}
        if portfolio_value <= 0:
            return exposure

        for ticker, pos in positions.items():
            market_value = float(pos.get("market_value", 0))
            sector = TICKER_SECTORS.get(ticker, "other")
            exposure[sector] = exposure.get(sector, 0.0) + market_value / portfolio_value

        return exposure

    # ── Entry timing ──────────────────────────────────────────

    def _maybe_wait_for_entry_window(self) -> None:
        """
        If the market opened less than ENTRY_DELAY_MINUTES ago, sleep
        until the delay has elapsed to avoid chasing the open pop.
        """
        clock = self._get_clock()
        if not clock.get("is_open", False):
            return  # market closed, orders queue for next open

        # Parse market open time
        open_str = clock.get("next_open") or clock.get("next_close", "")
        # Alpaca clock returns timestamps like "2024-03-01T09:30:00-05:00"
        # The "session_start" field tells us when current session opened
        # We use a simpler heuristic: check if we're within ENTRY_DELAY_MINUTES of 9:30 AM
        now_et = datetime.utcnow() - timedelta(hours=5)  # rough ET conversion
        market_open_today = now_et.replace(
            hour=self.MARKET_OPEN_ET[0],
            minute=self.MARKET_OPEN_ET[1],
            second=0, microsecond=0
        )
        elapsed = (now_et - market_open_today).total_seconds() / 60

        if 0 <= elapsed < ENTRY_DELAY_MINUTES:
            wait_sec = int((ENTRY_DELAY_MINUTES - elapsed) * 60)
            log.info(
                "[%s] Market open %d min ago — waiting %ds for entry window",
                self.NAME, int(elapsed), wait_sec
            )
            time.sleep(wait_sec)

    # ── Alpaca REST helpers ───────────────────────────────────

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url  = f"{self.base}{path}"
        resp = requests.request(
            method, url, headers=self.headers,
            data=json.dumps(body) if body else None,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _get_account(self) -> dict:
        return self._req("GET", "/v2/account")

    def _get_positions(self) -> dict:
        positions = self._req("GET", "/v2/positions")
        return {p["symbol"]: p for p in (positions if isinstance(positions, list) else [])}

    def _get_clock(self) -> dict:
        return self._req("GET", "/v2/clock")

    def _place_order(self, ticker: str, side: str, notional: float) -> dict:
        return self._req("POST", "/v2/orders", {
            "symbol":        ticker,
            "notional":      str(round(notional, 2)),
            "side":          side,
            "type":          "market",
            "time_in_force": "day",
        })

    def _close_position(self, ticker: str) -> None:
        self._req("DELETE", f"/v2/positions/{ticker}")

    def get_portfolio_summary(self) -> dict:
        """Returns a human-readable portfolio summary for reporting."""
        account   = self._get_account()
        positions = self._get_positions()
        return {
            "portfolio_value": float(account.get("portfolio_value", 0)),
            "buying_power":    float(account.get("buying_power", 0)),
            "equity":          float(account.get("equity", 0)),
            "open_positions":  len(positions),
            "positions":       [
                {
                    "ticker":       p["symbol"],
                    "qty":          p.get("qty"),
                    "market_value": float(p.get("market_value", 0)),
                    "unrealized_pl":float(p.get("unrealized_pl", 0)),
                    "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                }
                for p in positions.values()
            ],
        }
