"""
PerformanceAgent — real sector-specific alpha scoring per senator.
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from config import (
    ALPHA_DECAY_HALFLIFE_MONTHS,
    ALPHA_WINDOW_DAYS,
    KNOWN_STAR_SENATORS,
    PERFORMANCE_LOOKBACK_MONTHS,
    PRICE_DATA_CSV,
    SECTOR_ETF,
    TICKER_SECTORS,
    SENATOR_COMMITTEES,
)
from data_agent import Trade

log = logging.getLogger("PerformanceAgent")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    log.warning("yfinance not installed — alpha scores will be proxy-only.")

@dataclass
class SenatorProfile:
    name:              str
    total_trades:      int = 0
    winning_trades:    int = 0
    win_rate:          float = 0.0
    avg_alpha_30d:     float = 0.0
    overall_score:     float = 0.35
    is_top_trader:     bool = False
    is_chair:          bool = False
    committees:        list[str] = field(default_factory=list)
    subcommittees:     list[str] = field(default_factory=list)
    sector_scores:     dict[str, float] = field(default_factory=dict)

    def sector_score(self, sector: str) -> float:
        return self.sector_scores.get(sector, self.overall_score)

class PerformanceAgent:
    NAME = "PerformanceAgent"

    def __init__(self):
        self.price_df = self._load_price_csv()

    def _load_price_csv(self) -> Optional[pd.DataFrame]:
        path = Path(__file__).parent / PRICE_DATA_CSV
        if path.exists():
            try:
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
            except Exception as e:
                log.error("Failed to load price CSV: %s", e)
        return None

    def run(self, trades: list[Trade]) -> dict[str, SenatorProfile]:
        log.info("Scoring senator performance (%d trades)…", len(trades))

        by_senator = defaultdict(list)
        for t in trades:
            by_senator[t.senator].append(t)

        profiles = {}
        for name, s_trades in by_senator.items():
            profiles[name] = self._score_senator(name, s_trades)

        log.info("Profiled %d senators, %d confirmed top traders",
                 len(profiles), sum(1 for p in profiles.values() if p.is_top_trader))
        return profiles

    def _score_senator(self, name: str, trades: list[Trade]) -> SenatorProfile:
        profile = SenatorProfile(name=name, total_trades=len(trades))

        def normalize(n):
            n = n.replace(",", "").lower()
            return set(n.split())

        norm_name = normalize(name)

        for known, (coms, subcoms, is_chair) in SENATOR_COMMITTEES.items():
            norm_known = normalize(known)
            if norm_known.issubset(norm_name) or norm_name.issubset(norm_known):
                profile.committees    = coms
                profile.subcommittees = subcoms
                profile.is_chair      = is_chair
                break

        if _YF_AVAILABLE and trades:
            self._compute_real_alpha(profile, trades)
        else:
            self._compute_proxy_score(profile, trades)
        return profile

    def _compute_real_alpha(self, profile: SenatorProfile, trades: list[Trade]) -> None:
        buys = [t for t in trades if t.side == "buy"]
        if not buys:
            self._compute_proxy_score(profile, trades)
            return

        alphas_by_sector: dict[str, list[tuple[float, float]]] = defaultdict(list)
        all_weighted: list[tuple[float, float]] = []
        price_cache = {}
        today = datetime.today()

        for trade in buys:
            trade_dt = self._parse_date(trade.date)
            if trade_dt is None: continue

            months_ago = (today - trade_dt).days / 30
            decay_weight = math.exp(-math.log(2) * months_ago / ALPHA_DECAY_HALFLIFE_MONTHS)
            end_dt = trade_dt + timedelta(days=ALPHA_WINDOW_DAYS)
            if end_dt > today: continue

            ticker = trade.ticker
            sector = trade.sector or TICKER_SECTORS.get(ticker, "")
            etf = SECTOR_ETF.get(sector, "SPY")

            ret_stock = self._get_return(ticker, trade_dt, end_dt, price_cache)
            ret_etf   = self._get_return(etf,    trade_dt, end_dt, price_cache)

            if ret_stock is None or ret_etf is None: continue

            alpha = ret_stock - ret_etf
            all_weighted.append((alpha, decay_weight))
            if sector: alphas_by_sector[sector].append((alpha, decay_weight))

        if not all_weighted:
            self._compute_proxy_score(profile, trades)
            return

        total_w = sum(w for _, w in all_weighted)
        avg_alpha = sum(a * w for a, w in all_weighted) / total_w
        win_rate = sum(1 for a, _ in all_weighted if a > 0) / len(all_weighted)

        profile.win_rate = round(win_rate, 3)
        profile.avg_alpha_30d = round(avg_alpha * 100, 2)
        profile.winning_trades = sum(1 for a, _ in all_weighted if a > 0)

        raw_score = 0.5 + (avg_alpha * 3) + (win_rate - 0.5) * 0.4
        profile.overall_score = round(max(0.05, min(raw_score, 0.98)), 3)

        for sector, sector_alphas in alphas_by_sector.items():
            sw = sum(w for _, w in sector_alphas)
            sa = sum(a * w for a, w in sector_alphas) / sw
            swr = sum(1 for a, _ in sector_alphas if a > 0) / len(sector_alphas)
            raw = 0.5 + (sa * 3) + (swr - 0.5) * 0.4
            profile.sector_scores[sector] = round(max(0.05, min(raw, 0.98)), 3)

        profile.is_top_trader = (profile.overall_score >= 0.65 or profile.name in KNOWN_STAR_SENATORS)

    def _get_return(self, ticker: str, start: datetime, end: datetime, cache: dict) -> Optional[float]:
        if self.price_df is not None:
            try:
                ticker_df = self.price_df[self.price_df['Ticker'] == ticker]
                if not ticker_df.empty:
                    d_start = (ticker_df['Date'] - start).abs().idxmin()
                    d_end   = (ticker_df['Date'] - end).abs().idxmin()
                    p1 = float(ticker_df.loc[d_start, 'Close'])
                    p2 = float(ticker_df.loc[d_end, 'Close'])
                    return (p2 - p1) / p1
            except: pass
        
        key = f"{ticker}_{start.date()}_{end.date()}"
        if key in cache: return cache[key]
        try:
            # Silence yfinance warnings/errors
            import sys
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            
            df = yf.download(ticker, start=(start-timedelta(days=3)).strftime("%Y-%m-%d"), end=(end+timedelta(days=3)).strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            
            sys.stderr = old_stderr
            
            if df.empty: return None
            p_start, p_end = df["Close"].iloc[0], df["Close"].iloc[-1]
            ret = float((p_end - p_start) / p_start)
            cache[key] = ret
            return ret
        except: 
            sys.stderr = old_stderr
            return None

    def _compute_proxy_score(self, profile: SenatorProfile, trades: list[Trade]) -> None:
        freq_score = min(len(trades) / 20.0, 1.0)
        star_boost = 0.30 if profile.name in KNOWN_STAR_SENATORS else 0.0
        profile.overall_score = round(min(0.25 + 0.4 * freq_score + star_boost, 0.95), 3)
        profile.is_top_trader = profile.name in KNOWN_STAR_SENATORS

    def get_momentum(self, ticker: str, date: datetime) -> float:
        if self.price_df is None: return 0.0
        try:
            ticker_df = self.price_df[self.price_df['Ticker'] == ticker]
            if ticker_df.empty: return 0.0
            past_df = ticker_df[ticker_df['Date'] <= date].sort_values('Date')
            if len(past_df) < 10: return 0.0
            p_end = float(past_df.iloc[-1]['Close'])
            p_start = float(past_df.iloc[-10]['Close']) # 10 trading days
            return (p_end - p_start) / p_start
        except: return 0.0

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try: return datetime.strptime(date_str, fmt)
            except ValueError: continue
        return None
