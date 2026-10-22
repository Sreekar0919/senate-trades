"""
DataAgent — fetches and normalises Senate trade disclosures.

Source priority (automatic fallback chain):
  1. Senate Stock Watcher  (senatestockwatcher.com) — most reliable, no key needed
  2. Senate EFTS           (efts.senate.gov)        — official, sometimes unreliable DNS
  3. Quiver Quant          (quiverquant.com)         — requires API key
  4. Demo data             — 45 realistic trades so the pipeline always runs

Enhancements:
- Filer type parsing (senator vs spouse vs dependent — different weights)
- Bulk-rebalance noise detection (5+ same-sector trades same senator same day)
- Position add-on detection (senator already held ticker)
- Full pagination for EFTS
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import (
    DISCLOSURE_LOOKBACK_DAYS,
    QUIVER_API_KEY,
    TICKER_SECTORS,
    TRADE_DATA_CSV,
)

log = logging.getLogger("DataAgent")


@dataclass
class Trade:
    senator:            str
    ticker:             str
    side:               str     # "buy" | "sell"
    date:               str
    amount_range:       str
    asset_type:         str   = "Stock"
    filer_type:         str   = "senator"  # "senator"|"spouse"|"dependent"|"joint"
    filer_weight:       float = 1.0
    is_addon:           bool  = False
    sector:             str   = ""
    raw_amount_mid:     float = 0.0
    is_rebalance_noise: bool  = False
    excess_return:      float = 0.0
    state:              str   = ""


_AMOUNT_MIDS = {
    "$1,001 - $15,000":          8_000,
    "$15,001 - $50,000":        32_500,
    "$50,001 - $100,000":       75_000,
    "$100,001 - $250,000":     175_000,
    "$250,001 - $500,000":     375_000,
    "$500,001 - $1,000,000":   750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
}


class DataAgent:
    NAME = "DataAgent"

    SSW_URL    = "https://senatestockwatcher.com/api/transactions"
    EFTS_URL   = "https://efts.senate.gov/LATEST/search"
    QUIVER_URL = "https://api.quiverquant.com/beta/live/senatetrading"

    def run(
        self,
        days_back: int = DISCLOSURE_LOOKBACK_DAYS,
        source:    str = "csv",
    ) -> list[Trade]:
        log.info("Fetching disclosures [source=%s, days=%d]", source, days_back)
        
        # Priority 1: Local Data CSV (User requirement)
        trades = self._load_from_csv(days_back)
        if trades:
            log.info("Loaded %d trades from local CSV", len(trades))
            return self._enrich(trades)

        # Priority 2: Live APIs (Fallback)
        raw    = self._fetch_with_fallback(days_back, source)
        trades = [t for t in (self._parse(r) for r in raw) if t is not None]
        trades = self._enrich(trades)
        
        log.info(
            "%d valid trades fetched/parsed",
            len(trades)
        )
        return trades

    def _load_from_csv(self, days_back: int) -> list[Trade]:
        csv_path = Path(__file__).parent / TRADE_DATA_CSV
        if not csv_path.exists():
            log.error("CSV file not found: %s", csv_path)
            return []

        try:
            df = pd.read_csv(csv_path, encoding='latin1')
            cutoff = datetime.today() - timedelta(days=days_back)
            trades = []

            for _, row in df.iterrows():
                try:
                    # Parse date - handle potential errors gracefully
                    date_val = row['Traded']
                    if pd.isna(date_val): continue
                    trade_dt = pd.to_datetime(date_val)
                    if trade_dt < cutoff:
                        continue
                    
                    # Ticker: Take the first word, but don't restrict length
                    raw_ticker = str(row['Ticker']).split()[0].upper()
                    ticker = "".join(filter(str.isalnum, raw_ticker)) # Clean non-alphanumeric
                    if not ticker:
                        continue

                    side = "buy" if "Purchase" in str(row['Transaction']) else "sell"
                    
                    trades.append(Trade(
                        senator = str(row['Name']), # Using 'senator' field name for compatibility
                        ticker = ticker,
                        side = side,
                        date = trade_dt.strftime("%Y-%m-%d"),
                        amount_range = str(row['Trade_Size_USD']),
                        excess_return = float(row.get('excess_return', 0.0)),
                        state = str(row.get('State', "")),
                        sector = TICKER_SECTORS.get(ticker, "")
                    ))
                except Exception:
                    continue
            return trades
        except Exception as e:
            log.error("Failed to read CSV: %s", e)
            return []

    # ── Fallback chain ────────────────────────────────────────

    def _fetch_with_fallback(self, days_back: int, source: str) -> list[dict]:
        if source == "quiver":
            order = [self._fetch_quiver, self._fetch_ssw, self._fetch_efts]
        elif source == "efts":
            order = [self._fetch_efts, self._fetch_ssw]
        else:
            order = [self._fetch_ssw, self._fetch_efts]

        for fetcher in order:
            try:
                records = fetcher(days_back)
                if records:
                    return records
            except Exception as exc:
                log.error("%s API failed: %s", fetcher.__name__.replace("_fetch_", "").upper(), exc)

        log.info("All APIs failed — using demo data for demonstration")
        return self._demo_data(days_back)

    # ── Senate Stock Watcher ──────────────────────────────────

    def _fetch_ssw(self, days_back: int) -> list[dict]:
        resp = requests.get(self.SSW_URL, timeout=15)
        resp.raise_for_status()
        data  = resp.json()
        items = data.get("transactions", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []

        cutoff  = datetime.today() - timedelta(days=days_back)
        results = []
        for item in items:
            date_str = item.get("transaction_date") or item.get("transactionDate", "")
            try:
                if datetime.strptime(date_str[:10], "%Y-%m-%d") < cutoff:
                    continue
            except Exception:
                continue

            full_name  = item.get("senator") or item.get("name") or ""
            first_name = item.get("first_name") or (full_name.split()[0] if full_name else "")
            last_name  = item.get("last_name")  or (" ".join(full_name.split()[1:]) if full_name else "")

            results.append({
                "first_name":        first_name,
                "last_name":         last_name,
                "asset_description": item.get("ticker") or item.get("asset_description", ""),
                "type":              item.get("type") or item.get("transaction_type", ""),
                "transaction_date":  date_str[:10],
                "amount":            item.get("amount", ""),
                "asset_type":        item.get("asset_type", "Stock"),
                "filer_information": item.get("filer_type") or "Senator",
            })

        log.info("Senate Stock Watcher: %d trades in last %d days", len(results), days_back)
        return results

    # ── EFTS ──────────────────────────────────────────────────

    def _fetch_efts(self, days_back: int) -> list[dict]:
        from_d   = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_d     = datetime.today().strftime("%Y-%m-%d")
        records, offset = [], 0

        while True:
            params = {
                "q": '""', "dateRange": "custom",
                "fromDate": from_d, "toDate": to_d,
                "first": 200, "offset": offset,
            }
            resp  = requests.get(self.EFTS_URL, params=params, timeout=15)
            resp.raise_for_status()
            data  = resp.json()
            hits  = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {}).get("value", 0)
            records.extend(h["_source"] for h in hits)
            offset += 200
            if offset >= total or not hits:
                break

        log.info("EFTS: %d records", len(records))
        return records

    # ── Quiver Quant ──────────────────────────────────────────

    def _fetch_quiver(self, days_back: int) -> list[dict]:
        api_key = QUIVER_API_KEY
        if not api_key:
            raise ValueError("QUIVER_API_KEY not set")

        headers = {"Authorization": f"Token {api_key}"}
        resp    = requests.get(self.QUIVER_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        cutoff  = datetime.today() - timedelta(days=days_back)
        results = []
        for item in resp.json():
            try:
                if datetime.strptime(item["Date"], "%Y-%m-%d") < cutoff:
                    continue
            except Exception:
                continue
            results.append({
                "first_name":        item.get("Senator", "").split()[0],
                "last_name":         " ".join(item.get("Senator", "").split()[1:]),
                "asset_description": item.get("Ticker", ""),
                "type":              item.get("Transaction", ""),
                "transaction_date":  item.get("Date", ""),
                "amount":            item.get("Range", ""),
                "filer_information": "Senator",
            })

        log.info("Quiver Quant: %d trades", len(results))
        return results

    # ── Demo data ─────────────────────────────────────────────

    def _demo_data(self, days_back: int) -> list[dict]:
        """
        45 realistic trades — enough to generate committee hits,
        co-trading clusters, and multiple signals above threshold.
        Only used when all live APIs are unavailable.
        """
        today = datetime.today()
        def d(n): return (today - timedelta(days=n)).strftime("%Y-%m-%d")

        records = [
            # Defense cluster (Tuberville + Wicker + Reed co-buying)
            {"first_name":"Tommy","last_name":"Tuberville","asset_description":"LMT","type":"Purchase","transaction_date":d(12),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Tommy","last_name":"Tuberville","asset_description":"RTX","type":"Purchase","transaction_date":d(14),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Tommy","last_name":"Tuberville","asset_description":"NOC","type":"Purchase","transaction_date":d(10),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Roger","last_name":"Wicker",   "asset_description":"LMT","type":"Purchase","transaction_date":d(11),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Roger","last_name":"Wicker",   "asset_description":"GD", "type":"Purchase","transaction_date":d(13),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Jack", "last_name":"Reed",     "asset_description":"LMT","type":"Purchase","transaction_date":d(9), "amount":"$100,001 - $250,000","filer_information":"Senator"},
            {"first_name":"Jack", "last_name":"Reed",     "asset_description":"NOC","type":"Purchase","transaction_date":d(15),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Dan",  "last_name":"Sullivan", "asset_description":"RTX","type":"Purchase","transaction_date":d(8), "amount":"$15,001 - $50,000","filer_information":"Senator"},
            # Cybersecurity sub-cluster (Sullivan + King — both on cyber subcommittee)
            {"first_name":"Dan",  "last_name":"Sullivan", "asset_description":"CRWD","type":"Purchase","transaction_date":d(7),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Angus","last_name":"King",     "asset_description":"CRWD","type":"Purchase","transaction_date":d(6),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Angus","last_name":"King",     "asset_description":"PANW","type":"Purchase","transaction_date":d(8),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            # Palantir — defense + tech overlap
            {"first_name":"Tommy","last_name":"Tuberville","asset_description":"PLTR","type":"Purchase","transaction_date":d(5),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Jack", "last_name":"Reed",     "asset_description":"PLTR","type":"Purchase","transaction_date":d(6),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Angus","last_name":"King",     "asset_description":"PLTR","type":"Purchase","transaction_date":d(7),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            # Tech cluster (Warner + Rubio + Cantwell — all Commerce/Intel)
            {"first_name":"Mark", "last_name":"Warner",   "asset_description":"NVDA","type":"Purchase","transaction_date":d(20),"amount":"$100,001 - $250,000","filer_information":"Senator"},
            {"first_name":"Mark", "last_name":"Warner",   "asset_description":"MSFT","type":"Purchase","transaction_date":d(22),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Marco","last_name":"Rubio",    "asset_description":"NVDA","type":"Purchase","transaction_date":d(19),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Marco","last_name":"Rubio",    "asset_description":"PANW","type":"Purchase","transaction_date":d(21),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Maria","last_name":"Cantwell", "asset_description":"MSFT","type":"Purchase","transaction_date":d(17),"amount":"$100,001 - $250,000","filer_information":"Senator"},
            {"first_name":"Maria","last_name":"Cantwell", "asset_description":"AMZN","type":"Purchase","transaction_date":d(23),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            # Energy cluster (Manchin + Barrasso + Murkowski — all Energy committee)
            {"first_name":"Joe",  "last_name":"Manchin",  "asset_description":"XOM","type":"Purchase","transaction_date":d(30),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Joe",  "last_name":"Manchin",  "asset_description":"CVX","type":"Purchase","transaction_date":d(32),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"John", "last_name":"Barrasso", "asset_description":"XOM","type":"Purchase","transaction_date":d(28),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Lisa", "last_name":"Murkowski","asset_description":"CVX","type":"Purchase","transaction_date":d(29),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"John", "last_name":"Hoeven",   "asset_description":"XOM","type":"Purchase","transaction_date":d(27),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            # Finance cluster (Crapo + Scott + Paul — Finance committee)
            {"first_name":"Mike", "last_name":"Crapo",    "asset_description":"JPM","type":"Purchase","transaction_date":d(38),"amount":"$100,001 - $250,000","filer_information":"Senator"},
            {"first_name":"Tim",  "last_name":"Scott",    "asset_description":"BAC","type":"Purchase","transaction_date":d(41),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Tim",  "last_name":"Scott",    "asset_description":"GS", "type":"Purchase","transaction_date":d(39),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Rand", "last_name":"Paul",     "asset_description":"JPM","type":"Purchase","transaction_date":d(40),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            # Health cluster (Cassidy + Murray — HELP committee)
            {"first_name":"Bill", "last_name":"Cassidy",  "asset_description":"LLY","type":"Purchase","transaction_date":d(45),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Bill", "last_name":"Cassidy",  "asset_description":"ABBV","type":"Purchase","transaction_date":d(47),"amount":"$15,001 - $50,000","filer_information":"Senator"},
            {"first_name":"Patty","last_name":"Murray",   "asset_description":"LLY","type":"Purchase","transaction_date":d(46),"amount":"$100,001 - $250,000","filer_information":"Senator"},
            {"first_name":"Patty","last_name":"Murray",   "asset_description":"UNH","type":"Purchase","transaction_date":d(44),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            # Sell signals (Finance committee selling banks)
            {"first_name":"Elizabeth","last_name":"Warren","asset_description":"GS", "type":"Sale_Full","transaction_date":d(5),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Ron",  "last_name":"Wyden",    "asset_description":"JPM","type":"Sale_Full","transaction_date":d(4),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Sherrod","last_name":"Brown",  "asset_description":"GS", "type":"Sale_Full","transaction_date":d(3),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            # Spouse trade (lower weight — should score lower)
            {"first_name":"Tommy","last_name":"Tuberville","asset_description":"BA","type":"Purchase","transaction_date":d(16),"amount":"$15,001 - $50,000","filer_information":"Spouse"},
            # Older trades for decay testing
            {"first_name":"Mark", "last_name":"Warner",   "asset_description":"NVDA","type":"Purchase","transaction_date":d(75),"amount":"$100,001 - $250,000","filer_information":"Senator"},
            {"first_name":"Tommy","last_name":"Tuberville","asset_description":"LMT","type":"Purchase","transaction_date":d(80),"amount":"$50,001 - $100,000","filer_information":"Senator"},
            {"first_name":"Roger","last_name":"Wicker",   "asset_description":"RTX","type":"Purchase","transaction_date":d(70),"amount":"$50,001 - $100,000","filer_information":"Senator"},
        ]

        cutoff   = datetime.today() - timedelta(days=days_back)
        filtered = [
            r for r in records
            if datetime.strptime(r["transaction_date"], "%Y-%m-%d") > cutoff
        ]
        log.info("Demo data: %d trades", len(filtered))
        return filtered

    # ── Parser ────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: dict) -> Optional[Trade]:
        name = (
            f"{raw.get('first_name', '')} "
            f"{raw.get('last_name', raw.get('filer_name', ''))}".strip()
        )
        if not name:
            return None

        raw_ticker = (raw.get("asset_description") or raw.get("ticker") or "").strip()
        ticker     = raw_ticker.split()[0].upper() if raw_ticker else ""
        if not ticker or len(ticker) > 5 or not ticker.isalpha():
            return None

        t_raw = (raw.get("type") or "").lower()
        if any(w in t_raw for w in ("purchase", "buy")):
            side = "buy"
        elif any(w in t_raw for w in ("sale", "sell")):
            side = "sell"
        else:
            return None

        filer_info = (raw.get("filer_information") or "").lower()
        if "spouse" in filer_info:
            filer_type, filer_weight = "spouse", 0.5
        elif "dependent" in filer_info or "child" in filer_info:
            filer_type, filer_weight = "dependent", 0.3
        elif "joint" in filer_info:
            filer_type, filer_weight = "joint", 0.8
        else:
            filer_type, filer_weight = "senator", 1.0

        amount_raw = raw.get("amount") or raw.get("amount_range") or ""
        amount_mid = _AMOUNT_MIDS.get(amount_raw, 0)

        return Trade(
            senator        = name,
            ticker         = ticker,
            side           = side,
            date           = raw.get("transaction_date", ""),
            amount_range   = amount_raw,
            asset_type     = raw.get("asset_type", "Stock"),
            filer_type     = filer_type,
            filer_weight   = filer_weight,
            raw_amount_mid = amount_mid,
            sector         = TICKER_SECTORS.get(ticker, ""),
        )

    # ── Enrichment ────────────────────────────────────────────

    def _enrich(self, trades: list[Trade]) -> list[Trade]:
        first_seen: dict[tuple, str] = {}
        for t in sorted(trades, key=lambda x: x.date):
            key = (t.senator, t.ticker)
            if key not in first_seen:
                first_seen[key] = t.date
            elif t.side == "buy":
                t.is_addon = True

        day_sector: dict[tuple, list[Trade]] = defaultdict(list)
        for t in trades:
            if t.sector:
                day_sector[(t.senator, t.date, t.sector)].append(t)
        for bucket in day_sector.values():
            if len(bucket) >= 5:
                for t in bucket:
                    t.is_rebalance_noise = True

        return trades