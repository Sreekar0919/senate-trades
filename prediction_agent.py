"""
PredictionAgent — generates UPSTREAM signals BEFORE senators file trades.

Sources:
  1. Senate committee hearing calendar (senate.gov/committees)
     → upcoming hearings that mention sectors we track
  2. USASpending.gov contract awards
     → large new contracts in tracked sectors
  3. Congress.gov API — recent bill introductions/markup
     → bills that positively/negatively affect a sector
  4. C-SPAN / congress.gov transcript scraping (optional, uses Anthropic API)
     → company name mentions in hearing testimony

Returns a dict: ticker → PredictionSignal
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import (
    ANTHROPIC_API_KEY,
    COMMITTEE_SECTORS,
    PREDICTION_LOOKBACK_DAYS,
    SENATOR_COMMITTEES,
    TICKER_SECTORS,
)

log = logging.getLogger("PredictionAgent")


@dataclass
class PredictionSignal:
    ticker:           str
    sector:           str
    score:            float        # 0–1
    sources:          list[str]    = field(default_factory=list)
    # What triggered it
    upcoming_hearings:list[str]    = field(default_factory=list)
    contracts:        list[dict]   = field(default_factory=list)
    bill_mentions:    list[str]    = field(default_factory=list)
    transcript_hits:  list[str]    = field(default_factory=list)


class PredictionAgent:
    NAME = "PredictionAgent"

    # USASpending.gov awards endpoint
    USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    # Congress.gov API (free, no key needed for basic)
    CONGRESS_BILLS_URL = "https://api.congress.gov/v3/bill"

    # Senate committee calendar page (scrape)
    SENATE_CALENDAR_URL = "https://www.senate.gov/committees/hearings_meetings.htm"

    # Anthropic API for transcript analysis
    ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

    # Tickers we actively track for contract signals
    DEFENSE_TICKERS = ["LMT","RTX","NOC","GD","HII","KTOS","CACI","SAIC","BAH","LDOS","PLTR","BA"]
    HEALTH_TICKERS  = ["JNJ","PFE","MRK","ABBV","UNH","AMGN","MRNA","GILD","BIIB","REGN","LLY","BMY"]
    ENERGY_TICKERS  = ["XOM","CVX","COP","SLB","NEE","OXY","MPC","VLO","EOG"]
    TECH_TICKERS    = ["AAPL","MSFT","GOOGL","META","NVDA","AMZN","PANW","CRWD","ZS","NET"]

    # Keyword → sector mapping for transcript/bill analysis
    SECTOR_KEYWORDS: dict[str, list[str]] = {
        "defense":  ["defense","military","pentagon","armed forces","national security",
                     "cybersecurity","drone","missile","navy","air force","army"],
        "health":   ["healthcare","pharmaceutical","drug pricing","medicare","medicaid",
                     "fda","clinical trial","vaccine","biotech","prescription"],
        "energy":   ["energy","oil","gas","renewables","pipeline","lng","solar","wind",
                     "fossil fuel","climate","epa","emissions"],
        "tech":     ["technology","artificial intelligence","ai","semiconductor","chip",
                     "data privacy","antitrust","big tech","5g","broadband"],
        "finance":  ["banking","financial","interest rate","fed","securities","crypto",
                     "fintech","credit","debt ceiling","dodd-frank"],
    }

    def run(self) -> dict[str, PredictionSignal]:
        log.info("Running prediction layer (hearings + contracts + bills + transcripts)…")
        signals: dict[str, PredictionSignal] = {}

        # 1. USASpending contract awards
        self._scan_contracts(signals)

        # 2. Senate hearing calendar
        self._scan_hearings(signals)

        # 3. Congress.gov recent bills
        self._scan_bills(signals)

        # 4. Transcript analysis via LLM (optional)
        if ANTHROPIC_API_KEY:
            self._scan_transcripts(signals)

        # Normalise scores
        for sig in signals.values():
            sig.score = round(min(sig.score, 1.0), 3)

        log.info("Prediction layer: %d ticker signals generated", len(signals))
        return signals

    # ── 1. USASpending contract awards ────────────────────────

    def _scan_contracts(self, signals: dict[str, PredictionSignal]) -> None:
        log.info("  Scanning USASpending.gov contract awards…")
        days_ago = (datetime.today() - timedelta(days=PREDICTION_LOOKBACK_DAYS))
        date_str = days_ago.strftime("%Y-%m-%d")

        # Sector buckets: NAICS codes → tickers we watch
        sector_queries = [
            ("defense",  "336411,336412,336413,336414,541330,541712", self.DEFENSE_TICKERS),
            ("health",   "325412,325411,621111,621210",               self.HEALTH_TICKERS),
            ("energy",   "211120,211130,221118,486110",               self.ENERGY_TICKERS),
            ("tech",     "334413,518210,541511,541512,541519",        self.TECH_TICKERS),
        ]

        for sector, naics_codes, watch_tickers in sector_queries:
            try:
                payload = {
                    "filters": {
                        "time_period": [{"start_date": date_str,
                                         "end_date": datetime.today().strftime("%Y-%m-%d")}],
                        "award_type_codes": ["A","B","C","D"],
                        "naics_codes": naics_codes.split(","),
                    },
                    "fields": ["recipient_name","award_amount","action_date",
                               "awarding_agency_name","description"],
                    "limit": 25,
                    "page": 1,
                    "sort": "award_amount",
                    "order": "desc",
                }
                resp = requests.post(
                    self.USASPENDING_URL, json=payload, timeout=20
                )
                if resp.status_code != 200:
                    continue

                awards = resp.json().get("results", [])
                for award in awards:
                    recipient = (award.get("recipient_name") or "").upper()
                    amount    = award.get("award_amount") or 0

                    # Match recipient name to a ticker we track
                    for ticker in watch_tickers:
                        if ticker in recipient or self._company_match(ticker, recipient):
                            sig = signals.setdefault(
                                ticker, PredictionSignal(ticker=ticker, sector=sector, score=0)
                            )
                            # Score boost proportional to contract size (log scale)
                            boost = min(0.20 + 0.05 * (len(str(int(amount))) - 6), 0.40)
                            sig.score += boost
                            sig.sources.append("usaspending")
                            sig.contracts.append({
                                "amount":  amount,
                                "agency":  award.get("awarding_agency_name", ""),
                                "date":    award.get("action_date", ""),
                            })
                            log.info(
                                "    Contract hit: %s  $%s  (%s)",
                                ticker, f"{amount:,.0f}", award.get("awarding_agency_name", "")
                            )
            except Exception as exc:
                log.debug("USASpending error for %s: %s", sector, exc)

    @staticmethod
    def _company_match(ticker: str, recipient_name: str) -> bool:
        """Fuzzy company name → ticker matching."""
        mappings = {
            "LMT":  ["LOCKHEED", "LOCKHEED MARTIN"],
            "RTX":  ["RAYTHEON", "RTX"],
            "NOC":  ["NORTHROP", "NORTHROP GRUMMAN"],
            "GD":   ["GENERAL DYNAMICS"],
            "BA":   ["BOEING"],
            "HII":  ["HUNTINGTON INGALLS"],
            "KTOS": ["KRATOS"],
            "CACI": ["CACI"],
            "SAIC": ["SCIENCE APPLICATIONS", "SAIC"],
            "BAH":  ["BOOZ ALLEN"],
            "LDOS": ["LEIDOS"],
            "PLTR": ["PALANTIR"],
            "PFE":  ["PFIZER"],
            "MRK":  ["MERCK"],
            "JNJ":  ["JOHNSON"],
            "ABBV": ["ABBVIE"],
            "MRNA": ["MODERNA"],
            "LLY":  ["LILLY", "ELI LILLY"],
            "XOM":  ["EXXON"],
            "CVX":  ["CHEVRON"],
            "COP":  ["CONOCOPHILLIPS"],
            "SLB":  ["SCHLUMBERGER", "SLB"],
            "NVDA": ["NVIDIA"],
            "MSFT": ["MICROSOFT"],
            "GOOGL":["GOOGLE", "ALPHABET"],
            "AMZN": ["AMAZON"],
            "PANW": ["PALO ALTO"],
            "CRWD": ["CROWDSTRIKE"],
        }
        aliases = mappings.get(ticker, [])
        return any(alias in recipient_name for alias in aliases)

    # ── 2. Senate hearing calendar ────────────────────────────

    def _scan_hearings(self, signals: dict[str, PredictionSignal]) -> None:
        """
        Scrape senate.gov committee calendar for upcoming hearings.
        Detect sector keywords in hearing titles → boost relevant tickers.
        """
        log.info("  Scanning Senate hearing calendar…")
        try:
            resp = requests.get(self.SENATE_CALENDAR_URL, timeout=15)
            if resp.status_code != 200:
                return
            text = resp.text.lower()

            for sector, keywords in self.SECTOR_KEYWORDS.items():
                for kw in keywords:
                    if kw in text:
                        # Identify tickers in this sector
                        sector_tickers = [
                            t for t, s in TICKER_SECTORS.items() if s == sector
                        ]
                        for ticker in sector_tickers[:5]:   # top 5 per sector
                            sig = signals.setdefault(
                                ticker, PredictionSignal(ticker=ticker, sector=sector, score=0)
                            )
                            if "hearings" not in sig.sources:
                                sig.score += 0.15
                                sig.sources.append("hearings")
                                sig.upcoming_hearings.append(
                                    f"Keyword '{kw}' found in Senate hearing calendar"
                                )
                        break   # one keyword per sector is enough
        except Exception as exc:
            log.debug("Hearing calendar scrape error: %s", exc)

    # ── 3. Congress.gov bills ─────────────────────────────────

    def _scan_bills(self, signals: dict[str, PredictionSignal]) -> None:
        """
        Use congress.gov API to find recently introduced or passed bills.
        Match bill titles to sectors and boost/penalise relevant tickers.
        """
        log.info("  Scanning recent congressional bills…")
        congress_key = os.getenv("CONGRESS_API_KEY", "")
        if not congress_key:
            log.debug("CONGRESS_API_KEY not set — skipping bill scan. "
                      "Get a free key at api.congress.gov")
            return

        try:
            from_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00Z")
            params = {
                "fromDateTime": from_date,
                "sort":         "updateDate+desc",
                "limit":        50,
                "api_key":      congress_key,
            }
            resp = requests.get(self.CONGRESS_BILLS_URL, params=params, timeout=15)
            if resp.status_code != 200:
                return

            bills = resp.json().get("bills", [])
            for bill in bills:
                title = (bill.get("title") or "").lower()
                for sector, keywords in self.SECTOR_KEYWORDS.items():
                    if any(kw in title for kw in keywords):
                        sector_tickers = [
                            t for t, s in TICKER_SECTORS.items() if s == sector
                        ]
                        for ticker in sector_tickers[:5]:
                            sig = signals.setdefault(
                                ticker, PredictionSignal(ticker=ticker, sector=sector, score=0)
                            )
                            if "bills" not in sig.sources:
                                sig.score += 0.10
                                sig.sources.append("bills")
                                sig.bill_mentions.append(bill.get("title", ""))
                        break
        except Exception as exc:
            log.debug("Congress.gov bill scan error: %s", exc)

    # ── 4. Transcript analysis via Claude ─────────────────────

    def _scan_transcripts(self, signals: dict[str, PredictionSignal]) -> None:
        """
        Fetches recent C-SPAN hearing transcripts (via congress.gov) and uses
        Claude to extract company name mentions with sentiment context.
        """
        log.info("  Running LLM transcript analysis via Claude…")
        congress_key = os.getenv("CONGRESS_API_KEY", "")
        if not congress_key:
            log.debug("CONGRESS_API_KEY not set — skipping transcript LLM analysis")
            return

        # Fetch recent congressional records
        try:
            from_date = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%dT00:00:00Z")
            url = "https://api.congress.gov/v3/congressional-record"
            params = {
                "fromDateTime": from_date,
                "limit":        10,
                "api_key":      congress_key,
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return

            records = resp.json().get("dailyCongressionalRecord", [])
            # Take text snippets from each record
            snippets = []
            for record in records[:3]:  # limit to avoid too many API calls
                for section in record.get("fullIssue", {}).get("sections", []):
                    text = section.get("text", "")[:2000]  # first 2000 chars
                    if text:
                        snippets.append(text)

            if not snippets:
                return

            combined = "\n\n---\n\n".join(snippets)
            self._llm_extract_companies(combined, signals)

        except Exception as exc:
            log.debug("Transcript fetch error: %s", exc)

    def _llm_extract_companies(
        self, transcript_text: str, signals: dict[str, PredictionSignal]
    ) -> None:
        """Call Claude to extract company mentions with buy/sell/neutral context."""
        all_tickers = list(TICKER_SECTORS.keys())
        prompt = f"""Analyze this congressional hearing transcript excerpt.
Extract every mention of publicly-traded companies from this ticker list: {all_tickers}

For each company found, determine from context whether the mention is:
- POSITIVE (benefiting legislation, contracts, favorable regulation)
- NEGATIVE (scrutiny, pricing legislation, penalties, investigation)  
- NEUTRAL (just mentioned without directional context)

Return ONLY valid JSON, no other text:
{{
  "mentions": [
    {{"ticker": "NVDA", "sentiment": "positive", "context": "brief reason", "confidence": 0.8}}
  ]
}}

Transcript:
{transcript_text[:3000]}"""

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key":    ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            }
            body = {
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 800,
                "messages":   [{"role": "user", "content": prompt}],
            }
            resp = requests.post(self.ANTHROPIC_URL, headers=headers,
                                 json=body, timeout=30)
            if resp.status_code != 200:
                return

            content = resp.json().get("content", [{}])[0].get("text", "")
            # Strip markdown fences if present
            content = re.sub(r"```(?:json)?|```", "", content).strip()
            data    = json.loads(content)

            for mention in data.get("mentions", []):
                ticker     = mention.get("ticker", "").upper()
                sentiment  = mention.get("sentiment", "neutral")
                context    = mention.get("context", "")
                confidence = float(mention.get("confidence", 0.5))

                if ticker not in TICKER_SECTORS:
                    continue

                sig = signals.setdefault(
                    ticker,
                    PredictionSignal(
                        ticker=ticker, sector=TICKER_SECTORS[ticker], score=0
                    ),
                )

                boost = 0.0
                if sentiment == "positive":
                    boost = 0.20 * confidence
                elif sentiment == "negative":
                    boost = -0.15 * confidence   # negative signal still useful for sells

                sig.score += boost
                if "transcript_llm" not in sig.sources:
                    sig.sources.append("transcript_llm")
                sig.transcript_hits.append(f"{sentiment.upper()}: {context}")
                log.info(
                    "    Transcript: %s  %s  (conf=%.2f) — %s",
                    ticker, sentiment, confidence, context
                )

        except Exception as exc:
            log.debug("LLM transcript analysis error: %s", exc)
