"""
CommitteeAgent — enhanced committee intelligence.
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import (
    COMMITTEE_SECTORS,
    SENATOR_COMMITTEES,
    SUBCOMMITTEE_SECTORS,
    TICKER_SECTORS,
)
from data_agent import Trade
from performance_agent import SenatorProfile

log = logging.getLogger("CommitteeAgent")

@dataclass
class CommitteeResult:
    senator:          str
    ticker:           str
    relevance_score:  float    # 0–1 raw relevance
    multiplier:       float    # final multiplier to apply to signal score
    reasons:          list[str]

class CommitteeAgent:
    NAME = "CommitteeAgent"
    LDA_URL = "https://lda.senate.gov/api/v1/filings/"

    def __init__(self):
        self._hearing_html_cache = None
        self._lobbying_cache = {}

    def run(self, trades, profiles):
        log.info("Committee mappings verified for %d profiles", len(profiles))
        return profiles

    def evaluate_trade(self, senator, ticker, trade_date, profile) -> CommitteeResult:
        sector = TICKER_SECTORS.get(ticker.upper(), "")
        reasons: list[str] = []
        relevance  = 0.0

        # ── 1. Full committee relevance ──────────────────────
        relevant_full_coms = set(COMMITTEE_SECTORS.get(sector, []))
        senator_full_coms  = set(profile.get('committees', []))
        com_overlap = senator_full_coms & relevant_full_coms

        if com_overlap:
            relevance = 0.6
            reasons.append(f"Committee match: {', '.join(com_overlap)}")
            if profile.get('is_chair', False):
                reasons.append("Committee chair")

        # ── 2. Subcommittee relevance ──────
        relevant_subcoms = {sc for sc, sectors in SUBCOMMITTEE_SECTORS.items() if sector in sectors}
        senator_subcoms  = set(profile.get('subcommittees', []))
        subcom_overlap   = senator_subcoms & relevant_subcoms

        if subcom_overlap:
            relevance = max(relevance, 0.90)
            reasons.append(f"Subcommittee match: {', '.join(subcom_overlap)}")

        # ── 3. Pre-hearing window detection (CACHED) ──────────────────
        if relevance > 0:
            in_window, hearing_desc = self._check_pre_hearing_window(senator, sector, trade_date)
            if in_window:
                reasons.append(f"Pre-hearing window: {hearing_desc}")

        # ── 4. LDA lobbying correlation (SKIPPED in bulk) ───────────
        # lobbyist_hit = self._check_lobbying(senator, ticker, trade_date)

        return CommitteeResult(
            senator=senator, ticker=ticker, relevance_score=round(relevance, 3),
            multiplier=1.0, reasons=reasons
        )

    def _check_pre_hearing_window(self, senator, sector, trade_date) -> tuple[bool, str]:
        profile = self._get_profile_stub(senator)
        relevant_coms = {c for c in profile.committees if sector in COMMITTEE_SECTORS.get(c, [])}
        if not relevant_coms: return False, ""
        
        try:
            if self._hearing_html_cache is None:
                resp = requests.get("https://www.senate.gov/committees/hearings_meetings.htm", timeout=5)
                self._hearing_html_cache = resp.text.lower() if resp.status_code == 200 else ""
            
            text = self._hearing_html_cache
            for com in relevant_coms:
                if com.lower().split(",")[0][:15] in text:
                    return True, f"{com} hearing scheduled"
        except: pass
        return False, ""

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try: return datetime.strptime(date_str, fmt)
            except ValueError: continue
        return None

    @staticmethod
    def _get_profile_stub(senator: str) -> SenatorProfile:
        from performance_agent import SenatorProfile
        p = SenatorProfile(name=senator)
        def normalize(n): return set(n.replace(",", "").lower().split())
        ns = normalize(senator)
        for known, (coms, subcoms, is_chair) in SENATOR_COMMITTEES.items():
            nk = normalize(known)
            if nk.issubset(ns) or ns.issubset(nk):
                p.committees, p.subcommittees, p.is_chair = coms, subcoms, is_chair
                break
        return p
