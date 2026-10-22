"""
TiesAgent — scores geographic, political, and relationship ties between senators and tickers.

Features:
- Geographic Ties: Checks if the company is a major employer/headquartered in the senator's state.
- Political Ties: Uses a 'star' list and historical relevance.
- LLM Reasoning (Optional): Uses Anthropic Claude to reason about non-obvious ties.
- Caching: Persists LLM results to avoid redundant API costs.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from config import (
    ANTHROPIC_API_KEY,
    KNOWN_STAR_SENATORS,
    SENATOR_STATES,
    STATE_COMPANY_TIES,
    TICKER_SECTORS,
)

log = logging.getLogger("TiesAgent")

@dataclass
class TiesResult:
    senator: str
    ticker:  str
    score:   float
    rationale: str

class TiesAgent:
    NAME = "TiesAgent"
    CACHE_FILE = Path(__file__).parent / "data" / "ties_cache.json"

    def __init__(self):
        self.cache = self._load_cache()

    def run(self, trades, profiles) -> dict[tuple[str, str], float]:
        """
        Returns a mapping of (senator, ticker) -> ties_score.
        """
        log.info("Calculating geographic and political ties…")
        results = {}
        
        # We only need to score unique (senator, ticker) pairs from the trades
        unique_pairs = {(t.senator, t.ticker) for t in trades}
        
        for senator, ticker in unique_pairs:
            results[(senator, ticker)] = self.evaluate_tie(senator, ticker)
            
        self._save_cache()
        return results

    def evaluate_tie(self, senator: str, ticker: str) -> float:
        """
        Computes a tie score from 0 to 1.
        """
        if (senator, ticker) in self.cache:
            return self.cache[(senator, ticker)]

        score = 0.0
        reasons = []

        def normalize(n):
            return set(n.replace(",", "").lower().split())
        
        norm_senator = normalize(senator)

        # 1. Geographic Ties (Major presence in state)
        # Use fuzzy state lookup
        state = None
        for known_sen, s_state in SENATOR_STATES.items():
            if normalize(known_sen).issubset(norm_senator) or norm_senator.issubset(normalize(known_sen)):
                state = s_state
                break
        
        if state and ticker in STATE_COMPANY_TIES.get(state, []):
            score += 0.40
            reasons.append(f"Major {state} employer/presence")

        # 2. Political 'Star' Power
        is_star = False
        for star in KNOWN_STAR_SENATORS:
            if normalize(star).issubset(norm_senator) or norm_senator.issubset(normalize(star)):
                is_star = True
                break
        
        if is_star:
            score += 0.20
            reasons.append("Known high-performance trader")

        # 3. LLM Reasoning (Deep context)
        if ANTHROPIC_API_KEY:
            llm_score, llm_rationale = self._query_llm(senator, ticker)
            if llm_score is not None:
                # LLM score replaces rule-based if available as it is more nuanced
                score = llm_score
                reasons = [llm_rationale]

        final_score = round(min(score, 1.0), 3)
        self.cache[(senator, ticker)] = final_score
        return final_score

    def _query_llm(self, senator: str, ticker: str) -> tuple[Optional[float], str]:
        """
        Asks Claude to reason about the connection.
        """
        prompt = f"""Reason about the relationship between US Senator {senator} and the company {ticker}.
Consider:
- Geographic ties (HQ, major facilities, or jobs in their state).
- Legislative focus (do they lead committees or sponsor bills affecting this company's sector?).
- Historical trading patterns.

Provide a 'Tie Score' between 0.0 and 1.0 (where 0.5 is neutral, 0.8+ is a strong non-public information tie).
Return ONLY valid JSON:
{{
  "score": 0.85,
  "rationale": "Senator X represents State Y where Company Z has its largest manufacturing hub. Senator X is also a lead sponsor of the Z-Industry Support Act."
}}
"""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": "claude-3-haiku-20240307", # Use a cheaper model for batching
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=15)
            if resp.status_code == 200:
                content = resp.json()["content"][0]["text"]
                content = re.sub(r"```(?:json)?|```", "", content).strip()
                result = json.loads(content)
                return float(result.get("score", 0.5)), result.get("rationale", "")
        except Exception as e:
            log.warning(f"LLM Tie query failed for {senator}/{ticker}: {e}")
        
        return None, ""

    def _load_cache(self) -> dict:
        if self.CACHE_FILE.exists():
            try:
                raw = json.loads(self.CACHE_FILE.read_text())
                # Convert list of pairs back to dict keys
                return {(item[0], item[1]): item[2] for item in raw}
            except:
                return {}
        return {}

    def _save_cache(self):
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Convert dict keys to list of triplets for JSON
        serializable = [[k[0], k[1], v] for k, v in self.cache.items()]
        self.CACHE_FILE.write_text(json.dumps(serializable, indent=2))
