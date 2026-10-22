"""
SignalAgent — composite signal generation with Momentum (v4.1)
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from config import (
    BASE_ORDER_NOTIONAL,
    B_CHAIR,
    B_CONTRACT,
    B_PRE_HEARING,
    B_SUBCOMMITTEE,
    B_TOP_TRADER,
    B_TRANSCRIPT,
    KELLY_FRACTION,
    MAX_ORDER_NOTIONAL,
    MIN_SENATORS_TO_QUALIFY,
    MIN_SIGNAL_SCORE,
    MIN_SELL_SIGNAL_SCORE,
    TICKER_SECTORS,
    USE_KELLY_SIZING,
    W_COMMITTEE_SCORE,
    W_CROWD_SCORE,
    W_PERF_SCORE,
    W_PREDICTION_SCORE,
    W_TIES_SCORE,
    W_MOMENTUM_SCORE,
)
from data_agent import Trade
from performance_agent import SenatorProfile, PerformanceAgent
from committee_agent import CommitteeAgent
from prediction_agent import PredictionSignal
from ties_agent import TiesAgent
from model_agent import ModelAgent

log = logging.getLogger("SignalAgent")

@dataclass
class Signal:
    ticker:            str
    side:              str
    sector:            str
    score:             float
    senators:          list[str]
    committee_hits:    list[str]
    top_traders:       list[str]
    prediction_sources:list[str]
    co_trade_boost:    float     = 0.0
    suggested_notional:float     = BASE_ORDER_NOTIONAL
    rationale:         str       = ""
    score_breakdown:   dict      = field(default_factory=dict)
    model_score:       float     = 0.0
    senators_trades:   list[Trade] = field(default_factory=list)

class SignalAgent:
    NAME = "SignalAgent"

    def __init__(self, use_model: bool = False):
        self.committee_agent = CommitteeAgent()
        self.ties_agent      = TiesAgent()
        self.perf_agent      = PerformanceAgent() # Used for momentum
        self.model_agent     = ModelAgent(input_dim=6) if use_model else None
        self.use_model       = use_model

    def run(self, trades, profiles, prediction_signals, ties_scores, side="buy") -> list[Signal]:
        log.info("Generating %s signals (v4.1 Momentum Model)…", side)
        by_ticker = defaultdict(list)
        for t in [t for t in trades if t.side == side and not t.is_rebalance_noise]:
            by_ticker[t.ticker].append(t)

        signals = []
        for ticker, t_list in by_ticker.items():
            sig = self._build_signal(ticker, t_list, profiles, prediction_signals, ties_scores, side)
            if sig: signals.append(sig)

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def _build_signal(self, ticker, trades, profiles, pred_signals, ties_scores, side) -> Optional[Signal]:
        seen = {}
        for t in trades:
            if t.senator not in seen: seen[t.senator] = t
        unique_trades = list(seen.values())
        if len(unique_trades) < MIN_SENATORS_TO_QUALIFY: return None

        sector = TICKER_SECTORS.get(ticker, "")
        senators = [t.senator for t in unique_trades]

        # 1. Crowd (10%)
        crowd_score = min(len(unique_trades) / 5.0, 1.0)

        # 2. Performance (25%)
        perf_scores, top_traders = [], []
        for s in senators:
            p = profiles.get(s)
            # Accessing dictionary-based profile data
            score = p.get('avg_excess_return', 0.0) if p else (0.50 if side == "sell" else 0.35)
            perf_scores.append(score)
            if p and p.get('risk_profile') == 'AGGRESSIVE': top_traders.append(s)
        avg_perf = sum(perf_scores) / len(perf_scores)

        # 3. Committee (20%)
        best_com_score, applied_bonuses = 0.0, []
        for t in unique_trades:
            p = profiles.get(t.senator, SenatorProfile(name=t.senator))
            res = self.committee_agent.evaluate_trade(t.senator, ticker, t.date, p)
            best_com_score = max(best_com_score, res.relevance_score)
            if "chair" in str(res.reasons).lower(): applied_bonuses.append(B_CHAIR)
            if "subcommittee" in str(res.reasons).lower(): applied_bonuses.append(B_SUBCOMMITTEE)
            if "pre-hearing" in str(res.reasons).lower(): applied_bonuses.append(B_PRE_HEARING)

        # 4. Prediction (10%)
        pred = pred_signals.get(ticker)
        pred_score = pred.score if pred else 0.0
        if pred:
            if "usaspending" in pred.sources: applied_bonuses.append(B_CONTRACT)
            if "transcript_llm" in pred.sources: applied_bonuses.append(B_TRANSCRIPT)

        # 5. Ties (20%)
        rel_ties = [ties_scores.get((s, ticker), 0.0) for s in senators]
        avg_ties = sum(rel_ties) / len(rel_ties) if rel_ties else 0.0

        # 6. Momentum (15%) (New)
        # Use trade date for momentum context
        trade_dt = datetime.strptime(unique_trades[0].date, "%Y-%m-%d")
        momentum = self.perf_agent.get_momentum(ticker, trade_dt)
        # Normalize momentum (-20% to +20% -> 0 to 1)
        momentum_score = max(0.0, min(1.0, (momentum + 0.2) / 0.4))

        # Base Score
        features = [crowd_score, avg_perf, best_com_score, pred_score, avg_ties, momentum_score]
        raw_base = (W_CROWD_SCORE*crowd_score + W_PERF_SCORE*avg_perf + W_COMMITTEE_SCORE*best_com_score + 
                      W_PREDICTION_SCORE*pred_score + W_TIES_SCORE*avg_ties + W_MOMENTUM_SCORE*momentum_score)
        
        base_score = max(0.20, raw_base)

        if top_traders: applied_bonuses.append(B_TOP_TRADER)
        bonus_total = sum(set(applied_bonuses))
        
        final_score = base_score + (base_score * bonus_total)
        model_score = self.model_agent.predict(features) + (self.model_agent.predict(features) * bonus_total) if self.use_model else 0.0
        actual_score = round(min(model_score if self.use_model else final_score, 1.0), 4)

        return Signal(
            ticker=ticker, side=side, sector=sector, score=actual_score, senators=senators,
            committee_hits=[s for s in senators if profiles.get(s) and profiles[s].get('committees')],
            top_traders=top_traders, prediction_sources=pred.sources if pred else [],
            suggested_notional=self._kelly_notional(actual_score, avg_perf) if USE_KELLY_SIZING else BASE_ORDER_NOTIONAL,
            rationale=f"{side.upper()} {ticker} | Perf={avg_perf:.2f} | Mom={momentum:+.1%} | Bonus=+{bonus_total:.0%}",
            score_breakdown={"base": round(base_score, 3), "bonus": round(bonus_total, 3), "momentum": round(momentum, 3), "features": features},
            senators_trades=unique_trades
        )

    @staticmethod
    def _kelly_notional(score, win_rate):
        p = max(min(win_rate, 0.95), 0.30)
        kelly = (p * 2.5 - 1) / 1.5
        return round(min(BASE_ORDER_NOTIONAL + max(kelly, 0)*KELLY_FRACTION*(MAX_ORDER_NOTIONAL-BASE_ORDER_NOTIONAL), MAX_ORDER_NOTIONAL), 2)
