"""
BacktestAgent — validates the full signal pipeline on historical data.
Fixes look-ahead bias by using a rolling window for senator performance scoring.
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger("BacktestAgent")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False


@dataclass
class BacktestTrade:
    ticker:       str
    senator:      str
    entry_date:   str
    exit_date:    str
    entry_price:  float
    exit_price:   float
    signal_score: float
    return_pct:   float = 0.0
    spy_return:   float = 0.0
    alpha:        float = 0.0
    sector:       str   = ""

@dataclass
class BacktestResults:
    trades:             list[BacktestTrade] = field(default_factory=list)
    total_trades:       int   = 0
    winning_trades:     int   = 0
    win_rate:           float = 0.0
    avg_return:         float = 0.0
    avg_alpha:          float = 0.0
    sharpe_ratio:       float = 0.0
    max_drawdown:       float = 0.0
    total_return:       float = 0.0
    spy_total_return:   float = 0.0
    by_senator:         dict  = field(default_factory=dict)
    by_sector:          dict  = field(default_factory=dict)
    by_score_bucket:    dict  = field(default_factory=dict)


class BacktestAgent:
    NAME = "BacktestAgent"
    EXIT_DAYS = 30

    def run(self, years_back: int = 2, min_score: float = 0.40) -> BacktestResults:
        log.info("[%s] Running rolling-window backtest…", self.NAME)

        from data_agent import DataAgent
        from performance_agent import PerformanceAgent
        from signal_agent import SignalAgent
        from config import TICKER_SECTORS

        data_agent   = DataAgent()
        perf_agent   = PerformanceAgent()
        signal_agent = SignalAgent(use_model=False)

        # 1. Fetch historical trades
        trades = data_agent.run(days_back=years_back*365, source="csv")
        if not trades:
            return BacktestResults()

        # 2. Sort by date for rolling backtest
        trades.sort(key=lambda x: x.date)
        unique_dates = sorted(list({t.date for t in trades}))
        
        bt_trades: list[BacktestTrade] = []
        price_cache = {}

        for current_date in unique_dates:
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            
            # Historical context: only trades that happened BEFORE today
            past_trades = [t for t in trades if t.date < current_date]
            if not past_trades:
                continue

            # Update performance profiles using ONLY past data (Fixes Look-Ahead Bias)
            profiles = perf_agent.run(past_trades)
            
            # Ticker signals for trades on THIS date
            today_trades = [t for t in trades if t.date == current_date and t.side == "buy"]
            
            # Deduplicate by ticker for the signal agent
            by_ticker = defaultdict(list)
            for t in today_trades:
                by_ticker[t.ticker].append(t)
            
            for ticker, t_list in by_ticker.items():
                # We skip prediction signals and ties for simplified backtest unless we have historical data
                sig = signal_agent._build_signal(ticker, t_list, profiles, {}, {}, "buy")
                
                if sig and sig.score >= min_score:
                    # Simulate entry at current date closing price
                    exit_dt = current_dt + timedelta(days=self.EXIT_DAYS)
                    
                    # Ensure exit date isn't in the future
                    if exit_dt > datetime.today() - timedelta(days=2):
                        continue

                    entry_price = perf_agent._get_return(ticker, current_dt, current_dt, price_cache) # Actually returns 0
                    # Helper: use performance agent's price fetching logic
                    p1 = self._get_historical_price(perf_agent, ticker, current_dt)
                    p2 = self._get_historical_price(perf_agent, ticker, exit_dt)
                    s1 = self._get_historical_price(perf_agent, "SPY", current_dt)
                    s2 = self._get_historical_price(perf_agent, "SPY", exit_dt)

                    if None in (p1, p2, s1, s2):
                        continue

                    ret     = (p2 - p1) / p1
                    spy_ret = (s2 - s1) / s1
                    alpha   = ret - spy_ret

                    bt_trades.append(BacktestTrade(
                        ticker       = ticker,
                        senator      = sig.senators[0],
                        entry_date   = current_date,
                        exit_date    = exit_dt.strftime("%Y-%m-%d"),
                        entry_price  = round(p1, 2),
                        exit_price   = round(p2, 2),
                        signal_score = sig.score,
                        return_pct   = round(ret * 100, 2),
                        spy_return   = round(spy_ret * 100, 2),
                        alpha        = round(alpha * 100, 2),
                        sector       = sig.sector
                    ))

        return self._compute_stats(bt_trades)

    def _get_historical_price(self, perf_agent, ticker, date):
        # reuse performance agent's cache and CSV/YF logic
        if perf_agent.price_df is not None:
            df = perf_agent.price_df
            ticker_df = df[df['Ticker'] == ticker]
            if not ticker_df.empty:
                idx = (ticker_df['Date'] - date).abs().idxmin()
                return float(ticker_df.loc[idx, 'Close'])
        return None # Simplified for backtest reliability

    @staticmethod
    def _compute_stats(bt_trades: list[BacktestTrade]) -> BacktestResults:
        if not bt_trades: return BacktestResults()
        n = len(bt_trades)
        wins = sum(1 for t in bt_trades if t.return_pct > 0)
        returns = [t.return_pct for t in bt_trades]
        alphas = [t.alpha for t in bt_trades]
        avg_ret = sum(returns) / n
        avg_alpha = sum(alphas) / n
        
        # Total Return
        total_ret = 1.0
        for r in returns: total_ret *= (1 + r / 100)
        total_ret = (total_ret - 1) * 100

        spy_total = 1.0
        for t in bt_trades: spy_total *= (1 + t.spy_return / 100)
        spy_total = (spy_total - 1) * 100

        # Score bucket breakdown
        buckets = {"0.40-0.55": [], "0.55-0.70": [], "0.70-0.85": [], "0.85+": []}
        for t in bt_trades:
            if t.signal_score < 0.55: buckets["0.40-0.55"].append(t)
            elif t.signal_score < 0.70: buckets["0.55-0.70"].append(t)
            elif t.signal_score < 0.85: buckets["0.70-0.85"].append(t)
            else: buckets["0.85+"].append(t)
            
        bucket_stats = {k: {"trades": len(v), "avg_alpha": sum(x.alpha for x in v)/len(v) if v else 0} for k, v in buckets.items()}

        return BacktestResults(
            trades = bt_trades, total_trades = n, winning_trades = wins,
            win_rate = round(wins/n, 3), avg_return = round(avg_ret, 2),
            avg_alpha = round(avg_alpha, 2), total_return = round(total_ret, 2),
            spy_total_return = round(spy_total, 2), by_score_bucket = bucket_stats
        )

    def print_report(self, results: BacktestResults):
        sep = "─" * 64
        print(f"\n{sep}")
        print("ROLLING-WINDOW BACKTEST RESULTS (No Look-Ahead)")
        print(sep)
        print(f"  Trades:           {results.total_trades}")
        print(f"  Win Rate:         {results.win_rate:.1%}")
        print(f"  Avg Alpha (30d):  {results.avg_alpha:+.2f}%")
        print(f"  Total Return:     {results.total_return:+.2f}%")
        print(f"  SPY Return:       {results.spy_total_return:+.2f}%")
        print(sep)
        for k, v in results.by_score_bucket.items():
            print(f"  Score {k:10} | {v['trades']:3} trades | Alpha: {v['avg_alpha']:+.2f}%")
        print(sep + "\n")
