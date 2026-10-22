"""
QuantAgent v2 — Robust trading mechanics inspired by ECON2.0.
Handles position sizing, cash management, and realistic equity tracking.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_agent import Trade
from signal_agent import SignalAgent, Signal

log = logging.getLogger("QuantAgent")

class QuantAgent:
    NAME = "QuantAgent"

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        self.granular_history = []
        
        # INSTITUTIONAL RISK CONTROLS (Prevents unrealistic growth)
        self.max_position_size = 0.05  # Max 5% of portfolio per trade
        self.max_positions = 15        # Diversification
        self.min_trade_amount = 25.0
        self.stop_loss = -0.07         # 7% Stop loss
        self.take_profit = 0.25        # 25% Take profit
        
        # MARKET BASELINE (Realistic 10% annual growth for unused cash)
        self.daily_market_return = (1.10 ** (1/365)) - 1

    def run_simulation(self, trades: List[Trade], profiles_by_date: Dict[str, Dict], signal_agent: SignalAgent):
        log.info("Starting Realistic Quant Simulation ($%.2f capital)…", self.cash)
        trades.sort(key=lambda x: x.date)
        unique_dates = sorted(list({t.date for t in trades}))

        last_date = None
        
        # Initial state
        self._log_state(unique_dates[0], "INIT")

        for current_date in unique_dates:
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            
            # --- 1. MARKET GROWTH (Beta) ---
            if last_date:
                days_passed = (current_dt - datetime.strptime(last_date, "%Y-%m-%d")).days
                market_factor = (1 + self.daily_market_return) ** days_passed
                self.cash *= market_factor 
                for ticker in self.positions:
                    self.positions[ticker]["amount"] *= market_factor 
            
            # --- 2. EXIT MANAGEMENT ---
            self._process_exits(current_dt)
            
            # --- 3. DAILY SNAPSHOT ---
            total_value = self.get_total_value()
            self.equity_curve.append((current_date, total_value))

            # --- 4. ENTRY MANAGEMENT ---
            today_trades = [t for t in trades if t.date == current_date and t.side == "buy"]
            if today_trades:
                profiles = profiles_by_date.get(current_date, {})
                ticker_groups = {}
                for t in today_trades:
                    ticker_groups.setdefault(t.ticker, []).append(t)

                daily_signals = []
                for ticker, t_list in ticker_groups.items():
                    sig = signal_agent._build_signal(ticker, t_list, profiles, {}, {}, "buy")
                    # Higher conviction threshold for institutional quality
                    if sig and sig.score > 0.65: 
                        daily_signals.append(sig)
                
                # Prioritize strongest signals
                daily_signals.sort(key=lambda x: x.score, reverse=True)
                for sig in daily_signals:
                    self._execute_buy(sig, current_date)
            
            last_date = current_date

        self._print_results()

    def _log_state(self, date_str: str, action: str, ticker: str = ""):
        pos_val = sum(p["amount"] for p in self.positions.values())
        total = self.cash + pos_val
        self.granular_history.append({
            "Date": date_str,
            "Action": action,
            "Ticker": ticker,
            "Cash": round(self.cash, 2),
            "Positions_Value": round(pos_val, 2),
            "Total_Equity": round(total, 2),
            "Growth_Pct": round(((total - self.initial_capital) / self.initial_capital) * 100, 2)
        })

    def _execute_buy(self, sig: Signal, date_str: str):
        if sig.ticker in self.positions or len(self.positions) >= self.max_positions:
            return
        
        portfolio_value = self.get_total_value()
        # Conservative position sizing (max 5%)
        size_pct = min(self.max_position_size, (sig.score * 0.05)) 
        dollar_amount = min(portfolio_value * size_pct, self.cash * 0.5)
        
        if dollar_amount < self.min_trade_amount: return
        
        # Use historical alpha from data
        avg_alpha = sum(t.excess_return for t in sig.senators_trades) / len(sig.senators_trades)
        
        self.cash -= dollar_amount
        self.positions[sig.ticker] = {
            "amount": dollar_amount, 
            "entry_date": date_str, 
            "alpha": avg_alpha / 100.0,
            "score": sig.score
        }
        
        self.trade_history.append({"date": date_str, "ticker": sig.ticker, "action": "BUY", "amount": dollar_amount})
        self._log_state(date_str, "BUY", sig.ticker)

    def _process_exits(self, current_dt: datetime):
        to_remove = []
        date_str = current_dt.strftime("%Y-%m-%d")
        for ticker, pos in self.positions.items():
            # Standard 30-day holding period
            if (current_dt - datetime.strptime(pos["entry_date"], "%Y-%m-%d")).days >= 30:
                alpha_gain = max(self.stop_loss, min(self.take_profit, pos["alpha"]))
                realized_val = pos["amount"] * (1 + alpha_gain)
                
                self.cash += realized_val
                self.trade_history.append({"date": date_str, "ticker": ticker, "action": "SELL", "pnl": realized_val - pos["amount"], "return": alpha_gain})
                to_remove.append(ticker)
                
                self._log_state(date_str, "SELL", ticker)

        for t in to_remove: del self.positions[t]

    def _save_granular_log(self, filename: str = "granular_portfolio_history.csv"):
        df = pd.DataFrame(self.granular_history)
        # Ensure directory exists
        Path("logs").mkdir(exist_ok=True)
        path = Path("logs") / filename
        df.to_csv(path, index=False)
        log.info("Granular portfolio history saved to %s (Rows: %d)", path, len(df))

    def get_total_value(self):
        return self.cash + sum(p["amount"] for p in self.positions.values())

    def _print_results(self):
        final_val = self.get_total_value()
        total_ret = (final_val - self.initial_capital) / self.initial_capital
        print("\n" + "═"*68 + "\n      📈  REALISTIC QUANT AUDIT RESULTS  📈\n" + "═"*68)
        print(f"Final Value: ${final_val:,.2f} | Total Return: {total_ret:+.2%}")
        sells = [t for t in self.trade_history if t["action"] == "SELL"]
        if sells:
            print(f"Win Rate: {sum(1 for t in sells if t['pnl'] > 0) / len(sells):.1%}")
            print(f"Closed Trades: {len(sells)}")
        
        # Save simple plot
        dates = [d for d, v in self.equity_curve]
        values = [v for d, v in self.equity_curve]
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values)
        out_path = Path("visuals/portfolio_performance.png")
        out_path.parent.mkdir(exist_ok=True)
        plt.savefig(out_path)
