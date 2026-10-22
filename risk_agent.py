"""
RiskAgent — dynamic position sizing using volatility (ATR).
Ensures risk is uniform across trades regardless of stock volatility.
"""
import logging
import pandas as pd

log = logging.getLogger("RiskAgent")

class RiskAgent:
    NAME = "RiskAgent"

    def __init__(self, portfolio_risk: float = 0.02):
        self.portfolio_risk = portfolio_risk # Target 2% risk per position

    def calculate_size(self, ticker: str, price: float, portfolio_value: float, price_df: pd.DataFrame) -> int:
        """
        Calculates position size using ATR (Average True Range) risk-parity.
        """
        try:
            # Get last 14 days to calculate ATR
            ticker_df = price_df[price_df['Ticker'] == ticker].sort_values('Date').tail(15)
            if len(ticker_df) < 15:
                return 0

            # ATR = EMA of True Range
            tr = pd.concat([
                ticker_df['High'] - ticker_df['Low'],
                (ticker_df['High'] - ticker_df['Close'].shift(1)).abs(),
                (ticker_df['Low'] - ticker_df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.ewm(span=14).mean().iloc[-1]
            
            # Risk Parity Size = (Portfolio * Risk) / ATR
            dollar_risk = portfolio_value * self.portfolio_risk
            shares = int(dollar_risk / atr)
            
            # Limit position size to 15% of portfolio
            max_value = portfolio_value * 0.15
            if (shares * price) > max_value:
                shares = int(max_value / price)
                
            return max(0, shares)
        except Exception as e:
            log.warning(f"RiskAgent calculation error for {ticker}: {e}")
            return 0
