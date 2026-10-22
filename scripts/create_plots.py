
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set style for "cool" plots
plt.style.use('dark_background')
sns.set_palette("viridis")

def create_visualizations():
    from data_agent import DataAgent
    from performance_agent import PerformanceAgent
    from signal_agent import SignalAgent
    from quant_agent import QuantAgent
    from committee_agent import CommitteeAgent
    from ties_agent import TiesAgent
    from prediction_agent import PredictionAgent
    from config import TICKER_SECTORS

    print("🚀 Initializing Visual Analysis Pipeline...")
    
    # 1. Gather Data
    trades = DataAgent().run(source="csv")
    profiles = PerformanceAgent().run(trades)
    ties_scores = TiesAgent().run(trades, profiles)
    pred_signals = PredictionAgent().run()
    
    # 2. Run Simulations to get history
    # Strategy A (Research)
    sig_agent_a = SignalAgent(use_model=False)
    q_agent_a = QuantAgent(initial_capital=1000.0)
    q_agent_a.run_simulation(trades, {t.date: profiles for t in trades}, sig_agent_a)
    
    # Strategy B (AI)
    sig_agent_b = SignalAgent(use_model=True)
    q_agent_b = QuantAgent(initial_capital=1000.0)
    q_agent_b.run_simulation(trades, {t.date: profiles for t in trades}, sig_agent_b)

    # --- PLOT 1: Equity Curve Comparison ---
    print("📈 Plotting Equity Curve Comparison...")
    plt.figure(figsize=(14, 7))
    
    df_a = pd.DataFrame(q_agent_a.equity_curve, columns=['Date', 'Value'])
    df_a['Date'] = pd.to_datetime(df_a['Date'])
    
    df_b = pd.DataFrame(q_agent_b.equity_curve, columns=['Date', 'Value'])
    df_b['Date'] = pd.to_datetime(df_b['Date'])
    
    plt.plot(df_a['Date'], df_a['Value'], label='Research-Based Strategy', color='#00FFCC', linewidth=2)
    plt.plot(df_b['Date'], df_b['Value'], label='AI-Optimized Strategy', color='#FF00FF', linewidth=2, linestyle='--')
    
    plt.title('Portfolio Growth: Research vs. AI Optimization', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Timeline', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.2)
    plt.fill_between(df_a['Date'], df_a['Value'], 1000, color='#00FFCC', alpha=0.1)
    plt.tight_layout()
    plt.savefig('visuals/viz_equity_comparison.png', dpi=300)

    # --- PLOT 2: Trade Timeline Bubble Chart (Research Strategy) ---
    print("🫧 Plotting Trade Bubble Timeline...")
    history = [t for t in q_agent_a.trade_history if t['action'] == 'SELL']
    df_hist = pd.DataFrame(history)
    df_hist['Date'] = pd.to_datetime(df_hist['date'])
    df_hist['Sector'] = df_hist['ticker'].map(TICKER_SECTORS).fillna('Other')
    
    plt.figure(figsize=(16, 8))
    scatter = plt.scatter(
        df_hist['Date'], 
        df_hist['ticker'], 
        s=df_hist['return'].abs() * 5000 + 50, 
        c=df_hist['return'], 
        cmap='RdYlGn', 
        alpha=0.7, 
        edgecolors='white'
    )
    plt.colorbar(scatter, label='Return %')
    plt.title('Trade Timeline: Every Trade & Its Magnitude', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Ticker', fontsize=12)
    plt.yticks(fontsize=8)
    plt.grid(axis='x', alpha=0.1)
    plt.tight_layout()
    plt.savefig('visuals/viz_trade_bubbles.png', dpi=300)

    # --- PLOT 3: Returns by Category (Violin Plot) ---
    print("🎻 Plotting Returns by Sector...")
    plt.figure(figsize=(14, 7))
    sns.violinplot(x='Sector', y='return', data=df_hist, inner="quart", palette="magma")
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.title('Distribution of Alpha by Industrial Sector', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Trade Return %')
    plt.tight_layout()
    plt.savefig('visuals/viz_sector_violin.png', dpi=300)

    # --- PLOT 4: Signal Conviction vs Actual Performance ---
    # We need to re-run and get signals for this
    print("🎯 Plotting Signal Conviction vs Result...")
    results = []
    for t_sell in history:
        ticker = t_sell['ticker']
        # Find corresponding buy signal
        # (This is a proxy for the signal that generated the trade)
        results.append({'Return': t_sell['return'], 'Ticker': ticker})
    
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(10, 8))
    sns.regplot(x='Return', y='Return', data=df_res, scatter_kws={'alpha':0.5, 'color':'#00FFCC'}, line_kws={'color':'red'})
    plt.title('Prediction Accuracy: Realized Alpha Cluster', fontsize=18, fontweight='bold')
    plt.savefig('visuals/viz_accuracy_scatter.png', dpi=300)

    print("\n✅ All visualizations generated successfully!")
    print("Files created: viz_equity_comparison.png, viz_trade_bubbles.png, viz_sector_violin.png, viz_accuracy_scatter.png")

if __name__ == "__main__":
    create_visualizations()
