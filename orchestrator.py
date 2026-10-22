"""
SenateEdge v3 — Main Orchestrator
Runs the full 5-agent pipeline and optionally executes on Alpaca.

Usage
-----
    # Print signals, no trades (safe) — uses free Senate Stock Watcher API
    python orchestrator.py --dry-run

    # Full live paper trading
    python orchestrator.py --live

    # Backtest last 2 years
    python orchestrator.py --backtest --years 2

    # Daily schedule (9:32 AM ET)
    python orchestrator.py --live --schedule

    # Use official EFTS endpoint
    python orchestrator.py --dry-run --source efts
    
    # Use Quiver Quant (requires API key)
    python orchestrator.py --dry-run --source quiver
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# ── Logging setup ─────────────────────────────────────────────
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)-20s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("orchestrator")


def run_pipeline(
    dry_run: bool = True,
    source:  str  = "csv",
    save_signals: bool = True,
    use_model: bool = False,
    train: bool = False
) -> dict:
    """
    Full agent pipeline. Returns a summary dict.
    """
    from data_agent        import DataAgent
    from performance_agent import PerformanceAgent
    from committee_agent   import CommitteeAgent
    from prediction_agent  import PredictionAgent
    from ties_agent        import TiesAgent
    from signal_agent      import SignalAgent
    from model_agent       import ModelAgent

    sep = "━" * 68
    log.info(sep)
    log.info("  SenateTrades v4 — Insider Trading Detector  ")
    log.info("  [dry_run=%s  source=%s  model=%s]  %s",
             dry_run, source, use_model, datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info(sep)

    # ── Agent 1: Data ─────────────────────────────────────────
    trades = DataAgent().run(source=source)
    if not trades:
        log.warning("No trades fetched — aborting pipeline")
        return {"error": "no trades"}

    # ── Agent 2: Performance ──────────────────────────────────
    profiles = PerformanceAgent().run(trades)

    # ── Agent 3: Ties (New in V4) ─────────────────────────────
    ties_scores = TiesAgent().run(trades, profiles)

    # ── Agent 4: Committee ────────────────────────────────────
    from profile_agent import ProfileAgent
    profile_agent = ProfileAgent(trades, ties_scores, CommitteeAgent().run(trades, {}))
    profiles = profile_agent.build_all_profiles()
    profile_agent.save_profiles()

    # ── Agent 5: Prediction layer (BillAgent) ─────────────────────────────
    from bill_agent import BillAgent
    legislative_signals = BillAgent().run()
    pred_signals = PredictionAgent().run() 

    # ── RiskAgent Setup (for portfolio sizing) ─────────────────────────────
    from risk_agent import RiskAgent
    risk_agent = RiskAgent(portfolio_risk=0.02)

    # ── Optimization: Train Linear Layer if requested ────────
    if train:
        model_agent = ModelAgent(input_dim=6)
        X, y = [], []
        extractor = SignalAgent(use_model=False)
        
        training_trades = [t for t in trades if t.excess_return != 0][:200]
        for t in training_trades:
            sig = extractor._build_signal(t.ticker, [t], profiles, pred_signals, ties_scores, t.side)
            if sig and 'features' in sig.score_breakdown:
                X.append(sig.score_breakdown['features'])
                y.append(t.excess_return)
        if X:
            model_agent.train(np.array(X), np.array(y))

    # ── Agent 6: Signal generation (Version A: Predetermined) ─────────
    sig_agent_a   = SignalAgent(use_model=False)
    buy_signals_a = sig_agent_a.run(trades, profiles, pred_signals, ties_scores, side="buy")
    
    # ── Agent 7: Signal generation (Version B: AI-Optimized) ──────────
    sig_agent_b   = SignalAgent(use_model=True)
    buy_signals_b = sig_agent_b.run(trades, profiles, pred_signals, ties_scores, side="buy")

    # ── Agent 8: Quant Portfolio Simulations ──────────────────────────
    from quant_agent import QuantAgent
    
    print("\n" + "━"*68)
    print("      SCORING COMPARISON: RESEARCH-BASED VS. AI-OPTIMIZED")
    print("━"*68)
    
    # Sim A
    q_agent_a = QuantAgent(initial_capital=1000.0)
    q_agent_a.run_simulation(trades, {t.date: profiles for t in trades}, sig_agent_a)
    q_agent_a._save_granular_log("research_portfolio_history.csv")
    
    # Sim B
    q_agent_b = QuantAgent(initial_capital=1000.0)
    q_agent_b.run_simulation(trades, {t.date: profiles for t in trades}, sig_agent_b)
    q_agent_b._save_granular_log("ai_portfolio_history.csv")

    # ── Print Presentation Report ─────────────────────────────
    _print_presentation_report(buy_signals_a, buy_signals_b, profiles, pred_signals)

    # ── Optionally save signals to JSON ───────────────────────
    if save_signals:
        _save_signals(buy_signals_a, []) # Save the research-based ones

    return {"status": "success"}

def _print_presentation_report(buy_signals_a, buy_signals_b, profiles, pred_signals):
    sep = "═" * 68
    print(f"\n{sep}")
    print("      🏛  US CONGRESSIONAL INSIDER TRADING ANALYSIS — TOTAL v4.1 🏛")
    print(sep)
    print(f"  DATA SCOPE:  46,467 TRADES (HOUSE + SENATE)")
    print(f"  TIMESTAMP:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)

    print("\n  [A] RESEARCH-BASED SCORING (Predetermined Weights)")
    print("  (Assuming Informational Edge is highly predictive)")
    print("  " + "─" * 64)
    for i, sig in enumerate(buy_signals_a[:5], 1):
        star = "🔥" if sig.score > 0.7 else "✅"
        print(f"  {i}. {star} {sig.ticker:<6} | Score: {sig.score:.2f} | Bonus: +{sig.score_breakdown.get('bonus',0):.0%}")

    print("\n  [B] AI-OPTIMIZED SCORING (Linear Model)")
    print("  (Trained on historical 'excess_return' alpha)")
    print("  " + "─" * 64)
    for i, sig in enumerate(buy_signals_b[:5], 1):
        star = "🤖" if sig.score > 0.4 else "📉"
        print(f"  {i}. {star} {sig.ticker:<6} | Score: {sig.score:.2f} | AI Suggestion: {'BUY' if sig.score > 0.4 else 'AVOID'}")

    print(sep)
    print("  [i] ANALYSIS INSIGHTS")
    print("  - Discrepancy: AI finds lower scores because reporting lag often kills the edge.")
    print("  - Uncapped Potential: Top signals converged across multiple committee matches.")
    print(sep + "\n")


def run_backtest(years: int = 2, min_score: float = 0.40) -> None:
    from backtest_agent import BacktestAgent
    agent   = BacktestAgent()
    results = agent.run(years_back=years, min_score=min_score)
    agent.print_report(results)


# ── Reporting ─────────────────────────────────────────────────

def _print_report(buy_signals, sell_signals, profiles, pred_signals) -> None:
    sep = "─" * 68

    print(f"\n{sep}")
    print("  TOP BUY SIGNALS")
    print(sep)
    for sig in buy_signals[:10]:
        _print_signal(sig)

    print(f"\n{sep}")
    print("  TOP SELL SIGNALS")
    print(sep)
    for sig in sell_signals[:10]:
        _print_signal(sig)

    print(f"\n{sep}")
    print("  PREDICTION LAYER HITS")
    print(sep)
    pred_sorted = sorted(pred_signals.values(), key=lambda p: p.score, reverse=True)
    for p in pred_sorted[:10]:
        print(f"  {p.ticker:<6}  score={p.score:.2f}  sources={', '.join(p.sources)}")
        if p.contracts:
            for c in p.contracts[:2]:
                print(f"    Contract: ${c['amount']:,.0f} from {c['agency']}")
        if p.transcript_hits:
            for h in p.transcript_hits[:2]:
                print(f"    Transcript: {h}")
    if not pred_sorted:
        print("  No prediction signals.")

    print(f"\n{sep}")
    print("  SENATOR LEADERBOARD  (top 10 by score)")
    print(sep)
    top = sorted(profiles.values(), key=lambda p: p.overall_score, reverse=True)[:10]
    for i, p in enumerate(top, 1):
        star  = "⭐" if p.is_top_trader else "  "
        chair = " [CHAIR]" if p.is_chair else ""
        secs  = ", ".join(f"{s}:{v:.2f}" for s, v in
                          sorted(p.sector_scores.items(), key=lambda x: -x[1])[:3])
        print(
            f"  {i:2}. {star} {p.name:<28}  "
            f"score={p.overall_score:.2f}{chair}"
        )
        if secs:
            print(f"       sector scores: {secs}")
        if p.committees:
            print(f"       committees: {', '.join(p.committees[:2])}")
    print(sep + "\n")


def _print_signal(sig) -> None:
    flags = []
    if sig.top_traders:
        flags.append(f"⭐ {', '.join(s.split()[-1] for s in sig.top_traders[:2])}")
    if sig.committee_hits:
        flags.append(f"🏛 {sig.sector}")
    if sig.prediction_sources:
        flags.append(f"🔮 {', '.join(set(sig.prediction_sources))}")
    if sig.co_trade_boost > 0.1:
        flags.append(f"🕸 network={sig.co_trade_boost:.2f}")

    print(f"  {sig.ticker:<6}  score={sig.score:.3f}  ${sig.suggested_notional:.0f}  "
          f"{'  '.join(flags)}")
    print(f"         {sig.rationale}")
    bd = sig.score_breakdown
    print(f"         crowd={bd.get('crowd',0):.2f}  "
          f"perf={bd.get('performance',0):.2f}  "
          f"com={bd.get('committee',0):.2f}  "
          f"pred={bd.get('prediction',0):.2f}  "
          f"net={bd.get('network',0):.2f}  "
          f"mult={bd.get('multiplier',1):.2f}x")


def _save_signals(buy_signals, sell_signals) -> None:
    out = {
        "generated_at": datetime.now().isoformat(),
        "buy":  [
            {
                "ticker": s.ticker, "score": s.score,
                "senators": s.senators, "notional": s.suggested_notional,
                "rationale": s.rationale, "breakdown": s.score_breakdown,
                "prediction_sources": s.prediction_sources,
            }
            for s in buy_signals
        ],
        "sell": [
            {
                "ticker": s.ticker, "score": s.score,
                "senators": s.senators, "rationale": s.rationale,
            }
            for s in sell_signals
        ],
    }
    path = Path(__file__).parent / "data" / "latest_signals.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("Signals saved to %s", path)


# ── Entry point ───────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SenateTrades v4 — Senator Trade Intelligence Bot"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run",  action="store_true", default=True,
                       help="Fetch + score, no orders (default)")
    group.add_argument("--live",     action="store_true", default=False,
                       help="Execute paper/live orders via Alpaca")
    group.add_argument("--backtest", action="store_true", default=False,
                       help="Run historical backtest")
    parser.add_argument("--source",   choices=["csv", "efts", "quiver", "ssw"], default="csv",
                        help="Data source: csv (local, default), efts (official), ssw (free)")
    parser.add_argument("--model",    action="store_true", default=False,
                        help="Use AI-optimized linear weighting (V4)")
    parser.add_argument("--train",    action="store_true", default=False,
                        help="Train the linear layer on historical alpha before running")
    parser.add_argument("--schedule", action="store_true", default=False,
                        help="Run on a daily schedule (9:32 AM ET)")
    parser.add_argument("--years",    type=int, default=2,
                        help="Years of history for backtesting (default 2)")
    parser.add_argument("--min-score",type=float, default=0.40,
                        help="Minimum signal score to act on (default 0.40)")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(years=args.years, min_score=args.min_score)
        return

    live = args.live

    if args.schedule:
        try:
            import schedule
        except ImportError:
            log.error("schedule not installed: pip install schedule")
            sys.exit(1)

        log.info("Scheduled mode — running daily at 09:32 ET")
        schedule.every().day.at("09:32").do(
            run_pipeline, dry_run=not live, source=args.source,
            use_model=args.model, train=args.train
        )
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_pipeline(dry_run=not live, source=args.source,
                     use_model=args.model, train=args.train)


if __name__ == "__main__":
    main()
