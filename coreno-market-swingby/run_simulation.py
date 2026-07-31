from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from coreno_swingby.core import (
    AuditConfig,
    CorenoConfig,
    DEFAULT_TICKERS,
    StrategyConfig,
    compute_coreno_indicators,
    config_snapshot,
    run_false_positive_audit,
    run_strategy_backtest,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORENO market swing-by simulator")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default=(date.today() + timedelta(days=1)).isoformat())
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--placebo", type=int, default=1000)
    parser.add_argument("--surrogates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def download_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        list(DEFAULT_TICKERS.values()),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No market data downloaded from Yahoo Finance")
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        frame = raw[field].copy()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        frame = raw[[field]].copy()
    inverse = {ticker: name for name, ticker in DEFAULT_TICKERS.items()}
    frame = frame.rename(columns=inverse)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    required = {"SP500", "NIKKEI225", "VIX"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Required series unavailable: {sorted(missing)}")
    frame = frame.sort_index().dropna(how="all")
    frame = frame.reindex(pd.bdate_range(frame.index.min(), frame.index.max())).ffill(limit=2)
    return frame


def plot_results(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    strategy_returns: pd.DataFrame,
    audit: dict,
    output: Path,
) -> None:
    equity = (1 + strategy_returns.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    equity.plot(ax=ax, logy=True)
    ax.set_title("Equity curves (log scale)")
    ax.set_ylabel("Growth of 1.0")
    fig.tight_layout()
    fig.savefig(output / "equity_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    indicators[["CI", "Phi", "AC1"]].plot(ax=ax)
    alert_dates = audit["signal"].index[audit["signal"]["clustered_alert"]]
    for alert_date in alert_dates:
        ax.axvline(alert_date, alpha=0.08)
    ax.set_title("CORENO indicators and clustered alerts")
    fig.tight_layout()
    fig.savefig(output / "coreno_indicators.png", dpi=160)
    plt.close(fig)

    placebo = audit["placebo"]
    if not placebo.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        placebo["lift"].replace([np.inf, -np.inf], np.nan).dropna().hist(ax=ax, bins=40)
        ax.axvline(audit["summary"]["lift"], linewidth=2)
        ax.set_title("Circular-shift null distribution of alert lift")
        ax.set_xlabel("Lift")
        fig.tight_layout()
        fig.savefig(output / "placebo_lift.png", dpi=160)
        plt.close(fig)


def build_report(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    audit: dict,
    output: Path,
) -> None:
    summary = audit["summary"]
    latest = indicators.dropna(subset=["CI", "Phi", "AC1"]).iloc[-1]
    verdicts = [
        "PASS" if summary["placebo_p"] < 0.05 else "FAIL",
        "PASS" if summary["surrogate_p"] < 0.05 else "FAIL",
        "PASS" if summary["bootstrap_lift_lo"] > 1.0 else "FAIL",
        "PASS" if not summary["reverse_time_dominates"] else "FAIL",
        "PASS" if summary["fdr_significant_specs"] > 0 else "FAIL",
    ]
    overall = (
        "PROVISIONALLY SUPPORTED"
        if verdicts.count("PASS") >= 4
        else "NOT SUPPORTED / HIGH FALSE-POSITIVE RISK"
    )

    lines = [
        "# CORENO Market Swing-by Simulation Report",
        "",
        f"Data range: {prices.index.min().date()} to {prices.index.max().date()}",
        "",
        "## Primary falsification verdict",
        "",
        f"**{overall}**",
        "",
        f"- Clustered alerts: {int(summary['alerts'])}",
        f"- True positives: {int(summary['tp'])}",
        f"- False positives: {int(summary['fp'])}",
        f"- Precision: {summary['precision']:.3f}",
        f"- Recall: {summary['recall']:.3f}",
        f"- False-positive rate: {summary['fpr']:.3f}",
        f"- Lift over crisis prevalence: {summary['lift']:.3f}",
        f"- Crisis events captured: {int(summary['event_crises_captured'])} / {int(summary['event_crises'])}",
        f"- Event-level precision: {summary['event_precision']:.3f}",
        f"- Event-level recall: {summary['event_recall']:.3f}",
        f"- 95% block-bootstrap lift interval: [{summary['bootstrap_lift_lo']:.3f}, {summary['bootstrap_lift_hi']:.3f}]",
        f"- Circular-shift placebo p-value: {summary['placebo_p']:.4f}",
        f"- Phase-randomized surrogate p-value: {summary['surrogate_p']:.4f}",
        f"- Reverse-time lift: {summary['reverse_time_lift']:.3f}",
        f"- FDR-significant parameter specifications: {summary['fdr_significant_specs']} / {summary['fdr_total_specs']}",
        f"- False-positive categories: {summary['false_positive_categories']}",
        "",
        "## False-positive gates",
        "",
        f"1. Circular-shift label placebo: {verdicts[0]}",
        f"2. Phase-randomized market surrogate: {verdicts[1]}",
        f"3. Block-bootstrap lower lift > 1: {verdicts[2]}",
        f"4. Future association exceeds reverse-time association: {verdicts[3]}",
        f"5. At least one parameter specification survives BH-FDR 5%: {verdicts[4]}",
        "",
        "A failure is not repaired by selecting a more favorable threshold after seeing the full sample.",
        "",
        "## Latest computed state",
        "",
        f"- Date: {latest.name.date()}",
        f"- Phase: {latest['phase']}",
        f"- dCI/dt: {latest['dCI_dt']:.6f}",
        f"- lambda1: {latest['lambda1']:.4f}",
        f"- chi: {latest['chi']:.4f}",
        f"- AC1: {latest['AC1']:.4f}",
        f"- Phi: {latest['Phi']:.4f}",
        f"- CI: {latest['CI']:.4f}",
        f"- VIX: {latest['VIX']:.2f}",
        "",
        "## Backtest metrics",
        "",
        strategy_metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation limits",
        "",
        "- The CORENO variables are operational proxies, not physical gravity measurements.",
        "- Yahoo Finance revisions, missing holidays, local-currency returns, and index methodology changes can affect results.",
        "- The static index universe reduces constituent survivorship bias but introduces cross-market calendar and availability bias.",
        "- Crisis labels are drawdown definitions, not claims about economic causation.",
        "- Trading results are research outputs, not individualized investment advice.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.quick:
        args.bootstrap = min(args.bootstrap, 250)
        args.placebo = min(args.placebo, 250)
        args.surrogates = min(args.surrogates, 20)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prices = download_prices(args.start, args.end)
    prices.to_csv(output / "market_prices.csv")

    coreno_config = CorenoConfig()
    strategy_config = StrategyConfig()
    audit_config = AuditConfig(
        bootstrap_samples=args.bootstrap,
        placebo_samples=args.placebo,
        surrogate_samples=args.surrogates,
        seed=args.seed,
    )
    indicators = compute_coreno_indicators(prices, coreno_config)
    strategy_returns, strategy_metrics = run_strategy_backtest(
        prices, indicators, strategy_config
    )
    audit = run_false_positive_audit(prices, indicators, audit_config, coreno_config)

    indicators.to_csv(output / "daily_states.csv")
    strategy_returns.to_csv(output / "strategy_returns.csv")
    strategy_metrics.to_csv(output / "backtest_metrics.csv", index=False)
    for name in [
        "bootstrap",
        "placebo",
        "surrogates",
        "parameter_sweep",
        "walk_forward",
        "label_sensitivity",
        "false_positive_diagnostics",
        "crisis_events",
        "signal",
        "label",
    ]:
        audit[name].to_csv(output / f"{name}.csv")
    save_json(
        {
            "audit": audit["summary"],
            "config": config_snapshot(coreno_config, strategy_config, audit_config),
        },
        output / "summary.json",
    )
    plot_results(prices, indicators, strategy_returns, audit, output)
    build_report(prices, indicators, strategy_metrics, audit, output)
    print((output / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
