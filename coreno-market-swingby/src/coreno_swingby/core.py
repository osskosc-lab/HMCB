"""Public API for the CORENO market simulator."""

from .audit import (
    AuditConfig,
    classification_metrics,
    cluster_alerts,
    event_capture_metrics,
    make_future_drawdown_label,
    run_false_positive_audit as _run_false_positive_audit,
)
from .indicators import (
    DEFAULT_TICKERS,
    CorenoConfig,
    compute_coreno_indicators,
)
from .strategy import StrategyConfig, run_strategy_backtest
from .util import config_snapshot, save_json


def run_false_positive_audit(
    prices,
    indicators,
    config: AuditConfig | None = None,
    coreno_config: CorenoConfig | None = None,
):
    """Run the audit and enforce right-censoring-safe alert accounting.

    Alerts inside the final prediction horizon cannot yet be labelled true or
    false. They remain visible as ``censored_unscorable`` observations, but are
    excluded from false-positive counts and event-level precision.
    """
    cfg = config or AuditConfig()
    result = _run_false_positive_audit(prices, indicators, cfg, coreno_config)

    label = result["label"]["future_crisis"]
    signal = result["signal"]["clustered_alert"]
    crisis_events = result["crisis_events"]["crisis_crossing"]
    scorable_signal = signal & label.notna()
    result["summary"].update(
        event_capture_metrics(scorable_signal, crisis_events, cfg.horizon)
    )

    diagnostics = result["false_positive_diagnostics"].copy()
    if diagnostics.empty:
        categories = {}
        censored_alerts = 0
    else:
        censored = diagnostics["future_min_return"].isna()
        diagnostics.loc[censored, "category"] = "censored_unscorable"
        false_positive_rows = diagnostics.loc[~censored]
        categories = false_positive_rows["category"].value_counts().to_dict()
        censored_alerts = int(censored.sum())

    result["false_positive_diagnostics"] = diagnostics
    result["summary"]["false_positive_categories"] = categories
    result["summary"]["censored_alerts"] = censored_alerts

    counted_false_positives = int(sum(categories.values()))
    expected_false_positives = int(result["summary"]["fp"])
    if counted_false_positives != expected_false_positives:
        raise RuntimeError(
            "False-positive diagnostic count does not match the primary "
            f"confusion matrix: {counted_false_positives} != "
            f"{expected_false_positives}"
        )
    return result


__all__ = [
    "AuditConfig",
    "CorenoConfig",
    "DEFAULT_TICKERS",
    "StrategyConfig",
    "classification_metrics",
    "cluster_alerts",
    "compute_coreno_indicators",
    "config_snapshot",
    "make_future_drawdown_label",
    "run_false_positive_audit",
    "run_strategy_backtest",
    "save_json",
]
