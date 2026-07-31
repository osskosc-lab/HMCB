"""Public API for the CORENO market simulator."""

import numpy as np
import pandas as pd

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


def _f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _add_f1_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    denominator = frame["precision"] + frame["recall"]
    frame["f1"] = np.where(
        denominator > 0,
        2.0 * frame["precision"] * frame["recall"] / denominator,
        0.0,
    )
    return frame


def _rank_auc(label: pd.Series, score: pd.Series) -> float:
    aligned = pd.concat([label.rename("label"), score.rename("score")], axis=1).dropna()
    if aligned.empty:
        return np.nan
    observed = aligned["label"].astype(bool).to_numpy()
    values = aligned["score"].astype(float)
    positives = int(observed.sum())
    negatives = int((~observed).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = values.rank(method="average").to_numpy()
    rank_sum = float(ranks[observed].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _continuous_coreno_score(indicators: pd.DataFrame) -> pd.Series:
    slope_scale = (
        indicators["dCI_dt"]
        .abs()
        .rolling(252, min_periods=60)
        .median()
        .shift(1)
        .replace(0.0, np.nan)
    )
    slope_ratio = (indicators["dCI_dt"] / slope_scale).clip(-20, 20)
    slope_component = 1.0 / (1.0 + np.exp(-slope_ratio))
    ci_component = (
        indicators["CI"] / indicators["ci_high_threshold"].replace(0.0, np.nan)
    ).clip(0, 1)
    vix_component = 1.0 - (
        indicators["VIX"] / indicators["vix_threshold"].replace(0.0, np.nan)
    ).clip(0, 2) / 2.0
    components = pd.concat(
        [
            (indicators["Phi"] / 4.0).clip(0, 1).rename("phi"),
            ci_component.rename("ci"),
            ((indicators["AC1"] + 1.0) / 2.0).clip(0, 1).rename("ac1"),
            indicators["chi"].clip(0, 1).rename("chi"),
            slope_component.rename("slope"),
            vix_component.clip(0, 1).rename("quiet_vix"),
        ],
        axis=1,
    )
    return components.mean(axis=1, skipna=False).rename("coreno_score")


def _lead_time_metrics(
    alerts: pd.Series,
    crisis_events: pd.Series,
    horizon: int,
) -> dict[str, float]:
    alert_positions = np.flatnonzero(alerts.fillna(False).to_numpy(bool))
    event_positions = np.flatnonzero(crisis_events.fillna(False).to_numpy(bool))
    lead_times: list[int] = []
    for event in event_positions:
        candidates = alert_positions[
            (alert_positions >= event - horizon) & (alert_positions < event)
        ]
        if len(candidates):
            lead_times.append(int(event - candidates.max()))
    if not lead_times:
        return {
            "mean_lead_days": np.nan,
            "median_lead_days": np.nan,
            "min_lead_days": np.nan,
            "max_lead_days": np.nan,
        }
    values = np.asarray(lead_times, dtype=float)
    return {
        "mean_lead_days": float(values.mean()),
        "median_lead_days": float(np.median(values)),
        "min_lead_days": float(values.min()),
        "max_lead_days": float(values.max()),
    }


def run_false_positive_audit(
    prices,
    indicators,
    config: AuditConfig | None = None,
    coreno_config: CorenoConfig | None = None,
):
    """Run the audit with censoring-safe and benchmarked alert accounting.

    Alerts inside the final prediction horizon cannot yet be labelled true or
    false. They remain visible as ``censored_unscorable`` observations, but are
    excluded from false-positive counts, event precision, and annual rates.
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
    result["summary"].update(
        _lead_time_metrics(scorable_signal, crisis_events, cfg.horizon)
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

    summary = result["summary"]
    summary["f1"] = _f1(summary["precision"], summary["recall"])
    observed_years = summary["n"] / 252.0
    summary["alerts_per_year"] = (
        float(summary["alerts"] / observed_years) if observed_years > 0 else np.nan
    )
    summary["false_alarms_per_year"] = (
        float(summary["fp"] / observed_years) if observed_years > 0 else np.nan
    )

    coreno_score = _continuous_coreno_score(indicators)
    result["signal"]["coreno_score"] = coreno_score
    summary["coreno_roc_auc"] = _rank_auc(label, coreno_score)
    summary["vix_roc_auc"] = _rank_auc(label, indicators["VIX"])
    summary["auc_difference_vs_vix"] = (
        float(summary["coreno_roc_auc"] - summary["vix_roc_auc"])
        if np.isfinite(summary["coreno_roc_auc"])
        and np.isfinite(summary["vix_roc_auc"])
        else np.nan
    )

    vix_signal = cluster_alerts(
        (indicators["VIX"] > indicators["vix_threshold"]).fillna(False),
        cfg.alert_cooldown,
    )
    vix_metrics = classification_metrics(vix_signal, label)
    vix_metrics["f1"] = _f1(vix_metrics["precision"], vix_metrics["recall"])
    for key in ["alerts", "tp", "fp", "precision", "recall", "fpr", "f1", "lift"]:
        summary[f"vix_{key}"] = vix_metrics[key]

    for key in [
        "bootstrap",
        "placebo",
        "surrogates",
        "parameter_sweep",
        "walk_forward",
        "label_sensitivity",
    ]:
        result[key] = _add_f1_column(result[key])
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
