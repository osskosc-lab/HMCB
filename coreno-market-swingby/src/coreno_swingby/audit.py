from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .indicators import CorenoConfig, compute_coreno_indicators, expanding_quantile_shifted


@dataclass(frozen=True)
class AuditConfig:
    horizon: int = 60
    drawdown_threshold: float = 0.12
    alert_cooldown: int = 20
    bootstrap_samples: int = 1000
    placebo_samples: int = 1000
    surrogate_samples: int = 100
    block_length: int = 20
    seed: int = 42


def make_future_drawdown_label(price: pd.Series, horizon: int, threshold: float) -> pd.Series:
    values = price.to_numpy(dtype=float)
    labels = pd.Series(
        pd.NA,
        index=price.index,
        dtype="boolean",
        name=f"dd_{horizon}_{threshold:.3f}",
    )
    for position in range(len(values)):
        if not np.isfinite(values[position]) or position + horizon >= len(values):
            continue
        future = values[position + 1 : position + horizon + 1]
        future = future[np.isfinite(future)]
        if len(future) < max(5, int(0.8 * horizon)):
            continue
        labels.iloc[position] = bool(
            future.min() / values[position] - 1.0 <= -threshold
        )
    return labels


def future_min_return(price: pd.Series, horizon: int) -> pd.Series:
    values = price.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for position in range(len(values)):
        if not np.isfinite(values[position]) or position + horizon >= len(values):
            continue
        future = values[position + 1 : position + horizon + 1]
        future = future[np.isfinite(future)]
        if len(future) >= max(5, int(0.8 * horizon)):
            result[position] = future.min() / values[position] - 1.0
    return pd.Series(result, index=price.index, name="future_min_return")


def cluster_alerts(signal: pd.Series, cooldown: int) -> pd.Series:
    signal = signal.fillna(False).astype(bool)
    clustered = pd.Series(False, index=signal.index)
    last_position = -cooldown - 1
    for position, value in enumerate(signal.to_numpy()):
        if value and position - last_position > cooldown:
            clustered.iloc[position] = True
            last_position = position
    return clustered


def classification_metrics(signal: pd.Series, label: pd.Series) -> dict[str, float]:
    aligned = pd.concat([signal, label], axis=1).dropna()
    predicted = aligned.iloc[:, 0].astype(bool).to_numpy()
    observed = aligned.iloc[:, 1].astype(bool).to_numpy()
    tp = int(np.sum(predicted & observed))
    fp = int(np.sum(predicted & ~observed))
    fn = int(np.sum(~predicted & observed))
    tn = int(np.sum(~predicted & ~observed))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    prevalence = (tp + fn) / len(observed) if len(observed) else np.nan
    lift = precision / prevalence if prevalence and np.isfinite(precision) else np.nan
    odds_ratio, p_value = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return {
        "n": float(len(observed)),
        "alerts": float(tp + fp),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "prevalence": float(prevalence),
        "lift": float(lift),
        "odds_ratio": float(odds_ratio),
        "fisher_p": float(p_value),
    }


def crisis_crossing_events(
    price: pd.Series,
    threshold: float = 0.12,
    peak_window: int = 252,
    cooldown: int = 60,
) -> pd.Series:
    trailing_peak = price.rolling(
        peak_window, min_periods=max(60, peak_window // 2)
    ).max()
    drawdown = price / trailing_peak - 1.0
    crossing = (drawdown <= -threshold) & (drawdown.shift(1) > -threshold)
    return cluster_alerts(crossing.fillna(False), cooldown)


def event_capture_metrics(
    alerts: pd.Series,
    crisis_events: pd.Series,
    horizon: int,
) -> dict[str, float]:
    alert_positions = np.flatnonzero(alerts.fillna(False).to_numpy(bool))
    event_positions = np.flatnonzero(crisis_events.fillna(False).to_numpy(bool))
    alert_hits = sum(
        np.any((event_positions > alert) & (event_positions <= alert + horizon))
        for alert in alert_positions
    )
    event_hits = sum(
        np.any((alert_positions >= event - horizon) & (alert_positions < event))
        for event in event_positions
    )
    return {
        "event_alerts": float(len(alert_positions)),
        "event_crises": float(len(event_positions)),
        "event_true_alerts": float(alert_hits),
        "event_false_alerts": float(len(alert_positions) - alert_hits),
        "event_precision": float(alert_hits / len(alert_positions))
        if len(alert_positions)
        else np.nan,
        "event_recall": float(event_hits / len(event_positions))
        if len(event_positions)
        else np.nan,
        "event_crises_captured": float(event_hits),
    }


def diagnose_false_positives(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    alerts: pd.Series,
    horizon: int,
    threshold: float,
    min_assets: int,
) -> pd.DataFrame:
    future_loss = future_min_return(prices["SP500"], horizon)
    past_peak = prices["SP500"].rolling(
        horizon, min_periods=max(10, horizon // 2)
    ).max()
    past_drawdown = prices["SP500"] / past_peak - 1.0
    rows = []
    for date in alerts.index[alerts.fillna(False)]:
        loss = future_loss.loc[date]
        if np.isfinite(loss) and loss <= -threshold:
            continue
        reactive = bool(past_drawdown.loc[date] <= -0.08)
        near_miss = bool(np.isfinite(loss) and -threshold < loss <= -0.08)
        n_assets = indicators.loc[date, "n_assets"]
        sparse = bool(np.isfinite(n_assets) and n_assets <= min_assets + 1)
        if reactive:
            category = "post_shock_reactive"
        elif near_miss:
            category = "near_miss_8_to_12pct"
        elif sparse:
            category = "sparse_universe"
        else:
            category = "pure_false_alarm"
        rows.append(
            {
                "date": date,
                "category": category,
                "future_min_return": loss,
                "past_drawdown": past_drawdown.loc[date],
                "CI": indicators.loc[date, "CI"],
                "Phi": indicators.loc[date, "Phi"],
                "AC1": indicators.loc[date, "AC1"],
                "dCI_dt": indicators.loc[date, "dCI_dt"],
                "VIX": indicators.loc[date, "VIX"],
                "n_assets": n_assets,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "category",
                "future_min_return",
                "past_drawdown",
                "CI",
                "Phi",
                "AC1",
                "dCI_dt",
                "VIX",
                "n_assets",
            ]
        )
    return pd.DataFrame(rows).set_index("date")


def block_bootstrap_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = []
    total = 0
    while total < length:
        start = int(rng.integers(0, max(1, length - block_length + 1)))
        block = np.arange(start, min(length, start + block_length))
        blocks.append(block)
        total += len(block)
    return np.concatenate(blocks)[:length]


def block_bootstrap_metrics(
    signal: pd.Series,
    label: pd.Series,
    samples: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    aligned = pd.concat([signal, label], axis=1).dropna()
    predicted = aligned.iloc[:, 0].astype(bool).reset_index(drop=True)
    observed = aligned.iloc[:, 1].astype(bool).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        indices = block_bootstrap_indices(len(aligned), block_length, rng)
        rows.append(
            classification_metrics(
                predicted.iloc[indices].reset_index(drop=True),
                observed.iloc[indices].reset_index(drop=True),
            )
        )
    return pd.DataFrame(rows)


def circular_shift_placebo(
    signal: pd.Series,
    label: pd.Series,
    samples: int,
    min_shift: int,
    seed: int,
) -> pd.DataFrame:
    aligned = pd.concat([signal, label], axis=1).dropna()
    predicted = aligned.iloc[:, 0].astype(bool).reset_index(drop=True)
    observed = aligned.iloc[:, 1].astype(bool).reset_index(drop=True)
    length = len(aligned)
    if length <= 2 * min_shift + 1:
        return pd.DataFrame()
    allowed = np.arange(min_shift, length - min_shift)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        shift = int(rng.choice(allowed))
        shifted = pd.Series(np.roll(observed.to_numpy(), shift))
        row = classification_metrics(predicted, shifted)
        row["shift"] = shift
        rows.append(row)
    return pd.DataFrame(rows)


def phase_randomize(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return values.copy()
    clean = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    centered = clean - clean.mean()
    spectrum = np.fft.rfft(centered)
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    phases[0] = 0.0
    if len(clean) % 2 == 0:
        phases[-1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    output = np.fft.irfft(randomized, n=len(clean)) + clean.mean()
    output[~finite] = np.nan
    return output


def phase_randomized_prices(prices: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = pd.DataFrame(index=prices.index)
    for column in prices.columns:
        series = prices[column].astype(float)
        changes = np.log(series.clip(lower=1e-6)).diff().to_numpy()
        surrogate_changes = phase_randomize(changes, rng)
        start = float(series.dropna().iloc[0])
        reconstructed = start * np.exp(
            np.nancumsum(np.nan_to_num(surrogate_changes, nan=0.0))
        )
        reconstructed[series.isna().to_numpy()] = np.nan
        result[column] = reconstructed
    return result


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(p_values), dtype=float)
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for rank_from_end, index in enumerate(order[::-1], start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def parameter_sweep(
    indicators: pd.DataFrame,
    label: pd.Series,
    min_history: int,
) -> pd.DataFrame:
    rows = []
    for phi_quantile in [0.60, 0.70, 0.80]:
        phi_threshold = expanding_quantile_shifted(
            indicators["Phi"], phi_quantile, min_history
        )
        for ci_quantile in [0.85, 0.90, 0.95]:
            ci_threshold = expanding_quantile_shifted(
                indicators["CI"], ci_quantile, min_history
            )
            for ac1_quantile in [0.55, 0.65, 0.75]:
                ac1_threshold = expanding_quantile_shifted(
                    indicators["AC1"], ac1_quantile, min_history
                )
                for slope_quantile in [0.50, 0.60, 0.70]:
                    slope_threshold = expanding_quantile_shifted(
                        indicators["dCI_dt"], slope_quantile, min_history
                    )
                    signal = (
                        (indicators["Phi"] > phi_threshold)
                        & (indicators["CI"] < ci_threshold)
                        & (indicators["AC1"] > ac1_threshold)
                        & (indicators["dCI_dt"] > slope_threshold)
                        & (indicators["VIX"] < indicators["vix_threshold"])
                    ).fillna(False)
                    metrics = classification_metrics(cluster_alerts(signal, 20), label)
                    rows.append(
                        {
                            "phi_q": phi_quantile,
                            "ci_high_q": ci_quantile,
                            "ac1_q": ac1_quantile,
                            "slope_q": slope_quantile,
                            **metrics,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["fdr_q"] = bh_adjust(frame["fisher_p"].fillna(1.0))
    return frame.sort_values(
        ["fdr_q", "fisher_p", "lift"], ascending=[True, True, False]
    )


def walk_forward_audit(
    signal: pd.Series,
    label: pd.Series,
    purge_days: int = 60,
) -> pd.DataFrame:
    periods = [
        ("1990s", "1990-01-01", "1999-12-31"),
        ("2000-2007", "2000-01-01", "2007-12-31"),
        ("2008-2015", "2008-01-01", "2015-12-31"),
        ("2016-present", "2016-01-01", "2100-01-01"),
    ]
    rows = []
    for name, start, end in periods:
        index = signal.index[(signal.index >= start) & (signal.index <= end)]
        if len(index) <= 2 * purge_days:
            continue
        index = index[purge_days:-purge_days]
        rows.append(
            {"period": name, **classification_metrics(signal.loc[index], label.loc[index])}
        )
    return pd.DataFrame(rows)


def run_false_positive_audit(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    config: AuditConfig | None = None,
    coreno_config: CorenoConfig | None = None,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    cfg = config or AuditConfig()
    coreno_cfg = coreno_config or CorenoConfig()
    label = make_future_drawdown_label(
        prices["SP500"], cfg.horizon, cfg.drawdown_threshold
    )
    signal = cluster_alerts(indicators["late_sync"], cfg.alert_cooldown)
    primary = classification_metrics(signal, label)

    reversed_label = make_future_drawdown_label(
        prices["SP500"].iloc[::-1], cfg.horizon, cfg.drawdown_threshold
    ).iloc[::-1]
    reverse_time = classification_metrics(signal, reversed_label)

    bootstrap = block_bootstrap_metrics(
        signal,
        label,
        cfg.bootstrap_samples,
        cfg.block_length,
        cfg.seed,
    )
    placebo = circular_shift_placebo(
        signal,
        label,
        cfg.placebo_samples,
        cfg.horizon,
        cfg.seed + 1,
    )
    observed_lift = primary["lift"]
    placebo_p = (
        float(
            (1 + np.sum(placebo["lift"].fillna(-np.inf) >= observed_lift))
            / (len(placebo) + 1)
        )
        if len(placebo)
        else np.nan
    )

    label_rows = []
    for horizon, threshold in [(20, 0.08), (60, 0.12), (120, 0.18)]:
        alternative = make_future_drawdown_label(
            prices["SP500"], horizon, threshold
        )
        label_rows.append(
            {
                "horizon": horizon,
                "threshold": threshold,
                **classification_metrics(signal, alternative),
            }
        )
    label_sensitivity = pd.DataFrame(label_rows)

    surrogate_rows = []
    for index in range(cfg.surrogate_samples):
        surrogate_prices = phase_randomized_prices(
            prices, cfg.seed + 1000 + index
        )
        surrogate_indicators = compute_coreno_indicators(
            surrogate_prices, coreno_cfg
        )
        surrogate_label = make_future_drawdown_label(
            surrogate_prices["SP500"], cfg.horizon, cfg.drawdown_threshold
        )
        surrogate_signal = cluster_alerts(
            surrogate_indicators["late_sync"], cfg.alert_cooldown
        )
        surrogate_rows.append(
            {
                "surrogate": index,
                **classification_metrics(surrogate_signal, surrogate_label),
            }
        )
    surrogates = pd.DataFrame(surrogate_rows)
    surrogate_p = (
        float(
            (1 + np.sum(surrogates["lift"].fillna(-np.inf) >= observed_lift))
            / (len(surrogates) + 1)
        )
        if len(surrogates)
        else np.nan
    )

    sweep = parameter_sweep(indicators, label, coreno_cfg.min_history)
    walk_forward = walk_forward_audit(signal, label, cfg.horizon)
    crisis_events = crisis_crossing_events(
        prices["SP500"], cfg.drawdown_threshold, cooldown=cfg.horizon
    )
    event_metrics = event_capture_metrics(signal, crisis_events, cfg.horizon)
    diagnostics = diagnose_false_positives(
        prices,
        indicators,
        signal,
        cfg.horizon,
        cfg.drawdown_threshold,
        coreno_cfg.min_assets,
    )
    categories = (
        diagnostics["category"].value_counts().to_dict() if len(diagnostics) else {}
    )

    summary = {
        **primary,
        **event_metrics,
        "placebo_p": placebo_p,
        "surrogate_p": surrogate_p,
        "bootstrap_precision_lo": float(bootstrap["precision"].quantile(0.025)),
        "bootstrap_precision_hi": float(bootstrap["precision"].quantile(0.975)),
        "bootstrap_lift_lo": float(bootstrap["lift"].quantile(0.025)),
        "bootstrap_lift_hi": float(bootstrap["lift"].quantile(0.975)),
        "reverse_time_lift": reverse_time["lift"],
        "reverse_time_dominates": bool(
            reverse_time["lift"] >= primary["lift"]
        )
        if np.isfinite(reverse_time["lift"]) and np.isfinite(primary["lift"])
        else False,
        "fdr_significant_specs": int((sweep["fdr_q"] <= 0.05).sum()),
        "fdr_total_specs": int(len(sweep)),
        "false_positive_categories": categories,
    }
    return {
        "summary": summary,
        "bootstrap": bootstrap,
        "placebo": placebo,
        "surrogates": surrogates,
        "parameter_sweep": sweep,
        "walk_forward": walk_forward,
        "label_sensitivity": label_sensitivity,
        "false_positive_diagnostics": diagnostics,
        "crisis_events": crisis_events.to_frame("crisis_crossing"),
        "signal": signal.to_frame("clustered_alert"),
        "label": label.to_frame("future_crisis"),
    }
