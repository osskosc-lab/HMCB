from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


DEFAULT_TICKERS = {
    "SP500": "^GSPC",
    "NIKKEI225": "^N225",
    "VIX": "^VIX",
    "DOW": "^DJI",
    "NASDAQ": "^IXIC",
    "RUSSELL2000": "^RUT",
    "FTSE100": "^FTSE",
    "DAX": "^GDAXI",
    "CAC40": "^FCHI",
    "HANGSENG": "^HSI",
    "ASX200": "^AXJO",
}

PRICE_ASSETS = [name for name in DEFAULT_TICKERS if name != "VIX"]
TARGET_ASSETS = ["SP500", "NIKKEI225"]


@dataclass(frozen=True)
class CorenoConfig:
    corr_window: int = 60
    state_window: int = 60
    normalization_window: int = 756
    min_assets: int = 5
    min_history: int = 756
    phi_quantile: float = 0.70
    ci_low_quantile: float = 0.50
    ci_high_quantile: float = 0.90
    ac1_quantile: float = 0.65
    slope_quantile: float = 0.60
    vix_quantile: float = 0.80


def rolling_mad_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    median = series.rolling(window, min_periods=min_periods).median()
    mad = (series - median).abs().rolling(window, min_periods=min_periods).median()
    scale = 1.4826 * mad.replace(0.0, np.nan)
    return ((series - median) / scale).clip(-8, 8)


def sigmoid(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(x, values - values.mean()) / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def rolling_last_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def percentile(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return np.nan
        valid = values[np.isfinite(values)]
        if len(valid) < 2:
            return np.nan
        lower = np.sum(valid < values[-1])
        equal = np.sum(valid == values[-1])
        return float((lower + 0.5 * equal) / len(valid))

    return series.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def expanding_quantile_shifted(series: pd.Series, quantile: float, min_periods: int) -> pd.Series:
    return series.expanding(min_periods=min_periods).quantile(quantile).shift(1)


def spectral_metrics(
    window_returns: pd.DataFrame,
    min_assets: int,
) -> tuple[float, float, float, float, float]:
    required = max(10, int(0.8 * len(window_returns)))
    valid = window_returns.dropna(axis=1, thresh=required)
    if valid.shape[1] < min_assets:
        return (np.nan,) * 5
    corr = valid.corr().replace([np.inf, -np.inf], np.nan)
    corr = corr.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if corr.shape[0] < min_assets:
        return (np.nan,) * 5

    eigenvalues = np.clip(np.linalg.eigvalsh(corr.to_numpy(dtype=float)), 0.0, None)
    if eigenvalues.sum() <= 0:
        return (np.nan,) * 5
    lambda1 = float(eigenvalues.max())
    n_assets = float(len(eigenvalues))
    ci = lambda1 / n_assets
    matrix = corr.to_numpy()
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    mean_abs_corr = float(np.nanmean(np.abs(off_diagonal)))
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-15)))
    spectral_concentration = 1.0 - entropy / math.log(len(eigenvalues))
    return lambda1, ci, mean_abs_corr, spectral_concentration, n_assets


def compute_coreno_indicators(
    prices: pd.DataFrame,
    config: CorenoConfig | None = None,
) -> pd.DataFrame:
    """Compute causal CORENO proxies with all thresholds shifted one day."""
    cfg = config or CorenoConfig()
    missing = [name for name in TARGET_ASSETS + ["VIX"] if name not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prices = prices.sort_index().copy()
    market_prices = prices[[name for name in PRICE_ASSETS if name in prices.columns]].ffill(limit=2)
    returns = np.log(market_prices).diff().replace([np.inf, -np.inf], np.nan)

    rows: list[tuple[float, float, float, float, float]] = []
    for position in range(len(returns)):
        if position + 1 < cfg.corr_window:
            rows.append((np.nan,) * 5)
            continue
        window = returns.iloc[position + 1 - cfg.corr_window : position + 1]
        rows.append(spectral_metrics(window, cfg.min_assets))

    output = pd.DataFrame(
        rows,
        index=prices.index,
        columns=["lambda1", "CI", "mean_abs_corr", "spectral_concentration", "n_assets"],
    )
    output["dCI_dt"] = rolling_slope(output["CI"], 20)

    local_volatility = returns.rolling(60, min_periods=40).std().replace(0.0, np.nan)
    normalized_returns = returns.divide(local_volatility)
    global_factor = normalized_returns.mean(axis=1, skipna=True)
    stress_state = global_factor.abs().ewm(span=20, adjust=False, min_periods=20).mean()
    output["AC1"] = (
        stress_state.rolling(cfg.state_window, min_periods=40)
        .corr(stress_state.shift(1))
        .clip(-1, 1)
    )
    susceptibility = global_factor.rolling(20, min_periods=15).var() * normalized_returns.shape[1]
    output["chi"] = rolling_last_percentile(
        susceptibility,
        cfg.normalization_window,
        max(252, cfg.min_history // 2),
    )

    normal_min = max(252, cfg.min_history // 2)
    composite = (
        rolling_mad_z(output["CI"], cfg.normalization_window, normal_min)
        + rolling_mad_z(output["mean_abs_corr"], cfg.normalization_window, normal_min)
        + rolling_mad_z(output["spectral_concentration"], cfg.normalization_window, normal_min)
    ) / 3.0
    output["Phi"] = 4.0 * sigmoid(0.8 * composite)
    output["VIX"] = prices["VIX"]

    threshold_specs = {
        "phi_threshold": (output["Phi"], cfg.phi_quantile),
        "ci_low_threshold": (output["CI"], cfg.ci_low_quantile),
        "ci_high_threshold": (output["CI"], cfg.ci_high_quantile),
        "ac1_threshold": (output["AC1"], cfg.ac1_quantile),
        "slope_threshold": (output["dCI_dt"], cfg.slope_quantile),
        "vix_threshold": (output["VIX"], cfg.vix_quantile),
    }
    for name, (series, quantile) in threshold_specs.items():
        output[name] = expanding_quantile_shifted(series, quantile, cfg.min_history)

    output["late_sync"] = (
        (output["Phi"] > output["phi_threshold"])
        & (output["CI"] > output["ci_low_threshold"])
        & (output["CI"] < output["ci_high_threshold"])
        & (output["AC1"] > output["ac1_threshold"])
        & (output["dCI_dt"] > output["slope_threshold"])
        & (output["VIX"] < output["vix_threshold"])
    ).fillna(False)

    output["critical_lock"] = (
        (output["CI"] >= output["ci_high_threshold"])
        & ((output["VIX"] >= output["vix_threshold"]) | (output["dCI_dt"] > 0))
    ).fillna(False)

    output["phase"] = "diffuse"
    output.loc[output["Phi"] > output["phi_threshold"], "phase"] = "structure_formation"
    output.loc[output["late_sync"], "phase"] = "late_synchronization"
    output.loc[output["critical_lock"], "phase"] = "critical_lock"
    return output
