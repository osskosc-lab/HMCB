import numpy as np
import pandas as pd

from coreno_swingby.core import (
    AuditConfig,
    CorenoConfig,
    classification_metrics,
    cluster_alerts,
    compute_coreno_indicators,
    make_future_drawdown_label,
    run_false_positive_audit,
    run_strategy_backtest,
)


def synthetic_prices(n=2200, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2000-01-03", periods=n)
    common = rng.normal(0.0002, 0.008, n)
    data = {}
    names = [
        "SP500", "NIKKEI225", "DOW", "NASDAQ", "RUSSELL2000",
        "FTSE100", "DAX", "CAC40", "HANGSENG", "ASX200",
    ]
    for i, name in enumerate(names):
        idio = rng.normal(0, 0.006 + i * 0.0002, n)
        data[name] = 100 * np.exp(np.cumsum(0.65 * common + 0.35 * idio))
    data["VIX"] = np.clip(18 + 150 * np.abs(common) + rng.normal(0, 1.5, n), 9, 80)
    return pd.DataFrame(data, index=index)


def test_indicators_are_causal_and_finite_after_warmup():
    prices = synthetic_prices()
    indicators = compute_coreno_indicators(prices, CorenoConfig(min_history=500))
    tail = indicators.iloc[800:]
    assert tail["CI"].notna().mean() > 0.95
    assert ((tail["CI"] >= 0) & (tail["CI"] <= 1)).mean() > 0.95
    assert ((tail["Phi"] >= 0) & (tail["Phi"] <= 4)).mean() > 0.95
    assert indicators["phi_threshold"].iloc[:500].isna().all()


def test_future_drawdown_label_detects_known_drop_and_masks_tail():
    index = pd.bdate_range("2020-01-01", periods=50)
    price = pd.Series(100.0, index=index)
    price.iloc[20:25] = [98, 94, 90, 88, 87]
    label = make_future_drawdown_label(price, horizon=10, threshold=0.10)
    assert label.iloc[15]
    assert not label.iloc[30]
    assert label.iloc[-10:].isna().all()


def test_cluster_alerts_enforces_cooldown():
    signal = pd.Series([False, True, True, False, True, False, False, True])
    clustered = cluster_alerts(signal, cooldown=3)
    assert clustered.sum() == 2
    assert clustered.iloc[1]
    assert clustered.iloc[7]


def test_metrics_count_false_positives():
    signal = pd.Series([True, True, False, False])
    label = pd.Series([True, False, True, False])
    metrics = classification_metrics(signal, label)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1


def test_full_pipeline_small_audit():
    prices = synthetic_prices()
    config = CorenoConfig(min_history=500, normalization_window=500)
    indicators = compute_coreno_indicators(prices, config)
    returns, metrics = run_strategy_backtest(prices, indicators)
    assert set(returns.columns) == {
        "CORENO_swingby", "buy_hold_50_50", "VIX_25_cash", "MA200_filter"
    }
    assert len(metrics) == 4
    audit = run_false_positive_audit(
        prices,
        indicators,
        AuditConfig(bootstrap_samples=10, placebo_samples=10, surrogate_samples=2, seed=3),
        config,
    )
    assert "fp" in audit["summary"]
    assert "event_precision" in audit["summary"]
    assert "censored_alerts" in audit["summary"]
    assert len(audit["parameter_sweep"]) == 81
    assert sum(audit["summary"]["false_positive_categories"].values()) == int(
        audit["summary"]["fp"]
    )
    assert audit["summary"]["event_alerts"] <= audit["signal"][
        "clustered_alert"
    ].sum()
