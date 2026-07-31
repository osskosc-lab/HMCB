"""Public API for the CORENO market simulator."""

from .audit import (
    AuditConfig,
    classification_metrics,
    cluster_alerts,
    make_future_drawdown_label,
    run_false_positive_audit,
)
from .indicators import (
    DEFAULT_TICKERS,
    CorenoConfig,
    compute_coreno_indicators,
)
from .strategy import StrategyConfig, run_strategy_backtest
from .util import config_snapshot, save_json

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
