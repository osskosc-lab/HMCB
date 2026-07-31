from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import TARGET_ASSETS


@dataclass(frozen=True)
class StrategyConfig:
    trend_window: int = 126
    volatility_window: int = 20
    target_volatility: float = 0.10
    max_abs_weight: float = 1.0
    transaction_cost_bps: float = 5.0
    critical_mode: str = "cash"


def performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            key: np.nan
            for key in [
                "CAGR",
                "volatility",
                "Sharpe",
                "max_drawdown",
                "Calmar",
                "total_return",
            ]
        }
    equity = (1.0 + returns).cumprod()
    years = max(len(returns) / periods_per_year, 1 / periods_per_year)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = (
        float(
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(periods_per_year)
        )
        if returns.std(ddof=1) > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "CAGR": cagr,
        "volatility": volatility,
        "Sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "Calmar": calmar,
        "total_return": float(equity.iloc[-1] - 1.0),
    }


def run_strategy_backtest(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or StrategyConfig()
    returns = prices[TARGET_ASSETS].pct_change().fillna(0.0)
    trend = prices[TARGET_ASSETS].pct_change(cfg.trend_window)
    realized_volatility = (
        returns.rolling(
            cfg.volatility_window,
            min_periods=cfg.volatility_window,
        ).std()
        * np.sqrt(252)
    )
    direction = np.sign(trend).replace(0.0, np.nan).fillna(0.0)
    weights = direction * (
        cfg.target_volatility / realized_volatility.replace(0.0, np.nan)
    )
    weights = weights.clip(
        -cfg.max_abs_weight,
        cfg.max_abs_weight,
    ).fillna(0.0)
    weights = weights.div(
        weights.abs().sum(axis=1).clip(lower=1.0), axis=0
    )
    weights = weights.where(indicators["late_sync"], 0.0)
    if cfg.critical_mode == "cash":
        weights = weights.where(~indicators["critical_lock"], 0.0)

    # One-day delay is mandatory: today's close-derived state can trade tomorrow only.
    weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    gross_returns = (weights * returns).sum(axis=1)
    net_returns = gross_returns - turnover * (
        cfg.transaction_cost_bps / 10_000.0
    )

    buy_hold_weights = pd.DataFrame(
        0.5,
        index=returns.index,
        columns=TARGET_ASSETS,
    ).shift(1).fillna(0.0)
    buy_hold = (buy_hold_weights * returns).sum(axis=1)

    vix_weights = buy_hold_weights.where(
        prices["VIX"].shift(1) <= 25.0,
        0.0,
    )
    vix_rule = (vix_weights * returns).sum(axis=1)

    moving_average = prices[TARGET_ASSETS].rolling(
        200, min_periods=200
    ).mean()
    trend_weights = buy_hold_weights.where(
        prices[TARGET_ASSETS].shift(1) > moving_average.shift(1),
        0.0,
    )
    trend_rule = (trend_weights * returns).sum(axis=1)

    strategy_returns = pd.DataFrame(
        {
            "CORENO_swingby": net_returns,
            "buy_hold_50_50": buy_hold,
            "VIX_25_cash": vix_rule,
            "MA200_filter": trend_rule,
        }
    )
    rows = []
    for name in strategy_returns.columns:
        row = {
            "strategy": name,
            **performance_metrics(strategy_returns[name]),
        }
        row["annual_turnover"] = (
            float(turnover.mean() * 252)
            if name == "CORENO_swingby"
            else np.nan
        )
        rows.append(row)
    return strategy_returns, pd.DataFrame(rows)
