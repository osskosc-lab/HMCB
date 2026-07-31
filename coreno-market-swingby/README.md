# CORENO S-Class Market Swing-by Simulator

A falsification-first simulator for the CORENO market-state / market-black-hole analogy. It downloads daily market data from 1990 onward, constructs causal market-state proxies, backtests a swing-by allocation rule, and treats false positives as the primary research target.

## Research question

Does a high-structure, rising-correlation, pre-panic state identify future large S&P 500 drawdowns more often than expected from crisis prevalence, serial dependence, spectrum-preserving surrogate markets, and parameter search?

The simulator does **not** assume that the physical black-hole analogy is literally true. It operationalizes the analogy as measurable proxies:

- `lambda1`: maximum rolling correlation eigenvalue
- `CI`: `lambda1 / number_of_available_markets`
- `dCI_dt`: 20-day causal slope of CI
- `AC1`: lag-1 persistence of an EWMA global stress state
- `chi`: causal percentile of global-factor variance
- `Phi`: robustly normalized composite of correlation concentration, mean absolute correlation, and spectral concentration, scaled to 0–4

Because the historical formula that generated the supplied 2026 snapshot is not fully specified, values are **proxy-compatible, not numerically identical** to that snapshot. Regime thresholds use expanding quantiles shifted by one day to prevent look-ahead leakage.

## Data universe

Yahoo Finance daily series are used for S&P 500, Nikkei 225, VIX, Dow, Nasdaq Composite, Russell 2000, FTSE 100, DAX, CAC 40, Hang Seng, and ASX 200. The period begins on 1990-01-01 when available.

Using indices instead of present-day constituents reduces constituent survivorship bias. It does not eliminate local-calendar, currency, index-methodology, vendor-revision, or missing-series bias.

## False-positive audit

The run produces all of the following:

1. **Alert clustering** — repeated daily warnings within one episode count as one alert.
2. **Multiple crisis labels** — 20-day/8%, 60-day/12%, and 120-day/18% future drawdowns.
3. **Moving-block bootstrap** — confidence intervals preserve local serial dependence.
4. **Circular-shift placebo** — labels are shifted far away from their original dates.
5. **Phase-randomized surrogate markets** — return spectra are preserved while temporal phase structure is destroyed.
6. **Reverse-time control** — checks whether the indicator mostly reacts to past damage rather than predicts future damage.
7. **81-specification parameter sweep** — all p-values receive Benjamini-Hochberg FDR correction.
8. **Purged era-by-era audit** — 1990s, 2000–2007, 2008–2015, and 2016–present.
9. **False-positive decomposition** — post-shock reaction, 8–12% near miss, sparse universe, or pure false alarm.
10. **Event-level capture** — alerts are matched to distinct 12% drawdown crossings rather than daily crisis labels only.
11. **Transaction costs and one-day execution lag** — applied to the swing-by backtest.
12. **Baselines** — buy-and-hold, VIX>25 cash rule, and 200-day moving-average filter.

A model is not considered provisionally supported unless most independent falsification gates pass. A favorable threshold selected after seeing the entire sample is not accepted as a repair.

## Run locally

```bash
python -m pip install -e ".[test]"
pytest -q
python run_simulation.py --start 1990-01-01 --quick
```

Full audit:

```bash
python run_simulation.py \
  --start 1990-01-01 \
  --bootstrap 2000 \
  --placebo 2000 \
  --surrogates 200
```

Outputs are written to `outputs/`, including `REPORT.md`, `summary.json`, daily states, backtest metrics, parameter-sweep results, false-positive diagnostics, placebo distributions, surrogate results, and charts.

## Pre-registered primary definition

- Target: S&P 500
- Prediction horizon: 60 trading days
- Crisis: future minimum return from the signal date of at least -12%
- Alert: first `late_synchronization` observation after a 20-trading-day cooldown
- Primary statistic: precision lift over unconditional crisis prevalence
- Primary nulls: circular-shift labels and phase-randomized markets
- Multiplicity control: Benjamini-Hochberg FDR at 5%

## Investment limitation

This repository is an empirical research instrument. Backtests are sensitive to data, assumptions, costs, and regime changes and are not individualized investment advice.
