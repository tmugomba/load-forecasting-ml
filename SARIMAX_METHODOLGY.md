# SARIMAX Investigation — Methodology & Findings

This document preserves the SARIMAX investigation from Notebook 3. SARIMAX is **not included in the
live Streamlit app's model dropdown** — this file explains why, and records the real results obtained
before that decision was made.

## Why SARIMAX was tried

SARIMAX (Seasonal ARIMA with eXogenous variables) is a natural classical-statistics baseline for load
forecasting, and was included specifically to give the project an honest contrast between traditional
time-series methods and the feature-based ML models (Linear Regression, Random Forest, XGBoost).

## What we found on Bavaria (full validation, Jul–Dec 2022)

| Model | MAE (MWh) | RMSE (MWh) | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | 318.2 | 408.5 | 3.77% | 0.933 |
| Random Forest | 227.6 | 337.6 | 2.59% | 0.954 |
| **XGBoost** | **201.2** | **275.4** | **2.34%** | **0.969** |
| SARIMAX (final version) | 1180.8 | 1677.8 | 13.79% | **-0.133** |

A negative R² means SARIMAX performed *worse* than simply predicting the average demand for every hour.

## Remediation attempts (in order)

SARIMAX wasn't abandoned at the first bad result — three real fixes were tried, each addressing a
distinct, diagnosed problem:

1. **Enforced stationarity.** The first fit had an AR(1) coefficient of 1.055 — above 1.0, meaning the
   model was technically non-stationary and errors would compound and grow indefinitely. Refitting with
   `enforce_stationarity=True` brought this down to a stable 0.58, fixing the runaway-error risk.

2. **Walk-forward validation.** The initial approach forecast all 4,416 validation hours (6 months) in
   one blind shot. Switched to forecasting 24 hours at a time, feeding real actuals back into the model
   before advancing — much closer to how day-ahead forecasting is actually used operationally, and
   removes 6 months of unchecked error compounding.

3. **Trimmed collinear exogenous features.** `is_weekend`, `dow_sin`, and `dow_cos` all encode
   overlapping day-of-week information. The model's covariance matrix was severely ill-conditioned
   (condition number in the range of 10^18–10^22), and individual coefficients were unstable enough to
   flip sign between comparable model runs (e.g. `dow_sin` went from +6644 to -1184 between two fits on
   the same data). Trimmed to a smaller, less-overlapping exogenous feature set and increased AR order.

Each fix produced a measurable, real improvement (R² moved from -14.8 → -3.2 → -0.13 across the three
attempts) but never reached competitive territory.

## Root cause

At hourly resolution with both strong daily and weekly seasonality, SARIMAX's AR/seasonal-AR terms and
the calendar-based exogenous regressors (hour, day-of-week encodings) compete for the same underlying
signal. The model can't cleanly separate which part of the pattern belongs to its autoregressive memory
versus its exogenous inputs, producing unstable, sign-flipping coefficients and materially worse
forecasts than the tree-based models — which sidestep the issue entirely via the `lag_168h` feature
(same hour, one week ago), a direct, uncontested anchor to the weekly pattern.

## Attempted full 16-state precompute — technical failure

After the Bavaria investigation, an attempt was made to precompute SARIMAX walk-forward results for all
16 states (each state's fit + walk-forward validation took 30–50 minutes). The run reached state 15 of
16 (Schleswig-Holstein) successfully before crashing on Thuringia (`TH`) with an internal `statsmodels`
numerical error in its smoothed-covariance computation (unrelated to our code — a library-level edge
case). Because the original script only saved results at the very end of the full loop rather than
incrementally, this crash lost all progress from the several hours of compute already completed.

## Decision

Given that:
- SARIMAX was already confirmed the weakest model by a wide margin on Bavaria
- The per-state walk-forward fit is extremely slow (30–50 min/state), making it impractical to run live
  in the Streamlit app the way the other three models do
- The precompute approach hit a real, unresolved technical failure

**SARIMAX is excluded from the live app's model dropdown.** The investigation itself — diagnosing three
distinct real problems, fixing each one, and still reaching an honest negative conclusion — is preserved
here and in `notebooks/03_modeling.ipynb`, and referenced in the project README as a demonstration of
understanding *why* a classical method struggles at this problem's specific scale and seasonality,
rather than simply reporting a single number.