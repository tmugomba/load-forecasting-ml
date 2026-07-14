# ⚡ Electricity Load Forecasting — Germany

A machine learning project forecasting hourly electricity demand across Germany's 16 federal states,
comparing classical and modern approaches on real grid data. Built as part of a personal portfolio
demonstrating applied data science for energy infrastructure.

**[Live Dashboard →](#)** *(link added after deployment)*

---

## Why load forecasting matters

Electricity can't be stored cheaply at grid scale — supply has to match demand in real time, every 
hour, every day. Grid operators who forecast demand well can plan **proactively**: buying power ahead
of time at lower prices, scheduling maintenance during predictable low-demand windows, and integrating
variable renewables (solar, wind) around known demand shapes.

Operators who forecast poorly are stuck **reacting**: scrambling to buy expensive last-minute power
during unexpected peaks, running costly "peaker" plants on short notice, or in the worst case, risking
brownouts. The difference between a good and mediocre load forecast translates directly into cost,
grid stability, and how cleanly renewable generation can be integrated without wasteful over-provisioning.

This project builds and compares several forecasting approaches on real German grid data, at the same
granularity (hourly) and geographic detail (state-level) that a real grid operator would need.

---

## What this dashboard does

- Select any of Germany's **16 federal states**
- Select a model: **Linear Regression**, **Random Forest**, or **XGBoost**
- See live-computed validation performance (MAE, RMSE, MAPE, R²), an actual-vs-predicted chart, and
  (for tree-based models) feature importance — all genuinely computed on selection, not pre-cached

---

## Data

**Source:** [Electrical Load and Generation for Germany](https://www.kaggle.com/datasets/pythonafroz/electrical-load-and-generation-for-germany) (Kaggle)

- Hourly electricity demand and generation-by-source, 2019–2022 (4 full years)
- Regionalized to all 16 German federal states from national TSO (grid operator) data
- Built from real SMARD/ENTSO-E grid operator data using a documented regionalization methodology

---

## The 17 engineered features

Raw demand data alone isn't enough for a model to learn from — it needs to see the patterns that
actually drive demand. Each feature below was built and validated in `notebooks/02_feature_engineering.ipynb`.

### Calendar features
| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) |
| `month` | Calendar month (1–12) |
| `year` | Calendar year |
| `is_weekend` | 1 if Saturday/Sunday, else 0 — demand drops ~23% on weekends |
| `is_holiday` | 1 if a public holiday in the selected state, else 0 (state-specific — e.g. Bavaria observes extra Catholic holidays) |

### Cyclical encodings
Raw hour/day/month values have a hidden flaw: to a model, hour 23 and hour 0 look maximally far apart,
even though they're actually adjacent on the clock. Sine/cosine encoding fixes this by mapping each
cycle onto a circle.

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Circular encoding of hour of day (period = 24) |
| `dow_sin`, `dow_cos` | Circular encoding of day of week (period = 7) |
| `month_sin`, `month_cos` | Circular encoding of month of year (period = 12) |

### Lag features
Demand is highly autocorrelated — the single strongest predictors in this project.

| Feature | Description |
|---|---|
| `lag_24h` | Demand exactly 24 hours ago (yesterday, same hour) |
| `lag_168h` | Demand exactly 168 hours ago (same hour, one week ago) — the single strongest predictor found, correlating at r=0.93 with the target |

### Rolling statistics
Smoothed recent behavior rather than a single past point, shifted by 1 hour so the current hour never
leaks into its own features.

| Feature | Description |
|---|---|
| `rolling_mean_24h` | Average demand over the prior 24 hours |
| `rolling_std_24h` | Demand volatility over the prior 24 hours |
| `rolling_mean_168h` | Average demand over the prior 168 hours (1 week) |

---

## Methodology

1. **EDA & Cleaning** (`notebooks/01_eda_cleaning.ipynb`) — loaded 4 years of hourly data across 16
   states, found and fixed one data-quality issue (a 0 MWh logging glitch at the very first timestamp),
   and confirmed real seasonal/weekly demand patterns.
2. **Feature Engineering** (`notebooks/02_feature_engineering.ipynb`) — built the 17 features above,
   and split the data using a **time-based cutoff** (not random) — training on everything through
   June 2022, holding out the last 6 months (Jul–Dec 2022) as a genuinely unseen validation set.
3. **Modeling** (`notebooks/03_modeling.ipynb`) — trained and compared 4 models, then retrained the
   winner on the full dataset and generated a genuine forward forecast into 2023.

## Results

Validated on the held-out last 6 months of 2022 (Bavaria, never seen during training):

| Model | MAE (MWh) | RMSE (MWh) | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | 318.2 | 408.5 | 3.77% | 0.933 |
| Random Forest | 227.6 | 337.6 | 2.59% | 0.954 |
| **XGBoost** | **201.2** | **275.4** | **2.34%** | **0.969** |
| SARIMAX | 1180.8 | 1677.8 | 13.79% | -0.133 |

**XGBoost is the strongest model**, explaining ~97% of demand variance on genuinely unseen data.
Random Forest's feature importance confirmed `lag_168h` as the dominant signal (~84% of importance) —
it implicitly captures day-of-week and time-of-day patterns, since a Tuesday 11am one week ago is
still a Tuesday 11am.

**SARIMAX was investigated thoroughly and excluded from the live app** — not from a single bad run,
but after three separate remediation attempts (fixing a non-stationary AR root, switching to
walk-forward validation, trimming collinear exogenous features), each of which produced real
improvement but never reached competitive accuracy. Full write-up, reasoning, and results:
[`SARIMAX_METHODOLOGY.md`](SARIMAX_METHODOLOGY.md).

## Forward forecast — a real out-of-sample test

XGBoost was retrained on the full 4-year dataset and used to generate a **recursive forecast into
Jan–Jun 2023** (`data/forecast_2023_BY.csv`) — each hour's lag/rolling features built from the model's
own prior predictions once past the end of real data. The forecast shows no degradation over the
6-month horizon and correctly reproduces the real seasonal pattern (forecasted February average:
10,563 MWh vs. 10,455 MWh actual historical average).

**This forecast is being held specifically to check against real published 2023 German demand data
once available** — genuine out-of-sample validation, not just a held-out split of already-known data.

---

## Project structure

load-forecasting-ml/
├── data/                          # Cleaned data, features, forecasts (raw Kaggle file excluded, see below)
├── notebooks/
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py                # Reusable, state-parameterized feature engineering
│   ├── models.py                  # Model training, evaluation, recursive forecasting
│   └── precompute_sarimax.py      # SARIMAX batch script (not used in live app — see methodology doc)
├── app.py                         # Streamlit dashboard
├── requirements.txt
├── SARIMAX_METHODOLOGY.md
└── README.md

**Note:** the raw 66MB Kaggle source file (`time_series_federal_states_all.csv`) is excluded from this
repo via `.gitignore` — download it directly from the [Kaggle dataset page](https://www.kaggle.com/datasets/pythonafroz/electrical-load-and-generation-for-germany)
if reproducing `notebooks/01_eda_cleaning.ipynb` from scratch. The cleaned output
(`data/demand_by_state_cleaned.csv`) is included and is all `app.py` needs to run.

## Running locally

```bash
git clone https://github.com/tmugomba/load-forecasting-ml.git
cd load-forecasting-ml
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Tools

Python, pandas, scikit-learn, XGBoost, statsmodels, Streamlit, Plotly, Jupyter

## Data license

Dataset used under its [Kaggle listing](https://www.kaggle.com/datasets/pythonafroz/electrical-load-and-generation-for-germany) terms. Original source: German grid operator (TSO) data via SMARD/ENTSO-E.