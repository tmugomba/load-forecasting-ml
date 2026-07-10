"""
Precompute SARIMAX results for all 16 states.

Run this once, standalone (not part of the live app), since SARIMAX's
walk-forward validation is too slow to run per-click in the Streamlit app.
Produces two small files the app reads instantly:
  - data/sarimax_metrics.csv       (one row per state: MAE, RMSE, MAPE, R2)
  - data/sarimax_predictions.csv   (long format: state, datetime, actual, predicted)

Expect this to take a while — SARIMAX's walk-forward fit is the slowest part
of this whole project. Run it and step away rather than watching it live.
"""

import sys
import time
import pandas as pd

sys.path.append(".")  # run this from the project root: python src/precompute_sarimax.py
from src.features import build_features, time_split
from src.models import train_and_evaluate

STATES = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV",
          "NI", "NRW", "RP", "SL", "SN", "ST", "SH", "TH"]

metrics_rows = []
prediction_rows = []

for i, state in enumerate(STATES, start=1):
    print(f"[{i}/{len(STATES)}] Running SARIMAX for {state}...")
    start = time.time()

    features = build_features(state, cleaned_data_path="data/demand_by_state_cleaned.csv")
    train, val = time_split(features)

    result = train_and_evaluate("SARIMAX", train, val)

    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s | MAE: {result['metrics']['MAE']:.1f} | R2: {result['metrics']['R2']:.3f}")

    metrics_rows.append({"state": state, **result["metrics"]})

    preds_df = pd.DataFrame({
        "state": state,
        "datetime": result["predictions"].index,
        "actual": val["demand_mwh"].values,
        "predicted": result["predictions"].values
    })
    prediction_rows.append(preds_df)

# Save both outputs
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("data/sarimax_metrics.csv", index=False)
print("\nSaved data/sarimax_metrics.csv")
print(metrics_df)

predictions_df = pd.concat(prediction_rows, ignore_index=True)
predictions_df.to_csv("data/sarimax_predictions.csv", index=False)
print("\nSaved data/sarimax_predictions.csv:", predictions_df.shape)