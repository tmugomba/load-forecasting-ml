"""
Feature engineering for the Load Forecasting project.

Builds the same calendar, cyclical, lag, and rolling features used in
Notebook 2 — but callable for any of the 16 states, so the Streamlit app's
state dropdown can generate features on demand rather than only working
for the one state we happened to build a notebook around.
"""

import pandas as pd
import numpy as np
import holidays


def build_features(state_code: str, cleaned_data_path: str = "data/demand_by_state_cleaned.csv") -> pd.DataFrame:
    """
    Loads the cleaned demand data and builds the full feature set for one state.

    Parameters
    ----------
    state_code : str
        Two/three-letter German state code, e.g. "BY" for Bavaria, "SH" for
        Schleswig-Holstein. Must match a column in the cleaned data file.
    cleaned_data_path : str
        Path to demand_by_state_cleaned.csv, produced by Notebook 1.
        Defaults to a path relative to the project root (correct for app.py;
        notebooks calling this should pass "../data/demand_by_state_cleaned.csv" instead).

    Returns
    -------
    pd.DataFrame
        Indexed by datetime, with all engineered features plus the demand_mwh
        target column, NaN rows (first 168 hours) already dropped.
    """
    # Load the cleaned, all-states demand table (already fixed for the one
    # data-quality issue found in Notebook 1 — no zero-demand glitch)
    demand_all = pd.read_csv(cleaned_data_path, index_col=0, parse_dates=True)
    demand_all.index.name = "datetime"

    if state_code not in demand_all.columns:
        raise ValueError(f"State code '{state_code}' not found. Available: {list(demand_all.columns)}")

    demand = demand_all[state_code].copy()
    demand.name = "demand_mwh"

    # Start building the features table
    features = pd.DataFrame(index=demand.index)

    # --- Calendar features ---
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["year"] = features.index.year
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    # German public holidays — state-specific, since Bavaria etc. observe a
    # few extra Catholic holidays that other states don't
    de_holidays = holidays.Germany(state=state_code if state_code in holidays.Germany.subdivisions else None,
                                     years=range(demand.index.year.min(), demand.index.year.max() + 1))
    features["is_holiday"] = pd.Series(
        [d in de_holidays for d in features.index.date],
        index=features.index
    ).astype(int)

    # --- Cyclical encoding ---
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    # --- Lag features ---
    features["lag_24h"] = demand.shift(24)
    features["lag_168h"] = demand.shift(168)

    # --- Rolling statistics (shifted by 1 so the current hour never leaks into its own features) ---
    features["rolling_mean_24h"] = demand.shift(1).rolling(window=24).mean()
    features["rolling_std_24h"] = demand.shift(1).rolling(window=24).std()
    features["rolling_mean_168h"] = demand.shift(1).rolling(window=168).mean()

    # Attach the target
    features["demand_mwh"] = demand

    # Drop the warm-up period (first 168 hours with no valid lag_168h)
    return features.dropna()


def time_split(features: pd.DataFrame, split_date: str = "2022-07-01"):
    """
    Splits a features DataFrame into train/validation sets using a fixed
    cutoff date — never a random split, to avoid leaking future information
    into training.

    Returns
    -------
    (train, val) : tuple of pd.DataFrame
    """
    train = features[features.index < split_date]
    val = features[features.index >= split_date]
    return train, val