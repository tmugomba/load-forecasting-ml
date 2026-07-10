"""
Model training and forecasting for the Load Forecasting project.

Wraps the same 4 models compared in Notebook 3 (Linear Regression, Random
Forest, XGBoost, SARIMAX) into reusable functions, so app.py can train on
demand for whichever state/model combination the user selects.
"""

import pandas as pd
import numpy as np
import holidays
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "hour", "day_of_week", "month", "year", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h"
]

SARIMAX_EXOG_COLS = ["is_holiday", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_cos"]


def compute_metrics(y_true, y_pred) -> dict:
    """Returns the standard 4-metric evaluation used throughout this project."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate(model_name: str, train: pd.DataFrame, val: pd.DataFrame):
    """
    Trains one of the 4 models on `train` and evaluates on `val`.

    Parameters
    ----------
    model_name : str
        One of "Linear Regression", "Random Forest", "XGBoost", "SARIMAX"
    train, val : pd.DataFrame
        Output of features.time_split() — must include all FEATURE_COLS
        plus demand_mwh.

    Returns
    -------
    dict with keys: "predictions" (pd.Series), "metrics" (dict), "model" (fitted model object)
    """
    X_train, y_train = train[FEATURE_COLS], train["demand_mwh"]
    X_val, y_val = val[FEATURE_COLS], val["demand_mwh"]

    if model_name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_val), index=y_val.index)

    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_val), index=y_val.index)

    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_val), index=y_val.index)

    elif model_name == "SARIMAX":
        # Walk-forward validation, exactly as established in Notebook 3 —
        # single-shot forecasting was tried first and failed badly (R² < 0)
        exog_train = train[SARIMAX_EXOG_COLS]
        exog_val = val[SARIMAX_EXOG_COLS]

        fit = SARIMAX(
            y_train, exog=exog_train,
            order=(4, 0, 0), seasonal_order=(1, 0, 0, 24),
            enforce_stationarity=True, enforce_invertibility=True
        ).fit(disp=False)

        preds_list = []
        current_fit = fit
        for i in range(0, len(val), 24):
            exog_chunk = exog_val.iloc[i:i + 24]
            forecast = current_fit.get_forecast(steps=len(exog_chunk), exog=exog_chunk)
            preds_list.append(forecast.predicted_mean)
            actual_chunk = y_val.iloc[i:i + 24]
            current_fit = current_fit.append(actual_chunk, exog=exog_chunk, refit=False)

        preds = pd.concat(preds_list).reindex(y_val.index)
        model = fit  # store the initial fit; not used for further prediction outside this function

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    metrics = compute_metrics(y_val, preds)
    return {"predictions": preds, "metrics": metrics, "model": model}


def _build_calendar_features(ts, de_holidays: holidays.HolidayBase) -> dict:
    """Calendar features known in advance for a single future timestamp."""
    return {
        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month, "year": ts.year,
        "is_weekend": int(ts.dayofweek >= 5),
        "is_holiday": int(ts.date() in de_holidays),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
    }


def recursive_forecast(model, full_data: pd.DataFrame, state_code: str, future_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Forecasts forward hour-by-hour beyond the end of real data, feeding each
    prediction back into the history buffer so later lag/rolling features can
    use it — same approach validated in Notebook 3 for the Jan-Jun 2023 forecast.

    Only valid for the tree/linear models (not SARIMAX, which has its own
    internal state and isn't compatible with this feature-based buffer approach).
    """
    de_holidays = holidays.Germany(
        state=state_code if state_code in holidays.Germany.subdivisions else None,
        years=range(future_dates.min().year - 1, future_dates.max().year + 1)
    )

    buffer = full_data["demand_mwh"].copy()
    predictions = []

    for ts in future_dates:
        row = _build_calendar_features(ts, de_holidays)
        row["lag_24h"] = buffer.loc[ts - pd.Timedelta(hours=24)]
        row["lag_168h"] = buffer.loc[ts - pd.Timedelta(hours=168)]

        window_24 = buffer.loc[ts - pd.Timedelta(hours=24):ts - pd.Timedelta(hours=1)]
        window_168 = buffer.loc[ts - pd.Timedelta(hours=168):ts - pd.Timedelta(hours=1)]
        row["rolling_mean_24h"] = window_24.mean()
        row["rolling_std_24h"] = window_24.std()
        row["rolling_mean_168h"] = window_168.mean()

        X_row = pd.DataFrame([row])[FEATURE_COLS]
        pred = model.predict(X_row)[0]

        predictions.append(pred)
        buffer.loc[ts] = pred

    return pd.Series(predictions, index=future_dates, name="demand_forecast")