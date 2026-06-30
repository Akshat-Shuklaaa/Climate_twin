"""
predict.py — Reusable prediction API for next-day rainfall and temperature.

Loads saved RandomForest models and provides a clean function
predict_next_day(lat, lon, recent_history_df) that the dashboard and
what-if module both import.
"""

import os
import sys

import numpy as np
import pandas as pd
from joblib import load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

_MODELS = None
_FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "rainfall",
    "max_temp",
    "min_temp",
    "avg_temp",
    "day_of_year",
    "month",
    "rainfall_lag_1",
    "rainfall_lag_3",
    "rainfall_lag_7",
    "temp_lag_1",
    "temp_lag_3",
    "temp_lag_7",
    "rainfall_roll_3",
    "rainfall_roll_7",
    "temp_roll_3",
    "temp_roll_7",
    "temp_range",
]


def _load_models():
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    rain_path = os.path.join(CHECKPOINTS_DIR, "rainfall_model.joblib")
    temp_path = os.path.join(CHECKPOINTS_DIR, "temp_model.joblib")
    for p in [rain_path, temp_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model checkpoint not found: {p}. Run train.py first."
            )
    _MODELS = {
        "rainfall": load(rain_path),
        "temperature": load(temp_path),
    }
    return _MODELS


def _compute_derived_features(row):
    """Add derived features to a single row dict/Series."""
    row = row.copy()
    if pd.isna(row.get("avg_temp")):
        row["avg_temp"] = (row["max_temp"] + row["min_temp"]) / 2
    if pd.isna(row.get("temp_range")):
        row["temp_range"] = row["max_temp"] - row["min_temp"]
    return row


def predict_next_day(lat, lon, recent_history_df):
    """
    Predict next-day rainfall and average temperature for a given location.

    Parameters
    ----------
    lat : float
        Latitude of the grid cell.
    lon : float
        Longitude of the grid cell.
    recent_history_df : pd.DataFrame
        A DataFrame containing the most recent days of actual/known data
        for this location. Must include at least the columns:
        day, rainfall, max_temp, min_temp, avg_temp (optional),
        and enough history (>= 7 rows) to compute lag and rolling features.

    Returns
    -------
    (rainfall_pred, avg_temp_pred) : (float, float)
        Predicted next-day rainfall (mm) and average temperature (°C).
    """
    models = _load_models()

    # Filter and sort
    cell = recent_history_df[
        (recent_history_df["latitude"] == lat)
        & (recent_history_df["longitude"] == lon)
    ].copy()
    if len(cell) == 0:
        # Try without filtering (single-location data)
        cell = recent_history_df.copy()
    cell = cell.sort_values("day")

    # Take the most recent row to build the feature vector
    last_row = cell.iloc[-1].copy()

    # Compute derived features for the last known row
    last_row = _compute_derived_features(last_row)

    # Compute lags, rolling means from the history
    series_rain = cell["rainfall"].values
    series_temp = cell["avg_temp"].values if "avg_temp" in cell.columns else (
        (cell["max_temp"].values + cell["min_temp"].values) / 2
    )

    last_day = int(last_row["day"])
    next_day = last_day + 1

    # Build feature vector
    features = {
        "latitude": lat,
        "longitude": lon,
        "rainfall": last_row.get("rainfall", np.nan),
        "max_temp": last_row.get("max_temp", np.nan),
        "min_temp": last_row.get("min_temp", np.nan),
        "avg_temp": last_row.get("avg_temp", np.nan),
        "day_of_year": next_day,
        "month": min((next_day - 1) // 30 + 1, 12),
        "rainfall_lag_1": series_rain[-1] if len(series_rain) >= 1 else np.nan,
        "rainfall_lag_3": series_rain[-3] if len(series_rain) >= 3 else np.nan,
        "rainfall_lag_7": series_rain[-7] if len(series_rain) >= 7 else np.nan,
        "temp_lag_1": series_temp[-1] if len(series_temp) >= 1 else np.nan,
        "temp_lag_3": series_temp[-3] if len(series_temp) >= 3 else np.nan,
        "temp_lag_7": series_temp[-7] if len(series_temp) >= 7 else np.nan,
        "rainfall_roll_3": np.nanmean(series_rain[-3:]) if len(series_rain) >= 3 else np.nanmean(series_rain),
        "rainfall_roll_7": np.nanmean(series_rain[-7:]) if len(series_rain) >= 7 else np.nanmean(series_rain),
        "temp_roll_3": np.nanmean(series_temp[-3:]) if len(series_temp) >= 3 else np.nanmean(series_temp),
        "temp_roll_7": np.nanmean(series_temp[-7:]) if len(series_temp) >= 7 else np.nanmean(series_temp),
        "temp_range": last_row.get("max_temp", np.nan) - last_row.get("min_temp", np.nan),
    }

    X = np.array([[features[c] for c in _FEATURE_COLUMNS]])
    rain_pred = float(models["rainfall"].predict(X)[0])
    temp_pred = float(models["temperature"].predict(X)[0])

    return rain_pred, temp_pred
