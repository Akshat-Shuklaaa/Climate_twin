"""
Tests for feature engineering — validates lag/rolling features compute
correctly on a small synthetic DataFrame.
"""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _engineer_features(df):
    """Replicate the feature engineering logic from feature_engineering.py."""
    df = df.sort_values(["latitude", "longitude", "day"]).reset_index(drop=True)
    df["avg_temp"] = (df["max_temp"] + df["min_temp"]) / 2
    df["day_of_year"] = df["day"]
    df["month"] = ((df["day"] - 1) // 30) + 1
    df["month"] = df["month"].clip(upper=12)
    df["target_rainfall"] = df.groupby(["latitude", "longitude"])["rainfall"].shift(-1)
    df["target_avg_temp"] = df.groupby(["latitude", "longitude"])["avg_temp"].shift(-1)
    grouped = df.groupby(["latitude", "longitude"])
    df["rainfall_lag_1"] = grouped["rainfall"].shift(1)
    df["rainfall_lag_3"] = grouped["rainfall"].shift(3)
    df["rainfall_lag_7"] = grouped["rainfall"].shift(7)
    df["temp_lag_1"] = grouped["avg_temp"].shift(1)
    df["temp_lag_3"] = grouped["avg_temp"].shift(3)
    df["temp_lag_7"] = grouped["avg_temp"].shift(7)
    df["rainfall_roll_3"] = grouped["rainfall"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["rainfall_roll_7"] = grouped["rainfall"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["temp_roll_3"] = grouped["avg_temp"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["temp_roll_7"] = grouped["avg_temp"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    return df


def test_lag_features_compute_correctly():
    """Verify lag_1, lag_3, lag_7 shift appropriately per location."""
    np.random.seed(42)
    n_days = 20
    records = []
    for lat, lon in [(20.5, 78.5), (25.5, 80.5)]:
        for day in range(1, n_days + 1):
            records.append({
                "day": day,
                "latitude": lat,
                "longitude": lon,
                "rainfall": np.random.uniform(0, 50),
                "max_temp": np.random.uniform(25, 40),
                "min_temp": np.random.uniform(15, 25),
            })
    df = pd.DataFrame(records)
    result = _engineer_features(df)

    for cell_name, (lat, lon) in [("Cell A", (20.5, 78.5)), ("Cell B", (25.5, 80.5))]:
        cell = result[(result["latitude"] == lat) & (result["longitude"] == lon)].sort_values("day")

        # Check that lag_1[day] = value[day-1]
        for i in range(1, len(cell)):
            assert np.isclose(
                cell["rainfall_lag_1"].iloc[i],
                cell["rainfall"].iloc[i - 1],
            ), f"{cell_name}: rainfall_lag_1 failed at day {cell['day'].iloc[i]}"

        # Check that lag_3 works
        for i in range(3, len(cell)):
            assert np.isclose(
                cell["rainfall_lag_3"].iloc[i],
                cell["rainfall"].iloc[i - 3],
            ), f"{cell_name}: rainfall_lag_3 failed at day {cell['day'].iloc[i]}"

        # Check that rainfall_roll_3 is correct mean of last 3
        for i in range(2, len(cell)):
            expected = np.mean(cell["rainfall"].iloc[i - 2 : i + 1])
            assert np.isclose(
                cell["rainfall_roll_3"].iloc[i], expected
            ), f"{cell_name}: rainfall_roll_3 failed at day {cell['day'].iloc[i]}"


def test_lags_are_na_at_start():
    """Lag features should be NaN at the start of each location's time series."""
    df = pd.DataFrame({
        "day": [1, 2, 3],
        "latitude": [20.5, 20.5, 20.5],
        "longitude": [78.5, 78.5, 78.5],
        "rainfall": [10.0, 20.0, 30.0],
        "max_temp": [30.0, 31.0, 32.0],
        "min_temp": [20.0, 21.0, 22.0],
    })
    result = _engineer_features(df)
    assert pd.isna(result["rainfall_lag_1"].iloc[0])
    assert pd.isna(result["rainfall_lag_3"].iloc[0])
    assert pd.isna(result["rainfall_lag_7"].iloc[0])


def test_target_is_shifted():
    """target_rainfall[day] should equal rainfall[day+1]."""
    df = pd.DataFrame({
        "day": [1, 2, 3],
        "latitude": [20.5, 20.5, 20.5],
        "longitude": [78.5, 78.5, 78.5],
        "rainfall": [10.0, 20.0, 30.0],
        "max_temp": [30.0, 31.0, 32.0],
        "min_temp": [20.0, 21.0, 22.0],
    })
    result = _engineer_features(df)
    assert np.isclose(result["target_rainfall"].iloc[0], 20.0)
    assert np.isclose(result["target_rainfall"].iloc[1], 30.0)
    assert pd.isna(result["target_rainfall"].iloc[2])


def test_avg_temp_formula():
    df = pd.DataFrame({
        "day": [1], "latitude": [20.5], "longitude": [78.5],
        "rainfall": [10.0], "max_temp": [35.0], "min_temp": [25.0],
    })
    result = _engineer_features(df)
    assert np.isclose(result["avg_temp"].iloc[0], 30.0)


def test_temp_range():
    df = pd.DataFrame({
        "day": [1], "latitude": [20.5], "longitude": [78.5],
        "rainfall": [10.0], "max_temp": [35.0], "min_temp": [25.0],
    })
    result = _engineer_features(df)
    assert np.isclose(result["temp_range"].iloc[0], 10.0)


def test_month_assignment():
    df = pd.DataFrame({
        "day": [32, 60, 365],
        "latitude": [20.5, 20.5, 20.5],
        "longitude": [78.5, 78.5, 78.5],
        "rainfall": [10.0, 20.0, 30.0],
        "max_temp": [30.0, 31.0, 32.0],
        "min_temp": [20.0, 21.0, 22.0],
    })
    result = _engineer_features(df)
    # day 32 is ~ month 2, day 60 is ~ month 2, day 365 is ~ month 12
    assert result["month"].iloc[0] == 2
    assert result["month"].iloc[1] == 2
    assert result["month"].iloc[2] == 12
