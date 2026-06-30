"""
train.py — Train next-day rainfall and temperature prediction models.

Loads features_{2023,2024,2025}.csv, performs a time-based train/test split
per grid cell, trains RandomForestRegressor models, computes a persistence
baseline, and saves both models to src/models/checkpoints/.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

YEARS = [2023, 2024, 2025]
PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed"
)
CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoints"
)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
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

TARGET_RAIN = "target_rainfall"
TARGET_TEMP = "target_avg_temp"
# Persistence (baseline) columns
PERSIST_RAIN = "rainfall_lag_1"
PERSIST_TEMP = "temp_lag_1"

TEST_FRACTION = 0.15


def load_data():
    frames = []
    for year in YEARS:
        path = os.path.join(PROCESSED_DIR, f"features_{year}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}. Run feature engineering first."
            )
        df = pd.read_csv(path)
        df["year"] = year
        # Build a continuous day index across years
        offset = {2023: 0, 2024: 365, 2025: 365 + 366}
        df["global_day"] = df["day"] + offset[year]
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["latitude", "longitude", "global_day"]).reset_index(
        drop=True
    )
    print(f"[train] Loaded {len(data)} rows across {YEARS}")
    return data


def split_data(data):
    train_list, test_list = [], []
    for _, group in data.groupby(["latitude", "longitude"]):
        group = group.sort_values("global_day")
        n = len(group)
        split_idx = int(n * (1 - TEST_FRACTION))
        if split_idx < 2:
            split_idx = 2  # at least 2 training samples
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])

    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    print(f"[train] Train: {len(train)} rows, Test: {len(test)} rows")
    return train, test


def _persistence_baseline(y_true, y_persist):
    mask = ~(np.isnan(y_true) | np.isnan(y_persist))
    return y_true[mask], y_persist[mask]


def train_model(X, y, name):
    print(f"[train] Training {name} model with {X.shape[0]} samples, {X.shape[1]} features...")
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=42, verbose=0
    )
    model.fit(X, y)
    return model


def main():
    data = load_data()
    # Drop rows where targets are NaN (last day of each year has no target)
    data = data.dropna(subset=[TARGET_RAIN, TARGET_TEMP])
    print(f"[train] After dropping NaN targets: {len(data)} rows")

    train_df, test_df = split_data(data)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train_rain = train_df[TARGET_RAIN].values
    y_train_temp = train_df[TARGET_TEMP].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test_rain = test_df[TARGET_RAIN].values
    y_test_temp = test_df[TARGET_TEMP].values
    persist_test_rain = test_df[PERSIST_RAIN].values
    persist_test_temp = test_df[PERSIST_TEMP].values

    model_rain = train_model(X_train, y_train_rain, "rainfall")
    model_temp = train_model(X_train, y_train_temp, "temperature")

    rain_path = os.path.join(CHECKPOINTS_DIR, "rainfall_model.joblib")
    temp_path = os.path.join(CHECKPOINTS_DIR, "temp_model.joblib")
    dump(model_rain, rain_path)
    dump(model_temp, temp_path)
    print(f"[train] Saved models to {CHECKPOINTS_DIR}/")

    # Compute baseline on test set for immediate feedback
    rain_true, rain_persist = _persistence_baseline(y_test_rain, persist_test_rain)
    temp_true, temp_persist = _persistence_baseline(y_test_temp, persist_test_temp)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rain_pred = model_rain.predict(X_test)
    temp_pred = model_temp.predict(X_test)

    print()
    print("=" * 60)
    print("  Model  |  Target  |  RMSE  |  MAE  |  R²")
    print("=" * 60)
    for label, y_true, y_pred, p_true, p_pred in [
        ("RF", y_test_rain, rain_pred, rain_true, rain_persist),
        ("RF", y_test_temp, temp_pred, temp_true, temp_persist),
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  {label:4s} | {'rain':10s} | {rmse:.3f} | {mae:.3f} | {r2:.3f}")

    print("-" * 60)
    for label, y_true, y_pred in [
        ("Persistence", rain_true, rain_persist),
        ("Persistence", temp_true, temp_persist),
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  {label:10s} | {'rain':10s} | {rmse:.3f} | {mae:.3f} | {r2:.3f}")
    print("=" * 60)

    # Save test set with predictions for evaluation
    test_out = test_df.copy()
    test_out["pred_rainfall"] = rain_pred
    test_out["pred_avg_temp"] = temp_pred
    test_out["persist_rainfall"] = persist_test_rain
    test_out["persist_avg_temp"] = persist_test_temp
    test_out.to_csv(
        os.path.join(CHECKPOINTS_DIR, "test_predictions.csv"), index=False
    )
    print(f"[train] Saved test predictions to {CHECKPOINTS_DIR}/test_predictions.csv")


if __name__ == "__main__":
    main()
