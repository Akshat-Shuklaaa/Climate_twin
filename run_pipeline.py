"""
run_pipeline.py — End-to-end pipeline orchestrator.

Chains: load .GRD files → spatial merge → feature engineering → train → evaluate.
All existing files are left untouched; this script wraps the pipeline for reproducibility.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YEARS = [2023, 2024, 2025]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "src", "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "src", "models", "checkpoints")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def run_merge(year):
    climate_path = os.path.join(PROCESSED_DIR, f"climate_{year}.csv")
    if os.path.exists(climate_path):
        print(f"[pipeline] climate_{year}.csv exists — skipping merge")
        return

    print(f"[pipeline] Merging data for {year}...")
    # Import here so the script works without the full dataset present
    from src.data.climate_loaders import (
        load_rainfall,
        load_max_temp,
        load_min_temp,
        get_temperature_coordinates,
    )

    rainfall = load_rainfall(year)
    max_temp = load_max_temp(year)
    min_temp = load_min_temp(year)
    temp_lats, temp_lons = get_temperature_coordinates()
    days = rainfall.shape[0]

    records = []
    for day in range(days):
        for lat_idx in range(31):
            for lon_idx in range(31):
                max_t = max_temp[day, lat_idx, lon_idx]
                min_t = min_temp[day, lat_idx, lon_idx]
                if np.isnan(max_t) or np.isnan(min_t):
                    continue
                rain_lat_start = lat_idx * 4
                rain_lat_end = rain_lat_start + 4
                rain_lon_start = lon_idx * 4
                rain_lon_end = rain_lon_start + 4
                block = rainfall[
                    day, rain_lat_start:rain_lat_end, rain_lon_start:rain_lon_end
                ]
                avg_rain = np.nanmean(block)
                records.append(
                    [day + 1, temp_lats[lat_idx], temp_lons[lon_idx], avg_rain, max_t, min_t]
                )

    df = pd.DataFrame(
        records, columns=["day", "latitude", "longitude", "rainfall", "max_temp", "min_temp"]
    )
    df.to_csv(climate_path, index=False)
    print(f"[pipeline] Saved climate_{year}.csv ({len(df)} rows)")


def run_feature_engineering(year):
    features_path = os.path.join(PROCESSED_DIR, f"features_{year}.csv")
    if os.path.exists(features_path):
        print(f"[pipeline] features_{year}.csv exists — skipping feature engineering")
        return

    climate_path = os.path.join(PROCESSED_DIR, f"climate_{year}.csv")
    if not os.path.exists(climate_path):
        raise FileNotFoundError(
            f"{climate_path} not found. Run merge step first."
        )

    print(f"[pipeline] Engineering features for {year}...")
    df = pd.read_csv(climate_path)
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

    df = df.dropna()
    df.to_csv(features_path, index=False)
    print(f"[pipeline] Saved features_{year}.csv ({len(df)} rows, {len(df.columns)} columns)")


def run_training():
    print("[pipeline] Training models...")
    from src.models.train import main as train_main
    train_main()


def run_evaluation():
    print("[pipeline] Evaluating models...")
    from src.models.evaluate import main as eval_main
    eval_main()


def main():
    for year in YEARS:
        run_merge(year)

    for year in YEARS:
        run_feature_engineering(year)

    run_training()
    run_evaluation()

    print("[pipeline] Done.")


if __name__ == "__main__":
    main()
