import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

INPUT_FILE = "../data/processed/climate_2025.csv"
OUTPUT_FILE = "../data/processed/features_2025.csv"

print("Loading climate dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)

# =====================================================
# SORT DATA
# =====================================================

df = df.sort_values(
    by=["latitude", "longitude", "day"]
).reset_index(drop=True)

# =====================================================
# BASIC FEATURES
# =====================================================

print("Creating basic features...")

df["avg_temp"] = (
    df["max_temp"] + df["min_temp"]
) / 2

# Day of year

df["day_of_year"] = df["day"]

# Approximate month
# (good enough for now)

df["month"] = ((df["day"] - 1) // 30) + 1

df["month"] = df["month"].clip(upper=12)

# =====================================================
# TARGET VARIABLES
# =====================================================

print("Creating targets...")

df["target_rainfall"] = (
    df.groupby(["latitude", "longitude"])["rainfall"]
      .shift(-1)
)

df["target_avg_temp"] = (
    df.groupby(["latitude", "longitude"])["avg_temp"]
      .shift(-1)
)

# =====================================================
# RAINFALL LAG FEATURES
# =====================================================

print("Creating rainfall lag features...")

grouped = df.groupby(
    ["latitude", "longitude"]
)

df["rainfall_lag_1"] = (
    grouped["rainfall"].shift(1)
)

df["rainfall_lag_3"] = (
    grouped["rainfall"].shift(3)
)

df["rainfall_lag_7"] = (
    grouped["rainfall"].shift(7)
)

# =====================================================
# TEMPERATURE LAG FEATURES
# =====================================================

print("Creating temperature lag features...")

df["temp_lag_1"] = (
    grouped["avg_temp"].shift(1)
)

df["temp_lag_3"] = (
    grouped["avg_temp"].shift(3)
)

df["temp_lag_7"] = (
    grouped["avg_temp"].shift(7)
)

# =====================================================
# ROLLING FEATURES
# =====================================================

print("Creating rolling features...")

df["rainfall_roll_3"] = (
    grouped["rainfall"]
    .transform(
        lambda x: x.rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

df["rainfall_roll_7"] = (
    grouped["rainfall"]
    .transform(
        lambda x: x.rolling(
            window=7,
            min_periods=1
        ).mean()
    )
)

df["temp_roll_3"] = (
    grouped["avg_temp"]
    .transform(
        lambda x: x.rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

df["temp_roll_7"] = (
    grouped["avg_temp"]
    .transform(
        lambda x: x.rolling(
            window=7,
            min_periods=1
        ).mean()
    )
)

# =====================================================
# TEMPERATURE RANGE
# =====================================================

df["temp_range"] = (
    df["max_temp"] - df["min_temp"]
)

# =====================================================
# CLEAN DATA
# =====================================================

print("Dropping NaNs...")

df = df.dropna()

print("Final Shape:", df.shape)

# =====================================================
# SAVE
# =====================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:", OUTPUT_FILE)
print()

print("Columns:")
print(df.columns.tolist())

print()
print(df.head())