"""
Tests for spatial fusion logic — verifies the 4×4 block averaging produces
correct output shape and that records are correctly structured.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "data", "processed"
)


def _skip_if_no_processed():
    path = os.path.join(PROCESSED_DIR, "climate_2025.csv")
    if not os.path.exists(path):
        pytest.skip("Processed climate CSV not found — run merge_climate_data.py first")


def test_climate_csv_exists():
    path = os.path.join(PROCESSED_DIR, "climate_2025.csv")
    assert os.path.exists(path), f"Expected {path} to exist"


def test_climate_csv_schema():
    _skip_if_no_processed()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "climate_2025.csv"))
    expected_cols = ["day", "latitude", "longitude", "rainfall", "max_temp", "min_temp"]
    assert list(df.columns) == expected_cols, f"Got columns: {list(df.columns)}"


def test_climate_csv_dtypes():
    _skip_if_no_processed()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "climate_2025.csv"))
    assert df["day"].dtype in (np.int64, np.float64)
    assert df["rainfall"].dtype == np.float64
    assert df["max_temp"].dtype == np.float64


def test_climate_csv_value_ranges():
    _skip_if_no_processed()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "climate_2025.csv"))
    assert df["day"].min() >= 1
    assert df["day"].max() <= 366
    assert df["rainfall"].min() >= 0 or np.isnan(df["rainfall"].min())
    assert df["max_temp"].max() <= 60
    assert df["min_temp"].min() >= -10


def test_climate_csv_unique_cells():
    _skip_if_no_processed()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "climate_2025.csv"))
    cells = df[["latitude", "longitude"]].drop_duplicates()
    expected_cells = 31 * 31  # temperature grid is 31x31
    # Some cells may be excluded if all temps are NaN
    assert len(cells) <= expected_cells
    assert len(cells) > 0


def test_climate_csv_days():
    _skip_if_no_processed()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "climate_2025.csv"))
    days = df["day"].unique()
    assert len(days) == 365, f"Expected 365 unique days, got {len(days)}"
    assert sorted(days) == list(range(1, 366))


def test_climate_csv_no_duplicate_rows():
    _skip_if_no_processed()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "climate_2025.csv"))
    n_dupes = df.duplicated(subset=["day", "latitude", "longitude"]).sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate (day, lat, lon) rows"
