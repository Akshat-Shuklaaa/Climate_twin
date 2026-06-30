"""
Tests for src.data.climate_loaders — validates .GRD loading from actual data files,
leap year handling, coordinate grids, and NaN sentinel replacement.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.climate_loaders import (
    is_leap_year,
    load_rainfall,
    load_max_temp,
    load_min_temp,
    get_rainfall_coordinates,
    get_temperature_coordinates,
)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "Datasets")


def _skip_if_no_datasets():
    if not os.path.exists(DATASETS_DIR):
        pytest.skip("Datasets/ directory not found — skipping integration test")


def test_is_leap_year():
    assert is_leap_year(2020) is True
    assert is_leap_year(2021) is False
    assert is_leap_year(1900) is False
    assert is_leap_year(2000) is True
    assert is_leap_year(2024) is True
    assert is_leap_year(2025) is False


def test_rainfall_coordinates():
    lats, lons = get_rainfall_coordinates()
    assert len(lats) == 129
    assert len(lons) == 135
    assert np.isclose(lats[0], 6.5)
    assert np.isclose(lats[-1], 38.5)
    assert np.isclose(lons[0], 66.5)
    assert np.isclose(lons[-1], 100.0)


def test_temperature_coordinates():
    lats, lons = get_temperature_coordinates()
    assert len(lats) == 31
    assert len(lons) == 31
    assert np.isclose(lats[0], 7.5)
    assert np.isclose(lats[-1], 37.5)
    assert np.isclose(lons[0], 67.5)
    assert np.isclose(lons[-1], 97.5)


def test_load_rainfall_2025():
    _skip_if_no_datasets()
    data = load_rainfall(2025)
    assert data.shape == (365, 129, 135), f"Expected (365, 129, 135), got {data.shape}"
    assert np.isnan(data).any(), "Should contain NaN values (from -999 sentinel)"
    assert data.dtype == np.float64 or data.dtype == np.float32


def test_load_max_temp_2025():
    _skip_if_no_datasets()
    data = load_max_temp(2025)
    assert data.shape == (365, 31, 31), f"Expected (365, 31, 31), got {data.shape}"
    assert np.isnan(data).any(), "Should contain NaN values (from 99.9 sentinel)"
    valid = data[~np.isnan(data)]
    assert valid.max() <= 60, "Max temperature should be physically plausible (< 60°C)"
    assert valid.min() >= -10, "Min temperature should be physically plausible (> -10°C)"


def test_load_min_temp_2025():
    _skip_if_no_datasets()
    data = load_min_temp(2025)
    assert data.shape == (365, 31, 31)
    assert np.isnan(data).any()


def test_leap_year_2024_rainfall():
    """2024 is a leap year — should produce 366 days of rainfall data."""
    _skip_if_no_datasets()
    data = load_rainfall(2024)
    assert data.shape[0] == 366, f"Leap year 2024 should have 366 days, got {data.shape[0]}"


def test_non_leap_year_2023_rainfall():
    _skip_if_no_datasets()
    data = load_rainfall(2023)
    assert data.shape[0] == 365, f"Non-leap year 2023 should have 365 days, got {data.shape[0]}"


def test_load_rainfall_has_nan_variation():
    """Not every grid cell should be NaN — there should be valid rainfall data."""
    _skip_if_no_datasets()
    data = load_rainfall(2025)
    # Check day 1, there should be some cells with valid data
    day1 = data[0]
    assert not np.isnan(day1).all(), "Day 1 should have some valid grid cells"
    # At least some values should be >= 0 (rainfall is non-negative)
    assert (day1[~np.isnan(day1)] >= 0).all(), "Rainfall values should be non-negative"
