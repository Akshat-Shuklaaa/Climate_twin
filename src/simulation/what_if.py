"""
what_if.py — Scenario simulation for climate perturbation analysis.

Provides run_scenario() to perturb starting conditions (rainfall delta,
temperature delta) and compare the forward-simulated trajectory against
an unperturbed baseline.

Scenario types:
  - percentage_rainfall_change(delta)   : scale rainfall by (1 + delta)
  - absolute_temperature_shift(delta)   : add delta to temperature
  - heatwave_drought_preset()           : rainfall -30%, temp +3°C
"""

import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.simulation.digital_twin import DigitalTwinState


def _perturb_history(history_df, rainfall_delta=0.0, temp_delta=0.0):
    """
    Apply perturbations to a history DataFrame in place.

    Parameters
    ----------
    history_df : pd.DataFrame
        Must have 'rainfall' and 'avg_temp' columns.
    rainfall_delta : float
        Multiplicative factor for rainfall (0.2 = +20%).
    temp_delta : float
        Additive shift for temperature in °C.
    """
    df = history_df.copy()
    if rainfall_delta != 0:
        df["rainfall"] = df["rainfall"] * (1 + rainfall_delta)
        df["rainfall"] = df["rainfall"].clip(lower=0)
    if temp_delta != 0:
        df["avg_temp"] = df["avg_temp"] + temp_delta
        df["max_temp"] = df["max_temp"] + temp_delta
        df["min_temp"] = df["min_temp"] + temp_delta
    return df


def run_scenario(
    lat,
    lon,
    horizon_days=14,
    rainfall_delta=0.0,
    temp_delta=0.0,
    twin=None,
):
    """
    Run a what-if scenario for a single grid cell and return baseline +
    scenario trajectories for direct comparison and plotting.

    Parameters
    ----------
    lat, lon : float
        Grid cell coordinates.
    horizon_days : int
        Number of forward-simulation days.
    rainfall_delta : float
        Fractional change to rainfall (0.2 = +20%, -0.3 = -30%).
    temp_delta : float
        Absolute temperature shift in °C.
    twin : DigitalTwinState, optional
        An existing twin instance. Created fresh if None.

    Returns
    -------
    pd.DataFrame
        Columns: day, baseline_rainfall, scenario_rainfall,
                 baseline_temp, scenario_temp
        Sorted by day ascending.
    """
    if twin is None:
        twin = DigitalTwinState()

    # --- Baseline ---
    baseline_forecast = twin.get_forecast(lat, lon, horizon_days)
    if len(baseline_forecast) == 0:
        return pd.DataFrame()

    baseline = baseline_forecast[
        ["day", "rainfall", "avg_temp"]
    ].rename(columns={"rainfall": "baseline_rainfall", "avg_temp": "baseline_temp"})

    # --- Scenario ---
    # Build a second twin with perturbed initial history
    scenario_twin = DigitalTwinState()

    # Perturb the starting known data for this cell before advancing
    history = scenario_twin.get_current_state(lat, lon, n_days=14)
    perturbed = _perturb_history(history, rainfall_delta, temp_delta)

    # Replace the cell's known rainfall/temp in the twin's internal data
    mask = (
        (scenario_twin._data["latitude"] == lat)
        & (scenario_twin._data["longitude"] == lon)
        & (scenario_twin._data["day"].isin(perturbed["day"]))
    )
    if mask.any():
        for col in ["rainfall", "max_temp", "min_temp", "avg_temp"]:
            if col in perturbed.columns:
                scenario_twin._data.loc[mask, col] = perturbed[col].values

    # Recompute derived features in the underlying data after perturbation
    scenario_twin._data["avg_temp"] = (
        scenario_twin._data["max_temp"] + scenario_twin._data["min_temp"]
    ) / 2
    scenario_twin._data["temp_range"] = (
        scenario_twin._data["max_temp"] - scenario_twin._data["min_temp"]
    )

    # Recompute lag/rolling features for the perturbed data
    scenario_twin._data = scenario_twin._data.sort_values(
        ["latitude", "longitude", "day"]
    ).reset_index(drop=True)
    grouped = scenario_twin._data.groupby(["latitude", "longitude"])
    scenario_twin._data["rainfall_lag_1"] = grouped["rainfall"].shift(1)
    scenario_twin._data["rainfall_lag_3"] = grouped["rainfall"].shift(3)
    scenario_twin._data["rainfall_lag_7"] = grouped["rainfall"].shift(7)
    scenario_twin._data["temp_lag_1"] = grouped["avg_temp"].shift(1)
    scenario_twin._data["temp_lag_3"] = grouped["avg_temp"].shift(3)
    scenario_twin._data["temp_lag_7"] = grouped["avg_temp"].shift(7)
    scenario_twin._data["rainfall_roll_3"] = grouped["rainfall"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    scenario_twin._data["rainfall_roll_7"] = grouped["rainfall"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    scenario_twin._data["temp_roll_3"] = grouped["avg_temp"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    scenario_twin._data["temp_roll_7"] = grouped["avg_temp"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Ensure no lag columns have nulls from the perturbation shift
    # (they won't, but fill any edge cases forward)
    scenario_twin._data = scenario_twin._data.bfill()

    scenario_forecast = scenario_twin.get_forecast(lat, lon, horizon_days)
    if len(scenario_forecast) == 0:
        return baseline.assign(
            scenario_rainfall=np.nan, scenario_temp=np.nan
        )

    scenario = scenario_forecast[
        ["day", "rainfall", "avg_temp"]
    ].rename(columns={"rainfall": "scenario_rainfall", "avg_temp": "scenario_temp"})

    result = baseline.merge(scenario, on="day", how="outer").sort_values("day")
    return result.reset_index(drop=True)


# --- Convenience wrappers ---

def percentage_rainfall_change(delta):
    """Return a partial that applies a percentage rainfall change."""
    return lambda lat, lon, horizon_days=14, twin=None: run_scenario(
        lat, lon, horizon_days, rainfall_delta=delta, twin=twin
    )


def absolute_temperature_shift(delta):
    """Return a partial that applies an absolute temperature shift."""
    return lambda lat, lon, horizon_days=14, twin=None: run_scenario(
        lat, lon, horizon_days, temp_delta=delta, twin=twin
    )


def heatwave_drought_preset():
    """
    Combined 'heatwave + drought' scenario: rainfall -30%, temperature +3°C.
    Maps directly to the project's stated focus on drought evolution and extremes.
    """
    return lambda lat, lon, horizon_days=14, twin=None: run_scenario(
        lat, lon, horizon_days, rainfall_delta=-0.3, temp_delta=3.0, twin=twin
    )
