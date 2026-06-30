"""
digital_twin.py — Lightweight digital twin state layer for India's climate.

DigitalTwinState holds the most recent known climate data per grid cell and
can advance forward in time using the trained ML models, maintaining lag and
rolling features across simulation steps.

Scope limitation (PoC): A full digital twin would assimilate live incoming
observations continuously. This version snapshots the most recent dataset
and forward-simulates from there — no data assimilation, just replay+
prediction. This is intentional and appropriate for a proof-of-concept.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed"
)
FEATURES_FILE = os.path.join(PROCESSED_DIR, "features_2025.csv")


class DigitalTwinState:
    """
    Represents the current known climate state for all grid cells.

    Parameters
    ----------
    features_path : str, optional
        Path to the features CSV to load as the "known state". Defaults to
        features_2025.csv.
    """

    def __init__(self, features_path=None):
        path = features_path or FEATURES_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DigitalTwinState requires a features file at {path}. "
                "Run feature engineering first."
            )
        self._data = pd.read_csv(path)
        self._data = self._data.sort_values(
            ["latitude", "longitude", "day"]
        ).reset_index(drop=True)
        self._year = 2025  # snapshot year
        self._max_known_day = int(self._data["day"].max())
        self._simulated = []  # list of DataFrames for simulated days

        # Known cells
        self._cells = (
            self._data[["latitude", "longitude"]].drop_duplicates().values
        )

    @property
    def cells(self):
        """Nx2 array of (latitude, longitude) grid cells."""
        return self._cells

    def get_current_state(self, lat, lon, n_days=14):
        """
        Return the most recent n_days of known (non-simulated) data for a cell.

        Parameters
        ----------
        lat, lon : float
            Grid cell coordinates.
        n_days : int
            Number of recent days to return.

        Returns
        -------
        pd.DataFrame
            Sorted by day ascending, last n_days rows for this cell.
        """
        cell = self._data[
            (self._data["latitude"] == lat) & (self._data["longitude"] == lon)
        ].copy()
        if len(cell) == 0:
            return pd.DataFrame()
        cell = cell.sort_values("day")
        return cell.tail(n_days).reset_index(drop=True)

    def advance(self, days=1):
        """
        Roll the state forward by `days` simulation steps.

        For each step, uses predict_next_day() to forecast rainfall and temp
        for every cell, then appends the predicted values as if they were
        newly observed, updating lag/rolling features for the next step.

        Parameters
        ----------
        days : int
            Number of days to advance.
        """
        from src.models.predict import predict_next_day

        for _ in range(days):
            sim_rows = []
            next_day = self._max_known_day + 1 + len(self._simulated)

            for lat, lon in self._cells:
                # Gather history: known + previously simulated days
                history = self._data[
                    (self._data["latitude"] == lat)
                    & (self._data["longitude"] == lon)
                ].copy()

                if self._simulated:
                    sim_df = pd.concat(self._simulated, ignore_index=True)
                    sim_cell = sim_df[
                        (sim_df["latitude"] == lat)
                        & (sim_df["longitude"] == lon)
                    ]
                    history = pd.concat(
                        [history, sim_cell], ignore_index=True
                    ).drop_duplicates(subset=["day", "latitude", "longitude"])

                if len(history) < 2:
                    continue

                rain_pred, temp_pred = predict_next_day(lat, lon, history)

                sim_rows.append(
                    {
                        "day": next_day,
                        "latitude": lat,
                        "longitude": lon,
                        "rainfall": max(rain_pred, 0),  # non-negative
                        "max_temp": temp_pred,
                        "min_temp": temp_pred - 2,  # rough diurnal range
                        "avg_temp": temp_pred,
                    }
                )

            sim_df = pd.DataFrame(sim_rows)
            self._simulated.append(sim_df)

    def _forecast_cell(self, lat, lon, horizon_days):
        """
        Forecast a single cell for horizon_days without advancing the full grid.
        This is the fast path used by the dashboard.
        """
        from src.models.predict import predict_next_day

        history = self._data[
            (self._data["latitude"] == lat) & (self._data["longitude"] == lon)
        ].copy()
        if len(history) == 0:
            return pd.DataFrame()

        history = history.sort_values("day").reset_index(drop=True)
        start_day = int(history["day"].max())

        rows = []
        for step in range(1, horizon_days + 1):
            rain_pred, temp_pred = predict_next_day(lat, lon, history)
            next_day = start_day + step
            row = {
                "day": next_day,
                "latitude": lat,
                "longitude": lon,
                "rainfall": max(rain_pred, 0),
                "max_temp": temp_pred,
                "min_temp": temp_pred - 2,
                "avg_temp": temp_pred,
            }
            rows.append(row)
            # Append predicted row as new history for next step
            pred_df = pd.DataFrame([row])
            history = pd.concat(
                [history, pred_df], ignore_index=True
            ).tail(30)  # keep window bounded

        return pd.DataFrame(rows)

    def get_forecast(self, lat, lon, horizon_days=7):
        """
        Return a forecast trajectory for a single grid cell.

        Uses the fast per-cell simulation path. Does NOT advance the global
        state — this is a lightweight query for dashboard use.

        Parameters
        ----------
        lat, lon : float
            Grid cell coordinates.
        horizon_days : int
            Number of forecast days.

        Returns
        -------
        pd.DataFrame
            Columns: day, rainfall, max_temp, min_temp, avg_temp
            Sorted by day ascending.
        """
        result = self._forecast_cell(lat, lon, horizon_days)
        return result.sort_values("day").reset_index(drop=True)

    def get_current_dataframe(self):
        """
        Return the combined known + simulated data as a single DataFrame.
        Useful for dashboard map rendering.
        """
        frames = [self._data]
        if self._simulated:
            frames.append(pd.concat(self._simulated, ignore_index=True))
        return pd.concat(frames, ignore_index=True)
