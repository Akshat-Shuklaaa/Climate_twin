"""
evaluate.py — Compute and display evaluation metrics for trained models.

Reads the test_predictions.csv produced by train.py, computes RMSE/MAE/R²
for both the RandomForest model and the persistence baseline, generates
diagnostic plots, and saves the summary to outputs/.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoints"
)
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, label):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"model": label, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return {
        "model": label,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_scatter(y_true, y_pred, label, target_name, ax):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ax.scatter(y_true, y_pred, alpha=0.3, s=1)
    lims = [
        min(np.nanmin(y_true), np.nanmin(y_pred)),
        max(np.nanmax(y_true), np.nanmax(y_pred)),
    ]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{label} — {target_name}")
    ax.set_aspect("equal")


def plot_residuals(y_true, y_pred, global_day, label, target_name, ax):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(global_day))
    residuals = y_true[mask] - y_pred[mask]
    ax.scatter(global_day[mask], residuals, alpha=0.3, s=1)
    ax.axhline(0, color="r", linestyle="--", lw=1)
    ax.set_xlabel("Global day")
    ax.set_ylabel("Residual")
    ax.set_title(f"{label} — {target_name} residuals")


def plot_timeseries(df, lat, lon, ax):
    cell = df[(df["latitude"] == lat) & (df["longitude"] == lon)].sort_values("global_day")
    if len(cell) < 10:
        return
    ax.plot(cell["global_day"], cell["target_rainfall"], label="Actual rain", alpha=0.7)
    ax.plot(cell["global_day"], cell["pred_rainfall"], label="Predicted rain", alpha=0.7)
    ax.plot(cell["global_day"], cell["persist_rainfall"], label="Persistence rain", alpha=0.5, linestyle=":")
    ax.set_xlabel("Global day")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title(f"Rainfall time-series at ({lat:.1f}°N, {lon:.1f}°E)")
    ax.legend(fontsize=8)


def main():
    test_path = os.path.join(CHECKPOINTS_DIR, "test_predictions.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found. Run train.py (via run_pipeline.py) first."
        )

    df = pd.read_csv(test_path)
    print(f"[evaluate] Loaded {len(df)} test predictions")

    # Metrics for each combination
    metrics_list = [
        compute_metrics(df["target_rainfall"].values, df["pred_rainfall"].values, "RF-Rainfall"),
        compute_metrics(df["target_avg_temp"].values, df["pred_avg_temp"].values, "RF-Temperature"),
        compute_metrics(
            df["target_rainfall"].values, df["persist_rainfall"].values, "Persistence-Rainfall"
        ),
        compute_metrics(
            df["target_avg_temp"].values, df["persist_avg_temp"].values, "Persistence-Temperature"
        ),
    ]

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[evaluate] Saved metrics to {metrics_path}")

    # Console summary
    print()
    print("=" * 72)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("=" * 72)
    for _, row in metrics_df.iterrows():
        print(f"{row['model']:<22} {row['rmse']:>8.3f} {row['mae']:>8.3f} {row['r2']:>8.3f}")
    print("=" * 72)

    # Figures
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Model Diagnostics", fontsize=14)

    # Row 0: scatter plots
    plot_scatter(
        df["target_rainfall"].values, df["pred_rainfall"].values, "RF", "Rainfall", axes[0, 0]
    )
    plot_scatter(
        df["target_avg_temp"].values, df["pred_avg_temp"].values, "RF", "Avg Temp", axes[0, 1]
    )
    axes[0, 2].axis("off")

    # Row 1: residual plots
    plot_residuals(
        df["target_rainfall"].values,
        df["pred_rainfall"].values,
        df["global_day"].values,
        "RF",
        "Rainfall",
        axes[1, 0],
    )
    plot_residuals(
        df["target_avg_temp"].values,
        df["pred_avg_temp"].values,
        df["global_day"].values,
        "RF",
        "Avg Temp",
        axes[1, 1],
    )
    # Sample time-series for a central cell
    unique_cells = df[["latitude", "longitude"]].drop_duplicates()
    if len(unique_cells) > 0:
        sample_cell = unique_cells.iloc[len(unique_cells) // 2]
        plot_timeseries(df, sample_cell["latitude"], sample_cell["longitude"], axes[1, 2])

    plt.tight_layout()
    scatter_path = os.path.join(FIGURES_DIR, "diagnostics.png")
    plt.savefig(scatter_path, dpi=150)
    print(f"[evaluate] Saved diagnostic plots to {scatter_path}")
    plt.close()


if __name__ == "__main__":
    main()
