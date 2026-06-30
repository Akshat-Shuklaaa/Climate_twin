# Presentation Notes — Climate_twin

## Architecture (In Plain Language)

**Climate_twin** is an AI-powered digital twin of India's climate. It starts with raw weather data from the India Meteorological Department (IMD) — binary files containing daily rainfall, maximum temperature, and minimum temperature readings across India for 2023–2025.

The pipeline works in stages:
1. **Load:** Raw `.GRD` binary files are parsed into numerical arrays.
2. **Fuse:** Rainfall (measured at a finer 0.25° resolution) is averaged onto the same 1° grid as temperature, so both variables align spatially.
3. **Feature engineer:** For every grid cell, we compute lag features (yesterday's, 3-days-ago, 7-days-ago values), rolling averages (3-day and 7-day windows), and other derived quantities — giving the ML model temporal context.
4. **Train:** A RandomForest model learns to predict next-day rainfall and temperature from these features. A "persistence" baseline (predict tomorrow = today's value) is computed for comparison.
5. **Digital twin state layer:** A lightweight object holds the most recent climate state and can roll forward in time, using the model's predictions as if they were observations.
6. **What-if simulation:** Users can perturb starting conditions (e.g., "+2°C, -30% rainfall") and see how the forecast trajectory diverges from baseline.
7. **Dashboard:** A Streamlit app with a map of India's grid cells, forecast charts, what-if controls, and model performance metrics — all interactive and live.

## What Was Built vs Scoped Out

| Built | Scoped Out |
|---|---|
| Full data pipeline (IMD .GRD → features CSV) | INSAT satellite data integration (requires MOSDAC registration, HDF5 parsing, ~2-3 days effort) |
| RF model + persistence baseline for comparison | Deep learning (CNN/LSTM) — judged too complex for a PoC where explainability matters |
| Digital twin state layer (forward simulation) | Live data assimilation (current version snapshots the most recent dataset) |
| What-if scenarios (rainfall %, temp shift, combined preset) | Ocean-atmosphere coupling (ENSO indices, IOD) |
| Interactive Streamlit dashboard (4 tabs) | Real-time data feeds from IMD API |
| pytest test suite (31 tests) | |

## Model Performance vs Baseline

The key numbers:

| Metric | Rainfall (RF) | Rainfall (Persistence) | Temperature (RF) | Temperature (Persistence) |
|---|---|---|---|---|
| RMSE | 8.83 mm | 13.05 mm | 0.73 °C | 1.11 °C |
| MAE | 3.86 mm | 5.21 mm | 0.54 °C | 0.83 °C |
| R² | 0.350 | -0.418 | 0.976 | 0.944 |

**Why this matters:** The persistence baseline ("tomorrow will be the same as today") is surprisingly strong for temperature (R²=0.944) — because day-to-day temperature is very stable. The RF model still beats it by 34% on RMSE. For rainfall, persistence actually performs *worse* than predicting the long-term mean (R² negative), while RF achieves R²=0.350 — a meaningful signal for a chaotic variable.

## Evaluation Criteria Mapping

### 1. Problem Understanding
- **Addressed by:** The project tackles the stated challenge of building a digital twin for Indian climate using IMD data. The README's "Current Limitations" section honestly documents what's scoped in vs out.
- **Evidence:** Pipeline design decisions (time-based split, persistence baseline, per-grid-cell modeling) show understanding that climate data is spatiotemporal and auto-correlated.

### 2. Data Usage
- **Addressed by:** Three years (2023–2025) of IMD gridded rainfall and temperature data ingested from raw binary `.GRD` files. Spatial fusion from 0.25° rainfall to 1° temperature grid.
- **Evidence:** `climate_loaders.py` handles leap years, NaN sentinel replacement, coordinate grids. Processed data stored as CSVs for reproducibility.

### 3. Model Development
- **Addressed by:** RandomForestRegressor trained on engineered time-series features (lags, rolling averages). Time-based train/test split per grid cell to prevent data leakage.
- **Evidence:** `train.py` with clear feature/target separation, `predict.py` with a reusable API, persistence baseline as a controlled comparison.

### 4. Prediction Performance & Validation
- **Addressed by:** RMSE, MAE, R² reported for both model and baseline. Diagnostic plots (predicted vs actual, residuals, time-series) saved to `outputs/figures/`.
- **Evidence:** The model beats persistence on both targets. Temperature R²=0.976 is excellent; rainfall R²=0.350 is modest but shows clear value over persistence (which scores negative R²).

### 5. Digital Twin Concept
- **Addressed by:** `DigitalTwinState` class that maintains current climate state and can roll forward using model predictions, maintaining lag/rolling features across steps. The docstring explicitly scopes this as a PoC state layer.
- **Evidence:** `advance()`, `get_forecast()`, `get_current_state()` methods. The what-if module calls this to compare perturbed vs baseline trajectories.

### 6. Visualization
- **Addressed by:** Four-tab Streamlit dashboard with:
  - Interactive map of grid cells colored by rainfall/temperature
  - Forecast time-series charts with persistence overlay
  - What-if sliders with live scenario comparison
  - Model performance metrics table and diagnostic plots
- **Evidence:** `src/dashboard/app.py` uses Plotly and Streamlit for responsive, interactive visualization.

### 7. Innovation
- **Addressed by:** What-if scenario engine that perturbs starting conditions and forward-simulates the trajectory — users can explore "what if this region saw a heatwave + drought" interactively via the dashboard. The combined heatwave+drought preset maps directly to the problem statement's focus on drought evolution.
- **Evidence:** `run_scenario()` with `heatwave_drought_preset()`, integrated into the dashboard.

### 8. Communication
- **Addressed by:** README with ASCII architecture diagram, setup and run instructions, honest limitations section. This presentation notes document. Clean, commented code with module-level docstrings.
- **Evidence:** Every new module has a docstring explaining purpose and design rationale.

## Suggested Pitch Structure (3 Minutes)

1. **Hook (30s):** "India's climate is changing — droughts are intensifying, heatwaves are more frequent. We built a digital twin that can simulate what-if scenarios and forecast local climate conditions days ahead."

2. **What it does (45s):** Walk the pipeline: IMD data → feature engineering → ML model → digital twin → dashboard. Show the map and forecast.

3. **The key result (30s):** "Our model beats persistence baseline on both rainfall and temperature — for rainfall, RMSE drops from 13mm to 8.8mm; for temperature, R² reaches 0.976."

4. **What-if demo (45s):** "Here's the most powerful part — we can ask 'what if this region sees 30% less rainfall and 3°C higher temperatures?' and watch the simulation diverge from the baseline in real time."

5. **Scope & next steps (30s):** Honest about what's not included (INSAT, data assimilation, longer training period) — but clear that the architecture supports all of them as drop-in upgrades.
