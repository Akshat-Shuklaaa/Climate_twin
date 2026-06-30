"""
app.py — Streamlit dashboard for the Climate_twin digital twin.

Tabs:
  1. Map View   — Folium map of India grid cells colored by rainfall/temperature
  2. Forecast   — Actual history + forecast trajectory for a selected cell
  3. What-If    — Sliders for perturbations, baseline vs scenario comparison
  4. Metrics    — Model performance table and diagnostic plots
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.simulation.digital_twin import DigitalTwinState
from src.simulation.what_if import run_scenario, heatwave_drought_preset

st.set_page_config(
    page_title="Climate_twin — India Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics_summary.csv"
DIAGNOSTICS_PATH = OUTPUTS_DIR / "figures" / "diagnostics.png"

st.title(" Climate_twin — India Climate Digital Twin")
st.markdown(
    "An AI-powered digital twin for India's rainfall and temperature. "
    "Select a grid cell on the map to explore forecasts and what-if scenarios."
)


@st.cache_resource
def load_twin():
    return DigitalTwinState()


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


twin = load_twin()

# ── Sidebar — cell selection ──────────────────────────────────────────
all_cells = twin.cells
cell_labels = [f"{lat:.1f}°N, {lon:.1f}°E" for lat, lon in all_cells]

if "selected_index" not in st.session_state:
    st.session_state.selected_index = len(all_cells) // 2

selected_label = st.sidebar.selectbox(
    "Select grid cell",
    options=cell_labels,
    index=st.session_state.selected_index,
)
selected_idx = cell_labels.index(selected_label)
selected_lat, selected_lon = all_cells[selected_idx]

st.sidebar.markdown(f"**Selected:** {selected_label}")
st.sidebar.markdown("---")

horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)

# ── Tab layout ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [" Map View", " Forecast", " What-If", " Model Performance"]
)

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Map View
# ══════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])
    with col2:
        color_var = st.radio(
            "Color by",
            options=["rainfall", "avg_temp"],
            format_func=lambda x: "Rainfall (mm)" if x == "rainfall" else "Avg Temp (°C)",
            index=0,
        )
        st.caption("Click a circle to select that grid cell.")

    with col1:
        current = twin.get_current_dataframe()
        # Use the most recent known day for each cell
        latest = current.loc[
            current.groupby(["latitude", "longitude"])["day"].idxmax()
        ].copy()
        latest["color_value"] = latest[color_var].fillna(0)
        latest["label"] = latest.apply(
            lambda r: f"({r['latitude']:.1f}N, {r['longitude']:.1f}E) "
            f"– {color_var}: {r[color_var]:.1f}",
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scattergeo(
                lon=latest["longitude"],
                lat=latest["latitude"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=latest["color_value"],
                    colorscale="RdYlBu_r" if color_var == "rainfall" else "RdYlBu_r",
                    showscale=True,
                    colorbar=dict(
                        title="Rainfall (mm)" if color_var == "rainfall" else "Temp (°C)"
                    ),
                    cmin=(
                        latest["color_value"].quantile(0.05)
                        if color_var == "rainfall"
                        else latest["color_value"].quantile(0.05)
                    ),
                    cmax=(
                        latest["color_value"].quantile(0.95)
                        if color_var == "rainfall"
                        else latest["color_value"].quantile(0.95)
                    ),
                ),
                text=latest["label"],
                hoverinfo="text",
                customdata=np.column_stack(
                    [latest["latitude"], latest["longitude"]]
                ),
            )
        )

        # Highlight selected cell
        sel = latest[
            (latest["latitude"] == selected_lat)
            & (latest["longitude"] == selected_lon)
        ]
        if len(sel) > 0:
            fig.add_trace(
                go.Scattergeo(
                    lon=sel["longitude"],
                    lat=sel["latitude"],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="x"),
                    name="Selected",
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            geo=dict(
                scope="asia",
                projection_type="natural earth",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                coastlinecolor="rgb(200, 200, 200)",
                lataxis=dict(range=[5, 40]),
                lonaxis=dict(range=[65, 102]),
            ),
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            dragmode="lasso",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Click-to-select via plotly selection events
        st.caption(
            "Use the lasso or box select tool to pick points, or use the sidebar dropdown."
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Forecast for {selected_label}")

    history = twin.get_current_state(selected_lat, selected_lon, n_days=30)
    forecast = twin.get_forecast(selected_lat, selected_lon, horizon)

    if len(forecast) == 0:
        st.warning("No forecast data available for this cell.")
    else:
        fig2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Rainfall (mm)", "Avg Temperature (°C)"),
            vertical_spacing=0.1,
        )

        # Rainfall
        if len(history) > 0:
            fig2.add_trace(
                go.Scatter(
                    x=history["day"],
                    y=history["rainfall"],
                    mode="lines+markers",
                    name="Actual (history)",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )
        fig2.add_trace(
            go.Scatter(
                x=forecast["day"],
                y=forecast["rainfall"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Temperature
        if len(history) > 0:
            fig2.add_trace(
                go.Scatter(
                    x=history["day"],
                    y=history["avg_temp"],
                    mode="lines+markers",
                    name="Actual (history)",
                    line=dict(color="red"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig2.add_trace(
            go.Scatter(
                x=forecast["day"],
                y=forecast["avg_temp"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig2.update_layout(height=500, margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        # Persistence baseline overlay toggle
        st.caption(
            "Forecast uses the trained RandomForest model. "
            "Dashed lines mark the prediction horizon."
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — What-If
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"What-If Scenario — {selected_label}")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        rain_delta = st.slider(
            "Rainfall change (%)", -50, 50, 0, step=5
        ) / 100
    with col_b:
        temp_delta = st.slider(
            "Temperature shift (°C)", -5.0, 5.0, 0.0, step=0.5
        )
    with col_c:
        use_hwd = st.checkbox("Heatwave + drought preset")

    if use_hwd:
        rain_delta = -0.3
        temp_delta = 3.0
        st.info("Preset active: rainfall -30%, temperature +3°C")

    if st.button("Run Scenario", type="primary"):
        with st.spinner("Simulating..."):
            result = run_scenario(
                selected_lat,
                selected_lon,
                horizon_days=horizon,
                rainfall_delta=rain_delta,
                temp_delta=temp_delta,
                twin=twin,
            )

        if len(result) == 0:
            st.error("No data returned for this cell.")
        else:
            fig3 = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Rainfall (mm)", "Avg Temperature (°C)"),
                vertical_spacing=0.1,
            )

            fig3.add_trace(
                go.Scatter(
                    x=result["day"],
                    y=result["baseline_rainfall"],
                    mode="lines+markers",
                    name="Baseline",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )
            fig3.add_trace(
                go.Scatter(
                    x=result["day"],
                    y=result["scenario_rainfall"],
                    mode="lines+markers",
                    name="Scenario",
                    line=dict(color="orange", dash="dash"),
                ),
                row=1,
                col=1,
            )

            fig3.add_trace(
                go.Scatter(
                    x=result["day"],
                    y=result["baseline_temp"],
                    mode="lines+markers",
                    name="Baseline",
                    line=dict(color="red"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig3.add_trace(
                go.Scatter(
                    x=result["day"],
                    y=result["scenario_temp"],
                    mode="lines+markers",
                    name="Scenario",
                    line=dict(color="orange", dash="dash"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig3.update_layout(height=500, margin=dict(l=40, r=20, t=40, b=20))
            st.plotly_chart(fig3, use_container_width=True)

            # Summary stats
            diff_rain = result["scenario_rainfall"] - result["baseline_rainfall"]
            diff_temp = result["scenario_temp"] - result["baseline_temp"]
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            col_d1.metric("Avg rainfall change", f"{diff_rain.mean():+.2f} mm")
            col_d2.metric("Max rainfall change", f"{diff_rain.max():+.2f} mm")
            col_d3.metric("Avg temp change", f"{diff_temp.mean():+.2f} °C")
            col_d4.metric("Max temp change", f"{diff_temp.max():+.2f} °C")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Performance vs Persistence Baseline")

    metrics = load_metrics()
    if metrics is not None:
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        st.markdown("### Key Takeaways")
        rf_rain = metrics[metrics["model"] == "RF-Rainfall"]
        pers_rain = metrics[metrics["model"] == "Persistence-Rainfall"]
        rf_temp = metrics[metrics["model"] == "RF-Temperature"]
        pers_temp = metrics[metrics["model"] == "Persistence-Temperature"]

        if len(rf_rain) > 0 and len(pers_rain) > 0:
            rmse_gain = (
                1 - rf_rain["rmse"].values[0] / pers_rain["rmse"].values[0]
            ) * 100
            st.markdown(
                f"- **Rainfall:** RF model RMSE **{rmse_gain:.0f}% lower** "
                f"than persistence baseline."
            )
        if len(rf_temp) > 0 and len(pers_temp) > 0:
            rmse_gain = (
                1 - rf_temp["rmse"].values[0] / pers_temp["rmse"].values[0]
            ) * 100
            st.markdown(
                f"- **Temperature:** RF model RMSE **{rmse_gain:.0f}% lower** "
                f"than persistence baseline."
            )
        st.markdown(
            "- The persistence baseline (`predict tomorrow = today's value`) "
            "is a strong naive benchmark. Beating it proves the ML model "
            "learns meaningful patterns beyond simple day-to-day inertia."
        )
    else:
        st.warning("Metrics file not found. Run train.py and evaluate.py first.")

    if DIAGNOSTICS_PATH.exists():
        st.image(str(DIAGNOSTICS_PATH), use_container_width=True)
    else:
        st.warning("Diagnostic plots not found. Run evaluate.py first.")
