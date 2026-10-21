
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================================
# DECISION STUDIO | HEALTHCARE OPERATIONS ANALYTICS
# Educational dashboard using synthetic data only.
# ==========================================================

st.set_page_config(
    page_title="Decision Studio | Healthcare Analytics",
   
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the stylesheet from the same folder as app.py
css_path = Path(__file__).with_name("styles.css")

if css_path.exists():
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# -------------------- SIMULATION FUNCTIONS --------------------

def create_demand(days, seed=7):
    """Create illustrative patient demand."""
    rng = np.random.default_rng(seed)

    day = np.arange(days)
    weekly_pattern = np.array(
        [0.92, 0.88, 0.86, 0.90, 1.05, 1.18, 1.12]
    )

    seasonal = 1 + 0.12 * np.sin(2 * np.pi * day / 14)
    trend = 1 + 0.02 * day / max(days - 1, 1)

    arrivals = (
        95 * weekly_pattern[day % 7] * seasonal * trend
    )

    low_acuity = np.clip(
        0.63
        + rng.normal(0, 0.02, days)
        + 0.08 * np.sin(2 * np.pi * day / 14),
        0.25,
        0.90,
    )

    return pd.DataFrame({
        "Day": day + 1,
        "Expected arrivals": arrivals,
        "Low acuity probability": low_acuity,
    })


def run_simulation(
    demand,
    demand_multiplier,
    volatility,
    acuity_shift,
    capacity,
    staffing,
    triage,
    risk_threshold,
    runs,
    seed=123,
):
    """Run a simplified Monte Carlo operations simulation."""

    rng = np.random.default_rng(seed)
    records = []

    volatility = float(np.clip(volatility, 0, 0.50))
    gamma_shape = 1 / max(volatility ** 2, 1e-6)

    effective_capacity = max(
        1, capacity * (0.65 + 1.1 * staffing)
    )

    for run in range(1, runs + 1):
        queue = 0.0
        weighted_wait = 0.0
        total_patients = 0
        total_risk = 0.0
        overflow_days = 0
        target_miss_patients = 0

        for row in demand.itertuples(index=False):
            noise = rng.gamma(
                shape=gamma_shape,
                scale=1 / gamma_shape,
            )

            arrivals = int(
                rng.poisson(
                    row[1] * demand_multiplier * noise
                )
            )

            high_acuity = np.clip(
                1 - row[2] + acuity_shift * 0.25,
                0.05,
                0.85,
            )

            available = effective_capacity
            queue = max(0, queue + arrivals - available)

            daily_wait = (
                8
                + queue * 0.62
                + (arrivals / available) * 9
            )

            if triage:
                daily_wait *= 1 - 0.10 * high_acuity

            daily_wait += rng.normal(
                0, 2.2 + volatility * 3.5
            )
            daily_wait = max(2, daily_wait)

            # Illustrative risk score, not a clinical measurement.
            risk_score = np.clip(
                0.025
                + max(daily_wait - 15, 0) * 0.0017
                + high_acuity * 0.045
                + volatility * 0.02
                - (0.012 if triage else 0)
                + rng.normal(0, 0.008),
                0.005,
                0.35,
            )

            weighted_wait += daily_wait * arrivals
            total_patients += arrivals
            total_risk += risk_score * arrivals

            overflow_days += int(
                queue > effective_capacity * 0.35
            )

            # Count modeled patients above the waiting-time target.
            if daily_wait > 30:
                target_miss_patients += arrivals

        average_wait = (
            weighted_wait / max(total_patients, 1)
        )

        miss_percent = (
            100 * target_miss_patients
            / max(total_patients, 1)
        )

        risk_percent = (
            100 * total_risk / max(total_patients, 1)
        )

        cost = (
            8500
            + total_patients * 22
            + capacity * 18
            + staffing * 1700
            + overflow_days * 420
        )

        records.append({
            "Simulation run": run,
            "Average wait (minutes)": round(
                average_wait, 1
            ),
            "Target misses (%)": round(
                miss_percent, 1
            ),
            "Synthetic risk indicator (%)": round(
                risk_percent, 1
            ),
            "Overflow days": overflow_days,
            "Cost proxy (synthetic units)": round(cost),
        })

    return pd.DataFrame(records)


def style_chart(fig, height=285):
    """Apply consistent chart styling."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=12, b=12),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            size=11,
            color="#45617f",
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e7eff8",
        zeroline=False,
        linecolor="#d6e4f2",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e7eff8",
        zeroline=False,
        linecolor="#d6e4f2",
    )

    return fig


def metric_card(icon, label, value, note, color):
    """Render a reusable KPI card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon {color}">{icon}</div>
            <div class="metric-content">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- SIDEBAR CONTROLS --------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✚</div>
            <div>
                <div class="brand-name">Decision Studio</div>
                <div class="brand-sub">HEALTHCARE ANALYTICS</div>
            </div>
        </div>
        <div class="sidebar-heading">
            ⚙ Simulation Controls
        </div>
        """,
        unsafe_allow_html=True,
    )

    preset = st.selectbox(
        "Scenario preset",
        [
            "Balanced operations",
            "High patient demand",
            "Capacity constrained",
            "Surge response",
        ],
    )

    presets = {
        "Balanced operations": (1.00, 0.12, 0.00, 110, 1.00),
        "High patient demand": (1.25, 0.22, 0.08, 110, 1.00),
        "Capacity constrained": (1.05, 0.16, 0.03, 82, 0.82),
        "Surge response": (1.35, 0.28, 0.12, 125, 1.18),
    }

    defaults = presets[preset]

    period = st.selectbox(
        "Analysis window",
        ["7 days", "14 days", "21 days", "30 days"],
        index=2,
    )

    days = int(period.split()[0])

    st.markdown(
        '<div class="control-heading">Demand and risk</div>',
        unsafe_allow_html=True,
    )

    demand_multiplier = st.slider(
        "Patient demand multiplier",
        0.60, 1.60, defaults[0], 0.05,
        format="%.2f",
    )

    volatility = st.slider(
        "Demand volatility",
        0.00, 0.50, defaults[1], 0.01,
        format="%.2f",
    )

    acuity_shift = st.slider(
        "High-acuity shift",
        -0.15, 0.25, defaults[2], 0.01,
        format="%.2f",
    )

    st.markdown(
        '<div class="control-heading">Operational capacity</div>',
        unsafe_allow_html=True,
    )

    capacity = st.slider(
        "Daily service capacity",
        60, 160, defaults[3], 5,
    )

    staffing = st.slider(
        "Staffing level",
        0.50, 1.50, defaults[4], 0.05,
        format="%.2f",
    )

    triage = st.toggle(
        "Enable triage support",
        value=True,
    )

    st.markdown(
        '<div class="control-heading">Simulation settings</div>',
        unsafe_allow_html=True,
    )

    risk_threshold = st.slider(
        "Risk review threshold",
        0.05, 0.30, 0.15, 0.01,
        format="%.2f",
    )

    runs = st.selectbox(
        "Number of simulation runs",
        [25, 50, 100, 250],
        index=2,
    )

    run_clicked = st.button(
        "▶  Run simulation",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="sidebar-foot">
            ◉ Synthetic data only<br>
            Educational demonstration.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- RUN AND STORE SIMULATION --------------------

current_settings = (
    preset, days, demand_multiplier, volatility,
    acuity_shift, capacity, staffing, triage,
    risk_threshold, runs,
)

if "previous_settings" not in st.session_state:
    st.session_state.previous_settings = None

if "simulation_seed" not in st.session_state:
    st.session_state.simulation_seed = 123

if (
    current_settings != st.session_state.previous_settings
    or run_clicked
):
    st.session_state.simulation_seed += 1
    st.session_state.previous_settings = current_settings

demand_df = create_demand(days)

results = run_simulation(
    demand_df,
    demand_multiplier,
    volatility,
    acuity_shift,
    capacity,
    staffing,
    triage,
    risk_threshold,
    runs,
    seed=st.session_state.simulation_seed,
)

# Reproducible daily arrivals for the demand chart.
daily_rng = np.random.default_rng(88 + days)

demand_df["Simulated arrivals"] = np.maximum(
    0,
    np.round(
        demand_df["Expected arrivals"]
        * demand_multiplier
        * (
            1 + daily_rng.normal(
                0, volatility, days
            )
        )
    ),
).astype(int)

avg_wait = results["Average wait (minutes)"].mean()
miss_rate = results["Target misses (%)"].mean()
risk_rate = results["Synthetic risk indicator (%)"].mean()
overflow = results["Overflow days"].mean()
avg_cost = results["Cost proxy (synthetic units)"].mean()

target_achieved = max(0, 100 - miss_rate)
risk_flag = risk_rate >= risk_threshold * 100


# -------------------- HERO HEADER --------------------

st.markdown(
    """
    <section class="hero">
        <div class="hero-copy">
            <div class="eyebrow">
                HEALTHCARE OPERATIONS ANALYTICS
            </div>
            <h1>REDOS</h1>
            <h1>Decision Studio</h1>
            <p>
                Explore how patient demand, staffing, capacity,
                and triage decisions may affect waiting times
                and operational risk.
            </p>
            <div class="hero-note">
                ⚙ Interactive decision-support prototype using
                synthetic data. Results are illustrative and
                are not clinical predictions.
            </div>
        </div>
        <div class="hero-art">
            <div class="hero-cross">✚</div>
            <div class="hero-pulse">⌁〰</div>
            <div class="hero-medical">🩺</div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


# -------------------- KPI CARDS --------------------

st.markdown(
    '<div class="section-title">Operational overview</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4, gap="small")

with k1:
    metric_card(
        "◷", "Average waiting time",
        f"{avg_wait:.1f} min",
        "Across simulation runs", "blue",
    )

with k2:
    metric_card(
        "♢", "Wait-time target misses",
        f"{miss_rate:.1f}%",
        "Modeled patients above target", "mint",
    )

with k3:
    metric_card(
        "⚠", "Synthetic risk indicator",
        f"{risk_rate:.1f}%",
        "Model estimate only", "purple",
    )

with k4:
    metric_card(
        "♟", "Average overflow days",
        f"{overflow:.1f}",
        "Per simulation run", "teal",
    )

st.markdown(
    f"""
    <div class="micro-note">
        Results averaged across {runs} simulation runs and
        {days} days. Cost proxy: {avg_cost:,.0f}
        synthetic units per run.
        Risk review status: {"Threshold reached" if risk_flag else "Below review threshold"}.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- CHARTS 1, 2, 3 --------------------

c1, c2, c3 = st.columns(
    [1.28, 1.02, 1.02],
    gap="small",
)

with c1:
    st.markdown(
        '<div class="chart-title">1. Patient demand across the selected period</div>'
        '<div class="chart-subtitle">Expected daily patient arrivals</div>',
        unsafe_allow_html=True,
    )

    fig = px.line(
        demand_df,
        x="Day",
        y="Simulated arrivals",
        markers=True,
    )

    fig.update_traces(
        line=dict(color="#2878f0", width=2.5),
        marker=dict(size=5, color="#2878f0"),
    )

    fig.update_xaxes(title="Day")
    fig.update_yaxes(title="", rangemode="tozero")

    st.plotly_chart(
        style_chart(fig, 280),
        use_container_width=True,
        config={"displayModeBar": False},
    )


with c2:
    st.markdown(
        '<div class="chart-title">2. Waiting-time distribution</div>'
        '<div class="chart-subtitle">Distribution of average waiting times</div>',
        unsafe_allow_html=True,
    )

    fig = px.histogram(
        results,
        x="Average wait (minutes)",
        nbins=14,
    )

    fig.update_traces(
        marker_color="#7779f5",
        marker_line_color="#ffffff",
        marker_line_width=1.5,
    )

    fig.update_xaxes(title="Average waiting time (minutes)")
    fig.update_yaxes(title="Simulation runs")

    st.plotly_chart(
        style_chart(fig, 280),
        use_container_width=True,
        config={"displayModeBar": False},
    )


with c3:
    st.markdown(
        '<div class="chart-title">3. Wait-time target performance</div>'
        '<div class="chart-subtitle">Estimated target performance across runs</div>',
        unsafe_allow_html=True,
    )

    target_df = pd.DataFrame({
        "Outcome": ["Target misses", "Target achieved"],
        "Patients (%)": [miss_rate, target_achieved],
    })

    fig = px.bar(
        target_df,
        x="Outcome",
        y="Patients (%)",
        color="Outcome",
        text=target_df["Patients (%)"].map(
            lambda value: f"{value:.1f}%"
        ),
        color_discrete_map={
            "Target misses": "#f29b9b",
            "Target achieved": "#4297f5",
        },
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_xaxes(title="")
    fig.update_yaxes(
        title="Patients (%)",
        range=[0, 115],
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        style_chart(fig, 280),
        use_container_width=True,
        config={"displayModeBar": False},
    )


# -------------------- CHART 4 AND RESULTS TABLE --------------------

c4, c5 = st.columns(
    [1.08, 1.0],
    gap="small",
)

with c4:
    st.markdown(
        '<div class="chart-title">4. Waiting time versus synthetic risk indicator</div>'
        '<div class="chart-subtitle">Relationship between modeled waiting time and risk</div>',
        unsafe_allow_html=True,
    )

    fig = px.scatter(
        results,
        x="Average wait (minutes)",
        y="Synthetic risk indicator (%)",
        color="Cost proxy (synthetic units)",
        hover_data=[
            "Simulation run",
            "Overflow days",
            "Target misses (%)",
        ],
        color_continuous_scale="Viridis",
    )

    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.82,
            line=dict(width=0.5, color="white"),
        )
    )

    fig.update_xaxes(
        title="Average waiting time (minutes)"
    )

    fig.update_yaxes(
        title="Synthetic risk indicator (%)"
    )

    st.plotly_chart(
        style_chart(fig, 305),
        use_container_width=True,
        config={"displayModeBar": False},
    )


with c5:
    with st.expander(
        " View detailed simulation results",
        expanded=True,
    ):
        display_df = results.head(10).copy()

        display_df[
            "Cost proxy (synthetic units)"
        ] = display_df[
            "Cost proxy (synthetic units)"
        ].map(lambda value: f"{value:,}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=260,
        )

        st.download_button(
            "⬇  Download results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="decision_studio_results.csv",
            mime="text/csv",
        )


# -------------------- INTERPRETATION --------------------

st.markdown(
    f"""
    <section class="interpretation">
        <div class="info-icon">ⓘ</div>
        <div class="interpretation-content">
            <div class="interpretation-title">
                How to interpret these results
            </div>
            <div class="interpretation-grid">
                <div>
                    <b>• Waiting time:</b>
                    The model estimates {avg_wait:.1f} minutes
                    under the selected settings.
                </div>
                <div>
                    <b>• Capacity:</b>
                    Adjust capacity and staffing to explore
                    operational pressure.
                </div>
                <div>
                    <b>• Target misses:</b>
                    Approximately {miss_rate:.1f}% of modeled
                    patients exceed the illustrative target.
                </div>
                <div>
                    <b>• Triage:</b>
                    Toggle triage support to compare
                    the model's assumptions.
                </div>
            </div>
        </div>
    </section>

    <section class="warning">
        <div class="warning-icon">!</div>
        <div>
            This application uses synthetic data and simplified
            assumptions. Its risk indicator is not a measured
            clinical adverse-event rate. It has not been clinically
            validated and must not be used for patient triage,
            staffing decisions, diagnosis, or real-world healthcare
            planning without appropriate data, validation, and
            expert review.
        </div>
    </section>

    <footer class="app-footer">
        <b>Decision Studio</b> |
        Healthcare Analytics Portfolio Project
    </footer>
    """,
    unsafe_allow_html=True,
)
