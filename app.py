"""
Load Forecasting Dashboard — Streamlit app.

Two dropdowns (state, model) drive a live-computed forecast comparison,
backed by the same functions validated in Notebooks 1-3. Linear Regression,
Random Forest, and XGBoost all train live on selection.

A SARIMAX baseline was also investigated (see Notebook 3 and
SARIMAX_METHODOLOGY.md) but is excluded from this app: it was confirmed the
weakest performer, its walk-forward fit is too slow to run per-click, and a
full 16-state precompute attempt hit an unresolved statsmodels internal
failure. Excluded here for reliability and UX; fully documented elsewhere.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys

sys.path.append("src")
from features import build_features, time_split
from models import train_and_evaluate, FEATURE_COLS

# ---------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------
st.set_page_config(page_title="Load Forecasting — Germany", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0B1220; color: #E8EDF4; }
#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #E8EDF4 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #131B2E 0%, #0F1626 100%);
    border: 1px solid #24304A;
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 24px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.9rem;
    font-weight: 700;
    color: #E8EDF4;
    margin: 0 0 6px 0;
}
.hero-subtitle { color: #8593AD; font-size: 1rem; margin: 0; }
.hero-goal {
    color: #C4CDE0; font-size: 1.02rem; margin: 14px 0 0 0; width: 100%; line-height: 1.6;
}
.hero-badges { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
    background: #1B2438; border: 1px solid #2E3B57; border-radius: 20px;
    padding: 5px 14px; font-size: 0.78rem; color: #C4CDE0; font-family: 'JetBrains Mono', monospace;
}

/* Shimmering lightning bolt — a subtle brightness/glow pulse, like a live current flicker */
.bolt {
    display: inline-block;
    animation: bolt-shimmer 2.4s ease-in-out infinite;
    text-shadow: 0 0 6px rgba(242, 183, 5, 0.6);
}
@keyframes bolt-shimmer {
    0%, 100% { filter: brightness(1) drop-shadow(0 0 2px rgba(242, 183, 5, 0.4)); }
    45% { filter: brightness(1) drop-shadow(0 0 2px rgba(242, 183, 5, 0.4)); }
    50% { filter: brightness(1.9) drop-shadow(0 0 12px rgba(242, 183, 5, 0.9)); }
    55% { filter: brightness(1) drop-shadow(0 0 2px rgba(242, 183, 5, 0.4)); }
    70% { filter: brightness(1.5) drop-shadow(0 0 8px rgba(242, 183, 5, 0.7)); }
    75% { filter: brightness(1) drop-shadow(0 0 2px rgba(242, 183, 5, 0.4)); }
}

/* State summary strip */
.stat-strip { display: flex; gap: 16px; margin: 18px 0 6px 0; }
.stat-box {
    flex: 1; background-color: #131B2E; border: 1px solid #24304A; border-radius: 10px;
    padding: 14px 18px;
}
.stat-box-label { color: #8593AD; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.stat-box-value {
    font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700;
    color: #E8EDF4; margin-top: 4px;
}

/* Section headers with icon accent */
.section-header { display: flex; align-items: center; gap: 10px; margin: 8px 0 4px 0; }
.section-icon { width: 6px; height: 22px; background: #F2B705; border-radius: 3px; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: #E8EDF4; }

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #131B2E;
    border: 1px solid #24304A;
    border-radius: 10px;
    padding: 16px 18px;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #F2B705 !important;
}
div[data-testid="stMetricLabel"] { color: #8593AD !important; }

/* Selectboxes */
div[data-baseweb="select"] > div {
    background-color: #131B2E !important;
    border-color: #24304A !important;
    border-radius: 8px !important;
}

/* Animated transmission-line divider — the one signature flourish */
.transmission-line {
    height: 2px;
    margin: 30px 0;
    background: repeating-linear-gradient(90deg, #F2B705 0px, #F2B705 8px, transparent 8px, transparent 16px);
    background-size: 32px 2px;
    animation: flow 1.2s linear infinite;
    opacity: 0.55;
}
@keyframes flow {
    from { background-position: 0 0; }
    to { background-position: 32px 0; }
}

.subtitle { color: #8593AD; font-size: 0.95rem; }
.model-tag {
    display: inline-block; background: #1B2438; border: 1px solid #F2B705;
    color: #F2B705; border-radius: 6px; padding: 3px 10px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Data + config
# ---------------------------------------------------------------
STATE_NAMES = {
    "BW": "Baden-Württemberg", "BY": "Bavaria", "BE": "Berlin", "BB": "Brandenburg",
    "HB": "Bremen", "HH": "Hamburg", "HE": "Hesse", "MV": "Mecklenburg-Vorpommern",
    "NI": "Lower Saxony", "NRW": "North Rhine-Westphalia", "RP": "Rhineland-Palatinate",
    "SL": "Saarland", "SN": "Saxony", "ST": "Saxony-Anhalt", "SH": "Schleswig-Holstein",
    "TH": "Thuringia"
}
MODELS = ["Linear Regression", "Random Forest", "XGBoost"]

# Plain-English descriptions for every engineered feature — used in the
# feature importance chart's hover text and the reference expander below it
FEATURE_DESCRIPTIONS = {
    "hour": "Hour of day (0-23)",
    "day_of_week": "Day of week (0=Monday, 6=Sunday)",
    "month": "Calendar month (1-12)",
    "year": "Calendar year",
    "is_weekend": "1 if Saturday/Sunday, else 0",
    "is_holiday": "1 if a public holiday in this state, else 0",
    "hour_sin": "Sine-encoded hour — makes 11pm and midnight 'close' to a model",
    "hour_cos": "Cosine-encoded hour — paired with hour_sin for a full circular encoding",
    "dow_sin": "Sine-encoded day of week — makes Sunday and Monday 'close'",
    "dow_cos": "Cosine-encoded day of week — paired with dow_sin",
    "month_sin": "Sine-encoded month — makes December and January 'close'",
    "month_cos": "Cosine-encoded month — paired with month_sin",
    "lag_24h": "Demand exactly 24 hours ago (yesterday, same hour)",
    "lag_168h": "Demand exactly 168 hours ago (same hour, one week ago)",
    "rolling_mean_24h": "Average demand over the prior 24 hours",
    "rolling_std_24h": "Demand volatility over the prior 24 hours",
    "rolling_mean_168h": "Average demand over the prior 168 hours (1 week)",
}

# ---------------------------------------------------------------
# Hero
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
    <p class="hero-title"><span class="bolt">⚡</span> Electricity Load Forecasting</p>
    <p class="hero-subtitle">Hourly demand across 16 German federal states, 2019–2022 · 3 models trained and compared live</p>
    <p class="hero-goal">
        <strong>Goal:</strong> predict a state's next hour of electricity demand using calendar patterns
        and recent usage history, then compare how a simple baseline stacks up against more complex
        machine learning models on the same, genuinely held-out data.
    </p>
    <div class="hero-badges">
        <span class="badge">35,064 hourly readings</span>
        <span class="badge">4 years</span>
        <span class="badge">16 states</span>
        <span class="badge">17 engineered features</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    state_code = st.selectbox(
        "State", options=list(STATE_NAMES.keys()),
        format_func=lambda x: f"{STATE_NAMES[x]} ({x})", index=1,
        help="German federal states use 2-3 letter codes (e.g. BY = Bavaria, NRW = North Rhine-Westphalia). "
             "These aren't grid operators — Germany's 4 grid operators (50Hertz, Amprion, TenneT, TransnetBW) "
             "are a separate, different regional split not used in this dashboard."
    )
with col2:
    model_name = st.selectbox(
        "Model", options=MODELS,
        help="Linear Regression: simple baseline. Random Forest & XGBoost: tree-based ML, capture "
             "non-linear, interaction-driven patterns without needing them hand-engineered."
    )

# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_features(state):
    return build_features(state, cleaned_data_path="data/demand_by_state_cleaned.csv")

with st.spinner(f"Building features for {STATE_NAMES[state_code]}..."):
    features = get_features(state_code)
    train, val = time_split(features)

with st.spinner(f"Training {model_name} on {STATE_NAMES[state_code]}..."):
    result = train_and_evaluate(model_name, train, val)
    predictions = result["predictions"]
    metrics = result["metrics"]
    trained_model = result["model"]

# ---------------------------------------------------------------
# State summary strip — quick context before diving into model results
# ---------------------------------------------------------------
avg_demand = features["demand_mwh"].mean()
peak_demand = features["demand_mwh"].max()
peak_hour = int(features.groupby(features.index.hour)["demand_mwh"].mean().idxmax())
weekend_drop = (
    (features[features["is_weekend"] == 0]["demand_mwh"].mean() -
     features[features["is_weekend"] == 1]["demand_mwh"].mean())
    / features[features["is_weekend"] == 0]["demand_mwh"].mean() * 100
)

st.markdown(f"""
<div class="stat-strip">
    <div class="stat-box"><div class="stat-box-label">Avg hourly demand</div><div class="stat-box-value">{avg_demand:,.0f} MWh</div></div>
    <div class="stat-box"><div class="stat-box-label">Peak demand</div><div class="stat-box-value">{peak_demand:,.0f} MWh</div></div>
    <div class="stat-box"><div class="stat-box-label">Typical peak hour</div><div class="stat-box-value">{peak_hour}:00</div></div>
    <div class="stat-box"><div class="stat-box-label">Weekend drop</div><div class="stat-box-value">{weekend_drop:.1f}%</div></div>
</div>
""", unsafe_allow_html=True)
st.caption("Weekend drop = how much lower average demand is on Sat/Sun vs. weekdays, for this state")

st.markdown('<div class="transmission-line"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------
st.markdown(f"""
<div class="section-header">
    <div class="section-icon"></div>
    <div class="section-title">Validation performance <span class="model-tag">{model_name}</span></div>
</div>
""", unsafe_allow_html=True)
st.caption(f"Evaluated on held-out data: {val.index.min().date()} to {val.index.max().date()} (never seen during training)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"{metrics['MAE']:.1f} MWh",
          help="Mean Absolute Error — the average size of the model's error, in the same units as demand. "
               "Directly readable: 'predictions were off by this many MWh on average.'")
m2.metric("RMSE", f"{metrics['RMSE']:.1f} MWh",
          help="Root Mean Squared Error — like MAE, but penalizes large misses more heavily. A big gap "
               "between RMSE and MAE suggests occasional large errors, not just consistent small ones.")
m3.metric("MAPE", f"{metrics['MAPE']:.2f}%",
          help="Mean Absolute Percentage Error — error as a percentage of actual demand. Useful for "
               "comparing accuracy across states of very different sizes.")
m4.metric("R²", f"{metrics['R2']:.3f}",
          help="Coefficient of Determination — the share of demand's variation the model explains. "
               "1.0 = perfect, 0.0 = no better than predicting the average every time. Negative means worse than that.")

st.markdown('<div class="transmission-line"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Actual vs predicted chart — with area fill and richer hover
# ---------------------------------------------------------------
st.markdown(f"""
<div class="section-header">
    <div class="section-icon"></div>
    <div class="section-title">Actual vs predicted <span class="model-tag">{STATE_NAMES[state_code]} ({state_code})</span></div>
</div>
""", unsafe_allow_html=True)

window = st.slider("Days to display", min_value=7, max_value=180, value=14, step=7)
display_end = val.index.min() + pd.Timedelta(days=window)
actual_slice = val.loc[val.index < display_end, "demand_mwh"]
pred_slice = predictions.loc[predictions.index < display_end]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=actual_slice.index, y=actual_slice.values, name="Actual",
    line=dict(color="#E8EDF4", width=2),
    fill="tozeroy", fillcolor="rgba(232,237,244,0.05)",
    hovertemplate="%{x|%a %b %d, %H:00}<br>Actual: %{y:,.0f} MWh<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=pred_slice.index, y=pred_slice.values, name=model_name,
    line=dict(color="#F2B705", width=1.8, dash="dash"),
    hovertemplate="%{x|%a %b %d, %H:00}<br>Predicted: %{y:,.0f} MWh<extra></extra>"
))
fig.update_layout(
    plot_bgcolor="#131B2E", paper_bgcolor="#0B1220", font_color="#E8EDF4",
    xaxis=dict(gridcolor="#1E2A42", showgrid=True),
    yaxis=dict(gridcolor="#1E2A42", title="Demand (MWh)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
    height=460, margin=dict(t=30, l=10, r=10, b=10),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Feature importance (only for tree-based models)
# ---------------------------------------------------------------
if trained_model is not None and hasattr(trained_model, "feature_importances_"):
    st.markdown('<div class="transmission-line"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon"></div>
        <div class="section-title">Feature importance <span class="model-tag">{STATE_NAMES[state_code]} ({state_code})</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("What the model actually relied on to make predictions — hover any bar for what it means")

    importances = pd.Series(trained_model.feature_importances_, index=FEATURE_COLS).sort_values()
    descriptions = [FEATURE_DESCRIPTIONS.get(f, "") for f in importances.index]

    fig_imp = go.Figure(go.Bar(
        x=importances.values, y=importances.index, orientation="h",
        marker_color="#F2B705",
        customdata=descriptions,
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>Importance: %{x:.3f}<extra></extra>"
    ))
    fig_imp.update_layout(
        plot_bgcolor="#131B2E", paper_bgcolor="#0B1220", font_color="#E8EDF4",
        xaxis=dict(gridcolor="#1E2A42"), height=420, margin=dict(t=10, l=10, r=10, b=10)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("ⓘ What do these feature names mean?"):
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            st.markdown(f"**`{feat}`** — {desc}")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown('<div class="transmission-line"></div>', unsafe_allow_html=True)
st.caption(
    "Data: [Electrical Load and Generation for Germany (Kaggle)](https://www.kaggle.com/datasets/pythonafroz/electrical-load-and-generation-for-germany), "
    "hourly, 2019–2022. Models validated on the held-out last 6 months of 2022. "
    "A SARIMAX baseline was also investigated but excluded here — see SARIMAX_METHODOLOGY.md for the full write-up and results."
)