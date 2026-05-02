import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Heat Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f0f4ff !important;
    color: #1a1a2e !important;
  }

  .main .block-container {
    padding: 2.5rem 3rem !important;
    max-width: 1300px;
  }

  /* Sidebar — soft violet tint */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef0ff 0%, #f5f0ff 100%) !important;
    border-right: 1px solid #c9c3f0 !important;
  }
  [data-testid="stSidebar"] * { color: #1a1a2e !important; }

  /* Metric cards — each gets a colored top border via nth-child */
  [data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #dde3ff !important;
    border-top: 3px solid #6366f1 !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
  }
  [data-testid="metric-container"]:nth-child(2) { border-top-color: #06b6d4 !important; }
  [data-testid="metric-container"]:nth-child(3) { border-top-color: #f59e0b !important; }
  [data-testid="metric-container"]:nth-child(4) { border-top-color: #10b981 !important; }
  [data-testid="metric-container"]:nth-child(5) { border-top-color: #ec4899 !important; }

  [data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    color: #7c83b0 !important;
    text-transform: uppercase !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    color: #1a1a2e !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #dde3ff !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 3px !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7c83b0 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em !important;
    padding: 7px 18px !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    font-weight: 500 !important;
  }

  /* Divider */
  hr { border-color: #dde3ff !important; }

  /* Progress bars */
  .bar-row { margin-bottom: 0.65rem; }
  .bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #4a4a72;
    margin-bottom: 4px;
    font-family: 'DM Sans', sans-serif;
  }
  .bar-bg {
    background: #e0e4ff;
    border-radius: 3px;
    height: 6px;
  }
  .bar-fill { height: 6px; border-radius: 3px; }

  /* Info card */
  .info-card {
    background: #ffffff;
    border: 1px solid #dde3ff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .info-card .card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.12em;
    color: #9da3c8;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .info-card .card-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1.1;
  }
  .info-card .card-sub { font-size: 0.75rem; color: #7c83b0; margin-top: 0.25rem; }

  /* Status pill */
  .pill {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
  }
  .pill-safe    { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
  .pill-caution { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
  .pill-crisis  { background: #ffe4e6; color: #9f1239; border: 1px solid #fda4af; }

  /* Recommendation block */
  .rec-block {
    border-left: 4px solid #6366f1;
    padding: 1rem 1.2rem;
    background: #ffffff;
    border-radius: 0 10px 10px 0;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #2a2a4a;
  }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB THEME ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#f8f8ff',
    'axes.facecolor':    '#f8f8ff',
    'axes.edgecolor':    '#c9c3f0',
    'axes.labelcolor':   '#4a4a72',
    'axes.titlecolor':   '#1a1a2e',
    'xtick.color':       '#7c83b0',
    'ytick.color':       '#7c83b0',
    'grid.color':        '#e0e4ff',
    'grid.linestyle':    '-',
    'grid.alpha':        0.8,
    'text.color':        '#1a1a2e',
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#dde3ff',
    'legend.labelcolor': '#2a2a4a',
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ─── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv('/Users/madhu/Downloads/philly_data_center_analysis.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('philly_data_center_analysis.csv')
        except FileNotFoundError:
            np.random.seed(42)
            n = 120
            ndvi_v   = np.random.uniform(0.02, 0.85, n)
            imperv_v = np.clip(1 - ndvi_v * 0.8 + np.random.normal(0, 0.05, n), 0.1, 0.99)
            albedo_v = np.random.uniform(0.08, 0.50, n)
            temp_v   = 110 - ndvi_v*28 - albedo_v*18 + imperv_v*22 + np.random.normal(0, 1.5, n)
            df = pd.DataFrame({
                'NDVI (Vegetation)':     ndvi_v,
                'Impervious_Percent':    imperv_v,
                'Albedo (Reflectivity)': albedo_v,
                'Surface_Temp (°F)':     temp_v,
            })

    X = df[['NDVI (Vegetation)', 'Impervious_Percent', 'Albedo (Reflectivity)']]
    y = df['Surface_Temp (°F)']
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist(), df

model, feature_names, df = load_and_train()

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    st.markdown("<small style='color:#aaa'>Adjust parameters to simulate retrofit scenarios for the East Whiteland site.</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Vegetation**")
    ndvi = st.slider("NDVI — Tree Cover Index", 0.0, 1.0, 0.05, 0.01,
                     help="0 = bare concrete · 1 = dense canopy")

    st.markdown("**Reflectivity**")
    albedo = st.slider("Albedo — Cool Roof Rating", 0.0, 1.0, 0.15, 0.01,
                       help="Higher = more solar radiation reflected")

    st.markdown("**Hardscape**")
    impervious = st.slider("Impervious Surface %", 0.0, 1.0, 0.95, 0.01,
                           help="Concrete, asphalt, rooftops combined")

    st.markdown("---")
    st.markdown("<small style='color:#bbb;font-size:0.72rem'>Random Forest · 200 estimators<br>Features: NDVI, Albedo, Impervious %<br>Target: Surface Temp (°F)</small>", unsafe_allow_html=True)

# ─── PREDICTION & DERIVED STATS ────────────────────────────────────────────────
input_df      = pd.DataFrame([[ndvi, impervious, albedo]], columns=feature_names)
prediction    = model.predict(input_df)[0]
BASELINE      = 104.2
cooling       = BASELINE - prediction
delta_pct     = abs(cooling / BASELINE) * 100
heat_idx      = min(100, max(0, (prediction - 80) / 40 * 100))
green_score   = ndvi * 100
reflectivity  = albedo * 100
hardscape_pct = impervious * 100
carbon_offset = ndvi * 4.7
energy_savings = max(0, cooling * 2.3)
equity_score  = max(0, min(100, 100 - heat_idx * 0.8 + green_score * 0.3))

if prediction > 102:
    pill_class, pill_label = "pill-crisis",  "Crisis Zone"
elif prediction > 96:
    pill_class, pill_label = "pill-caution", "Caution"
else:
    pill_class, pill_label = "pill-safe",    "Within Safe Threshold"

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## Urban Heat Mitigation Simulator")
st.markdown(
    "<span style='font-size:0.85rem;color:#7c83b0'>Equitable Retrofit Initiative · East Whiteland Data Center · Chester County, PA</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─── PRIMARY METRICS ───────────────────────────────────────────────────────────
st.markdown("<span style='font-size:0.7rem;letter-spacing:0.12em;color:#8b5cf6;font-family:monospace;text-transform:uppercase'>Live Scenario Output</span>", unsafe_allow_html=True)
st.markdown(" ")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Surface Temp", f"{prediction:.1f}°F",
              delta=f"{-cooling:+.1f}°F vs baseline", delta_color="inverse")
with c2:
    st.metric("Cooling Effect", f"{cooling:.1f}°F", delta="from 104.2°F baseline")
with c3:
    st.metric("Heat Stress Index", f"{heat_idx:.0f} / 100")
with c4:
    st.metric("Energy Savings Est.", f"{energy_savings:.0f} kWh/day", delta="per 1,000 sq ft")
with c5:
    st.metric("Carbon Offset", f"{carbon_offset:.1f} t/ac/yr", delta="CO₂ sequestration")

st.markdown(
    f"<div style='margin:1rem 0'><span class='pill {pill_class}'>{pill_label}</span>"
    f"&nbsp;&nbsp;<span style='font-size:0.8rem;color:#888'>Malvern Hunt neighborhood · {prediction:.1f}°F predicted</span></div>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─── MAIN CONTENT ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("**Parameter Breakdown**")
    st.markdown(" ")

    params = [
        ("Vegetation (NDVI)",   f"{green_score:.0f}%",    "#10b981", green_score / 100),
        ("Albedo Rating",       f"{reflectivity:.0f}%",   "#6366f1", reflectivity / 100),
        ("Impervious Surface",  f"{hardscape_pct:.0f}%",  "#f43f5e", hardscape_pct / 100),
        ("Water Runoff Risk",   f"{impervious*100:.0f}%", "#f59e0b", impervious),
        ("Equity Health Score", f"{equity_score:.0f}/100","#8b5cf6", equity_score / 100),
    ]
    for label, val, color, frac in params:
        st.markdown(f"""
        <div class="bar-row">
          <div class="bar-label"><span>{label}</span><span>{val}</span></div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{min(frac,1)*100:.1f}%;background:{color}"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(f"""
    <div class="info-card">
      <div class="card-label">Thermal Reduction</div>
      <div class="card-value">{delta_pct:.1f}%</div>
      <div class="card-sub">{BASELINE}°F → {prediction:.1f}°F</div>
    </div>
    <div class="info-card">
      <div class="card-label">Annual CO₂ Offset</div>
      <div class="card-value">{carbon_offset:.2f} t/ac</div>
      <div class="card-sub">from tree canopy sequestration</div>
    </div>
    <div class="info-card">
      <div class="card-label">Current Scenario</div>
      <div class="card-sub" style="font-size:0.8rem;color:#555;line-height:1.8;margin-top:0.4rem">
        NDVI: {ndvi:.2f}<br>
        Albedo: {albedo:.2f}<br>
        Impervious: {impervious:.0%}<br>
        Predicted Temp: {prediction:.1f}°F
      </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Saturation Curve", "Distribution"])

    # ── Tab 1: Feature Importance ──────────────────────────────────────────────
    with tab1:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        importances = model.feature_importances_
        labels = ['Vegetation (NDVI)', 'Impervious Surface', 'Albedo']
        colors = ['#10b981', '#f43f5e', '#6366f1']

        bars = ax.barh(range(len(labels)), importances, color=colors,
                       height=0.45, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        for bar, imp in zip(bars, importances):
            ax.text(imp + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{imp*100:.1f}%', va='center', fontsize=9.5, color='#555')
        ax.set_xlabel("Relative influence on surface temperature")
        ax.set_xlim(0, max(importances) * 1.3)
        ax.axvline(1/3, color='#ddd', linewidth=1, linestyle='--')
        ax.set_title("Which factor drives cooling the most?", fontsize=12, pad=10)
        ax.grid(axis='x', alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Dashed line = equal-influence baseline (33.3%). Features to the right punch above their weight.")

    # ── Tab 2: Saturation Curve ────────────────────────────────────────────────
    with tab2:
        ndvi_range  = np.linspace(0, 1, 80)
        curve_preds = model.predict(pd.DataFrame({
            'NDVI (Vegetation)':     ndvi_range,
            'Impervious_Percent':    [impervious] * 80,
            'Albedo (Reflectivity)': [albedo] * 80,
        }))

        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        ax2.fill_between(ndvi_range, curve_preds, curve_preds.max() + 2,
                         alpha=0.10, color='#10b981')
        ax2.plot(ndvi_range, curve_preds, color='#10b981', linewidth=2.5)
        ax2.scatter([ndvi], [prediction], color='#f43f5e', s=90, zorder=5,
                    marker='o', label=f'Current scenario: {prediction:.1f}°F')
        ax2.axhline(102, color='#f43f5e', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='Crisis threshold (102°F)')
        ax2.axhline(96,  color='#f59e0b', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='Caution threshold (96°F)')

        diffs = np.diff(curve_preds)
        sat_candidates = np.where(diffs > -0.15)[0]
        if len(sat_candidates) > 0:
            sat_idx = sat_candidates[0]
            ax2.axvline(ndvi_range[sat_idx], color='#8b5cf6', linestyle=':',
                        linewidth=1.8, alpha=0.8,
                        label=f'Diminishing returns ~ NDVI {ndvi_range[sat_idx]:.2f}')

        ax2.set_xlabel("NDVI (tree cover index)")
        ax2.set_ylabel("Predicted surface temp (°F)")
        ax2.set_title("Tree planting vs cooling — where returns diminish", fontsize=12, pad=10)
        ax2.legend(fontsize=8.5, framealpha=0.9)
        ax2.grid(True, alpha=0.4)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.caption("Blue dotted line marks where each additional unit of NDVI yields less than 0.15°F of cooling.")

    # ── Tab 3: Distribution ────────────────────────────────────────────────────
    with tab3:
        fig3, axes = plt.subplots(1, 2, figsize=(7, 3.8))

        sc = axes[0].scatter(
            df['NDVI (Vegetation)'], df['Surface_Temp (°F)'],
            c=df['Albedo (Reflectivity)'], cmap='cool',
            alpha=0.6, s=30, linewidths=0
        )
        axes[0].scatter([ndvi], [prediction], color='#f43f5e', s=90,
                        zorder=5, label='Your scenario', marker='D')
        axes[0].set_xlabel("NDVI")
        axes[0].set_ylabel("Surface Temp (°F)")
        axes[0].set_title("NDVI vs Temperature\n(color = albedo)", fontsize=10)
        axes[0].legend(fontsize=8)
        plt.colorbar(sc, ax=axes[0], label='Albedo', shrink=0.85)

        axes[1].hist(df['Surface_Temp (°F)'], bins=20,
                     color='#6366f1', alpha=0.70, edgecolor='white', linewidth=0.5)
        axes[1].axvline(prediction, color='#f43f5e', linewidth=2.2,
                        label=f'Your scenario: {prediction:.1f}°F')
        axes[1].axvline(BASELINE, color='#f59e0b', linewidth=1.8,
                        linestyle='--', label=f'Baseline: {BASELINE}°F')
        axes[1].set_xlabel("Surface Temp (°F)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Temp distribution\nin training dataset", fontsize=10)
        axes[1].legend(fontsize=8)

        fig3.tight_layout(pad=1.5)
        st.pyplot(fig3)
        st.caption("Red diamond marks your current scenario against the full training data range.")

st.markdown("---")

# ─── RECOMMENDATION ────────────────────────────────────────────────────────────
st.markdown("<span style='font-size:0.7rem;letter-spacing:0.12em;color:#8b5cf6;font-family:monospace;text-transform:uppercase'>Retrofit Assessment</span>", unsafe_allow_html=True)
st.markdown(" ")

rec_col, _ = st.columns([2, 1])
with rec_col:
    if cooling > 8:
        border_color = "#10b981"
        msg = (
            f"This configuration reduces the thermal burden by **{cooling:.1f}°F** ({delta_pct:.1f}% below baseline), "
            f"moving the Malvern Hunt neighborhood into safe territory. The combination of NDVI {ndvi:.2f} and "
            f"albedo {albedo:.2f} delivers compounding benefits beyond either strategy alone — validating the "
            f"Equitable Retrofit Initiative's multi-pronged approach."
        )
    elif cooling > 2:
        border_color = "#f59e0b"
        msg = (
            f"A **{cooling:.1f}°F** reduction is meaningful but not yet sufficient to fully neutralize the data "
            f"center's thermal footprint. Consider raising NDVI above 0.4 and albedo above 0.3. Current hardscape "
            f"at {hardscape_pct:.0f}% remains the primary driver of excess heat."
        )
    elif cooling <= 0:
        border_color = "#f43f5e"
        msg = (
            f"This configuration worsens the heat island. At {prediction:.1f}°F, the scenario exceeds baseline "
            f"by {abs(cooling):.1f}°F. High impervious surface ({hardscape_pct:.0f}%) with minimal vegetation "
            f"is the most thermally hostile outcome. Immediate intervention is needed before deployment."
        )
    else:
        border_color = "#f59e0b"
        msg = (
            f"Only **{cooling:.1f}°F** of cooling achieved — below the initiative's minimum threshold. "
            f"Prioritize impervious surface reduction and canopy planting before finalizing the retrofit plan."
        )

    st.markdown(
        f"<div class='rec-block' style='border-left-color:{border_color}'>{msg}</div>",
        unsafe_allow_html=True
    )