"""
Lower Control Arm — Material Optimization
Multi-criteria decision analysis for automotive suspension component material selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="mat_opt (anjnim)",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        h1, h2, h3, h4 {
            color: #c9d1d9;
            letter-spacing: 0.03em;
        }
        [data-testid="metric-container"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 16px;
        }
        hr { border-color: #30363d; }
        .stSlider label { color: #8b949e; font-size: 0.82rem; }
        .winner-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid #3b82f6;
            border-radius: 10px;
            padding: 24px 32px;
            margin-top: 8px;
        }
        .winner-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6b7280;
            margin-bottom: 4px;
        }
        .winner-name {
            font-size: 2rem;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 8px;
        }
        .winner-meta {
            font-size: 1rem;
            color: #9ca3af;
        }
        .winner-meta span {
            color: #e5e7eb;
            font-weight: 600;
        }
        /* 2x2 KPI grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            height: 100%;
        }
        .kpi-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
        }
        .kpi-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: #6b7280;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #e5e7eb;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .kpi-unit {
            font-size: 0.78rem;
            font-weight: 400;
            color: #6b7280;
            margin-left: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Materials database ────────────────────────────────────────────────────────
# density          g/cm3       — lower is better (weight)
# yield_strength   MPa         — higher is better
# youngs_modulus   GPa         — higher is better (stiffness)
# fatigue_strength MPa         — higher is better
# cost_index       1–10        — lower is better
# corrosion_score  1–10        — higher is better
# manufacturability 1–10       — higher is better

MATERIALS = {
    "Steel": {
        "density": 7.85,
        "yield_strength": 350.0,
        "youngs_modulus": 200.0,
        "fatigue_strength": 220.0,
        "cost_index": 2.5,
        "corrosion_score": 3.0,
        "manufacturability": 8.5,
    },

    "Stainless Steel": {
        "density": 7.90,
        "yield_strength": 290.0,
        "youngs_modulus": 193.0,
        "fatigue_strength": 240.0,
        "cost_index": 5.5,
        "corrosion_score": 9.0,
        "manufacturability": 7.0,
    },

    "HSLA Steel": {
        "density": 7.80,
        "yield_strength": 700.0,
        "youngs_modulus": 205.0,
        "fatigue_strength": 420.0,
        "cost_index": 3.5,
        "corrosion_score": 4.0,
        "manufacturability": 7.5,
    },

    "Aluminium": {
        "density": 2.70,
        "yield_strength": 250.0,
        "youngs_modulus": 69.0,
        "fatigue_strength": 95.0,
        "cost_index": 4.0,
        "corrosion_score": 7.5,
        "manufacturability": 8.0,
    },

    "Titanium": {
        "density": 4.43,
        "yield_strength": 880.0,
        "youngs_modulus": 114.0,
        "fatigue_strength": 510.0,
        "cost_index": 9.5,
        "corrosion_score": 9.5,
        "manufacturability": 4.0,
    },

    "Cast Iron": {
        "density": 7.20,
        "yield_strength": 200.0,
        "youngs_modulus": 130.0,
        "fatigue_strength": 110.0,
        "cost_index": 2.0,
        "corrosion_score": 4.0,
        "manufacturability": 9.0,
    },

    "CFRP Composite": {
        "density": 1.60,
        "yield_strength": 600.0,
        "youngs_modulus": 70.0,
        "fatigue_strength": 350.0,
        "cost_index": 9.0,
        "corrosion_score": 9.0,
        "manufacturability": 3.5,
    },
}

df_raw = (
    pd.DataFrame(MATERIALS)
    .T.reset_index()
    .rename(columns={"index": "Material"})
)

# ── Sidebar — priority weights ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Priority Weights")
    st.markdown("Adjust the relative importance of each criterion.")
    st.markdown("---")

    w_weight   = st.slider("Low Weight",           min_value=0, max_value=10, value=7)
    w_cost     = st.slider("Low Cost",             min_value=0, max_value=10, value=5)
    w_strength = st.slider("Strength",             min_value=0, max_value=10, value=8)
    w_fatigue  = st.slider("Fatigue Life",         min_value=0, max_value=10, value=7)
    w_stiff    = st.slider("Stiffness",            min_value=0, max_value=10, value=6)
    w_corr     = st.slider("Corrosion Resistance", min_value=0, max_value=10, value=4)
    w_mfg      = st.slider("Manufacturability",    min_value=0, max_value=10, value=5)

    total_w = w_weight + w_cost + w_strength + w_fatigue + w_stiff + w_corr + w_mfg
    if total_w == 0:
        st.error("At least one weight must be non-zero.")
        st.stop()

    st.markdown("---")
    st.markdown(
        f"<small style='color:#6b7280'>Weight sum: "
        f"<b style='color:#9ca3af'>{total_w}</b></small>",
        unsafe_allow_html=True,
    )

# ── Normalisation helpers ─────────────────────────────────────────────────────

def minmax_norm(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalise a series to [0, 1]. Invert when lower raw value is better."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - lo) / (hi - lo)
    return (1.0 - norm) if invert else norm


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color string to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ── Scoring ───────────────────────────────────────────────────────────────────
df = df_raw.copy()

df["n_weight"]   = minmax_norm(df["density"],          invert=True)
df["n_cost"]     = minmax_norm(df["cost_index"],        invert=True)
df["n_strength"] = minmax_norm(df["yield_strength"])
df["n_fatigue"]  = minmax_norm(df["fatigue_strength"])
df["n_stiff"]    = minmax_norm(df["youngs_modulus"])
df["n_corr"]     = minmax_norm(df["corrosion_score"])
df["n_mfg"]      = minmax_norm(df["manufacturability"])

SCORE_COLS = ["n_weight", "n_cost", "n_strength", "n_fatigue", "n_stiff", "n_corr", "n_mfg"]

raw_weights = np.array(
    [w_weight, w_cost, w_strength, w_fatigue, w_stiff, w_corr, w_mfg],
    dtype=float,
)
norm_weights = raw_weights / raw_weights.sum()

df["Score"]     = df[SCORE_COLS].values @ norm_weights
df["Score_pct"] = (df["Score"] * 100).round(2)
df = df.sort_values("Score", ascending=False).reset_index(drop=True)
df["Rank"] = df.index + 1

winner = df.iloc[0]
top3   = df.head(3)

# ── Chart constants ───────────────────────────────────────────────────────────
ACCENT = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

BASE_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#c9d1d9", family="Segoe UI"),
    margin=dict(l=40, r=40, t=50, b=40),
)

AXIS_STYLE = dict(gridcolor="#1f2937", tickfont=dict(color="#6b7280"))

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# Lower Control Arm — Material Optimization")
st.markdown(
    "<p style='color:#6b7280; font-size:0.9rem; margin-top:-12px;'>"
    "Multi-criteria decision analysis for automotive suspension component material selection"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Winner card + 2x2 KPI grid ───────────────────────────────────────────────
col_win, col_metrics = st.columns([1, 1.6], gap="large")

with col_win:
    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-label">Recommended Material</div>
            <div class="winner-name">{winner["Material"]}</div>
            <div class="winner-meta">
                Composite Score: <span>{winner["Score_pct"]:.2f} / 100</span><br>
                Rank: <span>1 of {len(df)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_metrics:
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Density</div>
                <div class="kpi-value">{winner['density']:.2f}<span class="kpi-unit">g/cm\u00b3</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Yield Strength</div>
                <div class="kpi-value">{winner['yield_strength']:.0f}<span class="kpi-unit">MPa</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Young's Modulus</div>
                <div class="kpi-value">{winner['youngs_modulus']:.1f}<span class="kpi-unit">GPa</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Fatigue Strength</div>
                <div class="kpi-value">{winner['fatigue_strength']:.0f}<span class="kpi-unit">MPa</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
st.markdown("---")

# ── Row 1: Ranked bar chart + Radar chart ─────────────────────────────────────
col_bar, col_radar = st.columns(2)

with col_bar:
    st.markdown("#### Material Rankings")

    bar_colors = [ACCENT[0] if i == 0 else "#374151" for i in range(len(df))]

    fig_bar = go.Figure(
        go.Bar(
            x=df["Score_pct"],
            y=df["Material"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in df["Score_pct"]],
            textposition="outside",
            textfont=dict(color="#9ca3af", size=11),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>",
        )
    )
    fig_bar.update_layout(
        **BASE_LAYOUT,
        height=360,
        xaxis=dict(
            title="Composite Score (0 - 100)",
            range=[0, 108],
            **AXIS_STYLE,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#c9d1d9"),
            gridcolor="#1f2937",
        ),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_radar:
    st.markdown("#### Top 3 — Criterion Profile")

    CRITERIA_LABELS = [
        "Low Weight", "Low Cost", "Strength",
        "Fatigue Life", "Stiffness", "Corrosion", "Manufacturability",
    ]
    # Close the polygon by repeating the first point
    closed_labels = CRITERIA_LABELS + [CRITERIA_LABELS[0]]

    fig_radar = go.Figure()
    for i, (_, row) in enumerate(top3.iterrows()):
        vals = [row[c] for c in SCORE_COLS]
        closed_vals = vals + [vals[0]]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=closed_vals,
                theta=closed_labels,
                fill="toself",
                name=row["Material"],
                line=dict(color=ACCENT[i], width=2),
                fillcolor=hex_to_rgba(ACCENT[i], 0.12),
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>",
            )
        )
    fig_radar.update_layout(
        **BASE_LAYOUT,
        height=360,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="#374151",
                tickfont=dict(color="#6b7280", size=9),
            ),
            angularaxis=dict(
                gridcolor="#374151",
                tickfont=dict(color="#9ca3af", size=10),
            ),
        ),
        legend=dict(
            font=dict(color="#c9d1d9", size=10),
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Row 2: Scatter plots ──────────────────────────────────────────────────────
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("#### Strength vs. Density")

    fig_sv = px.scatter(
        df,
        x="density",
        y="yield_strength",
        text="Material",
        size="Score_pct",
        size_max=28,
        color="Score_pct",
        color_continuous_scale=[[0.0, "#1f2937"], [0.5, "#3b82f6"], [1.0, "#10b981"]],
        hover_data={"Score_pct": ":.2f", "density": ":.2f", "yield_strength": ":.0f"},
    )
    fig_sv.update_traces(
        textposition="top center",
        textfont=dict(color="#9ca3af", size=9),
        marker=dict(line=dict(width=1, color="#374151")),
    )
    fig_sv.update_layout(
        **BASE_LAYOUT,
        height=360,
        xaxis=dict(title="Density (g/cm\u00b3)", **AXIS_STYLE),
        yaxis=dict(title="Yield Strength (MPa)", **AXIS_STYLE),
        coloraxis_colorbar=dict(
            title=dict(text="Score", font=dict(color="#9ca3af")),
            tickfont=dict(color="#6b7280"),
        ),
    )
    st.plotly_chart(fig_sv, use_container_width=True)

with col_s2:
    st.markdown("#### Cost vs. Performance")

    fig_cp = px.scatter(
        df,
        x="cost_index",
        y="Score_pct",
        text="Material",
        size="yield_strength",
        size_max=28,
        color="density",
        color_continuous_scale=[[0.0, "#10b981"], [0.5, "#3b82f6"], [1.0, "#ef4444"]],
        hover_data={"Score_pct": ":.2f", "cost_index": ":.1f", "density": ":.2f"},
        labels={"density": "Density (g/cm\u00b3)"},
    )
    fig_cp.update_traces(
        textposition="top center",
        textfont=dict(color="#9ca3af", size=9),
        marker=dict(line=dict(width=1, color="#374151")),
    )
    fig_cp.update_layout(
        **BASE_LAYOUT,
        height=360,
        xaxis=dict(title="Cost Index (1 = cheapest)", **AXIS_STYLE),
        yaxis=dict(title="Composite Score (0 - 100)", **AXIS_STYLE),
        coloraxis_colorbar=dict(
            title=dict(text="Density (g/cm\u00b3)", font=dict(color="#9ca3af")),
            tickfont=dict(color="#6b7280"),
        ),
    )
    st.plotly_chart(fig_cp, use_container_width=True)

# ── Data table ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Full Materials Data")

DISPLAY_MAP = {
    "Rank":             "Rank",
    "Material":         "Material",
    "density":          "Density (g/cm\u00b3)",
    "yield_strength":   "Yield Str. (MPa)",
    "youngs_modulus":   "E (GPa)",
    "fatigue_strength": "Fatigue Str. (MPa)",
    "cost_index":       "Cost Index",
    "corrosion_score":  "Corrosion",
    "manufacturability":"Manufacturability",
    "Score_pct":        "Score",
}

df_display = df[list(DISPLAY_MAP.keys())].rename(columns=DISPLAY_MAP)

FORMAT_MAP = {
    "Density (g/cm\u00b3)":  "{:.2f}",
    "Yield Str. (MPa)":       "{:.0f}",
    "E (GPa)":                "{:.1f}",
    "Fatigue Str. (MPa)":     "{:.0f}",
    "Cost Index":             "{:.1f}",
    "Corrosion":              "{:.1f}",
    "Manufacturability":      "{:.1f}",
    "Score":                  "{:.2f}",
}

st.dataframe(
    df_display.style
        .format(FORMAT_MAP)
        .background_gradient(subset=["Score"], cmap="Blues"),
    use_container_width=True,
    hide_index=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#374151; font-size:0.75rem; text-align:center;'>"
    "Lower Control Arm Material Optimization &nbsp;|&nbsp; "
    "Multi-Criteria Decision Analysis &nbsp;|&nbsp; "
    "Engineering values are representative approximations for comparative analysis."
    "</p>",
    unsafe_allow_html=True,
)
