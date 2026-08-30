import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_processing import (
    clean_raw_dataframe, engineer_features, encode_for_model, score_dataframe,
)
from src.strategy_engine import recommend_action, recommend_batch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# DESIGN TOKENS
# ============================================================
BG = "#0B1120"
SURFACE = "#131B2E"
SURFACE_ALT = "#0F1626"
BORDER = "#1F2A44"
TEXT = "#E8ECF4"
MUTED = "#8593AD"
GOLD = "#C9A24B"
RISK_RED = "#E5484D"
WARN_AMBER = "#F0A93B"
SAFE_GREEN = "#34C77B"
RETAIN_BLUE = "#3B82F6"

st.set_page_config(
    page_title="ChurnIQ | AI Customer Churn Prediction & Retention Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp {{ background-color: {BG}; color: {TEXT}; font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; color: {TEXT} !important; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background-color: transparent; }}
header[data-testid="stHeader"] button {{ visibility: visible !important; color: {TEXT} !important; }}
.main .block-container {{ padding-top: 1.5rem; max-width: 1350px; }}

section[data-testid="stSidebar"] {{ background-color: {SURFACE_ALT}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

.sidebar-brand {{ display: flex; align-items: center; gap: 10px; padding: 4px 4px 18px 4px; border-bottom: 1px solid {BORDER}; margin-bottom: 14px; }}
.sidebar-mark {{ width: 32px; height: 32px; border-radius: 9px; background: linear-gradient(135deg, {RETAIN_BLUE}, {GOLD}); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; }}
.sidebar-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02rem; color: {TEXT}; line-height: 1.1; }}
.sidebar-sub {{ font-size: 0.7rem; color: {MUTED}; }}
.sidebar-section-label {{ font-size: 0.68rem; color: {MUTED}; letter-spacing: 1.5px; text-transform: uppercase; margin: 4px 4px 6px 4px; }}

/* Sidebar nav styled from a radio widget */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background-color: transparent; border: 1px solid transparent; border-radius: 8px;
    padding: 9px 12px; width: 100%; transition: background-color 0.15s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background-color: rgba(255,255,255,0.04); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
    background-color: rgba(201,162,75,0.12); border: 1px solid {GOLD};
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: 0.88rem; color: {TEXT}; font-weight: 500; }}

.hero-label {{ font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 2px; font-size: 0.75rem; color: {MUTED}; margin-bottom: 6px; }}
.hero-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; color: {TEXT}; line-height: 1.25; margin-bottom: 6px; }}
.hero-sub {{ font-size: 1rem; color: {MUTED}; max-width: 720px; line-height: 1.55; margin-bottom: 4px; }}
.hero-number {{ font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 600; color: {GOLD}; line-height: 1.1; }}
.hero-underline {{ height: 3px; width: 140px; background: linear-gradient(90deg, {GOLD}, transparent); margin: 10px 0 18px 0; }}

.kpi-card {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 18px 20px; }}
.kpi-label {{ font-size: 0.76rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: {TEXT}; }}
.kpi-value.gold {{ color: {GOLD}; }}
.kpi-value.red {{ color: {RISK_RED}; }}
.kpi-value.green {{ color: {SAFE_GREEN}; }}

/* Tier-1 cards: the north-star numbers get more visual weight than Tier-2 */
.kpi-card.tier1 {{ padding: 22px 24px; border: 1px solid {BORDER}; border-top: 3px solid {RISK_RED}; }}
.kpi-card.tier1 .kpi-label {{ font-size: 0.82rem; }}
.kpi-card.tier1 .kpi-value {{ font-size: 2.1rem; }}
.kpi-card.tier2 {{ padding: 14px 18px; background-color: {SURFACE_ALT}; }}
.kpi-card.tier2 .kpi-value {{ font-size: 1.25rem; }}
.tier-label {{ font-size: 0.7rem; color: {MUTED}; letter-spacing: 1.5px; text-transform: uppercase; margin: 4px 0 10px 2px; }}

.freshness-pill {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.74rem; color: {MUTED}; background: rgba(255,255,255,0.03); border: 1px solid {BORDER}; padding: 4px 12px; border-radius: 20px; margin-bottom: 14px; }}
.freshness-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {SAFE_GREEN}; display: inline-block; }}

/* Key Insights & Threats cards */
.insight-card {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-left: 4px solid {RISK_RED}; border-radius: 10px; padding: 16px 18px; height: 100%; }}
.insight-card.severity-medium {{ border-left-color: {WARN_AMBER}; }}
.insight-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; font-weight: 700; color: {RISK_RED}; margin-bottom: 4px; }}
.insight-card.severity-medium .insight-value {{ color: {WARN_AMBER}; }}
.insight-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.92rem; color: {TEXT}; margin-bottom: 6px; line-height: 1.3; }}
.insight-text {{ font-size: 0.8rem; color: {MUTED}; line-height: 1.5; }}
.insight-severity-tag {{ display: inline-block; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: {RISK_RED}; margin-bottom: 8px; }}
.insight-card.severity-medium .insight-severity-tag {{ color: {WARN_AMBER}; }}

.section-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; color: {TEXT}; border-left: 3px solid {GOLD}; padding-left: 10px; margin: 8px 0 14px 0; }}
.section-caption {{ font-size: 0.82rem; color: {MUTED}; margin: -8px 0 16px 14px; }}

.badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px; }}
.badge-high {{ background: rgba(229,72,77,0.15); color: {RISK_RED}; border: 1px solid {RISK_RED}; }}
.badge-medium {{ background: rgba(240,169,59,0.15); color: {WARN_AMBER}; border: 1px solid {WARN_AMBER}; }}
.badge-low {{ background: rgba(52,199,123,0.15); color: {SAFE_GREEN}; border: 1px solid {SAFE_GREEN}; }}

.active-source-pill {{ display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; background: rgba(201,162,75,0.12); color: {GOLD}; border: 1px solid {GOLD}; margin-bottom: 14px; }}

div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; color: {GOLD}; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border: 1px solid {BORDER}; }}
button[kind="primary"] {{ background: {GOLD} !important; color: {BG} !important; border: none !important; font-weight: 700 !important; }}

/* ---------------- HERO WALKER ANIMATION ---------------- */
.walker-wrap {{
    background: linear-gradient(180deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 26px 30px 14px 30px;
    margin: 6px 0 26px 0; overflow: hidden; perspective: 700px;
}}
.walker-row {{
    display: flex; align-items: flex-end; gap: 20px; height: 96px;
    transform: rotateX(6deg); transform-style: preserve-3d;
}}
.walker {{ position: relative; height: 84px; animation: bodyBob 2.2s ease-in-out infinite; }}
.walker.standing {{ width: 32px; }}
.walker.runner {{ width: 54px; }}
.walker svg {{ width: 100%; height: 100%; display: block; filter: drop-shadow(0 6px 5px rgba(0,0,0,0.35)); }}
.walker.leaving {{ animation: bodyBob 1.1s ease-in-out infinite, walkAcross 2.6s ease-in infinite; }}
.exit-door {{ flex-shrink: 0; width: 50px; height: 84px; margin-left: 4px; align-self: flex-end; }}
.exit-door svg {{ width: 100%; height: 100%; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.4)); }}
.walker-caption {{ font-size: 0.78rem; color: {MUTED}; margin-top: 10px; text-align: center; }}

@keyframes bodyBob {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-3px); }} }}
@keyframes walkAcross {{
    0% {{ transform: translateX(0) translateY(0); opacity: 1; }}
    70% {{ opacity: 1; }}
    100% {{ transform: translateX(140px) translateY(-4px); opacity: 0; }}
}}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Inter", color=TEXT, size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

STANDING_COLORS = ["#3B82F6", "#D6467D", "#2C2F36", "#F0A93B", "#1D9E75"]
RUNNER_COLORS = ["#3B82F6", "#F0A93B", "#1D9E75"]


# ============================================================
# HERO ANIMATION — office people running out an exit door
# ============================================================
def _standing_person_svg(color: str) -> str:
    return f"""
    <div class="walker standing">
        <svg viewBox="0 0 32 84" xmlns="http://www.w3.org/2000/svg">
            <circle cx="16" cy="10" r="8" fill="{color}"/>
            <rect x="6" y="20" width="20" height="34" rx="6" fill="{color}"/>
            <polygon points="16,20 13,30 16,34 19,30" fill="#0B1120" opacity="0.3"/>
            <rect x="7" y="54" width="8" height="26" rx="2" fill="{color}"/>
            <rect x="17" y="54" width="8" height="26" rx="2" fill="{color}"/>
        </svg>
    </div>
    """


def _running_person_svg(color: str, delay: float) -> str:
    return f"""
    <div class="walker runner leaving" style="animation-delay:{delay}s;">
        <svg viewBox="0 0 54 84" xmlns="http://www.w3.org/2000/svg">
            <line x1="0" y1="42" x2="12" y2="42" stroke="{color}" stroke-width="3" opacity="0.5" stroke-linecap="round"/>
            <line x1="0" y1="50" x2="10" y2="50" stroke="{color}" stroke-width="3" opacity="0.32" stroke-linecap="round"/>
            <line x1="0" y1="58" x2="7" y2="58" stroke="{color}" stroke-width="3" opacity="0.18" stroke-linecap="round"/>
            <g transform="translate(12,0) rotate(-14 16 40)">
                <circle cx="16" cy="10" r="8" fill="{color}"/>
                <rect x="7" y="20" width="18" height="28" rx="6" fill="{color}"/>
                <rect x="-3" y="25" width="15" height="11" rx="3" fill="{color}"/>
                <rect x="6" y="48" width="8" height="26" rx="2" fill="{color}" transform="rotate(26 10 61)"/>
                <rect x="18" y="48" width="8" height="26" rx="2" fill="{color}" transform="rotate(-32 22 61)"/>
            </g>
        </svg>
    </div>
    """


def _exit_door_svg() -> str:
    return f"""
    <div class="exit-door">
        <svg viewBox="0 0 54 84" xmlns="http://www.w3.org/2000/svg">
            <rect x="4" y="2" width="46" height="80" rx="3" fill="{RISK_RED}"/>
            <rect x="9" y="7" width="36" height="70" rx="2" fill="#C23B3F"/>
            <circle cx="40" cy="44" r="2.4" fill="{RISK_RED}"/>
            <text x="27" y="46" text-anchor="middle" font-size="10" font-weight="700" fill="white"
                  font-family="Inter, sans-serif" transform="rotate(-90 27 44)">EXIT</text>
        </svg>
    </div>
    """


def render_churn_animation():
    standing = "".join(_standing_person_svg(c) for c in STANDING_COLORS)
    running = "".join(_running_person_svg(c, delay=i * 0.4) for i, c in enumerate(RUNNER_COLORS))
    html = f"""
    <div class="walker-wrap">
        <div class="walker-row">{standing}{running}{_exit_door_svg()}</div>
        <div class="walker-caption">Every customer who churns is walking out the door — and taking real revenue with them.</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def kpi_card(label, value, css_class=""):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value {css_class}">{value}</div></div>'


# ============================================================
# ASSET LOADING
# ============================================================
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    cols = joblib.load(os.path.join(MODELS_DIR, "model_columns.pkl"))
    explainer_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else None
    return model, cols, explainer


@st.cache_data
def load_company_data():
    path = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def compute_performance_on_active(_model, model_cols, df):
    """Same metrics as compute_model_performance, but scored directly against
    whatever dataset is currently active (e.g. an uploaded file) instead of a
    held-out split of the company data. Requires the file to have an actual
    Churn column — there's no way to measure accuracy without ground truth."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score,
    )

    full = clean_raw_dataframe(df.copy())
    full["Churn_bin"] = full["Churn"].map({"Yes": 1, "No": 0})
    engineered = engineer_features(full)
    X = pd.get_dummies(engineered.drop(columns=["Churn", "Churn_bin"]), drop_first=True)
    X = X.reindex(columns=model_cols, fill_value=0)
    y = full["Churn_bin"]

    preds = _model.predict(X)
    probs = _model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs) if y.nunique() > 1 else float("nan"),
        "cm": confusion_matrix(y, preds),
        "n_test": len(y),
    }


@st.cache_data
def compute_model_performance(_model, model_cols, df):
    """Reproduces the notebook's exact train/test split (test_size=0.2,
    random_state=42, stratify=y) against the ORIGINAL company dataset the
    model was trained on, then scores real accuracy/precision/recall/F1/
    ROC-AUC + confusion matrix. This is a live recomputation, not hardcoded
    numbers, so it stays correct even if the model is retrained."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score,
    )

    full = clean_raw_dataframe(df.copy())
    full["Churn_bin"] = full["Churn"].map({"Yes": 1, "No": 0})
    engineered = engineer_features(full)
    X = pd.get_dummies(engineered.drop(columns=["Churn", "Churn_bin"]), drop_first=True)
    X = X.reindex(columns=model_cols, fill_value=0)
    y = full["Churn_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preds = _model.predict(X_test)
    probs = _model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "cm": confusion_matrix(y_test, preds),
        "n_test": len(y_test),
    }


def _segment_rate(df, scored, mask):
    """Rate for a segment: uses the ACTUAL Churn column when present (ground
    truth), otherwise falls back to the model's predicted probability — so
    this works even on an uploaded file that has no Churn column at all."""
    if mask.sum() == 0:
        return None
    if "Churn" in df.columns:
        sub = df.loc[mask, "Churn"]
        return (sub == "Yes").mean() * 100
    return scored.loc[mask, "Churn_Probability"].mean() * 100


def compute_risk_factors(df, scored):
    """Ranks the biggest churn risk factors present in whatever dataset is
    currently active. Skips any factor whose column isn't in the file, so
    this never breaks on a partial/uploaded CSV."""
    factors = []

    if "Contract" in df.columns:
        r = _segment_rate(df, scored, df["Contract"] == "Month-to-month")
        if r is not None:
            factors.append(("Month-to-month contract", r))

    if "tenure" in df.columns:
        r = _segment_rate(df, scored, df["tenure"] <= 12)
        if r is not None:
            factors.append(("New customers (0-12 months)", r))

    if "InternetService" in df.columns and "TechSupport" in df.columns:
        r = _segment_rate(df, scored, (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No"))
        if r is not None:
            factors.append(("Fiber + no tech support", r))

    if "PaymentMethod" in df.columns:
        r = _segment_rate(df, scored, df["PaymentMethod"] == "Electronic check")
        if r is not None:
            factors.append(("Pays by electronic check", r))

    if "OnlineSecurity" in df.columns:
        r = _segment_rate(df, scored, df["OnlineSecurity"] == "No")
        if r is not None:
            factors.append(("No online security add-on", r))

    if "SeniorCitizen" in df.columns:
        r = _segment_rate(df, scored, df["SeniorCitizen"] == 1)
        if r is not None:
            factors.append(("Senior citizens", r))

    factors.sort(key=lambda x: x[1])
    return factors


try:
    model, model_cols, explainer = load_assets()
except Exception as e:
    st.error(f"Model assets not found in `{MODELS_DIR}`. Details: {e}")
    st.stop()

company_df = load_company_data()

# ============================================================
# SESSION STATE
# ============================================================
if "active_df" not in st.session_state:
    st.session_state.active_df = company_df
    st.session_state.active_label = "Full Company Dataset"

# ============================================================
# SIDEBAR — brand + navigation
# ============================================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-mark">◆</div>
            <div>
                <div class="sidebar-name">ChurnIQ</div>
                <div class="sidebar-sub">AI-Powered Customer Churn Prediction &amp; Retention Analytics</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate", label_visibility="collapsed",
        options=[
            "🏠  Home",
            "📁  Upload Data",
            "🧠  Why Customers Churn",
            "🎯  Customer Simulator",
            "🏆  Retention Priorities",
            "📈  Model Performance",
            "🔎  Customer Explorer",
        ],
    )

    st.markdown('<div class="sidebar-section-label" style="margin-top:20px;">Active Dataset</div>', unsafe_allow_html=True)
    n_active = len(st.session_state.active_df) if st.session_state.active_df is not None else 0
    st.markdown(
        f'<span class="active-source-pill">{st.session_state.active_label} · {n_active:,} rows</span>',
        unsafe_allow_html=True,
    )

active_df = st.session_state.active_df
active_label = st.session_state.active_label
has_actual_churn = active_df is not None and "Churn" in active_df.columns
scored = score_dataframe(active_df, model, model_cols) if active_df is not None else None

# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠  Home":
    st.markdown('<div class="hero-label">Customer Churn Prediction &amp; Revenue Retention Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Know which customers are about to leave — before they do.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">ChurnIQ is an AI-powered churn prediction and customer retention analytics '
        'platform. It scores every customer\'s churn risk, translates that risk into real monthly revenue at '
        'stake, and recommends the exact retention action to take — turning a prediction into a decision your '
        'team can act on today.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    render_churn_animation()

    if active_df is not None:
        total_revenue_risk = scored["RevenueAtRisk"].sum()
        if has_actual_churn:
            churn_rate = (active_df["Churn"] == "Yes").mean() * 100
        else:
            churn_rate = scored["Churn_Probability"].mean() * 100
        high_risk_count = (scored["RiskLevel"] == "High").sum()
        total_monthly_rev = pd.to_numeric(active_df["MonthlyCharges"], errors="coerce").sum()
        avg_value = pd.to_numeric(active_df["TotalCharges"], errors="coerce").mean() if "TotalCharges" in active_df.columns else None
        revenue_protected = total_monthly_rev - total_revenue_risk

        st.markdown(
            f'<span class="freshness-pill"><span class="freshness-dot"></span> '
            f'{active_label} · updated {datetime.now().strftime("%b %d, %I:%M %p")} · '
            f'{len(active_df):,} records</span>',
            unsafe_allow_html=True,
        )
        if not has_actual_churn:
            st.caption("This file has no Churn column, so churn rate here is the model's average predicted risk, not an actual rate.")

        st.markdown('<div class="tier-label">Tier 1 · Headline Numbers</div>', unsafe_allow_html=True)
        t1a, t1b, t1c = st.columns(3)
        t1a.markdown(kpi_card("Monthly Revenue at Risk", f"${total_revenue_risk:,.0f}", "red"), unsafe_allow_html=True)
        t1b.markdown(kpi_card("Churn Rate" if has_actual_churn else "Avg. Predicted Risk", f"{churn_rate:.1f}%", "red"), unsafe_allow_html=True)
        t1c.markdown(kpi_card("Annualized Revenue Loss", f"${total_revenue_risk*12:,.0f}", "red"), unsafe_allow_html=True)
        # reuse tier1 styling via CSS class on the outer card
        st.markdown("<style>div[data-testid='column'] .kpi-card{border-top:3px solid #E5484D;padding:22px 24px;} div[data-testid='column'] .kpi-value{font-size:2rem;}</style>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="tier-label">Tier 2 · Supporting Numbers</div>', unsafe_allow_html=True)
        t2a, t2b, t2c, t2d = st.columns(4)
        t2a.markdown(kpi_card("Total Customers", f"{len(active_df):,}"), unsafe_allow_html=True)
        t2b.markdown(kpi_card("High-Risk Accounts", f"{high_risk_count:,}", "red"), unsafe_allow_html=True)
        t2c.markdown(kpi_card("Avg. Customer Value", f"${avg_value:,.0f}" if avg_value else "—"), unsafe_allow_html=True)
        t2d.markdown(kpi_card("Revenue Protected*", f"${revenue_protected:,.0f}", "green"), unsafe_allow_html=True)
        st.caption("*Revenue Protected = total monthly billing minus revenue currently at risk.")

        # ---------------- TOP RISK FACTORS (chart, not paragraphs) ----------------
        st.markdown('<div class="section-title" style="margin-top:30px;">Top Churn Risk Factors</div>', unsafe_allow_html=True)
        factors = compute_risk_factors(active_df, scored)
        if factors:
            labels = [f[0] for f in factors]
            values = [f[1] for f in factors]
            colors = [RISK_RED if v == max(values) else (WARN_AMBER if v > sum(values)/len(values) else GOLD) for v in values]
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h", marker_color=colors,
                text=[f"{v:.1f}%" for v in values], textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="Churn Rate (%)")
            st.plotly_chart(fig, use_container_width=True, key="chart_risk_factors")
            st.caption("Highest-risk customer segments in this dataset, ranked by churn rate.")
        else:
            st.info("Not enough matching columns in this file to rank risk factors.")

        # ---------------- SUPPORTING VISUALS ----------------
        st.markdown('<div class="section-title" style="margin-top:20px;">Where Churn Is Concentrated</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">By Contract Type</div>', unsafe_allow_html=True)
            if "Contract" in active_df.columns:
                tmp = active_df.copy()
                tmp["_val"] = (tmp["Churn"] == "Yes").astype(float) * 100 if has_actual_churn else scored["Churn_Probability"].values * 100
                grp = tmp.groupby("Contract")["_val"].mean().reset_index()
                fig = go.Figure(go.Bar(
                    x=grp["Contract"], y=grp["_val"], marker_color=[RISK_RED, WARN_AMBER, SAFE_GREEN][:len(grp)],
                    text=[f"{v:.1f}%" for v in grp["_val"]], textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=280, yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig, use_container_width=True, key="chart_by_contract")
            else:
                st.info("No Contract column in this file.")
        with c2:
            st.markdown('<div class="section-title">Risk Level Distribution</div>', unsafe_allow_html=True)
            risk_counts = scored["RiskLevel"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig = go.Figure(go.Pie(
                labels=risk_counts.index, values=risk_counts.values, hole=0.6,
                marker_colors=[SAFE_GREEN, WARN_AMBER, RISK_RED], textfont=dict(color=TEXT),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig, use_container_width=True, key="chart_risk_pie")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="section-title">By Internet Service</div>', unsafe_allow_html=True)
            if "InternetService" in active_df.columns:
                tmp = active_df.copy()
                tmp["_val"] = (tmp["Churn"] == "Yes").astype(float) * 100 if has_actual_churn else scored["Churn_Probability"].values * 100
                grp2 = tmp.groupby("InternetService")["_val"].mean().reset_index()
                fig = go.Figure(go.Bar(
                    x=grp2["_val"], y=grp2["InternetService"], orientation="h", marker_color=RISK_RED,
                    text=[f"{v:.1f}%" for v in grp2["_val"]], textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=250, xaxis_title="Churn Rate (%)")
                st.plotly_chart(fig, use_container_width=True, key="chart_by_internet")
            else:
                st.info("No Internet Service column in this file.")
        with c4:
            st.markdown('<div class="section-title">By Tenure</div>', unsafe_allow_html=True)
            if "tenure" in active_df.columns:
                tmp = active_df.copy()
                tmp["tenure_bucket"] = pd.cut(tmp["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"])
                tmp["_val"] = (tmp["Churn"] == "Yes").astype(float) * 100 if has_actual_churn else scored["Churn_Probability"].values * 100
                trend = tmp.groupby("tenure_bucket", observed=True)["_val"].mean().reset_index()
                fig = go.Figure(go.Scatter(
                    x=trend["tenure_bucket"], y=trend["_val"], mode="lines+markers",
                    line=dict(color=RISK_RED, width=3), marker=dict(size=9, color=RISK_RED),
                    fill="tozeroy", fillcolor="rgba(229,72,77,0.08)",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=250, yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig, use_container_width=True, key="chart_by_tenure")
            else:
                st.info("No tenure column in this file.")
    else:
        st.warning("No data loaded. Go to Upload Data to load a file.")

# ============================================================
# PAGE: UPLOAD DATA
# ============================================================
elif page == "📁  Upload Data":
    st.markdown('<div class="section-title">Upload Customer Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Uploading a file here updates the whole dashboard — Home, '
        'Why Customers Churn, Retention Priorities, and Customer Explorer all switch to this data. '
        'Model Performance always evaluates against the original company dataset.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
<div class="insight-card severity-medium">
<b>⚠ Important — this model expects a specific schema.</b><br>
The model was trained only on the Telco Customer Churn dataset's columns and category values.
It does <b>not</b> automatically adapt to a different telecom provider's data if the column names,
number of columns, or category labels are different (e.g. different contract-type names, extra
services, different currency fields). To score correctly, your CSV needs these exact columns:
<br><br>
<code>tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod,
OnlineSecurity, TechSupport, OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies,
PaperlessBilling, SeniorCitizen, gender, Partner, Dependents, PhoneService, MultipleLines</code>
<br><br>
Categorical columns also need matching values (e.g. Contract must be one of
"Month-to-month", "One year", "Two year"). A file with a different schema will either error out
or silently produce unreliable scores — to support a genuinely different telecom company's data
layout, the model would need to be retrained on that schema.
</div>
""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.session_state.active_df = new_df
            st.session_state.active_label = f"Uploaded: {uploaded_file.name}"
            st.success(f"Loaded {len(new_df):,} customer records. This file now drives the dashboard.")
            st.rerun()
    with col_b:
        if st.button("↺ Reset to Full Company Dataset"):
            st.session_state.active_df = company_df
            st.session_state.active_label = "Full Company Dataset"
            st.rerun()

    if st.session_state.active_df is not None:
        st.markdown('<div class="section-title" style="margin-top:20px;">Preview — Active Dataset</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.active_df.head(25), use_container_width=True)

# ============================================================
# PAGE: WHY CUSTOMERS CHURN (SHAP)
# ============================================================
elif page == "🧠  Why Customers Churn":
    st.markdown('<div class="section-title">Global Churn Drivers</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">Based on: {active_label}</div>', unsafe_allow_html=True)
    if explainer is None:
        st.info("SHAP explainer not found.")
    elif active_df is None:
        st.info("No active dataset available.")
    else:
        sample_df = clean_raw_dataframe(active_df.sample(min(300, len(active_df)), random_state=42))
        sample_eng = engineer_features(sample_df)
        sample_encoded = encode_for_model(sample_eng, model_cols)
        shap_values = explainer.shap_values(sample_encoded)
        vals = shap_values[:, :, 1] if len(np.array(shap_values).shape) == 3 else shap_values
        importance = pd.DataFrame({
            "feature": sample_encoded.columns, "impact": np.abs(vals).mean(axis=0),
        }).sort_values("impact", ascending=True).tail(15)
        fig = go.Figure(go.Bar(x=importance["impact"], y=importance["feature"], orientation="h", marker_color=GOLD))
        fig.update_layout(**PLOTLY_LAYOUT, height=500, xaxis_title="Average Impact on Churn Probability")
        st.plotly_chart(fig, use_container_width=True, key="chart_shap_importance")
        top_feature = importance.iloc[-1]["feature"]
        st.markdown(
            f'<div class="section-caption">In plain terms: <b style="color:{TEXT}">{top_feature}</b> has the single '
            f'largest effect on predicted churn risk across this {min(300, len(active_df))}-customer sample.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE: CUSTOMER SIMULATOR
# ============================================================
elif page == "🎯  Customer Simulator":
    st.markdown('<div class="section-title">Customer Attribute Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Adjust attributes to see predicted churn risk update in real time.</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with sc2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with sc3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])

    raw_record = {
        "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": tenure * monthly_charges,
        "Contract": contract, "InternetService": internet, "PaymentMethod": payment,
        "OnlineSecurity": security, "TechSupport": tech_support,
        "OnlineBackup": "No", "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "PaperlessBilling": paperless, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "gender": "Male", "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    }
    input_df = pd.DataFrame([raw_record])
    encoded_input = encode_for_model(engineer_features(input_df), model_cols)
    prob = model.predict_proba(encoded_input)[0][1]
    rev_risk = monthly_charges * prob

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(kpi_card("Churn Risk", f"{prob*100:.1f}%", "red" if prob >= 0.5 else ""), unsafe_allow_html=True)
    m2.markdown(kpi_card("Monthly Billing", f"${monthly_charges:.2f}"), unsafe_allow_html=True)
    m3.markdown(kpi_card("Revenue at Risk", f"${rev_risk:.2f}", "red" if prob >= 0.35 else "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended Retention Action</div>', unsafe_allow_html=True)
    result = recommend_action(raw_record, prob)
    risk_level = result["risk_level"]
    badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[risk_level]
    st.markdown(f'<span class="badge {badge_class}">{risk_level.upper()} RISK</span>', unsafe_allow_html=True)
    st.write(result["action"])

# ============================================================
# PAGE: RETENTION PRIORITIES
# ============================================================
elif page == "🏆  Retention Priorities":
    st.markdown('<div class="section-title">Priority Action List</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">Based on: {active_label}</div>', unsafe_allow_html=True)
    if active_df is None:
        st.info("Dataset required for this view.")
    else:
        priority_df = scored.copy()
        priority_df["Contract"] = active_df["Contract"].values if "Contract" in active_df.columns else "—"
        priority_df["tenure"] = active_df["tenure"].values if "tenure" in active_df.columns else np.nan
        priority_df["TotalCharges"] = pd.to_numeric(active_df["TotalCharges"], errors="coerce").values
        median_value = priority_df["TotalCharges"].median()

        def priority_tier(row):
            high_value = row["TotalCharges"] > median_value if pd.notna(row["TotalCharges"]) else False
            if row["RiskLevel"] == "High" and high_value:
                return "P1 · Urgent"
            elif row["RiskLevel"] == "High":
                return "P3 · Automated Offer"
            elif row["RiskLevel"] == "Medium" and high_value:
                return "P2 · Loyalty Outreach"
            return "P4 · Monitor"

        priority_df["Priority"] = priority_df.apply(priority_tier, axis=1)
        top_priority = priority_df.sort_values("RevenueAtRisk", ascending=False).head(25)
        st.dataframe(
            top_priority[["Contract", "tenure", "MonthlyCharges", "Churn_Probability", "RevenueAtRisk", "RiskLevel", "Priority"]],
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download Priority List (CSV)",
            top_priority.to_csv(index=False).encode("utf-8"),
            "priority_accounts.csv", "text/csv",
        )

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "📈  Model Performance":
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">How well the model actually predicts churn, tested on data it never saw during training.</div>',
        unsafe_allow_html=True,
    )
    if company_df is None:
        st.info("Company dataset required to compute performance metrics.")
    else:
        perf = compute_model_performance(model, model_cols, company_df)

        p1, p2, p3, p4, p5 = st.columns(5)
        for col, label, value in zip(
            [p1, p2, p3, p4, p5],
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            [perf["accuracy"], perf["precision"], perf["recall"], perf["f1"], perf["roc_auc"]],
        ):
            col.markdown(kpi_card(label, f"{value*100:.1f}%", "gold"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = perf["cm"]
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Predicted: Stay", "Predicted: Churn"], y=["Actual: Stay", "Actual: Churn"],
                colorscale=[[0, SURFACE], [1, GOLD]],
                text=cm, texttemplate="%{text}", textfont=dict(size=18, color=TEXT),
                showscale=False,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True, key="chart_confusion_company")
        with c2:
            st.markdown('<div class="section-title">What This Means</div>', unsafe_allow_html=True)
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            st.markdown(f"""
- **{tp}** at-risk customers caught in time
- **{fn}** churners the model missed
- **{fp}** loyal customers wrongly flagged (a wasted offer, not a big cost)
- **{tn}** loyal customers correctly left alone

The model favors **catching churners over being precise** — missing a real churner costs more
than a few unnecessary retention offers.
            """)
        st.caption(f"Tested on {perf['n_test']:,} customers held out from training.")

        # ---------------- PERFORMANCE ON ACTIVE / UPLOADED DATA ----------------
        st.markdown('<div class="section-title" style="margin-top:34px;">Performance on Active Dataset</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-caption">Currently loaded: {active_label}</div>', unsafe_allow_html=True)

        if active_label == "Full Company Dataset":
            st.info("Upload a file on the Upload Data page to see the model's performance measured on that file specifically.")
        elif not has_actual_churn:
            st.warning(
                "This file has no `Churn` column, so accuracy can't be measured — there's no real outcome "
                "to check predictions against. What you can see instead: the risk scores themselves, on the "
                "Home page and Customer Explorer."
            )
        else:
            active_perf = compute_performance_on_active(model, model_cols, active_df)
            ap1, ap2, ap3, ap4, ap5 = st.columns(5)
            for col, label, value in zip(
                [ap1, ap2, ap3, ap4, ap5],
                ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                [active_perf["accuracy"], active_perf["precision"], active_perf["recall"],
                 active_perf["f1"], active_perf["roc_auc"]],
            ):
                display_val = "—" if pd.isna(value) else f"{value*100:.1f}%"
                col.markdown(kpi_card(label, display_val, "gold"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ac1, ac2 = st.columns([1, 1.2])
            with ac1:
                cm2 = active_perf["cm"]
                fig2 = go.Figure(go.Heatmap(
                    z=cm2, x=["Predicted: Stay", "Predicted: Churn"], y=["Actual: Stay", "Actual: Churn"],
                    colorscale=[[0, SURFACE], [1, GOLD]],
                    text=cm2, texttemplate="%{text}", textfont=dict(size=18, color=TEXT),
                    showscale=False,
                ))
                fig2.update_layout(**PLOTLY_LAYOUT, height=300)
                st.plotly_chart(fig2, use_container_width=True, key="chart_confusion_active")
                st.caption(f"Measured on all {active_perf['n_test']:,} rows of the uploaded file — not a held-out sample.")
            with ac2:
                st.markdown('<div class="section-title">What This Means For This File</div>', unsafe_allow_html=True)
                tn2, fp2, fn2, tp2 = cm2[0, 0], cm2[0, 1], cm2[1, 0], cm2[1, 1]
                total2 = tn2 + fp2 + fn2 + tp2
                st.markdown(f"""
- **{tp2:,}** at-risk customers in this file caught correctly
- **{fn2:,}** actual churners this file's model run missed
- **{fp2:,}** loyal customers flagged by mistake (a wasted offer, not a real loss)
- **{tn2:,}** loyal customers correctly left alone
- Out of **{total2:,}** customers in this file, the model correctly called **{tp2+tn2:,}** of them

In plain terms: if this file represents real customers, retention outreach based on these
scores would reach most of the people actually at risk — with some false alarms mixed in,
which is the intended trade-off (see "What This Means" above).
                """)

# ============================================================
# PAGE: CUSTOMER EXPLORER
# ============================================================
elif page == "🔎  Customer Explorer":
    st.markdown('<div class="section-title">Customer Explorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">Based on: {active_label} — filter, inspect, and export.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">This table lists every customer with their churn risk score and revenue '
        'at stake, so you can drill into any slice of the business instead of just the top-line numbers. Filter '
        'by contract, risk level, or tenure below, then export the exact list your retention team should work '
        'from today.</div>',
        unsafe_allow_html=True,
    )

    if active_df is None:
        st.info("No active dataset available.")
    else:
        explorer_df = scored.copy()
        for col in ["Contract", "InternetService", "PaymentMethod", "tenure", "gender"]:
            if col in active_df.columns:
                explorer_df[col] = active_df[col].values

        f1, f2, f3 = st.columns(3)
        with f1:
            contracts = sorted(explorer_df["Contract"].dropna().unique()) if "Contract" in explorer_df.columns else []
            pick_contract = st.multiselect("Contract Type", contracts, default=contracts)
        with f2:
            risk_levels = ["Low", "Medium", "High"]
            pick_risk = st.multiselect("Risk Level", risk_levels, default=risk_levels)
        with f3:
            if "tenure" in explorer_df.columns:
                t_min, t_max = int(explorer_df["tenure"].min()), int(explorer_df["tenure"].max())
                tenure_range = st.slider("Tenure Range (Months)", t_min, t_max, (t_min, t_max))
            else:
                tenure_range = None

        filtered = explorer_df.copy()
        if pick_contract and "Contract" in filtered.columns:
            filtered = filtered[filtered["Contract"].isin(pick_contract)]
        if pick_risk:
            filtered = filtered[filtered["RiskLevel"].astype(str).isin(pick_risk)]
        if tenure_range and "tenure" in filtered.columns:
            filtered = filtered[filtered["tenure"].between(tenure_range[0], tenure_range[1])]

        st.markdown(f'<div class="section-caption">{len(filtered):,} of {len(explorer_df):,} customers match these filters.</div>', unsafe_allow_html=True)
        display_cols = [c for c in ["Contract", "tenure", "MonthlyCharges", "TotalCharges", "Churn_Probability", "RevenueAtRisk", "RiskLevel"] if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].sort_values("RevenueAtRisk", ascending=False),
            use_container_width=True, height=460,
        )
        st.download_button(
            "⬇ Download Filtered Results (CSV)",
            filtered[display_cols].to_csv(index=False).encode("utf-8"),
            "customer_explorer_results.csv", "text/csv",
        )
