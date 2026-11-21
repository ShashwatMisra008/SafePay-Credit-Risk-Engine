import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafePay Bank | Executive Credit Risk & Underwriting Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CORPORATE DARK FINTECH CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0B1120;
            color: #F8FAFC;
        }
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1480px;
        }
        header[data-testid="stHeader"] {
            background-color: rgba(11, 17, 32, 0.88);
            backdrop-filter: blur(8px);
        }
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0;
        }
        .hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #064E3B 100%);
            border: 1px solid #334155;
            padding: 1.5rem 2rem;
            border-radius: 14px;
            color: #F8FAFC;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45);
        }
        .hero-kicker {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #34D399;
            margin-bottom: 0.3rem;
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: #FFFFFF;
            letter-spacing: -0.4px;
        }
        .hero-subtitle {
            font-size: 0.93rem;
            color: #CBD5E1;
            margin: 0;
            line-height: 1.45;
        }
        .badge-row {
            margin-top: 0.85rem;
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .tech-badge {
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.38);
            padding: 0.22rem 0.72rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #6EE7B7;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: #111827;
            padding: 0.4rem;
            border-radius: 10px;
            border: 1px solid #1E293B;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.6rem;
            border-radius: 8px;
            padding: 0 1.15rem;
            color: #94A3B8;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #10B981 !important;
            border: 1px solid #334155;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111827;
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            padding: 0.65rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
            border: 1px solid #1E293B;
            padding: 0.95rem 1.1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }
        .decision-approve {
            background: linear-gradient(90deg, rgba(6, 78, 59, 0.65) 0%, rgba(16, 185, 129, 0.15) 100%);
            border: 1px solid #059669;
            border-left: 6px solid #10B981;
            padding: 1rem 1.3rem;
            border-radius: 10px;
            color: #ECFDF5;
            font-size: 0.96rem;
            margin: 0.85rem 0 1.25rem 0;
        }
        .decision-reject {
            background: linear-gradient(90deg, rgba(127, 29, 29, 0.65) 0%, rgba(220, 38, 38, 0.15) 100%);
            border: 1px solid #DC2626;
            border-left: 6px solid #EF4444;
            padding: 1rem 1.3rem;
            border-radius: 10px;
            color: #FEF2F2;
            font-size: 0.96rem;
            margin: 0.85rem 0 1.25rem 0;
        }
        .econ-card {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 0.9rem;
            font-size: 0.83rem;
            line-height: 1.65;
            color: #CBD5E1;
            margin-top: 0.75rem;
        }
        .section-heading {
            font-size: 1.12rem;
            font-weight: 700;
            color: #E2E8F0;
            margin: 1.1rem 0 0.65rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

ARTIFACT_PATH = "safepay_credit_risk_engine_v1.joblib"

# Core model features expected by the trained pipeline
MODEL_BASE_COLS = [
    "age", "annual_income_lakhs", "loan_amount_lakhs", "credit_score",
    "credit_utilization_pct", "late_payments_12m", "inquiries_6m",
    "employment_type", "city_tier", "debt_to_income_ratio",
    "delinquency_utilization_idx"
]


# -----------------------------------------------------------------------------
# 1. AUTOMATED FEATURE ENGINEERING ENGINE (8 DERIVED INTERDEPENDENT COLUMNS)
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically derives 8 interdependent financial & behavioral risk metrics
    from raw borrower attributes for live underwriting and portfolio analytics.
    """
    data = df.copy()

    # 1. Debt-to-Income Ratio (Core Model Feature)
    data["debt_to_income_ratio"] = (
        data["loan_amount_lakhs"] / np.maximum(data["annual_income_lakhs"], 0.1)
    ).round(3)

    # 2. Delinquency-Utilization Interaction Index (Core Model Feature)
    data["delinquency_utilization_idx"] = (
        (data["late_payments_12m"] + 1) * (data["credit_utilization_pct"] / 100.0)
    ).round(3)

    # 3. Monthly Income in INR Thousands
    data["monthly_income_k"] = ((data["annual_income_lakhs"] * 100.0) / 12.0).round(2)

    # 4. Estimated Monthly EMI in INR Thousands (24-month tenure @ 14.5% APR)
    r = 0.145 / 12.0
    n_months = 24
    emi_factor = (r * ((1 + r) ** n_months)) / (((1 + r) ** n_months) - 1)
    data["estimated_emi_k"] = ((data["loan_amount_lakhs"] * 100.0) * emi_factor).round(2)

    # 5. Fixed Obligation to Income Ratio (FOIR / EMI Burden %)
    data["emi_to_income_pct"] = (
        (data["estimated_emi_k"] / np.maximum(data["monthly_income_k"], 1.0)) * 100.0
    ).round(1)

    # 6. Net Annual Disposable Income After Debt Service (₹ Lakhs)
    data["net_disposable_lakhs"] = np.maximum(
        data["annual_income_lakhs"] - ((data["estimated_emi_k"] * 12.0) / 100.0), 0.0
    ).round(2)

    # 7. Available Credit Headroom (%)
    data["credit_headroom_pct"] = (100.0 - data["credit_utilization_pct"]).round(1)

    # 8. Credit Inquiry Velocity (Inquiries per credit-active year since age 18)
    credit_active_yrs = np.maximum(data["age"] - 18, 1)
    data["inquiry_velocity_idx"] = ((data["inquiries_6m"] / credit_active_yrs) * 10.0).round(2)

    return data


# -----------------------------------------------------------------------------
# 2. 12,000-RECORD PORTFOLIO GENERATOR & MODEL TRAINER
# -----------------------------------------------------------------------------
@st.cache_data
def generate_portfolio_dataset(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Generates a 12,000-account Indian credit portfolio with realistic correlations."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 65, n_samples)
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Freelancer"], size=n_samples, p=[0.60, 0.27, 0.13]
    )
    city_tier = rng.choice(
        ["Tier-1", "Tier-2", "Tier-3"], size=n_samples, p=[0.48, 0.33, 0.19]
    )

    # Income scales realistically with age and city tier
    tier_mult = np.where(city_tier == "Tier-1", 1.18, np.where(city_tier == "Tier-2", 0.98, 0.82))
    age_mult = 1.0 + np.clip((age - 25) * 0.012, 0.0, 0.45)
    annual_income_lakhs = np.round(
        rng.lognormal(mean=2.05, sigma=0.52, size=n_samples) * tier_mult * age_mult, 2
    )

    loan_amount_lakhs = np.round(annual_income_lakhs * rng.uniform(0.15, 0.95, n_samples), 2)
    credit_score = np.clip(rng.normal(712, 66, n_samples), 450, 880).astype(int)
    credit_utilization_pct = np.clip(rng.beta(2.2, 3.4, n_samples) * 100, 2, 99).round(1)
    late_payments_12m = rng.poisson(0.72, n_samples)
    inquiries_6m = rng.poisson(1.15, n_samples)

    dti = loan_amount_lakhs / (annual_income_lakhs + 0.1)
    emp_risk = np.where(
        employment_type == "Freelancer", 0.55, np.where(employment_type == "Self-Employed", 0.25, 0.0)
    )

    logit = (
        -3.1
        + 1.85 * dti
        + 0.028 * (credit_utilization_pct - 40)
        + 0.68 * late_payments_12m
        + 0.32 * inquiries_6m
        - 0.012 * (credit_score - 680)
        + emp_risk
        + 0.015 * (late_payments_12m * credit_utilization_pct / 10)
    )
    default_flag = (rng.random(n_samples) < (1 / (1 + np.exp(-logit)))).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"SP-{100000 + i}" for i in range(n_samples)],
        "age": age,
        "annual_income_lakhs": annual_income_lakhs,
        "loan_amount_lakhs": loan_amount_lakhs,
        "credit_score": credit_score,
        "credit_utilization_pct": credit_utilization_pct,
        "late_payments_12m": late_payments_12m,
        "inquiries_6m": inquiries_6m,
        "employment_type": employment_type,
        "city_tier": city_tier,
        "actual_default": default_flag
    })
    return engineer_features(df)


def _train_model_bundle(portfolio_df: pd.DataFrame) -> dict:
    X = portfolio_df[MODEL_BASE_COLS]
    y = portfolio_df["actual_default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ["employment_type", "city_tier"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    if HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=imbalance_ratio, eval_metric="logloss", random_state=42
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_depth=4,
            l2_regularization=2.0, class_weight="balanced", random_state=42
        )

    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]

    best_t, best_prof = 0.50, -np.inf
    for t in np.linspace(0.10, 0.90, 81):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        prof = (tn * 12000) - (fp * 4000) - (fn * 65000)
        if prof > best_prof:
            best_prof, best_t = prof, float(t)

    artifact = {
        "pipeline": pipe,
        "optimal_threshold": best_t,
        "feature_columns": MODEL_BASE_COLS,
        "metrics": {
            "test_pr_auc": float(average_precision_score(y_test, probs)),
            "test_roc_auc": float(roc_auc_score(y_test, probs))
        }
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


@st.cache_resource
def load_model_bundle(_portfolio_df: pd.DataFrame):
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH)
    return _train_model_bundle(_portfolio_df)


# Load 12,000-account dataset and score entire portfolio live
portfolio_df = generate_portfolio_dataset(n_samples=12000)
bundle = load_model_bundle(portfolio_df)
pipeline = bundle["pipeline"]
default_optimal_cutoff = bundle["optimal_threshold"]
metrics = bundle.get("metrics", {})

# Compute live probabilities across all 12,000 borrowers
portfolio_df["default_prob"] = pipeline.predict_proba(portfolio_df[MODEL_BASE_COLS])[:, 1]
portfolio_df["default_risk_pct"] = (portfolio_df["default_prob"] * 100).round(2)
portfolio_df["risk_tier"] = pd.cut(
    portfolio_df["default_prob"],
    bins=[-0.01, 0.20, 0.40, 0.65, 1.01],
    labels=["1. Prime Low-Risk", "2. Near-Prime Watch", "3. Subprime High-Risk", "4. Critical Default Risk"]
)


# -----------------------------------------------------------------------------
# DARK PLOTLY & MATPLOTLIB HELPERS
# -----------------------------------------------------------------------------
def apply_dark_plotly(fig, height: int = 330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        font=dict(color="#E2E8F0", size=12),
        margin=dict(l=20, r=20, t=42, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1")),
        xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
    )
    return fig


def make_risk_gauge(prob: float, threshold: float) -> go.Figure:
    bar_color = "#EF4444" if prob >= threshold else "#10B981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": "#F8FAFC"}},
            delta={
                "reference": round(threshold * 100, 1),
                "increasing": {"color": "#F87171"},
                "decreasing": {"color": "#34D399"},
                "suffix": "% vs Cutoff",
                "font": {"size": 13}
            },
            title={"text": "BORROWER DEFAULT PROBABILITY", "font": {"size": 13, "color": "#94A3B8"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                    "tickwidth": 1,
                    "tickcolor": "#64748B",
                    "tickfont": {"color": "#94A3B8", "size": 11}
                },
                "bar": {"color": bar_color, "thickness": 0.30},
                "bgcolor": "#0F172A",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(16, 185, 129, 0.16)"},
                    {"range": [threshold * 100, 100], "color": "rgba(239, 68, 68, 0.16)"}
                ],
                "threshold": {
                    "line": {"color": "#FBBF24", "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100
                }
            }
        )
    )
    fig.update_layout(
        height=310,
        margin=dict(l=25, r=25, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"}
    )
    return fig


def style_shap_dark(fig):
    fig.patch.set_facecolor("#111827")
    for ax in fig.axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#CBD5E1", labelsize=9)
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#CBD5E1")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        for text in ax.texts:
            text.set_color("#F8FAFC")
            text.set_fontsize(8.5)


# -----------------------------------------------------------------------------
# SIDEBAR: GLOBAL POLICY CONTROLS & DEPARTMENT FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏦 SafePay Risk Console")
    st.caption("Enterprise Credit Intelligence • 12,000 Accounts")
    st.divider()

    st.markdown("#### ⚙️ Decision Policy Threshold")
    optimal_val = round(float(default_optimal_cutoff), 2)

    if "cutoff_slider" not in st.session_state:
        st.session_state.cutoff_slider = optimal_val

    def reset_cutoff():
        st.session_state.cutoff_slider = optimal_val

    active_threshold = st.slider(
        "Default Probability Cutoff",
        min_value=0.10,
        max_value=0.90,
        key="cutoff_slider",
        step=0.01,
        help="Live policy slider: updates KPIs, Net Profit, and Approval rates across all 12,000 accounts."
    )

    st.button("↺ Reset to Profit-Optimal Cutoff", on_click=reset_cutoff, use_container_width=True)

    st.divider()
    st.markdown("#### 🏢 Department Portfolio Filters")
    selected_tiers = st.multiselect(
        "City Tier", ["Tier-1", "Tier-2", "Tier-3"], default=["Tier-1", "Tier-2", "Tier-3"]
    )
    selected_emp = st.multiselect(
        "Employment Segment", ["Salaried", "Self-Employed", "Freelancer"],
        default=["Salaried", "Self-Employed", "Freelancer"]
    )

    st.markdown(
        f"""
        <div class="econ-card">
            <div style="color:#34D399;font-weight:700;margin-bottom:5px;">
                🎯 Profit-Optimal Cutoff: <code>{default_optimal_cutoff:.2f}</code>
            </div>
            <div style="font-weight:700;color:#F8FAFC;margin-bottom:3px;">💰 Unit Economics (INR):</div>
            <div>• Repaid Loan (TN): <span style="color:#34D399;font-weight:700;">+₹12,000</span></div>
            <div>• Rejected Good (FP): <span style="color:#FBBF24;font-weight:700;">-₹4,000</span></div>
            <div>• Approved Default (FN): <span style="color:#F87171;font-weight:700;">-₹65,000</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("PR-AUC", f"{metrics.get('test_pr_auc', 0.0):.3f}")
    col_m2.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0.0):.3f}")


# Apply sidebar department filters to the 12,000-record portfolio
filtered_df = portfolio_df[
    (portfolio_df["city_tier"].isin(selected_tiers if selected_tiers else ["Tier-1", "Tier-2", "Tier-3"])) &
    (portfolio_df["employment_type"].isin(selected_emp if selected_emp else ["Salaried", "Self-Employed", "Freelancer"]))
].copy()

filtered_df["policy_decision"] = np.where(
    filtered_df["default_prob"] >= active_threshold, "Rejected / Flagged", "Approved"
)


# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-kicker">SAFEPAY BANK • EXECUTIVE &amp; DEPARTMENT RISK COMMAND CENTER</div>
        <div class="hero-title">🏦 Enterprise Credit Portfolio &amp; Automated Underwriting Suite</div>
        <p class="hero-subtitle">
            Live portfolio intelligence across <strong>{len(filtered_df):,} active borrowers</strong> with <strong>8 automated derived risk indicators</strong>,
            cost-matrix policy simulation, and borrower-level SHAP explainability.
        </p>
        <div class="badge-row">
            <span class="tech-badge">📊 {len(filtered_df):,} Live Accounts</span>
            <span class="tech-badge">⚙️ 8 Auto-Derived Financial Ratios</span>
            <span class="tech-badge">💰 Net Profit Policy Simulator</span>
            <span class="tech-badge">🔍 SHAP Regulatory Explainability</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ Executive Command Center (CXO View)",
    "🔬 Column & Feature Intelligence (Dept. Heads)",
    "🎯 Live Single Applicant Underwriting & SHAP",
    "📂 Batch Scoring & 12k Portfolio Explorer"
])


# =============================================================================
# TAB 1: EXECUTIVE COMMAND CENTER (COMPANY HEAD / CXO VIEW)
# =============================================================================
with tab1:
    # Live Confusion Matrix & Net Profit Calculations on Filtered Portfolio
    y_true_live = filtered_df["actual_default"].values
    y_prob_live = filtered_df["default_prob"].values
    y_pred_live = (y_prob_live >= active_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_live, y_pred_live, labels=[0, 1]).ravel()
    net_profit_cr = ((tn * 12000) - (fp * 4000) - (fn * 65000)) / 1e7
    loss_prevented_cr = (tp * 65000) / 1e7
    approval_rate = (y_pred_live == 0).mean() * 100
    default_capture_rate = (tp / max(tp + fn, 1)) * 100
    disbursed_cr = filtered_df.loc[y_pred_live == 0, "loan_amount_lakhs"].sum() / 100.0

    st.markdown('<div class="section-heading">1. Live Portfolio Financial & Risk KPIs (Updates with Policy Slider)</div>', unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Net Portfolio Profit", f"₹{net_profit_cr:.2f} Cr", delta=f"Cutoff: {active_threshold:.2f}")