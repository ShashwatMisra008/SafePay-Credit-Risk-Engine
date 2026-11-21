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
    average_precision_score, roc_auc_score, confusion_matrix, precision_score
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
    page_title="SafePay Bank | Global Credit Risk & Underwriting Intelligence",
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
            max-width: 1500px;
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
            padding: 1.45rem 2rem;
            border-radius: 14px;
            color: #F8FAFC;
            margin-bottom: 1.2rem;
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
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: #FFFFFF;
            letter-spacing: -0.4px;
        }
        .hero-subtitle {
            font-size: 0.92rem;
            color: #CBD5E1;
            margin: 0;
            line-height: 1.45;
        }
        .badge-row {
            margin-top: 0.8rem;
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
            padding: 0 1.1rem;
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
            padding: 0.9rem 1.05rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.5rem !important;
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

ARTIFACT_PATH = "safepay_credit_risk_engine_v2.joblib"

MODEL_BASE_COLS = [
    "age", "annual_income_lakhs", "loan_amount_lakhs", "credit_score",
    "credit_utilization_pct", "late_payments_12m", "inquiries_6m",
    "employment_type", "city_tier", "product_line", "acquisition_channel",
    "country", "debt_to_income_ratio", "delinquency_utilization_idx"
]

GEO_CATALOG = [
    ("India", "Maharashtra", "Mumbai", "Tier-1", 0.16),
    ("India", "Karnataka", "Bengaluru", "Tier-1", 0.15),
    ("India", "Delhi NCR", "New Delhi", "Tier-1", 0.11),
    ("India", "Uttar Pradesh", "Greater Noida", "Tier-2", 0.09),
    ("India", "Telangana", "Hyderabad", "Tier-1", 0.10),
    ("India", "Tamil Nadu", "Chennai", "Tier-1", 0.08),
    ("India", "Maharashtra", "Pune", "Tier-2", 0.08),
    ("India", "Gujarat", "Ahmedabad", "Tier-2", 0.06),
    ("India", "West Bengal", "Kolkata", "Tier-2", 0.05),
    ("UAE", "Dubai", "Dubai", "Tier-1", 0.05),
    ("Singapore", "Central Region", "Singapore", "Tier-1", 0.04),
    ("United Kingdom", "Greater London", "London", "Tier-1", 0.03),
]


# -----------------------------------------------------------------------------
# 1. AUTOMATED FEATURE ENGINEERING ENGINE (8 DERIVED INTERDEPENDENT COLUMNS)
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically computes 8 interdependent financial & behavioral risk metrics."""
    data = df.copy()

    # 1. Debt-to-Income Ratio
    data["debt_to_income_ratio"] = (
        data["loan_amount_lakhs"] / np.maximum(data["annual_income_lakhs"], 0.1)
    ).round(3)

    # 2. Delinquency-Utilization Interaction Index
    data["delinquency_utilization_idx"] = (
        (data["late_payments_12m"] + 1) * (data["credit_utilization_pct"] / 100.0)
    ).round(3)

    # 3. Monthly Income in INR Thousands
    data["monthly_income_k"] = ((data["annual_income_lakhs"] * 100.0) / 12.0).round(2)

    # 4. Product-Aware APR & Estimated Monthly EMI (₹ Thousands, 24-month tenure)
    apr_map = {
        "Personal Term Loan": 0.145,
        "BNPL Pay-Later": 0.180,
        "Instant Credit Line": 0.165,
        "MSME Working Capital": 0.155,
        "Auto Upgrade Loan": 0.115,
    }
    apr = data["product_line"].map(apr_map).fillna(0.145)
    r = apr / 12.0
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

    # 8. Credit Inquiry Velocity Index (Inquiries per credit-active decade)
    credit_active_yrs = np.maximum(data["age"] - 18, 1)
    data["inquiry_velocity_idx"] = ((data["inquiries_6m"] / credit_active_yrs) * 10.0).round(2)

    return data


# -----------------------------------------------------------------------------
# 2. 12,000-RECORD GLOBAL & INDIA PORTFOLIO DATASET GENERATOR
# -----------------------------------------------------------------------------
@st.cache_data
def generate_portfolio_dataset(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    geo_probs = [g[4] for g in GEO_CATALOG]
    geo_indices = rng.choice(len(GEO_CATALOG), size=n_samples, p=geo_probs)
    countries = [GEO_CATALOG[i][0] for i in geo_indices]
    states = [GEO_CATALOG[i][1] for i in geo_indices]
    cities = [GEO_CATALOG[i][2] for i in geo_indices]
    city_tiers = [GEO_CATALOG[i][3] for i in geo_indices]

    age = rng.integers(21, 65, n_samples)
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Freelancer"], size=n_samples, p=[0.58, 0.28, 0.14]
    )
    product_line = rng.choice(
        ["Personal Term Loan", "BNPL Pay-Later", "Instant Credit Line", "MSME Working Capital", "Auto Upgrade Loan"],
        size=n_samples,
        p=[0.32, 0.24, 0.20, 0.14, 0.10]
    )
    acquisition_channel = rng.choice(
        ["Mobile App Organic", "Partner Merchant", "Pre-Approved Offer", "DSA / Branch"],
        size=n_samples,
        p=[0.40, 0.25, 0.20, 0.15]
    )

    # Income scales with geography, city tier, and age
    country_mult = np.where(np.array(countries) == "India", 1.0, 1.85)
    tier_mult = np.where(np.array(city_tiers) == "Tier-1", 1.15, 0.90)
    age_mult = 1.0 + np.clip((age - 25) * 0.012, 0.0, 0.45)

    annual_income_lakhs = np.round(
        rng.lognormal(mean=2.05, sigma=0.50, size=n_samples) * country_mult * tier_mult * age_mult, 2
    )
    loan_amount_lakhs = np.round(annual_income_lakhs * rng.uniform(0.15, 0.92, n_samples), 2)
    credit_score = np.clip(rng.normal(714, 65, n_samples), 450, 880).astype(int)
    credit_utilization_pct = np.clip(rng.beta(2.2, 3.4, n_samples) * 100, 2, 99).round(1)
    late_payments_12m = rng.poisson(0.70, n_samples)
    inquiries_6m = rng.poisson(1.12, n_samples)

    # Realistic risk effects by segment, product, channel, and geography
    dti = loan_amount_lakhs / (annual_income_lakhs + 0.1)
    emp_risk = np.where(employment_type == "Freelancer", 0.52, np.where(employment_type == "Self-Employed", 0.22, 0.0))
    prod_risk = np.where(
        product_line == "BNPL Pay-Later", 0.38,
        np.where(product_line == "Instant Credit Line", 0.22,
                 np.where(product_line == "Auto Upgrade Loan", -0.35, 0.0))
    )
    chan_risk = np.where(
        acquisition_channel == "Partner Merchant", 0.25,
        np.where(acquisition_channel == "Pre-Approved Offer", -0.30, 0.0)
    )
    geo_risk = np.where(np.array(countries) == "India", 0.08, -0.25)

    logit = (
        -3.15
        + 1.80 * dti
        + 0.027 * (credit_utilization_pct - 40)
        + 0.66 * late_payments_12m
        + 0.30 * inquiries_6m
        - 0.012 * (credit_score - 680)
        + emp_risk + prod_risk + chan_risk + geo_risk
        + 0.014 * (late_payments_12m * credit_utilization_pct / 10)
    )
    default_flag = (rng.random(n_samples) < (1 / (1 + np.exp(-logit)))).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"SP-{100000 + i}" for i in range(n_samples)],
        "country": countries,
        "state": states,
        "city_name": cities,
        "city_tier": city_tiers,
        "product_line": product_line,
        "acquisition_channel": acquisition_channel,
        "employment_type": employment_type,
        "age": age,
        "annual_income_lakhs": annual_income_lakhs,
        "loan_amount_lakhs": loan_amount_lakhs,
        "credit_score": credit_score,
        "credit_utilization_pct": credit_utilization_pct,
        "late_payments_12m": late_payments_12m,
        "inquiries_6m": inquiries_6m,
        "actual_default": default_flag
    })
    df = engineer_features(df)

    # Automated Customer Segment classification
    df["customer_segment"] = np.select(
        [
            (df["annual_income_lakhs"] >= 18.0) & (df["credit_score"] >= 740),
            (df["employment_type"] == "Self-Employed") | (df["product_line"] == "MSME Working Capital"),
            (df["employment_type"] == "Freelancer") | (df["credit_score"] < 640),
        ],
        ["Affluent Prime", "MSME & Self-Employed", "Emerging / High-Beta"],
        default="Mass Salaried Retail"
    )
    return df


def _train_model_bundle(portfolio_df: pd.DataFrame) -> dict:
    X = portfolio_df[MODEL_BASE_COLS]
    y = portfolio_df["actual_default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ["employment_type", "city_tier", "product_line", "acquisition_channel", "country"]

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
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        prof = (int(tn) * 12000) - (int(fp) * 4000) - (int(fn) * 65000)
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


portfolio_df = generate_portfolio_dataset(n_samples=12000)
bundle = load_model_bundle(portfolio_df)
pipeline = bundle["pipeline"]
default_optimal_cutoff = bundle["optimal_threshold"]
metrics = bundle.get("metrics", {})

# Live scoring across all 12,000 accounts
portfolio_df["default_prob"] = pipeline.predict_proba(portfolio_df[MODEL_BASE_COLS])[:, 1]
portfolio_df["default_risk_pct"] = (portfolio_df["default_prob"] * 100).round(2)
portfolio_df["risk_tier"] = pd.cut(
    portfolio_df["default_prob"],
    bins=[-0.01, 0.20, 0.40, 0.65, 1.01],
    labels=["1. Prime Low-Risk", "2. Near-Prime Watch", "3. Subprime High-Risk", "4. Critical Default Risk"]
).astype(str)


# -----------------------------------------------------------------------------
# DARK PLOTLY & MATPLOTLIB HELPERS
# -----------------------------------------------------------------------------
def apply_dark_plotly(fig, height: int = 320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        font=dict(color="#E2E8F0", size=12),
        margin=dict(l=20, r=20, t=38, b=20),
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
# SIDEBAR: GLOBAL POLICY CONTROLS & MULTI-DIMENSION FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏦 SafePay Risk Console")
    st.caption("Global Credit Intelligence • 12,000 Accounts")
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
        help="Live policy slider: updates all KPIs, Net Profit, and graphs across all tabs."
    )
    st.button("↺ Reset to Profit-Optimal Cutoff", on_click=reset_cutoff, use_container_width=True)

    st.divider()
    st.markdown("#### 🌍 Enterprise Slice Filters")

    all_countries = sorted(portfolio_df["country"].unique().tolist())
    sel_countries = st.multiselect("Country", all_countries, default=all_countries)

    avail_states = sorted(portfolio_df[portfolio_df["country"].isin(sel_countries or all_countries)]["state"].unique().tolist())
    sel_states = st.multiselect("State / Region", avail_states, default=avail_states)

    avail_cities = sorted(portfolio_df[portfolio_df["state"].isin(sel_states or avail_states)]["city_name"].unique().tolist())
    sel_cities = st.multiselect("City", avail_cities, default=avail_cities)

    all_products = sorted(portfolio_df["product_line"].unique().tolist())
    sel_products = st.multiselect("Product Line", all_products, default=all_products)

    all_segments = sorted(portfolio_df["customer_segment"].unique().tolist())
    sel_segments = st.multiselect("Customer Segment", all_segments, default=all_segments)

    st.markdown(
        f"""
        <div class="econ-card">
            <div style="color:#34D399;font-weight:700;margin-bottom:4px;">
                🎯 Optimal Cutoff: <code>{default_optimal_cutoff:.2f}</code>
            </div>
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


# Apply live filters to 12,000-record dataset
filtered_df = portfolio_df[
    (portfolio_df["country"].isin(sel_countries if sel_countries else all_countries)) &
    (portfolio_df["state"].isin(sel_states if sel_states else avail_states)) &
    (portfolio_df["city_name"].isin(sel_cities if sel_cities else avail_cities)) &
    (portfolio_df["product_line"].isin(sel_products if sel_products else all_products)) &
    (portfolio_df["customer_segment"].isin(sel_segments if sel_segments else all_segments))
].copy()

if len(filtered_df) == 0:
    filtered_df = portfolio_df.copy()

filtered_df["policy_decision"] = np.where(
    filtered_df["default_prob"] >= active_threshold, "Rejected / Flagged", "Approved"
)
filtered_df["unit_net_profit_inr"] = np.select(
    [
        (filtered_df["policy_decision"] == "Approved") & (filtered_df["actual_default"] == 0),
        (filtered_df["policy_decision"] != "Approved") & (filtered_df["actual_default"] == 0),
        (filtered_df["policy_decision"] == "Approved") & (filtered_df["actual_default"] == 1),
    ],
    [12000, -4000, -65000],
    default=0
)


# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-kicker">SAFEPAY BANK • GLOBAL EXECUTIVE &amp; DEPARTMENT RISK COMMAND CENTER</div>
        <div class="hero-title">🏦 Enterprise Credit Portfolio &amp; Automated Underwriting Suite</div>
        <p class="hero-subtitle">
            Live multi-region intelligence across <strong>{len(filtered_df):,} active accounts</strong> (India, UAE, Singapore, UK) with
            <strong>8 automated derived risk indicators</strong>, live policy simulation, and borrower-level SHAP explainability.
        </p>
        <div class="badge-row">
            <span class="tech-badge">📊 {len(filtered_df):,} Live Filtered Accounts</span>
            <span class="tech-badge">🌍 {filtered_df['city_name'].nunique()} Cities • {filtered_df['product_line'].nunique()} Products</span>
            <span class="tech-badge">⚙️ 8 Auto-Derived Financial Ratios</span>
            <span class="tech-badge">💰 Net Profit Policy Simulator</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ Executive Command Center (CXO View)",
    "🔬 Department & Column Analytics (Dept. Heads)",
    "🎯 Live Single Applicant Underwriting & SHAP",
    "📂 12k Portfolio Explorer & Batch Scoring"
])


# =============================================================================
# TAB 1: EXECUTIVE COMMAND CENTER (COMPANY HEAD / CXO VIEW)
# =============================================================================
with tab1:
    y_true_live = filtered_df["actual_default"].values
    y_prob_live = filtered_df["default_prob"].values
    y_pred_live = (y_prob_live >= active_threshold).astype(int)

    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y_true_live, y_pred_live, labels=[0, 1]).ravel()]
    net_profit_cr = ((tn * 12000) - (fp * 4000) - (fn * 65000)) / 1e7
    loss_prevented_cr = (tp * 65000) / 1e7
    approval_rate = float((y_pred_live == 0).mean() * 100)
    default_capture_rate = (tp / max(tp + fn, 1)) * 100.0
    prec_pct = float(precision_score(y_true_live, y_pred_live, zero_division=0) * 100)
    disbursed_cr = float(filtered_df.loc[y_pred_live == 0, "loan_amount_lakhs"].sum() / 100.0)
    avg_dti = float(filtered_df["debt_to_income_ratio"].mean())

    st.markdown('<div class="section-heading">1. Live Executive Portfolio KPIs (Updates Dynamically with Policy Slider & Filters)</div>', unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Net Portfolio Profit", f"₹{net_profit_cr:.2f} Cr", delta=f"Cutoff: {active_threshold:.2f}")
    e2.metric("Write-Off Loss Prevented", f"₹{loss_prevented_cr:.2f} Cr", delta=f"{tp:,} Defaults Blocked")
    e3.metric("Loan Approval Rate", f"{approval_rate:.1f}%", delta=f"{tn + fn:,} Approved")
    e4.metric("Default Capture (Recall)", f"{default_capture_rate:.1f}%", delta=f"Precision: {prec_pct:.1f}%")
    e5.metric("Approved Disbursal Vol.", f"₹{disbursed_cr:.2f} Cr", delta=f"Avg DTI: {avg_dti:.2f}x")

    # Row 2: Live Profit Curve + Product-Line Net Profit & Risk
    col_p1, col_p2 = st.columns([1.25, 1.15], gap="medium")

    with col_p1:
        with st.container(border=True):
            st.markdown("##### 💰 Policy Threshold vs. Net Portfolio Profit (₹ Crores)")
            t_grid = np.linspace(0.10, 0.90, 41)
            prof_curve = []
            for t in t_grid:
                p_t = (y_prob_live >= t).astype(int)
                tn_t, fp_t, fn_t, _ = [int(x) for x in confusion_matrix(y_true_live, p_t, labels=[0, 1]).ravel()]
                prof_curve.append(((tn_t * 12000) - (fp_t * 4000) - (fn_t * 65000)) / 1e7)

            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(
                x=t_grid, y=prof_curve, mode="lines", name="Net Profit (₹ Cr)",
                line=dict(color="#10B981", width=3), fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.14)"
            ))
            fig_prof.add_vline(x=active_threshold, line_dash="dash", line_color="#FBBF24",
                               annotation_text=f"Active Cutoff ({active_threshold:.2f})")
            fig_prof.update_xaxes(title="Default Probability Cutoff")
            fig_prof.update_yaxes(title="Net Profit (₹ Crores)")
            st.plotly_chart(apply_dark_plotly(fig_prof, 320), use_container_width=True)

    with col_p2:
        with st.container(border=True):
            st.markdown("##### 📦 Live Net Profit (₹ Lakhs) & Approval % by Product Line")
            prod_grp = filtered_df.groupby("product_line", as_index=False).agg(
                Net_Profit_Lakhs=("unit_net_profit_inr", lambda s: round(s.sum() / 1e5, 2)),
                Approval_Pct=("policy_decision", lambda s: round((s == "Approved").mean() * 100, 1)),
                Accounts=("applicant_id", "count")
            ).sort_values("Net_Profit_Lakhs", ascending=True)

            fig_prod = px.bar(
                prod_grp, y="product_line", x="Net_Profit_Lakhs", orientation="h",
                text="Approval_Pct", color="Net_Profit_Lakhs",
                color_continuous_scale=["#EF4444", "#FBBF24", "#10B981"]
            )
            fig_prod.update_traces(texttemplate="₹%{x:.1f}L (%{text}% Appr)", textposition="outside")
            fig_prod.update_layout(coloraxis_showscale=False)
            fig_prod.update_xaxes(title="Net Profit (₹ Lakhs)")
            fig_prod.update_yaxes(title="")
            st.plotly_chart(apply_dark_plotly(fig_prod, 320), use_container_width=True)

    # Row 3: City/State Live Performance + Underwriting Outcome Donut
    col_b1, col_b2 = st.columns([1.35, 1.0], gap="medium")

    with col_b1:
        with st.container(border=True):
            st.markdown("##### 🏙️ Live City-Wise Portfolio Profit (₹ Lakhs) & Default Risk")
            city_grp = filtered_df.groupby(["city_name", "country"], as_index=False).agg(
                Accounts=("applicant_id", "count"),
                Net_Profit_Lakhs=("unit_net_profit_inr", lambda s: round(s.sum() / 1e5, 2)),
                Actual_Default_Pct=("actual_default", lambda s: round(s.mean() * 100, 1)),
                Approval_Pct=("policy_decision", lambda s: round((s == "Approved").mean() * 100, 1))
            ).sort_values("Net_Profit_Lakhs", ascending=False)

            fig_city = px.bar(
                city_grp, x="city_name", y="Net_Profit_Lakhs", color="Actual_Default_Pct",
                text="Accounts", color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                labels={"city_name": "City", "Net_Profit_Lakhs": "Net Profit (₹ Lakhs)", "Actual_Default_Pct": "Default %"}
            )
            fig_city.update_traces(texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(apply_dark_plotly(fig_city, 320), use_container_width=True)

    with col_b2:
        with st.container(border=True):
            st.markdown("##### 🛡️ Live Underwriting Decision Matrix")
            outcome_df = pd.DataFrame({
                "Outcome": ["Approved Good (TN)", "Blocked Default (TP)", "Rejected Good (FP)", "Missed Default (FN)"],
                "Count": [tn, tp, fp, fn]
            })
            fig_out = px.pie(
                outcome_df, names="Outcome", values="Count", hole=0.52,
                color="Outcome",
                color_discrete_map={
                    "Approved Good (TN)": "#10B981",
                    "Blocked Default (TP)": "#3B82F6",
                    "Rejected Good (FP)": "#F59E0B",
                    "Missed Default (FN)": "#EF4444"
                }
            )
            fig_out.update_traces(textinfo="percent+label")
            st.plotly_chart(apply_dark_plotly(fig_out, 320), use_container_width=True)


# =============================================================================
# TAB 2: DEPARTMENT & COLUMN INTELLIGENCE LAB (DEPT. HEADS VIEW)
# =============================================================================
with tab2:
    st.markdown('<div class="section-heading">1. Live Interactive Column-by-Column Analyzer (15 Raw & Auto-Derived Columns)</div>', unsafe_allow_html=True)

    all_analyzable_cols = {
        "credit_score": "CIBIL / Credit Score (Raw)",
        "debt_to_income_ratio": "Debt-to-Income Ratio [Auto-Derived]",
        "emi_to_income_pct": "Monthly EMI Burden / FOIR % [Auto-Derived]",
        "credit_utilization_pct": "Credit Utilization % (Raw)",
        "credit_headroom_pct": "Available Credit Headroom % [Auto-Derived]",
        "delinquency_utilization_idx": "Delinquency-Utilization Index [Auto-Derived]",
        "late_payments_12m": "Late Payments in 12M (Raw)",
        "inquiries_6m": "Hard Credit Inquiries in 6M (Raw)",
        "inquiry_velocity_idx": "Credit Inquiry Velocity Index [Auto-Derived]",
        "annual_income_lakhs": "Annual Income ₹ Lakhs (Raw)",
        "net_disposable_lakhs": "Net Annual Disposable Income ₹ Lakhs [Auto-Derived]",
        "loan_amount_lakhs": "Requested Loan ₹ Lakhs (Raw)",
        "estimated_emi_k": "Estimated Monthly EMI ₹k [Auto-Derived]",
        "age": "Applicant Age in Years (Raw)",
        "unit_net_profit_inr": "Account Net Profit Contribution ₹ [Live Policy Output]"
    }

    sel_col = st.selectbox(
        "Select Any Column to Inspect Live Distribution, Policy Approval Rate & Default Escalation:",
        options=list(all_analyzable_cols.keys()),
        format_func=lambda k: all_analyzable_cols[k]
    )

    c_left, c_right = st.columns(2, gap="medium")
    with c_left:
        with st.container(border=True):
            st.markdown(f"##### 📊 Live Distribution by Policy Decision: `{sel_col}`")
            fig_dist = px.histogram(
                filtered_df, x=sel_col, color="policy_decision", barmode="overlay", nbins=35,
                color_discrete_map={"Approved": "#10B981", "Rejected / Flagged": "#EF4444"},
                opacity=0.72
            )
            st.plotly_chart(apply_dark_plotly(fig_dist, 310), use_container_width=True)

    with c_right:
        with st.container(border=True):
            st.markdown(f"##### 📈 Actual Default % vs. Model Risk % Across Deciles of `{sel_col}`")
            temp_df = filtered_df[[sel_col, "actual_default", "default_prob"]].copy()
            temp_df["Decile"] = pd.qcut(temp_df[sel_col], q=8, duplicates="drop").astype(str)
            decile_grp = temp_df.groupby("Decile", observed=False, as_index=False).agg(
                Actual_Default_Pct=("actual_default", lambda s: round(s.mean() * 100, 2)),
                Predicted_Risk_Pct=("default_prob", lambda s: round(s.mean() * 100, 2))
            )
            fig_dec = go.Figure()
            fig_dec.add_trace(go.Bar(
                x=decile_grp["Decile"], y=decile_grp["Actual_Default_Pct"],
                name="Actual Default %", marker_color="#EF4444"
            ))
            fig_dec.add_trace(go.Scatter(
                x=decile_grp["Decile"], y=decile_grp["Predicted_Risk_Pct"],
                name="Model Predicted Risk %", mode="lines+markers", line=dict(color="#38BDF8", width=3)
            ))
            fig_dec.update_yaxes(title="Rate (%)")
            st.plotly_chart(apply_dark_plotly(fig_dec, 310), use_container_width=True)

    st.markdown('<div class="section-heading">2. Cross-Department Segment, Channel & State Intelligence (All Live)</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        with st.container(border=True):
            st.markdown("##### 📣 Acquisition Channel Approval vs. Default %")
            ch_df = filtered_df.groupby("acquisition_channel", as_index=False).agg(
                Approval_Pct=("policy_decision", lambda s: round((s == "Approved").mean() * 100, 1)),
                Default_Pct=("actual_default", lambda s: round(s.mean() * 100, 1))
            )
            fig_ch = go.Figure()
            fig_ch.add_trace(go.Bar(x=ch_df["acquisition_channel"], y=ch_df["Approval_Pct"], name="Approval %", marker_color="#10B981"))
            fig_ch.add_trace(go.Bar(x=ch_df["acquisition_channel"], y=ch_df["Default_Pct"], name="Default %", marker_color="#EF4444"))
            fig_ch.update_layout(barmode="group")
            st.plotly_chart(apply_dark_plotly(fig_ch, 290), use_container_width=True)

    with g2:
        with st.container(border=True):
            st.markdown("##### 👥 Customer Segment Net Profit (₹ Lakhs)")
            seg_df = filtered_df.groupby("customer_segment", as_index=False).agg(
                Net_Profit_Lakhs=("unit_net_profit_inr", lambda s: round(s.sum() / 1e5, 2)),
                Avg_CIBIL=("credit_score", "mean")
            )
            fig_seg = px.bar(
                seg_df, x="customer_segment", y="Net_Profit_Lakhs", text="Net_Profit_Lakhs",
                color="customer_segment", color_discrete_sequence=["#10B981", "#38BDF8", "#F59E0B", "#A78BFA"]
            )
            fig_seg.update_traces(texttemplate="₹%{text:.1f}L", textposition="outside")
            fig_seg.update_layout(showlegend=False)
            st.plotly_chart(apply_dark_plotly(fig_seg, 290), use_container_width=True)

    with g3:
        with st.container(border=True):
            st.markdown("##### 🗺️ State × Product Default Risk Heatmap (%)")
            piv = filtered_df.pivot_table(
                index="state", columns="product_line", values="actual_default", aggfunc="mean"
            ) * 100
            fig_hm = px.imshow(
                piv.round(1), text_auto=True, color_continuous_scale=["#064E3B", "#D97706", "#991B1B"],
                labels=dict(x="Product", y="State", color="Default %")
            )
            st.plotly_chart(apply_dark_plotly(fig_hm, 290), use_container_width=True)

    g4, g5, g6 = st.columns(3, gap="medium")
    with g4:
        with st.container(border=True):
            st.markdown("##### ⏰ Late Payments (12M) Default Escalation")
            lp_df = filtered_df.groupby("late_payments_12m", as_index=False).agg(
                Default_Pct=("actual_default", lambda s: round(s.mean() * 100, 1)),
                Count=("applicant_id", "count")
            )
            lp_df = lp_df[lp_df["Count"] >= 10]
            fig_lp = px.line(lp_df, x="late_payments_12m", y="Default_Pct", markers=True)
            fig_lp.update_traces(line=dict(color="#F87171", width=3), marker=dict(size=8))
            st.plotly_chart(apply_dark_plotly(fig_lp, 280), use_container_width=True)

    with g5:
        with st.container(border=True):
            st.markdown("##### 🧮 Auto-FOIR (EMI Burden %) vs. DTI Risk")
            sample_sc = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
            fig_sc = px.scatter(
                sample_sc, x="debt_to_income_ratio", y="emi_to_income_pct",
                color="policy_decision",
                color_discrete_map={"Approved": "#10B981", "Rejected / Flagged": "#EF4444"},
                opacity=0.72
            )
            st.plotly_chart(apply_dark_plotly(fig_sc, 280), use_container_width=True)

    with g6:
        with st.container(border=True):
            st.markdown("##### 💵 Auto-Disposable Income vs. Outcome")
            disp_df = filtered_df.copy()
            disp_df["Outcome"] = np.where(disp_df["actual_default"] == 1, "Defaulted", "Repaid")
            fig_box = px.box(
                disp_df, x="Outcome", y="net_disposable_lakhs", color="Outcome",
                color_discrete_map={"Repaid": "#10B981", "Defaulted": "#EF4444"}
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(apply_dark_plotly(fig_box, 280), use_container_width=True)


# =============================================================================
# TAB 3: LIVE SINGLE APPLICANT UNDERWRITING & SHAP
# =============================================================================
with tab3:
    st.markdown('<div class="section-heading">1. Enter Applicant Profile (All 8 Derived Ratios Compute Live)</div>', unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            geo_choice = st.selectbox("City & Country", [f"{g[2]} ({g[1]}, {g[0]})" for g in GEO_CATALOG])
            sel_geo = [g for g in GEO_CATALOG if f"{g[2]} ({g[1]}, {g[0]})" == geo_choice][0]
            product_line = st.selectbox("Product Line", all_products)
            acquisition_channel = st.selectbox("Acquisition Channel", ["Mobile App Organic", "Partner Merchant", "Pre-Approved Offer", "DSA / Branch"])

        with c2:
            age = st.number_input("Applicant Age (Years)", min_value=21, max_value=70, value=32)
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
            credit_score = st.slider("CIBIL / Credit Score", min_value=450, max_value=880, value=725)

        with c3:
            annual_income_lakhs = st.number_input("Annual Income (₹ Lakhs)", min_value=1.0, max_value=150.0, value=14.0, step=0.5)
            loan_amount_lakhs = st.number_input("Requested Loan (₹ Lakhs)", min_value=0.5, max_value=60.0, value=3.5, step=0.5)
            credit_utilization_pct = st.slider("Credit Utilization (%)", min_value=1.0, max_value=99.0, value=30.0, step=1.0)

        with c4:
            late_payments_12m = st.number_input("Late Payments (Last 12 Months)", min_value=0, max_value=15, value=0)
            inquiries_6m = st.number_input("Hard Credit Inquiries (6M)", min_value=0, max_value=15, value=1)

    raw_input = pd.DataFrame([{
        "country": sel_geo[0],
        "state": sel_geo[1],
        "city_name": sel_geo[2],
        "city_tier": sel_geo[3],
        "product_line": product_line,
        "acquisition_channel": acquisition_channel,
        "employment_type": employment_type,
        "age": age,
        "annual_income_lakhs": annual_income_lakhs,
        "loan_amount_lakhs": loan_amount_lakhs,
        "credit_score": credit_score,
        "credit_utilization_pct": credit_utilization_pct,
        "late_payments_12m": late_payments_12m,
        "inquiries_6m": inquiries_6m
    }])
    processed_input = engineer_features(raw_input)

    default_prob = float(pipeline.predict_proba(processed_input[MODEL_BASE_COLS])[0, 1])
    is_rejected = default_prob >= active_threshold
    expected_ev = ((1.0 - default_prob) * 12000.0) - (default_prob * 65000.0)

    st.markdown('<div class="section-heading">2. Automated Decision & Live Derived Financial Ratios</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Default Risk", f"{default_prob * 100:.1f}%", delta=f"{(default_prob - active_threshold)*100:+.1f}% vs Cutoff", delta_color="inverse")
    k2.metric("Expected Loan EV", f"₹{expected_ev:,.0f}")
    k3.metric("Auto-DTI Ratio", f"{processed_input['debt_to_income_ratio'].iloc[0]:.2f}x")
    k4.metric("Est. Monthly EMI", f"₹{processed_input['estimated_emi_k'].iloc[0]:.1f}k")
    k5.metric("Auto-FOIR Burden", f"{processed_input['emi_to_income_pct'].iloc[0]:.1f}%")
    k6.metric("Disposable Income", f"₹{processed_input['net_disposable_lakhs'].iloc[0]:.2f}L")

    if is_rejected:
        st.markdown(
            f"""
            <div class="decision-reject">
                🚨 <strong>DECISION: REJECT / ROUTE TO MANUAL UNDERWRITING</strong> —
                Predicted default probability (<strong>{default_prob:.1%}</strong>) exceeds SafePay's risk threshold (<strong>{active_threshold:.1%}</strong>).
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="decision-approve">
                ✅ <strong>DECISION: AUTO-APPROVE CREDIT LINE</strong> —
                Predicted default probability (<strong>{default_prob:.1%}</strong>) is within SafePay's risk threshold (<strong>{active_threshold:.1%}</strong>).
                Expected Net Profit: <strong>₹{expected_ev:,.0f}</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-heading">3. Risk Gauge & Regulatory SHAP Explainability</div>', unsafe_allow_html=True)
    g_col, s_col = st.columns([1.0, 1.5], gap="medium")

    with g_col:
        with st.container(border=True):
            st.plotly_chart(make_risk_gauge(default_prob, active_threshold), use_container_width=True)
            st.caption("Yellow marker indicates SafePay's active policy cutoff threshold.")

    with s_col:
        with st.container(border=True):
            prep_step = pipeline.named_steps["prep"]
            clf_step = pipeline.named_steps["clf"]

            num_cols = processed_input[MODEL_BASE_COLS].select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = ["employment_type", "city_tier", "product_line", "acquisition_channel", "country"]
            encoded_cats = prep_step.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
            feature_names = num_cols + encoded_cats

            transformed_row = pd.DataFrame(prep_step.transform(processed_input[MODEL_BASE_COLS]), columns=feature_names)
            explainer = shap.TreeExplainer(clf_step)
            shap_exp = explainer(transformed_row)

            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            shap.plots.waterfall(shap_exp[0], show=False)
            style_shap_dark(fig)
            plt.tight_layout()
            st.pyplot(fig)


# =============================================================================
# TAB 4: 12,000-ACCOUNT LIVE PORTFOLIO EXPLORER & BATCH CSV SCORING
# =============================================================================
with tab4:
    st.markdown('<div class="section-heading">Live 12,000-Account Enterprise Portfolio & Batch CSV Scoring</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Select Data Source:",
        ["📊 Explore Live Filtered SafePay Portfolio (With All 8 Auto-Derived Columns)", "📤 Upload Custom Applicant CSV"],
        horizontal=True
    )

    with st.container(border=True):
        if mode.startswith("📤"):
            uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])
            if uploaded_file is not None:
                raw_batch = pd.read_csv(uploaded_file)
                scored_batch = engineer_features(raw_batch)
                probs = pipeline.predict_proba(scored_batch[MODEL_BASE_COLS])[:, 1]
                scored_batch["default_risk_pct"] = (probs * 100).round(2)
                scored_batch["policy_decision"] = np.where(probs >= active_threshold, "Rejected / Flagged", "Approved")
            else:
                st.info("Upload a CSV with columns: country, state, city_name, city_tier, product_line, acquisition_channel, employment_type, age, annual_income_lakhs, loan_amount_lakhs, credit_score, credit_utilization_pct, late_payments_12m, inquiries_6m.")
                scored_batch = filtered_df.head(50)
        else:
            scored_batch = filtered_df

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Active Records", f"{len(scored_batch):,}")
        b2.metric("Auto-Approved", f"{int((scored_batch['policy_decision'] == 'Approved').sum()):,}")
        b3.metric("Flagged / Rejected", f"{int((scored_batch['policy_decision'] != 'Approved').sum()):,}")
        b4.metric("Avg Auto-FOIR %", f"{float(scored_batch['emi_to_income_pct'].mean()):.1f}%")

        st.dataframe(scored_batch.head(500), use_container_width=True, height=380)

        csv_bytes = scored_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Scored Portfolio CSV (With All Automated Columns)",
            data=csv_bytes,
            file_name="safepay_12k_enterprise_portfolio.csv",
            mime="text/csv"
        )