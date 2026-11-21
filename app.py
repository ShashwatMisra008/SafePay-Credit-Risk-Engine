import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafePay Bank | Institutional Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CORPORATE DARK FINTECH CSS (BLOOMBERG / RAZORPAYX / STRIPE RADAR INSPIRED)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Global Dark Slate Canvas */
        .stApp {
            background-color: #0B1120;
            color: #F8FAFC;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1400px;
        }
        /* Top Header Bar Styling */
        header[data-testid="stHeader"] {
            background-color: rgba(11, 17, 32, 0.85);
            backdrop-filter: blur(8px);
        }
        /* Sidebar Dark FinTech Panel */
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0;
        }
        /* Executive Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #064E3B 100%);
            border: 1px solid #334155;
            padding: 1.65rem 2rem;
            border-radius: 14px;
            color: #F8FAFC;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45);
        }
        .hero-kicker {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            color: #34D399;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0 0 0.4rem 0;
            color: #FFFFFF;
            letter-spacing: -0.4px;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #CBD5E1;
            margin: 0;
            line-height: 1.5;
        }
        .badge-row {
            margin-top: 0.95rem;
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .tech-badge {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #6EE7B7;
        }
        /* Navigation Tabs */
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
            padding: 0 1.2rem;
            color: #94A3B8;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #10B981 !important;
            border: 1px solid #334155;
        }
        /* Dark Elevated Cards & Borders */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111827;
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            padding: 0.5rem;
        }
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
            border: 1px solid #1E293B;
            padding: 1rem 1.15rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.65rem !important;
            font-weight: 700 !important;
        }
        /* Decision Banners */
        .decision-approve {
            background: linear-gradient(90deg, rgba(6, 78, 59, 0.65) 0%, rgba(16, 185, 129, 0.15) 100%);
            border: 1px solid #059669;
            border-left: 6px solid #10B981;
            padding: 1.05rem 1.35rem;
            border-radius: 10px;
            color: #ECFDF5;
            font-size: 0.98rem;
            margin: 0.85rem 0 1.35rem 0;
        }
        .decision-reject {
            background: linear-gradient(90deg, rgba(127, 29, 29, 0.65) 0%, rgba(220, 38, 38, 0.15) 100%);
            border: 1px solid #DC2626;
            border-left: 6px solid #EF4444;
            padding: 1.05rem 1.35rem;
            border-radius: 10px;
            color: #FEF2F2;
            font-size: 0.98rem;
            margin: 0.85rem 0 1.35rem 0;
        }
        /* Sidebar Unit Economics Box */
        .econ-card {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 0.95rem;
            font-size: 0.84rem;
            line-height: 1.65;
            color: #CBD5E1;
            margin-top: 0.75rem;
        }
        .section-heading {
            font-size: 1.12rem;
            font-weight: 700;
            color: #E2E8F0;
            margin: 1.1rem 0 0.6rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

ARTIFACT_PATH = "safepay_credit_risk_engine_v1.joblib"


# -----------------------------------------------------------------------------
# FEATURE ENGINEERING & MODEL LOADER
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["debt_to_income_ratio"] = (
        data["loan_amount_lakhs"] / data["annual_income_lakhs"]
    ).round(3)
    data["delinquency_utilization_idx"] = (
        (data["late_payments_12m"] + 1) * (data["credit_utilization_pct"] / 100.0)
    ).round(3)
    return data


def _build_fallback_artifact() -> dict:
    rng = np.random.default_rng(42)
    n_samples = 5000

    age = rng.integers(21, 65, n_samples)
    annual_income_lakhs = np.round(rng.lognormal(mean=2.1, sigma=0.55, size=n_samples), 2)
    loan_amount_lakhs = np.round(annual_income_lakhs * rng.uniform(0.15, 0.95, n_samples), 2)
    credit_score = np.clip(rng.normal(710, 65, n_samples), 450, 880).astype(int)
    credit_utilization_pct = np.clip(rng.beta(2.2, 3.5, n_samples) * 100, 2, 99).round(1)
    late_payments_12m = rng.poisson(0.7, n_samples)
    inquiries_6m = rng.poisson(1.1, n_samples)
    employment_type = rng.choice(["Salaried", "Self-Employed", "Freelancer"], size=n_samples, p=[0.62, 0.26, 0.12])
    city_tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], size=n_samples, p=[0.50, 0.32, 0.18])

    dti = loan_amount_lakhs / (annual_income_lakhs + 0.1)
    emp_risk = np.where(employment_type == "Freelancer", 0.55, np.where(employment_type == "Self-Employed", 0.25, 0.0))

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
        "age": age,
        "annual_income_lakhs": annual_income_lakhs,
        "loan_amount_lakhs": loan_amount_lakhs,
        "credit_score": credit_score,
        "credit_utilization_pct": credit_utilization_pct,
        "late_payments_12m": late_payments_12m,
        "inquiries_6m": inquiries_6m,
        "employment_type": employment_type,
        "city_tier": city_tier,
        "default": default_flag
    })
    df = engineer_features(df)

    X = df.drop(columns=["default"])
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

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

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
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
        "feature_columns": X.columns.tolist(),
        "metrics": {
            "test_pr_auc": float(average_precision_score(y_test, probs)),
            "test_roc_auc": float(roc_auc_score(y_test, probs))
        }
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


@st.cache_resource
def load_model_bundle():
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH)
    return _build_fallback_artifact()


def make_risk_gauge(prob: float, threshold: float) -> go.Figure:
    """Dark-theme institutional risk gauge with clean tick spacing."""
    bar_color = "#EF4444" if prob >= threshold else "#10B981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 38, "color": "#F8FAFC"}},
            delta={
                "reference": round(threshold * 100, 1),
                "increasing": {"color": "#F87171"},
                "decreasing": {"color": "#34D399"},
                "suffix": "% vs Cutoff",
                "font": {"size": 14}
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
    """Restyles Matplotlib SHAP waterfall figure to match the Dark Slate theme."""
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


bundle = load_model_bundle()
pipeline = bundle["pipeline"]
default_optimal_cutoff = bundle["optimal_threshold"]
metrics = bundle.get("metrics", {})


# -----------------------------------------------------------------------------
# SIDEBAR: POLICY THRESHOLD & UNIT ECONOMICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏦 SafePay Risk Console")
    st.caption("Institutional Credit & Fraud Engine v1.0")
    st.divider()

    st.markdown("#### ⚙️ Decision Policy Threshold")
    if "cutoff_slider" not in st.session_state:
        st.session_state.cutoff_slider = round(float(default_optimal_cutoff), 2)

    active_threshold = st.slider(
        "Default Probability Cutoff",
        min_value=0.10,
        max_value=0.90,
        key="cutoff_slider",
        step=0.01,
        help="Applicants with default risk >= cutoff are flagged for rejection."
    )

    if st.button("↺ Reset to Profit-Optimal Cutoff", use_container_width=True):
        st.session_state.cutoff_slider = round(float(default_optimal_cutoff), 2)
        st.rerun()

    st.markdown(
        f"""
        <div class="econ-card">
            <div style="color:#34D399;font-weight:700;margin-bottom:6px;">
                🎯 Profit-Optimal Cutoff: <code>{default_optimal_cutoff:.2f}</code>
            </div>
            <div style="font-weight:700;color:#F8FAFC;margin-bottom:4px;">💰 Unit Economics (INR):</div>
            <div>• Repaid Loan (TN): <span style="color:#34D399;font-weight:700;">+₹12,000</span></div>
            <div>• Rejected Good (FP): <span style="color:#FBBF24;font-weight:700;">-₹4,000</span></div>
            <div>• Approved Default (FN): <span style="color:#F87171;font-weight:700;">-₹65,000</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("#### 📊 Validated Test Metrics")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("PR-AUC", f"{metrics.get('test_pr_auc', 0.0):.3f}")
    col_m2.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0.0):.3f}")


# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-kicker">SAFEPAY BANK • INSTITUTIONAL RISK INTELLIGENCE</div>
        <div class="hero-title">🏦 Credit Default & Automated Underwriting Engine</div>
        <p class="hero-subtitle">
            Real-time borrower risk scoring powered by a leakage-free Scikit-Learn &amp; Gradient Boosting pipeline,
            cost-matrix threshold optimization, and regulatory SHAP explainability.
        </p>
        <div class="badge-row">
            <span class="tech-badge">⚡ XGBoost / GBDT</span>
            <span class="tech-badge">💰 Cost-Matrix Calibrated</span>
            <span class="tech-badge">🔍 SHAP Explainable AI</span>
            <span class="tech-badge">🛡️ Leakage-Free Pipeline</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Applicant Underwriting & SHAP",
    "📂 Batch Portfolio Scoring (CSV)",
    "📐 Model Architecture & Economics"
])


# -----------------------------------------------------------------------------
# TAB 1: REAL-TIME SINGLE APPLICANT SCORING + GAUGE + SHAP WATERFALL
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-heading">1. Applicant Financial & Credit Profile</div>', unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            age = st.number_input("Applicant Age (Years)", min_value=21, max_value=70, value=32)
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
            city_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2", "Tier-3"])

        with c2:
            annual_income_lakhs = st.number_input("Annual Income (₹ Lakhs)", min_value=1.0, max_value=100.0, value=12.0, step=0.5)
            loan_amount_lakhs = st.number_input("Requested Loan (₹ Lakhs)", min_value=0.5, max_value=50.0, value=3.5, step=0.5)
            credit_score = st.slider("CIBIL / Credit Score", min_value=450, max_value=880, value=725)

        with c3:
            credit_utilization_pct = st.slider("Credit Utilization (%)", min_value=1.0, max_value=99.0, value=32.0, step=1.0)
            late_payments_12m = st.number_input("Late Payments (Last 12 Months)", min_value=0, max_value=15, value=0)
            inquiries_6m = st.number_input("Hard Credit Inquiries (Last 6 Months)", min_value=0, max_value=15, value=1)

    raw_input = pd.DataFrame([{
        "age": age,
        "annual_income_lakhs": annual_income_lakhs,
        "loan_amount_lakhs": loan_amount_lakhs,
        "credit_score": credit_score,
        "credit_utilization_pct": credit_utilization_pct,
        "late_payments_12m": late_payments_12m,
        "inquiries_6m": inquiries_6m,
        "employment_type": employment_type,
        "city_tier": city_tier
    }])
    processed_input = engineer_features(raw_input)

    default_prob = float(pipeline.predict_proba(processed_input)[0, 1])
    is_rejected = default_prob >= active_threshold
    expected_ev = ((1 - default_prob) * 12000) - (default_prob * 65000)

    st.markdown('<div class="section-heading">2. Automated Underwriting Decision & Risk Telemetry</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted Default Risk", f"{default_prob * 100:.2f}%", delta=f"{(default_prob - active_threshold)*100:+.2f}% vs Cutoff", delta_color="inverse")
    k2.metric("Policy Threshold", f"{active_threshold * 100:.0f}%")
    k3.metric("Debt-to-Income (DTI)", f"{processed_input['debt_to_income_ratio'].iloc[0]:.2f}x")
    k4.metric("Expected Loan Value", f"₹{expected_ev:,.0f}")

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

            num_cols = processed_input.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = ["employment_type", "city_tier"]
            encoded_cats = prep_step.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
            feature_names = num_cols + encoded_cats

            transformed_row = pd.DataFrame(prep_step.transform(processed_input), columns=feature_names)
            explainer = shap.TreeExplainer(clf_step)
            shap_exp = explainer(transformed_row)

            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            shap.plots.waterfall(shap_exp[0], show=False)
            style_shap_dark(fig)
            plt.tight_layout()
            st.pyplot(fig)


# -----------------------------------------------------------------------------
# TAB 2: BATCH PORTFOLIO SCORING (CSV UPLOAD)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-heading">Batch Credit Application Scoring</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file of loan applicants or test using the pre-loaded sample portfolio below.")

    sample_batch = pd.DataFrame([
        {"age": 34, "annual_income_lakhs": 18.5, "loan_amount_lakhs": 3.0, "credit_score": 795,
         "credit_utilization_pct": 18.0, "late_payments_12m": 0, "inquiries_6m": 0,
         "employment_type": "Salaried", "city_tier": "Tier-1"},
        {"age": 26, "annual_income_lakhs": 5.2, "loan_amount_lakhs": 4.5, "credit_score": 560,
         "credit_utilization_pct": 88.0, "late_payments_12m": 3, "inquiries_6m": 4,
         "employment_type": "Freelancer", "city_tier": "Tier-3"},
        {"age": 41, "annual_income_lakhs": 11.0, "loan_amount_lakhs": 4.2, "credit_score": 690,
         "credit_utilization_pct": 52.0, "late_payments_12m": 1, "inquiries_6m": 1,
         "employment_type": "Self-Employed", "city_tier": "Tier-2"},
        {"age": 29, "annual_income_lakhs": 9.5, "loan_amount_lakhs": 2.0, "credit_score": 745,
         "credit_utilization_pct": 24.0, "late_payments_12m": 0, "inquiries_6m": 1,
         "employment_type": "Salaried", "city_tier": "Tier-2"}
    ])

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])
        batch_df = pd.read_csv(uploaded_file) if uploaded_file is not None else sample_batch

        scored_batch = engineer_features(batch_df)
        batch_probs = pipeline.predict_proba(scored_batch)[:, 1]

        scored_batch["default_risk_pct"] = (batch_probs * 100).round(2)
        scored_batch["credit_decision"] = np.where(batch_probs >= active_threshold, "🚨 REJECT / REVIEW", "✅ AUTO-APPROVE")

        b1, b2, b3 = st.columns(3)
        b1.metric("Total Applicants", len(scored_batch))
        b2.metric("Auto-Approved", int((batch_probs < active_threshold).sum()))
        b3.metric("Flagged / Rejected", int((batch_probs >= active_threshold).sum()))

        st.dataframe(scored_batch, use_container_width=True)

        csv_bytes = scored_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Scored Decisions CSV",
            data=csv_bytes,
            file_name="safepay_scored_applicants.csv",
            mime="text/csv"
        )


# -----------------------------------------------------------------------------
# TAB 3: ARCHITECTURE & UNIT ECONOMICS
# -----------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-heading">End-to-End Machine Learning & Financial Calibration Design</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="medium")
    with a1:
        with st.container(border=True):
            st.markdown("#### 🛡️ Statistical & ML Pipeline Guardrails")
            st.markdown(
                """
                - **Multicollinearity Governance (VIF):** Removed `monthly_income_k` ($VIF > 1000$) to eliminate collinearity with `annual_income_lakhs`.
                - **Leakage-Free Preprocessing:** `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) is encapsulated inside `sklearn.pipeline.Pipeline` so scaling parameters are learned strictly from training folds.
                - **Class Imbalance Mitigation:** Benchmarked `class_weight="balanced"`, `SMOTE` (`imblearn.pipeline.Pipeline`), and cost-sensitive `scale_pos_weight` using **PR-AUC (`average_precision`)**.
                """
            )
    with a2:
        with st.container(border=True):
            st.markdown("#### 💰 Financial Cost-Matrix & Regulatory AI")
            st.markdown(
                """
                - **Profit-Optimal Cutoff:** Replaces the default `0.50` probability threshold with the cutoff maximizing Net Bank Profit:
                  $$\\text{Net Profit} = (\\text{TN} \\times ₹12,000) - (\\text{FP} \\times ₹4,000) - (\\text{FN} \\times ₹65,000)$$
                - **Explainable Underwriting:** Uses `shap.TreeExplainer` to generate auditable, feature-level risk attributions for every single borrower decision.
                """
            )