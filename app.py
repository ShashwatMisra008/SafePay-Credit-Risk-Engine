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
# PAGE CONFIG & CUSTOM FINTECH CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafePay Bank | Credit Risk & Underwriting Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        /* Fix top padding so hero banner is never clipped */
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 2rem !important;
        }
        /* Executive FinTech Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #0F2537 0%, #1B4965 60%, #2C7DA0 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 37, 55, 0.15);
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.3px;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #E0F2FE;
            margin: 0;
        }
        .badge-row {
            margin-top: 0.75rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .tech-badge {
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #FFFFFF;
        }
        /* Metric Card Styling */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        /* Decision Banners */
        .decision-approve {
            background: linear-gradient(90deg, #ECFDF5 0%, #D1FAE5 100%);
            border-left: 5px solid #059669;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            color: #064E3B;
            font-size: 1rem;
            margin: 0.75rem 0 1.25rem 0;
        }
        .decision-reject {
            background: linear-gradient(90deg, #FEF2F2 0%, #FEE2E2 100%);
            border-left: 5px solid #DC2626;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            color: #7F1D1D;
            font-size: 1rem;
            margin: 0.75rem 0 1.25rem 0;
        }
        /* Sidebar Unit Economics Box */
        .econ-card {
            background-color: #F1F5F9;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            padding: 0.85rem;
            font-size: 0.85rem;
            line-height: 1.55;
            color: #1E293B;
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
    """Renders an executive Plotly speedometer gauge for borrower default risk."""
    bar_color = "#DC2626" if prob >= threshold else "#059669"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 34, "color": "#0F2537"}},
            delta={
                "reference": round(threshold * 100, 1),
                "increasing": {"color": "#DC2626"},
                "decreasing": {"color": "#059669"},
                "suffix": "% vs Cutoff"
            },
            title={"text": "Default Probability Score", "font": {"size": 15, "color": "#334155"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, threshold * 100], "color": "#ECFDF5"},
                    {"range": [threshold * 100, 100], "color": "#FEF2F2"}
                ],
                "threshold": {
                    "line": {"color": "#0F2537", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100
                }
            }
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


bundle = load_model_bundle()
pipeline = bundle["pipeline"]
default_optimal_cutoff = bundle["optimal_threshold"]
metrics = bundle.get("metrics", {})


# -----------------------------------------------------------------------------
# SIDEBAR: POLICY THRESHOLD & UNIT ECONOMICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 SafePay Risk Engine")
    st.caption("Automated Credit Underwriting & SHAP")
    st.divider()

    st.markdown("### ⚙️ Decision Policy Threshold")
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
            <strong>🎯 Profit-Optimal Cutoff:</strong> <code>{default_optimal_cutoff:.2f}</code><br><br>
            <strong>💰 Unit Economics (INR):</strong><br>
            • Good Loan Repaid (TN): <span style="color:#059669;font-weight:600;">+₹12,000</span><br>
            • Good Loan Rejected (FP): <span style="color:#D97706;font-weight:600;">-₹4,000</span><br>
            • Default Approved (FN): <span style="color:#DC2626;font-weight:600;">-₹65,000</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### 📊 Validated Test Metrics")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("PR-AUC", f"{metrics.get('test_pr_auc', 0.0):.3f}")
    col_m2.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0.0):.3f}")


# -----------------------------------------------------------------------------
# HERO BANNER (FIXES CLIPPED TITLE)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🏦 SafePay Bank — Credit Default & Underwriting Engine</div>
        <p class="hero-subtitle">
            Real-time borrower risk scoring powered by a leakage-free Scikit-Learn &amp; Gradient Boosting pipeline,
            cost-matrix threshold optimization, and regulatory SHAP explainability.
        </p>
        <div class="badge-row">
            <span class="tech-badge">XGBoost / GBDT</span>
            <span class="tech-badge">Cost-Matrix Calibrated</span>
            <span class="tech-badge">SHAP Explainable AI</span>
            <span class="tech-badge">Leakage-Free Pipeline</span>
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
    st.markdown("#### 1. Applicant Financial & Credit Profile")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
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

    st.markdown("#### 2. Automated Underwriting Decision & Risk Telemetry")

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

    # Side-by-Side Risk Gauge & SHAP Waterfall
    st.markdown("#### 3. Risk Gauge & Regulatory SHAP Explainability")
    g_col, s_col = st.columns([1.05, 1.45])

    with g_col:
        with st.container(border=True):
            st.plotly_chart(make_risk_gauge(default_prob, active_threshold), use_container_width=True)

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
            plt.tight_layout()
            st.pyplot(fig)


# -----------------------------------------------------------------------------
# TAB 2: BATCH PORTFOLIO SCORING (CSV UPLOAD)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("#### Batch Credit Application Scoring")
    st.markdown("Upload a CSV file of loan applicants or test using the pre-loaded sample batch below.")

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
    st.markdown("#### End-to-End Machine Learning & Financial Calibration Design")
    with st.container(border=True):
        st.markdown(
            """
            1. **Multicollinearity Governance (VIF):** Removed `monthly_income_k` ($VIF > 1000$) to eliminate collinearity with `annual_income_lakhs`.
            2. **Leakage-Free Preprocessing:** `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) is encapsulated inside `sklearn.pipeline.Pipeline` so scaling parameters are learned strictly from training folds.
            3. **Class Imbalance Handling:** Benchmarked `class_weight="balanced"`, `SMOTE` (`imblearn.pipeline.Pipeline`), and cost-sensitive `scale_pos_weight` in XGBoost/LightGBM using **PR-AUC (`average_precision`)**.
            4. **Financial Cost-Matrix Thresholding:** Replaces the default `0.50` probability cutoff with the threshold maximizing expected Net Bank Profit:
               $$\\text{Net Profit} = (\\text{TN} \\times ₹12,000) - (\\text{FP} \\times ₹4,000) - (\\text{FN} \\times ₹65,000)$$
            5. **Regulatory Explainability:** Uses `shap.TreeExplainer` to provide exact feature-level attributions for every credit decision.
            """
        )