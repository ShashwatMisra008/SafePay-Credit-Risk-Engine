# 🏦 SafePay Bank — Global Credit Risk & Automated Underwriting Engine

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://safepay-credit-risk-engine-shashwat-misra.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Leak--Free%20Pipeline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20GBDT-10B981)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainable%20AI-SHAP-8B5CF6)](https://shap.readthedocs.io/)

> ### 🚀 [**Launch Live Interactive Dashboard → SafePay Credit Risk Engine**](https://safepay-credit-risk-engine-shashwat-misra.streamlit.app/)
> *Click the badge or link above to test real-time borrower underwriting, simulate Net Portfolio Profit across 12,000 accounts, inspect SHAP waterfall explanations, and score custom CSV batches directly in your browser.*

---

## 📌 Executive Summary
**SafePay Bank** manages personal credit lines, BNPL (Buy Now, Pay Later), and MSME working capital accounts across **India, UAE, Singapore, and the UK**. Approving high-risk borrowers creates severe principal write-offs (`-₹65,000` per default), while overly rigid credit rules reject profitable borrowers (`+₹12,000` net interest per repaid loan).

This repository delivers an **end-to-end, leakage-free machine learning pipeline** and a **4-tab Dark FinTech Streamlit Command Center (`app.py`)** built for **CXOs, Risk Department Heads, and Underwriting Officers**.

---

## 🖥️ Interactive Dashboard Modules (`app.py`)

| Dashboard Tab | Target Audience | Live Capabilities |
| :--- | :--- | :--- |
| **🏛️ 1. Executive Command Center** | **CEO / Chief Risk Officer** | Real-time KPIs (`Net Portfolio Profit ₹ Cr`, `Loss Prevented`, `Approval Rate`, `Recall`, `Disbursed Volume`), Profit Optimization Curve, Product-Line Profit Bar Chart, City-Wise Risk Breakdown, and Underwriting Outcome Donut across **12,000 accounts**. |
| **🔬 2. Department & Column Analytics** | **Risk & Credit Dept. Heads** | Interactive column-by-column analyzer across **15 raw & auto-derived features**, Decile Default-Escalation Curves, Acquisition Channel comparison, Customer Segment profitability, and State × Product risk heatmaps. |
| **🎯 3. Live Underwriting & SHAP** | **Underwriting & Compliance** | Instant borrower scoring across 12 global cities & 5 product lines, live calculation of 8 financial ratios, Plotly Risk Speedometer Gauge, and regulatory **SHAP Waterfall** feature attributions. |
| **📂 4. 12k Portfolio & Batch Scoring** | **Operations & Analytics** | Filterable 12,000-account enterprise portfolio explorer + drag-and-drop CSV batch scoring with one-click scored CSV export. |

---

## ⚙️ Automated Feature Engineering (8 Derived Risk Indicators)
Raw applicant inputs are automatically transformed inside `engineer_features()` into 8 interdependent financial and behavioral risk signals:

1. **`debt_to_income_ratio`:** Requested Loan Amount ÷ Annual Income (`x` multiple).
2. **`estimated_emi_k`:** Product-aware monthly EMI (₹ Thousands) computed using product-specific APRs over a 24-month tenure.
3. **`emi_to_income_pct` (FOIR):** Fixed Obligation to Income Ratio (`Monthly EMI ÷ Monthly Income %`).
4. **`net_disposable_lakhs`:** Annual post-debt disposable income (`₹ Lakhs`).
5. **`delinquency_utilization_idx`:** Non-linear interaction of `(Late Payments + 1) × Utilization Rate`.
6. **`credit_headroom_pct`:** Available unutilized revolving credit buffer (`100% - Utilization %`).
7. **`inquiry_velocity_idx`:** Hard credit inquiries normalized per credit-active decade.
8. **`customer_segment` & `risk_tier`:** Automated segmentation into *Affluent Prime*, *Mass Salaried Retail*, *MSME & Self-Employed*, and *Emerging / High-Beta* cohorts.

---

## 🏗️ End-to-End ML Architecture & Unit Economics

```text
Enterprise Borrower Portfolio (12,000 Accounts • 12 Cities • 5 Products)
       │
       ▼
1. Statistical Governance & Feature Engineering
   ├── Multicollinearity Audit (VIF) ──► Eliminated redundant `monthly_income_k` (VIF > 1000)
   └── Automated Ratios ──► DTI, FOIR %, Disposable Income, Delinquency-Utilization Index
       │
       ▼
2. Leakage-Free Preprocessing (`ColumnTransformer`)
   ├── Numeric Features ──► `StandardScaler` (fit strictly inside training folds)
   └── Categorical Features ──► `OneHotEncoder(drop="first", handle_unknown="ignore")`
       │
       ▼
3. Imbalanced Classification & Cross-Validation (`StratifiedKFold`, k=5)
   ├── Benchmarked L2 Logistic Regression, Random Forest + SMOTE (`ImbPipeline`), XGBoost & LightGBM
   └── Optimized for Precision-Recall AUC (`average_precision`) & ROC-AUC
       │
       ▼
4. Financial Cost-Matrix Threshold Optimization
   └── Maximizes Net Bank Profit:  Net Profit = (TN × ₹12,000) - (FP × ₹4,000) - (FN × ₹65,000)
       │
       ▼
5. Regulatory Explainability (SHAP) & Streamlit Cloud Deployment
   └── `shap.TreeExplainer` Waterfall Attributions + Live Policy Simulator (`app.py`)
