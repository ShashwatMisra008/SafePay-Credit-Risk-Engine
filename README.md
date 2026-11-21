# 🏦 SafePay Bank — Global Credit Risk & Automated Underwriting Engine

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://safepay-credit-risk-engine-shashwat-misra.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ShashwatMisra008/SafePay-Credit-Risk-Engine)
[![Google Colab](https://img.shields.io/badge/Google_Colab-ML_Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://github.com/ShashwatMisra008/SafePay-Credit-Risk-Engine/blob/main/End_to_End_Supervised_ML_Capstone_SafePay_Bank.ipynb)

### 🛠️ Core Tech Stack & Libraries Used
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized_Math-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistical_Tests-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-VIF_Audit-3F51B5?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Leak--Free_Pipeline-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-10B981?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-Histogram_GBDT-0284C7?style=flat-square)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-EA580C?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-8B5CF6?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-SHAP_Visuals-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-4C72B0?style=flat-square)
![Joblib](https://img.shields.io/badge/Joblib-Model_Serialization-D97706?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

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
1. Statistical Governance & Feature Engineering (Statsmodels, SciPy, Pandas, NumPy)
   ├── Multicollinearity Audit (VIF) ──► Eliminated redundant `monthly_income_k` (VIF > 1000)
   └── Automated Ratios ──► DTI, FOIR %, Disposable Income, Delinquency-Utilization Index
       │
       ▼
2. Leakage-Free Preprocessing (Scikit-Learn `ColumnTransformer`)
   ├── Numeric Features ──► `StandardScaler` (fit strictly inside training folds)
   └── Categorical Features ──► `OneHotEncoder(drop="first", handle_unknown="ignore")`
       │
       ▼
3. Imbalanced Classification & Cross-Validation (`StratifiedKFold`, k=5)
   ├── Benchmarked L2 Logistic Regression, Random Forest + SMOTE (`ImbPipeline`), XGBoost & LightGBM
   └── Tuned via `RandomizedSearchCV` optimizing Precision-Recall AUC (`average_precision`) & ROC-AUC
       │
       ▼
4. Financial Cost-Matrix Threshold Optimization
   └── Maximizes Net Bank Profit:  Net Profit = (TN × ₹12,000) - (FP × ₹4,000) - (FN × ₹65,000)
       │
       ▼
5. Regulatory Explainability & Cloud Deployment (SHAP, Plotly, Joblib, Streamlit)
   └── `shap.TreeExplainer` Waterfall Attributions + Live Policy Simulator (`app.py`)
