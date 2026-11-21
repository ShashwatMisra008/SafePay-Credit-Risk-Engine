### 🏦 SafePay Bank — Global Credit Risk & Automated Underwriting Engine
[![Live App](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-10B981?style=for-the-badge&logo=streamlit&logoColor=white)](https://safepay-credit-risk-engine-shashwat-misra.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ShashwatMisra008/SafePay-Credit-Risk-Engine)

An end-to-end supervised Machine Learning pipeline and 4-tab Dark FinTech Streamlit Command Center built for CXOs, Risk Department Heads, and Underwriting Officers across **12,000+ global borrower accounts** (India, UAE, Singapore, UK).

* **Automated Feature Engineering:** Dynamically computes **8 interdependent financial & behavioral risk ratios** (`debt_to_income_ratio`, `emi_to_income_pct` / FOIR, `net_disposable_lakhs`, `delinquency_utilization_idx`, `credit_headroom_pct`, `inquiry_velocity_idx`, and automated `risk_tier` segmentation).
* **Leakage-Free ML & Cost-Matrix Calibration:** Eliminates multicollinearity via VIF governance, encapsulates `ColumnTransformer` + `SMOTE` / cost-sensitive XGBoost & LightGBM pipelines, and optimizes decision thresholds directly against bank unit economics (`+₹12,000` repaid profit vs. `-₹65,000` default write-off).
* **Regulatory Explainability & Live Simulation:** Features an interactive Policy Threshold Simulator, Plotly Risk Speedometer Gauge, borrower-level **SHAP Waterfall** attributions, and batch CSV scoring.

**Tech Stack:** `Python` • `Scikit-Learn` • `XGBoost` • `LightGBM` • `Imbalanced-Learn (SMOTE)` • `SHAP` • `Plotly` • `Streamlit`

## 📌 Executive Summary
**SafePay Bank** processes personal credit lines and BNPL (Buy Now, Pay Later) accounts across India. With a **~23.7% baseline default rate**, approving high-risk borrowers causes severe principal write-offs (`-₹65,000` per default), while overly conservative rules reject profitable, low-risk customers (`+₹12,000` net interest per good loan).

This repository contains a **leakage-free, explainable machine learning pipeline** and an interactive **Streamlit Underwriting Web App** (`app.py`) that benchmarks linear, bagging, and gradient-boosted ensembles, optimizes the classification decision threshold directly against a **financial cost-benefit matrix**, and exports a serialized `.joblib` inference artifact.

---

## 🏗️ End-to-End ML Architecture

```text
Raw Borrower Data (5,000 Applicants)
       │
       ▼
1. Statistical Audit & Feature Engineering
   ├── Variance Inflation Factor (VIF) Audit ──► Dropped redundant `monthly_income_k` (VIF > 1000)
   └── Domain Ratios ──► `debt_to_income_ratio` & `delinquency_utilization_idx`
       │
       ▼
2. Leak-Free Preprocessing (`ColumnTransformer`)
   ├── Numeric Features ──► `StandardScaler` (fit strictly on training folds)
   └── Categorical Features ──► `OneHotEncoder(drop="first")`
       │
       ▼
3. Multi-Model Benchmarking & Hyperparameter Tuning (`StratifiedKFold`, k=5)
   ├── Baseline: L2 Logistic Regression (`class_weight="balanced"`)
   ├── Bagging: Random Forest + `SMOTE` (`imblearn.pipeline.Pipeline`)
   └── Boosting: Cost-Sensitive `XGBoost` & `LightGBM` tuned via `RandomizedSearchCV` (PR-AUC)
       │
       ▼
4. Financial Profit-Maximizing Threshold Optimization
   └── Replaces default `0.50` probability cutoff with Net Profit optimal threshold
       │
       ▼
5. Regulatory Explainability (SHAP) & Deployment
   ├── Global `shap.summary_plot` + Borrower-level `shap.plots.waterfall`
   └── Serialized artifact (`safepay_credit_risk_engine_v1.joblib`) powering `app.py`
```

---

## 💰 Business Cost-Benefit Threshold Optimization
Decisions are calibrated to maximize **Net Bank Profit (₹ Lakhs)**:

| Prediction Outcome | Business Meaning | Financial Impact (INR) |
| :--- | :--- | :---: |
| **True Negative (TN)** | Approve good borrower who repays | **+₹12,000** |
| **False Positive (FP)** | Reject good borrower (lost interest) | **-₹4,000** |
| **False Negative (FN)** | Approve defaulter (principal write-off) | **-₹65,000** |
| **True Positive (TP)** | Flag & reject high-risk defaulter | **₹0** |

---

## 📂 Repository Structure
* `End_to_End_Supervised_ML_Capstone_SafePay_Bank.ipynb` — Complete Jupyter/Colab notebook with EDA, VIF audit, model benchmarking, profit optimization, and SHAP analysis.
* `app.py` — Interactive Streamlit Underwriting & Batch Scoring Web Application.
* `safepay_credit_risk_engine_v1.joblib` — Serialized production pipeline, optimal profit threshold, and validation telemetry.
* `sample_applicants.csv` — Sample borrower dataset for testing batch CSV scoring in the Streamlit UI.
* `requirements.txt` — Python dependencies for Streamlit Cloud deployment.

---

## 🚀 How to Run Locally or on Streamlit Cloud
```bash
pip install -r requirements.txt
streamlit run app.py
```
