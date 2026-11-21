# 🏦 SafePay Bank — End-to-End Credit Default & Risk Scoring Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline%20%7C%20ColumnTransformer-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-brightgreen)
![Explainability](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)
![Streamlit](https://img.shields.io/badge/App-Streamlit%20Cloud-red)

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
