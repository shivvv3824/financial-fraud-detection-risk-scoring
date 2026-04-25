# Financial Fraud Detection & Risk Scoring

Enterprise-grade machine learning and BI workflow for detecting card transaction fraud in a highly imbalanced environment (~284K records), then operationalizing model outputs into an analyst triage experience.

---

## 1) Business Objective

Fraud operations teams need to prioritize a small set of high-risk transactions from a very large transaction stream.  
This project delivers:

- A leakage-safe ML pipeline for fraud detection.
- Probability-based risk scoring and risk tiering (`Low`, `Medium`, `High`, `Critical`).
- A Tableau-ready analyst workflow dataset for investigation and triage.

---

## 2) Technical Architecture

### End-to-End Flow

1. **Ingest raw transactions** (`data/raw/creditcard.csv`).
2. **EDA + preprocessing** with strict split-first logic.
3. **Imbalance mitigation** using SMOTE on training data only.
4. **Modeling**:
   - Baseline: Logistic Regression
   - Optimized: XGBoost (RandomizedSearchCV)
5. **Evaluation** using PR curve, ROC curve, confusion matrix, and fraud-class metrics.
6. **Risk scoring** with continuous fraud probability + tier assignment.
7. **Export** scored test set for Tableau analyst review workflow.

---

## 3) Methodology Details

### Data Preprocessing & EDA (`src/data_preprocessing_eda.py`)

- Adds temporal derivatives:
  - `hour_of_day` from elapsed transaction seconds
  - `day_index` from elapsed days
- Generates EDA artifacts:
  - Class imbalance chart
  - Fraud trend by hour-of-day
  - Amount anomalies by class (log-scale boxplot)
  - Correlation heatmap of selected variables
- Applies `RobustScaler` to `Amount` and `Time` (fit on train only).
- Persists train/validation/test processed datasets and scaler artifact.

### Modeling & Resampling (`src/train_models.py`)

- Applies SMOTE strictly to training data only.
- Trains:
  - Logistic Regression baseline (`class_weight="balanced"`)
  - XGBoost with randomized hyperparameter search optimizing average precision
- Calibrates decision threshold from validation set (best F1 from PR tradeoff).
- Evaluates on test set and saves:
  - PR curve
  - ROC curve
  - Confusion matrix
  - Metrics JSON summary

### Risk Scoring & Export (`src/export_for_tableau.py`)

- Recreates leak-safe split logic to recover original (unscaled) test features.
- Uses trained XGBoost + scaler to generate:
  - `risk_score` (fraud probability)
  - `predicted_label` (thresholded class)
  - `risk_tier` buckets:
    - `< 0.25` -> `Low`
    - `0.25-0.49` -> `Medium`
    - `0.50-0.74` -> `High`
    - `>= 0.75` -> `Critical`
  - `analyst_priority_rank` descending by risk
- Exports clean CSV for Tableau ingestion.

---

## 4) Performance Targets

The project is configured with benchmark targets for production-style tuning:

- **Precision-Recall AUC target:** `0.97`
- **Fraud flagging accuracy target:** `94.2%`

> These are data-dependent and should be interpreted as optimization goals rather than guaranteed outputs on every run.  
> Use threshold calibration, richer feature engineering, and more extensive hyperparameter search to move toward or beyond target levels.

---

## 5) Latest Benchmark Snapshot (Real Dataset Run)

Latest benchmark generated from `models/metrics_summary.json` using the final frozen tuning run:

```bash
python src/train_models.py --search-iterations 40 --cv-folds 3
```

### Logistic Regression (Baseline)

- Accuracy: `0.9993`
- PR-AUC: `0.7344`
- ROC-AUC: `0.9628`
- Fraud precision: `0.8143`
- Fraud recall: `0.7703`
- Fraud F1: `0.7917`

### XGBoost (Optimized)

- Accuracy: `0.9996`
- PR-AUC: `0.8509`
- ROC-AUC: `0.9731`
- Fraud precision: `0.9375`
- Fraud recall: `0.8108`
- Fraud F1: `0.8696`
- Decision threshold: `0.8575`

### Interpretation

- XGBoost materially improves minority-class capture and precision over baseline logistic regression.
- The PR-AUC target (`0.97`) is intentionally ambitious; reaching it generally requires richer feature engineering, larger search budgets, calibrated cost-sensitive thresholding, and often additional context features (merchant/channel/device/geography).
- A deeper 40x3 search reached the same best score as the 15x3 run, indicating current feature space/label signal has likely plateaued under this modeling family.

### Final Portfolio Model Configuration

- Training command: `python src/train_models.py --search-iterations 40 --cv-folds 3`
- Class imbalance strategy: `SMOTE(sampling_strategy=0.25, k_neighbors=5)` on train split only
- Decision threshold (XGBoost): `0.8575`
- Production artifact: `models/xgboost_fraud_detector.joblib`

### Next Steps to Reach PR-AUC > 0.90

- Engineer behavioral velocity features (transaction count and amount over rolling windows by card/account) to improve fraud separability.
- Add segment context features (merchant type, channel, device fingerprint, geo-distance, and historical risk flags) where available.
- Evaluate stronger ensemble families (LightGBM, CatBoost, stacking with calibrated meta-learner) under the same leak-safe split protocol.
- Replace pure F1 thresholding with cost-sensitive optimization aligned to fraud operations economics (false-positive analyst cost vs fraud-loss avoidance).
- Introduce probability calibration (`CalibratedClassifierCV` or isotonic regression) before risk tiering to improve score reliability for triage decisions.
- Implement time-aware validation (forward chaining) and stability checks to reduce performance drift between historical and future fraud patterns.

### Governance and Production Hardening

- Add model monitoring KPIs: PR-AUC drift, fraud recall drift, score distribution shift, and alert volume volatility.
- Define SLA-based escalation rules by risk tier (for example: `Critical` reviewed within 15 minutes).
- Track analyst feedback outcomes (`confirmed_fraud`, `false_alert`) and close the loop with periodic retraining.
- Version model artifacts, thresholds, and feature schemas to ensure reproducible fraud governance audits.

---

## 6) Tableau Dashboard Architecture (Analyst Triage Workflow)

Design your Tableau workbook around **triage-first investigation**:

### A) Executive Risk Overview (Top Row)

- KPI cards:
  - Total transactions in scope
  - Total predicted fraud
  - Fraud capture rate (recall proxy)
  - Precision proxy (true fraud in flagged set)
- Global filters:
  - Date/day index
  - Hour of day
  - Risk tier
  - Predicted label

### B) Temporal Fraud Monitoring (Middle-Left)

- **Chart 1:** Line chart of fraud volume by `hour_of_day`
- **Chart 2:** Dual-axis chart of:
  - Fraud count
  - Fraud rate (% flagged or actual)
- **Purpose:** detect shift changes and attack windows

### C) High-Risk Segment Diagnostics (Middle-Right)

If fields like merchant category/type are present in your dataset:

- **Bar chart:** Fraud count by merchant category (sorted descending)
- **Heatmap:** Risk tier by category/type

If merchant fields are unavailable (as in anonymized PCA datasets):

- Build segment proxies using:
  - `Amount` bins (low/mid/high value transactions)
  - Hour clusters (`late night`, `business hours`, etc.)
  - PCA feature anomaly bands (e.g., extreme `V14`, `V17`, `V10`)

### D) Analyst Investigation Queue (Bottom Section)

- **Primary table:** Filterable transaction queue
  - Columns: transaction ID/index, `risk_score`, `risk_tier`, `predicted_label`, `Amount`, `hour_of_day`, selected feature vectors
  - Default sort: `analyst_priority_rank` ascending (highest risk first)
- **Default filter state:** `risk_tier = Critical`
- **Action links:**
  - Click row -> detail pane updates with full transaction profile
  - Analyst decision simulation field (`Approve`, `Escalate`, `Block`)

### E) Triage UX Recommendations

- Use color semantics:
  - `Low`: green
  - `Medium`: amber
  - `High`: orange
  - `Critical`: red
- Keep queue and filters always visible to mimic SOC/fraud-desk operations.
- Add quick-toggle parameter to switch threshold sensitivity and watch queue volume change in real time.

---

## 7) Repository Structure

See `project_structure.txt` for the canonical structure.

---

## 8) Quick Start

### Prerequisites

- Python 3.10+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Place dataset

Copy source CSV to:

`data/raw/creditcard.csv`

### Run pipeline

```bash
python src/data_preprocessing_eda.py \
  --input data/raw/creditcard.csv \
  --processed-dir data/processed \
  --figures-dir reports/figures
```

```bash
python src/train_models.py \
  --processed-dir data/processed \
  --models-dir models \
  --figures-dir reports/figures
```

```bash
python src/export_for_tableau.py \
  --raw-input data/raw/creditcard.csv \
  --processed-dir data/processed \
  --models-dir models \
  --output-csv data/processed/scored_transactions_for_tableau.csv
```

---

## 9) Key Outputs

- Processed datasets:
  - `data/processed/train_processed.csv`
  - `data/processed/valid_processed.csv`
  - `data/processed/test_processed.csv`
- Model artifacts:
  - `models/logistic_regression.joblib`
  - `models/xgboost_fraud_detector.joblib`
  - `models/metrics_summary.json`
- Evaluation visuals:
  - PR/ROC curves and confusion matrices in `reports/figures/`
- Tableau input:
  - `data/processed/scored_transactions_for_tableau.csv`

---

## 10) Portfolio Positioning

This project demonstrates practical strengths across:

- Imbalanced classification strategy (SMOTE + threshold optimization)
- Model benchmarking and explainable evaluation diagnostics
- Risk score operationalization for fraud operations teams
- BI delivery mindset through analyst-first dashboard architecture

It is intentionally structured to mirror production analytics handoffs between data science and fraud intelligence teams.
