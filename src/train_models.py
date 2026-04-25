"""
Phase 2: Machine Learning Pipeline, Resampling, and Risk Scoring.

This script:
1) Loads processed train/valid/test datasets.
2) Applies SMOTE ONLY to training data.
3) Trains a baseline Logistic Regression and tuned XGBoost model.
4) Evaluates models with PR-AUC, ROC-AUC, confusion matrix, and diagnostic plots.
5) Produces risk scores and risk tiers for analyst workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


sns.set_theme(style="whitegrid")


def load_processed_data(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(processed_dir / "train_processed.csv")
    valid = pd.read_csv(processed_dir / "valid_processed.csv")
    test = pd.read_csv(processed_dir / "test_processed.csv")
    return train, valid, test


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=random_state, sampling_strategy=0.25, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def train_logistic_baseline(X_train_res: pd.DataFrame, y_train_res: pd.Series, random_state: int) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1200,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=None,
    )
    model.fit(X_train_res, y_train_res)
    return model


def tune_xgboost(
    X_train_res: pd.DataFrame,
    y_train_res: pd.Series,
    random_state: int,
    n_iter: int = 8,
    cv_folds: int = 2,
) -> XGBClassifier:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_estimators=350,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
    )

    param_grid = {
        "n_estimators": [250, 350, 450, 550],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2, 0.4],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.8, 1.0, 1.2, 1.5],
    }

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv_folds,
        random_state=random_state,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train_res, y_train_res)
    return search.best_estimator_


def select_threshold_by_f1(y_true: pd.Series, y_scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precision * recall) / np.clip((precision + recall), 1e-10, None)
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_scores: np.ndarray,
    threshold: float,
    figures_dir: Path,
) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # PR curve
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#d62728", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_name.lower().replace(' ', '_')}_pr_curve.png", dpi=180)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_name.lower().replace(' ', '_')}_roc_curve.png", dpi=180)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix @ threshold={threshold:.3f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=180)
    plt.close()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision_fraud": float(report.get("1", {}).get("precision", 0.0)),
        "recall_fraud": float(report.get("1", {}).get("recall", 0.0)),
        "f1_fraud": float(report.get("1", {}).get("f1-score", 0.0)),
    }


def assign_risk_tier(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.5:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Critical"


def score_transactions(model: XGBClassifier, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    risk_scores = model.predict_proba(X)[:, 1]
    pred_labels = (risk_scores >= threshold).astype(int)
    scored = X.copy()
    scored["risk_score"] = risk_scores
    scored["predicted_label"] = pred_labels
    scored["risk_tier"] = scored["risk_score"].apply(assign_risk_tier)
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud models and generate risk scores.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Path to processed data directory.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory to store trained models and metrics.")
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"), help="Directory to store model evaluation figures.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--manual-threshold", type=float, default=None, help="Optional decision threshold override.")
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=8,
        help="RandomizedSearchCV iterations (lower is faster, higher is more thorough).",
    )
    parser.add_argument("--cv-folds", type=int, default=2, help="Cross-validation folds for XGBoost hyperparameter search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = load_processed_data(args.processed_dir)

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(valid_df)
    X_test, y_test = split_xy(test_df)

    # SMOTE is intentionally applied only on training data to prevent leakage.
    X_train_res, y_train_res = apply_smote(X_train, y_train, args.random_state)

    lr_model = train_logistic_baseline(X_train_res, y_train_res, args.random_state)
    xgb_model = tune_xgboost(
        X_train_res,
        y_train_res,
        args.random_state,
        n_iter=args.search_iterations,
        cv_folds=args.cv_folds,
    )

    lr_valid_scores = lr_model.predict_proba(X_valid)[:, 1]
    xgb_valid_scores = xgb_model.predict_proba(X_valid)[:, 1]

    lr_threshold = args.manual_threshold if args.manual_threshold is not None else select_threshold_by_f1(y_valid, lr_valid_scores)
    xgb_threshold = args.manual_threshold if args.manual_threshold is not None else select_threshold_by_f1(y_valid, xgb_valid_scores)

    lr_metrics = evaluate_model("Logistic Regression", y_test, lr_model.predict_proba(X_test)[:, 1], lr_threshold, args.figures_dir)
    xgb_metrics = evaluate_model("XGBoost", y_test, xgb_model.predict_proba(X_test)[:, 1], xgb_threshold, args.figures_dir)

    # Score full test set for downstream analyst dashboarding
    scored_test = score_transactions(xgb_model, X_test, xgb_threshold)
    scored_test["true_label"] = y_test.values
    scored_test.to_csv(args.models_dir / "scored_test_preview.csv", index=False)

    joblib.dump(lr_model, args.models_dir / "logistic_regression.joblib")
    joblib.dump(xgb_model, args.models_dir / "xgboost_fraud_detector.joblib")

    metrics = {
        "target_reference": {
            "target_pr_auc": 0.97,
            "target_fraud_flagging_accuracy": 0.942,
            "note": (
                "Targets are benchmark goals and data-dependent. "
                "Use hyperparameter search, threshold calibration, and feature engineering to approach them."
            ),
        },
        "logistic_regression_test_metrics": lr_metrics,
        "xgboost_test_metrics": xgb_metrics,
    }

    with open(args.models_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model training complete.")
    print(f"Saved models to: {args.models_dir}")
    print(f"Saved metrics to: {args.models_dir / 'metrics_summary.json'}")
    print(f"Saved figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
