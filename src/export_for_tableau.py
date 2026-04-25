"""
Phase 3: Export scored dataset for Tableau analyst workflow.

Exports a clean CSV containing:
- original transaction features (unscaled for analyst readability),
- true labels,
- predicted labels,
- risk score probabilities,
- risk tiers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_with_features(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
    df["day_index"] = (df["Time"] // (24 * 3600)).astype(int)
    return df


def split_like_training(df: pd.DataFrame, test_size: float, valid_size: float, random_state: int):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    _X_train, _X_valid, _y_train, _y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_size / (1.0 - test_size),
        random_state=random_state,
        stratify=y_train_valid,
    )
    return X_test, y_test


def assign_risk_tier(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.5:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Critical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scored fraud transactions for Tableau.")
    parser.add_argument("--raw-input", type=Path, default=Path("data/raw/creditcard.csv"), help="Raw source dataset.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed artifacts directory.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Trained models directory.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/scored_transactions_for_tableau.csv"),
        help="Export path for Tableau CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--valid-size", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold used for predicted label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_with_features(args.raw_input)
    X_test_original, y_test = split_like_training(raw_df, args.test_size, args.valid_size, args.random_state)

    scaler = joblib.load(args.processed_dir / "robust_scaler.joblib")
    model = joblib.load(args.models_dir / "xgboost_fraud_detector.joblib")

    X_test_model = X_test_original.copy()
    scale_columns = [c for c in ["Amount", "Time"] if c in X_test_model.columns]
    X_test_model[scale_columns] = scaler.transform(X_test_model[scale_columns])

    risk_scores = model.predict_proba(X_test_model)[:, 1]
    predicted = (risk_scores >= args.threshold).astype(int)

    export_df = X_test_original.copy()
    export_df["true_label"] = y_test.values
    export_df["predicted_label"] = predicted
    export_df["risk_score"] = np.round(risk_scores, 6)
    export_df["risk_tier"] = pd.Series(risk_scores).apply(assign_risk_tier).values
    export_df["analyst_priority_rank"] = export_df["risk_score"].rank(method="first", ascending=False).astype(int)

    export_df.to_csv(args.output_csv, index=False)
    print(f"Tableau export created at: {args.output_csv}")
    print(f"Rows exported: {len(export_df):,}")


if __name__ == "__main__":
    main()
