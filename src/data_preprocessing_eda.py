"""
Phase 1: Data Preprocessing & Exploratory Data Analysis (EDA).

This script:
1) Loads the credit card transactions dataset.
2) Performs EDA focused on fraud imbalance, temporal behavior, and anomalies.
3) Builds a leak-safe train/validation/test split.
4) Fits preprocessing components ONLY on training data.
5) Saves transformed datasets and fitted preprocessors for downstream modeling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


sns.set_theme(style="whitegrid")


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load source data and perform basic schema checks."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {"Time", "Amount", "Class"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretive time features from seconds since first transaction.

    Standard Kaggle-style dataset stores `Time` in elapsed seconds. We map it to:
    - hour_of_day: cyclical interpretation for analyst behavior
    - day_index: sequence day in dataset
    """
    out = df.copy()
    out["hour_of_day"] = ((out["Time"] // 3600) % 24).astype(int)
    out["day_index"] = (out["Time"] // (24 * 3600)).astype(int)
    return out


def run_eda(df: pd.DataFrame, figures_dir: Path) -> Dict[str, float]:
    """Create enterprise-style EDA artifacts for fraud diagnostics."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_counts = df["Class"].value_counts().sort_index()
    fraud_rate = class_counts.get(1, 0) / max(len(df), 1)

    # 1) Class imbalance bar chart
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(data=df, x="Class", hue="Class", legend=False, palette="viridis")
    ax.set_title("Class Distribution (0 = Legit, 1 = Fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=180)
    plt.close()

    # 2) Fraud by hour-of-day
    hourly = (
        df.groupby(["hour_of_day", "Class"], as_index=False)
        .size()
        .pivot(index="hour_of_day", columns="Class", values="size")
        .fillna(0)
        .rename(columns={0: "legit_count", 1: "fraud_count"})
    )
    hourly["fraud_rate_pct"] = (
        hourly["fraud_count"] / (hourly["legit_count"] + hourly["fraud_count"]).replace(0, np.nan)
    ) * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x=hourly.index,
        y=hourly["fraud_count"],
        marker="o",
        color="#d62728",
        ax=ax1,
        label="Fraud Count",
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Fraud Count", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_title("Temporal Fraud Trend by Hour of Day")

    ax2 = ax1.twinx()
    sns.lineplot(
        x=hourly.index,
        y=hourly["fraud_rate_pct"],
        marker="s",
        color="#1f77b4",
        ax=ax2,
        label="Fraud Rate (%)",
    )
    ax2.set_ylabel("Fraud Rate (%)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "fraud_trend_by_hour.png", dpi=180)
    plt.close()

    # 3) Amount anomaly diagnostics by class
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="Class", y="Amount", palette="Set2", showfliers=False)
    plt.yscale("log")
    plt.title("Transaction Amount Distribution by Class (Log Scale)")
    plt.xlabel("Class")
    plt.ylabel("Amount (log scale)")
    plt.tight_layout()
    plt.savefig(figures_dir / "amount_distribution_by_class.png", dpi=180)
    plt.close()

    # 4) Correlation heatmap for selected fields
    corr_features = [c for c in ["Amount", "Time", "hour_of_day", "V1", "V2", "V3", "V4", "V10", "V14", "V17", "Class"] if c in df.columns]
    corr = df[corr_features].corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap (Selected Features)")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap_selected.png", dpi=180)
    plt.close()

    metrics = {
        "total_transactions": float(len(df)),
        "fraud_transactions": float(class_counts.get(1, 0)),
        "fraud_rate_pct": float(fraud_rate * 100),
        "max_amount": float(df["Amount"].max()),
        "median_amount": float(df["Amount"].median()),
    }
    return metrics


def split_data(df: pd.DataFrame, test_size: float, valid_size: float, random_state: int) -> SplitData:
    """
    Split data into train/valid/test while preserving class ratio.
    No oversampling is applied here; this guards against leakage.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    valid_relative_size = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_relative_size,
        random_state=random_state,
        stratify=y_train_valid,
    )

    return SplitData(X_train, X_valid, X_test, y_train, y_valid, y_test)


def fit_and_apply_preprocessing(split: SplitData) -> Tuple[SplitData, RobustScaler]:
    """
    Apply robust scaling to Amount (and optionally Time) fitted only on train set.
    """
    scaler = RobustScaler()
    scale_columns = [col for col in ["Amount", "Time"] if col in split.X_train.columns]

    X_train = split.X_train.copy()
    X_valid = split.X_valid.copy()
    X_test = split.X_test.copy()

    X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_valid[scale_columns] = scaler.transform(X_valid[scale_columns])
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])

    transformed = SplitData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=split.y_train,
        y_valid=split.y_valid,
        y_test=split.y_test,
    )
    return transformed, scaler


def save_outputs(split: SplitData, scaler: RobustScaler, output_dir: Path) -> None:
    """Persist processed data and fitted scaler for model training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split.X_train.assign(Class=split.y_train.values).to_csv(output_dir / "train_processed.csv", index=False)
    split.X_valid.assign(Class=split.y_valid.values).to_csv(output_dir / "valid_processed.csv", index=False)
    split.X_test.assign(Class=split.y_test.values).to_csv(output_dir / "test_processed.csv", index=False)
    joblib.dump(scaler, output_dir / "robust_scaler.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing and EDA for fraud detection.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/creditcard.csv"), help="Path to raw dataset CSV.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Output directory for processed datasets.")
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"), help="Output directory for EDA figures.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--valid-size", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input)
    df = add_temporal_features(df)
    metrics = run_eda(df, args.figures_dir)
    split = split_data(df, test_size=args.test_size, valid_size=args.valid_size, random_state=args.random_state)
    transformed_split, scaler = fit_and_apply_preprocessing(split)
    save_outputs(transformed_split, scaler, args.processed_dir)

    metrics_path = args.processed_dir / "eda_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)
    print(f"EDA and preprocessing completed. Outputs stored in: {args.processed_dir}")
    print(f"Figures saved in: {args.figures_dir}")
    print(f"EDA metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
