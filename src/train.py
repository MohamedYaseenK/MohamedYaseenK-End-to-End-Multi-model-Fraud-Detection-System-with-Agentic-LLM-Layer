"""Trains XGBoost on prepped features. Saves model artifact to models/.
Prints PR-AUC and recall-at-fixed-precision (NOT accuracy — data is imbalanced).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_prep import build_model_matrix, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_PRECISION = 0.90


def train_test_split_stratified(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def recall_at_precision(y_true, y_scores, target_precision: float) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    eligible = recalls[precisions >= target_precision]
    return eligible.max() if len(eligible) > 0 else 0.0


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_scores = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_scores)
    recall_at_90 = recall_at_precision(y_test, y_scores, TARGET_PRECISION)

    return {
        "pr_auc": pr_auc,
        f"recall_at_{int(TARGET_PRECISION * 100)}pct_precision": recall_at_90,
    }


def evaluate_naive_baseline(df_test: pd.DataFrame) -> dict:
    precision = precision_score(df_test["isFraud"], df_test["isFlaggedFraud"])
    recall = recall_score(df_test["isFraud"], df_test["isFlaggedFraud"])
    return {"precision": precision, "recall": recall}


def save_model(model: xgb.XGBClassifier, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def run_training() -> None:
    prepped_df = run_pipeline()
    X, y = build_model_matrix(prepped_df)

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    test_idx = X_test.index
    baseline_metrics = evaluate_naive_baseline(prepped_df.loc[test_idx])

    save_model(model)

    print("XGBoost model:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nNaive isFlaggedFraud baseline:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    run_training()