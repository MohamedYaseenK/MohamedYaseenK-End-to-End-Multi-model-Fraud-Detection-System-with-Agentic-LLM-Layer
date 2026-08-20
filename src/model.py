"""Thin serving wrapper: load trained model, predict_proba(transaction) -> score.
Does NOT know how to train. Imported by api.py only.
"""

import pickle
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"

FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "dest_balance_untouched",
]

FLAG_THRESHOLD = 0.5

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def build_feature_row(transaction: dict) -> pd.DataFrame:
    dest_balance_untouched = int(
        transaction["oldbalanceDest"] == 0
        and transaction["newbalanceDest"] == 0
        and transaction["amount"] > 0
    )

    row = {
        "type": 1 if transaction["type"] == "TRANSFER" else 0,
        "amount": transaction["amount"],
        "oldbalanceOrg": transaction["oldbalanceOrg"],
        "newbalanceOrig": transaction["newbalanceOrig"],
        "oldbalanceDest": transaction["oldbalanceDest"],
        "newbalanceDest": transaction["newbalanceDest"],
        "dest_balance_untouched": dest_balance_untouched,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def score_transaction(transaction: dict) -> dict:
    model = load_model()
    X = build_feature_row(transaction)
    fraud_probability = float(model.predict_proba(X)[0, 1])
    is_flagged = fraud_probability >= FLAG_THRESHOLD

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_flagged": is_flagged,
    }