"""CSV -> feature engineering -> SQLite load.
Single source of truth for feature engineering (used at train time AND serve time).
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
DB_PATH = PROJECT_ROOT / "data" / "transactions.db"

FRAUD_ELIGIBLE_TYPES = ["TRANSFER", "CASH_OUT"]

MODEL_FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "dest_balance_untouched",
]
TARGET_COLUMN = "isFraud"


def load_raw_data(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected PaySim CSV at {csv_path}. "
            "Download from https://www.kaggle.com/datasets/ealaxi/paysim1 "
            "and place it at that path."
        )
    return pd.read_csv(csv_path)


def filter_to_fraud_eligible_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"].isin(FRAUD_ELIGIBLE_TYPES)].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dest_balance_untouched"] = (
        (df["oldbalanceDest"] == 0)
        & (df["newbalanceDest"] == 0)
        & (df["amount"] > 0)
    ).astype(int)
    return df


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[MODEL_FEATURE_COLUMNS].copy()
    X["type"] = X["type"].map({"CASH_OUT": 0, "TRANSFER": 1})
    y = df[TARGET_COLUMN].copy()
    return X, y


def save_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_name_orig ON transactions(nameOrig)"
        )
        conn.commit()
    finally:
        conn.close()


def run_pipeline() -> pd.DataFrame:
    raw_df = load_raw_data()
    filtered_df = filter_to_fraud_eligible_types(raw_df)
    prepped_df = engineer_features(filtered_df)
    save_to_sqlite(prepped_df)
    return prepped_df


if __name__ == "__main__":
    prepped = run_pipeline()
    print(f"Rows after filtering to TRANSFER/CASH_OUT: {len(prepped):,}")
    print(f"Fraud rate after filtering: {prepped['isFraud'].mean() * 100:.3f}%")
    print(f"Saved to: {DB_PATH}")

    X, y = build_model_matrix(prepped)
    print(f"\nModel feature matrix shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")