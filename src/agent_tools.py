"""The 3 agent tools, testable in isolation without any LLM call:
- get_user_history(user_id)
- get_user_profile(user_id)
- write_case_report(findings)
"""
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "transactions.db"

HISTORY_LIMIT = 10


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_user_history(name_orig: str) -> list[dict]:
    """
    Return this user's most recent transactions, most recent first.
    Gives the agent behavioral context to judge whether the flagged
    transaction is consistent with the account's normal activity.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT step, type, amount, oldbalanceOrg, newbalanceOrig,
                   nameDest, isFraud
            FROM transactions
            WHERE nameOrig = ?
            ORDER BY step DESC
            LIMIT ?
            """,
            (name_orig, HISTORY_LIMIT),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_user_profile(name_orig: str) -> dict:
    """
    Return a summary profile for this user: how many transactions
    they've made, their average amount, and whether any past
    transaction from this account was labeled fraud.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_transactions,
                AVG(amount) AS avg_amount,
                MAX(amount) AS max_amount,
                SUM(isFraud) AS prior_fraud_count
            FROM transactions
            WHERE nameOrig = ?
            """,
            (name_orig,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def write_case_report(
    risk_level: str,
    key_indicators: list[str],
    recommended_action: str,
    summary: str,
) -> dict:
    """
    Forces the agent's findings into a fixed schema instead of free
    text. This is what makes the agent's output structured, logged,
    and renderable in the Streamlit demo rather than an unparseable
    paragraph.
    """
    valid_risk_levels = {"LOW", "MEDIUM", "HIGH"}
    if risk_level not in valid_risk_levels:
        raise ValueError(f"risk_level must be one of {valid_risk_levels}")

    return {
        "risk_level": risk_level,
        "key_indicators": key_indicators,
        "recommended_action": recommended_action,
        "summary": summary,
    }