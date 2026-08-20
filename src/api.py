"""FastAPI app.
POST /predict     -> fast XGBoost score+flag (called on every transaction)
POST /investigate -> slow LLM agent report (called only on flagged transactions)
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import score_transaction

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    type: str = Field(..., examples=["TRANSFER"])
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_flagged: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ScoreResponse)
def predict(transaction: Transaction):
    return score_transaction(transaction.model_dump())