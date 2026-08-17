"""FastAPI app.
POST /predict     -> fast XGBoost score+flag (called on every transaction)
POST /investigate -> slow LLM agent report (called only on flagged transactions)
"""
