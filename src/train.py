"""Trains XGBoost on prepped features. Saves model artifact to models/.
Prints PR-AUC and recall-at-fixed-precision (NOT accuracy — data is imbalanced).
"""
