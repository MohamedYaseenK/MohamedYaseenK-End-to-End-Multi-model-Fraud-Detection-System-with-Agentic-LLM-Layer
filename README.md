# Fraud Detection System with an Agentic Investigation Layer

An end-to-end fraud detection system combining a real-time XGBoost classifier with an LLM agent that autonomously investigates flagged transactions and produces a structured case report.

**Live demo:** `streamlit run app.py` → score a transaction → flagged cases trigger an autonomous investigation.

---

## Problem

Most fraud systems answer one question: *is this transaction fraud?* That's necessary but incomplete — a flagged transaction still needs a human analyst to pull the account's history, check its profile, and write up a case before anyone can act on it. That manual step is slow and doesn't scale.

This project builds both halves:

1. **A fast classifier** (XGBoost) that scores every transaction in real time.
2. **An agent** that only activates on flagged transactions, autonomously pulls account history and profile data via SQL, and writes a structured investigation report — the work a human analyst would otherwise do by hand.

---

## Architecture

```
Transaction
     │
     ▼
┌─────────────┐     flagged      ┌──────────────────────┐
│  XGBoost    │ ───────────────▶ │   LLM Agent           │
│  Classifier │                  │   (LangChain)          │
│  (FastAPI)  │                  │                        │
└─────────────┘                  │  Tools:                │
     │                           │  • get_user_history     │
     │ not flagged               │  • get_user_profile     │
     ▼                           │  • write_case_report    │
   Done                          └──────────┬─────────────┘
                                             │
                                             ▼
                                    Structured Case Report
                                    (risk level, indicators,
                                     recommendation, summary)
```

**Why two stages, not one LLM doing everything:** the classifier must be fast and cheap since it runs on every transaction; the agent only needs to run on the small minority that get flagged, where speed matters less than investigation quality. Letting an LLM score every transaction directly would be slower, more expensive, and worse at numerical pattern detection than a purpose-built classifier.

---

## Dataset

**PaySim** — a public, synthetic mobile-money transaction simulator (6.36M rows), chosen over alternatives for three reasons:

- **Realistic class imbalance** (0.13% fraud overall) — forces genuine handling of imbalanced classification rather than a toy balanced dataset.
- **Raw, interpretable balance fields** (not anonymized PCA components like most public fraud datasets) — allows real feature engineering grounded in how fraud actually behaves, not black-box vectors.
- **A built-in naive rule (`isFlaggedFraud`)** — gives a legitimate, verifiable baseline to beat, rather than an assumed one.

*Limitation, stated honestly:* PaySim is a synthetic simulation, not real transaction logs (which are never publicly available for privacy reasons). It's the standard academic/industry proxy specifically because it preserves realistic fraud statistical patterns.

---

## EDA — key findings that shaped every downstream decision

| Finding | Evidence | Decision |
|---|---|---|
| Fraud occurs only in `TRANSFER` / `CASH_OUT` | 0% fraud rate in CASH_IN, DEBIT, PAYMENT | Filter training data to these two types only |
| Destination balance often stays at zero despite a real transfer | **49.6%** of fraud vs **0.06%** of non-fraud show this pattern | Engineered as `dest_balance_untouched` — the strongest single feature |
| Origin balance mismatch (`oldbalance - amount ≠ newbalance`) | Fraud mismatch rate (1.5%) *lower* than non-fraud (94%) | Tested, then dropped — no separation |
| Account transaction velocity (repeat senders) | Nearly identical distribution for fraud vs non-fraud (mean ≈ 1.0 both) | Tested, then dropped — accounts are largely one-time senders in this dataset |
| Naive `isFlaggedFraud` rule (amount > ₹200,000) | 100% precision, but only **0.18%** recall | Used as the baseline to beat, never as a model feature (would leak fraud-adjacent logic) |

---

## Model & Results

**XGBoost classifier**, trained on the filtered TRANSFER/CASH_OUT subset with `scale_pos_weight` to handle class imbalance. Evaluated on PR-AUC and recall at a fixed 90% precision — not accuracy, which is meaningless at this class imbalance (predicting "never fraud" gets 99.87% accuracy while catching zero fraud).

| Metric | Naive rule (`isFlaggedFraud`) | XGBoost model |
|---|---|---|
| Precision | 100% | 90% (fixed target) |
| Recall | 0.18% | **81.4%** |
| PR-AUC | — | 0.94 |

**In plain terms:** the dataset's built-in rule catches roughly 3 fraud cases out of every 1,600. The trained model catches about 1,300 — an **~450x improvement in recall**, while keeping false positives to 1 in 10 flagged transactions.

**Feature importance validates the EDA, not just the metric.** Holding a transaction constant and only updating `newbalanceDest` from `0.00` to match the transferred amount drops the model's fraud probability from **99.98% to 27.56%** — direct, live confirmation that the model learned the exact signal (`dest_balance_untouched`) identified during EDA, not a spurious correlation.

---

## The Agentic Layer

When a transaction is flagged, an agent (LangChain + Claude/Gemini) is given the transaction and three tools:

| Tool | Purpose |
|---|---|
| `get_user_history` | Pulls the account's recent transactions from SQLite |
| `get_user_profile` | Returns summary stats — transaction count, average amount, prior fraud flags |
| `write_case_report` | Forces the agent's findings into a fixed schema (risk level, key indicators, recommended action, summary) instead of free text |

The agent decides which tools to call, reasons over the results, and always finishes by calling `write_case_report` — producing a consistent, parseable, demoable output every time rather than an unstructured paragraph.

**Example output (real, from a flagged transaction):**
> **Risk Level:** HIGH
> **Key Indicators:** XGBoost fraud probability 99.98% · TRANSFER type · zero prior transaction history
> **Recommended Action:** Block transaction, freeze account, require KYC verification
> **Summary:** New/dormant account executing an immediate large outgoing transfer — consistent with a mule-account fraud pattern.

---

## Tech Stack

`Python` · `XGBoost` · `SQLite` · `FastAPI` · `LangChain` · `Streamlit` · `Docker`

---

## Project Structure

```
fraud-agent/
├── app.py                 # Streamlit demo
├── Dockerfile
├── requirements.txt
├── data/
│   ├── raw/                # PaySim CSV (not committed — see Setup)
│   └── transactions.db      # SQLite, built by data_prep.py
├── models/
│   └── xgb_fraud_model.pkl  # trained model, built by train.py
├── notebooks/
│   └── eda.ipynb
└── src/
    ├── data_prep.py         # raw CSV -> filtered, feature-engineered SQLite table
    ├── train.py              # trains XGBoost, evaluates vs. naive baseline, saves model
    ├── model.py               # thin serving wrapper (load model, score a transaction)
    ├── api.py                 # FastAPI /predict endpoint
    ├── agent_tools.py         # 3 SQL-querying tools, testable without any LLM
    └── agent.py               # LangChain agent orchestration
```

---

## Setup & Run

**1. Download the dataset**
Get PaySim from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at `data/raw/paysim.csv`.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your LLM API key**
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```

**4. Train the model**
```bash
cd src
python train.py
```
This builds the SQLite database, trains XGBoost, prints evaluation metrics, and saves the model to `models/`.

**5. Run the demo**
```bash
streamlit run app.py
```

---

## Run with Docker

```bash
docker build -t fraud-agent .
docker run -p 8080:8080 --env-file .env fraud-agent
```
Open `http://localhost:8080`.

---

## Design Decisions & Scope

Built as an MVP — deliberately simple, defensible choices over unnecessary complexity:

| Decision | Reasoning |
|---|---|
| XGBoost only, no ensemble | One well-tuned model demonstrates the pipeline; ensembling is a tuning problem, not an architecture one |
| Batch scoring via API, not a streaming pipeline (Kafka) | Simulates real-time fine for an MVP; streaming is infrastructure, not a modeling concern |
| One agent, three tools, no multi-agent orchestration | Easier to explain and debug; equally demonstrates tool-use competency |
| Single Docker container | Simpler to ship and reason about; a production version would split scoring and UI into independently scaling services |

---

## Troubleshooting

- **Blank Streamlit page, no errors:** check that `app.py` actually has content (`cat app.py`) — an empty file produces exactly this symptom with no traceback.
- **`APIConnectionError` from the LLM call:** if on a VPN with TLS interception, disconnect it — Python's certificate store often doesn't trust VPN-injected certificates even when `curl`/browsers do.
- **`FileNotFoundError` for the model file:** run `python src/train.py` first — the model isn't included in the repo and must be trained locally.