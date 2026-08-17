#!/usr/bin/env bash
# template.sh — scaffolds the fraud-agent project structure
# Usage: bash template.sh [project-name]

set -e

PROJECT_NAME="${1:-fraud-agent}"

echo "Scaffolding $PROJECT_NAME ..."

mkdir -p "$PROJECT_NAME"/{data/raw,src,models,notebooks,tests}

# --- root files ---
touch "$PROJECT_NAME/README.md"
touch "$PROJECT_NAME/requirements.txt"
touch "$PROJECT_NAME/.env.example"
touch "$PROJECT_NAME/app.py"

cat > "$PROJECT_NAME/.gitignore" <<'EOF'
# data
data/raw/*
data/*.db
!data/raw/.gitkeep

# secrets
.env

# models
models/*.pkl
!models/.gitkeep

# python
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
.venv/

# os
.DS_Store
EOF

cat > "$PROJECT_NAME/.env.example" <<'EOF'
ANTHROPIC_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
EOF

cat > "$PROJECT_NAME/requirements.txt" <<'EOF'
pandas
numpy
scikit-learn
xgboost
fastapi
uvicorn
streamlit
langchain
langchain-anthropic
python-dotenv
pydantic
EOF

# --- src package ---
touch "$PROJECT_NAME/src/__init__.py"

cat > "$PROJECT_NAME/src/data_prep.py" <<'EOF'
"""CSV -> feature engineering -> SQLite load.
Single source of truth for feature engineering (used at train time AND serve time).
"""
EOF

cat > "$PROJECT_NAME/src/train.py" <<'EOF'
"""Trains XGBoost on prepped features. Saves model artifact to models/.
Prints PR-AUC and recall-at-fixed-precision (NOT accuracy — data is imbalanced).
"""
EOF

cat > "$PROJECT_NAME/src/model.py" <<'EOF'
"""Thin serving wrapper: load trained model, predict_proba(transaction) -> score.
Does NOT know how to train. Imported by api.py only.
"""
EOF

cat > "$PROJECT_NAME/src/agent_tools.py" <<'EOF'
"""The 3 agent tools, testable in isolation without any LLM call:
- get_user_history(user_id)
- get_user_profile(user_id)
- write_case_report(findings)
"""
EOF

cat > "$PROJECT_NAME/src/agent.py" <<'EOF'
"""LangChain agent orchestration only — no tool logic here, just prompts + loop."""
EOF

cat > "$PROJECT_NAME/src/api.py" <<'EOF'
"""FastAPI app.
POST /predict     -> fast XGBoost score+flag (called on every transaction)
POST /investigate -> slow LLM agent report (called only on flagged transactions)
"""
EOF

# --- notebooks ---
touch "$PROJECT_NAME/notebooks/eda.ipynb"

# --- tests ---
cat > "$PROJECT_NAME/tests/test_pipeline.py" <<'EOF'
"""Sanity checks: model loads, predict returns a valid probability, tools return valid schema."""
EOF

# --- keep-empty-dir placeholders ---
touch "$PROJECT_NAME/data/raw/.gitkeep"
touch "$PROJECT_NAME/models/.gitkeep"

echo "Done. Structure:"
find "$PROJECT_NAME" -not -path '*/.git*' | sort