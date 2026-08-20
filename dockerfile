FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY models/ ./models/
COPY data/transactions.db ./data/transactions.db

ENV PORT=8080
EXPOSE 8080

CMD streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.headless=true