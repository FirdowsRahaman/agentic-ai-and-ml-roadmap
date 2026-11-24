# Starter Kit 2: Full-Lifecycle Local MLOps Pipeline ($0 Cloud Cost)

[← Back to Portfolio Projects](../README.md)

A production-structured **Local Project 4** starter scaffold demonstrating:
1. **Leak-Free Training Pipeline + Safe Zero-Pickle Serialization** (`StandardScaler` fit strictly on `X_train` + L2-regularized Logistic Classifier saved via `allow_pickle=False` `.npz`, evaluated via `PR-AUC`)
2. **Startup-Loaded `FastAPI` Microservice** (`lifespan` context manager + strict `Pydantic` validation)
3. **Built-In `Population Stability Index (PSI)` Drift Endpoint** (`/drift`)
4. **Non-Root Security `Dockerfile` + `pytest` Test Suite**

## Quickstart

```bash
cd projects/intermediate/starter-local-mlops
pip install -r requirements.txt
python -m src.train
pytest -q
uvicorn src.app:app --reload
```

## Container Build & Run

```bash
docker build -t local-churn-mlops .
docker run -p 8000:8000 local-churn-mlops
```
