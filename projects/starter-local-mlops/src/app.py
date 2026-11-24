"""Production FastAPI service with lifespan loading, Pydantic schemas, and PSI drift detection."""
from __future__ import annotations

from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.train import MODEL_PATH, LeakFreeChurnPipeline, train_and_save

STATE: dict = {}


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 8) -> float:
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_p = np.clip(np.histogram(expected, bins=edges)[0] / len(expected), 1e-6, None)
    act_p = np.clip(np.histogram(actual, bins=edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        train_and_save()
    pipeline, baseline_spend = LeakFreeChurnPipeline.load_npz(MODEL_PATH)
    STATE["pipeline"] = pipeline
    STATE["baseline_spend"] = baseline_spend
    yield
    STATE.clear()


app = FastAPI(title="Local Churn MLOps API", lifespan=lifespan)


class ChurnRequest(BaseModel):
    tenure_months: float = Field(ge=0.0, le=240.0)
    monthly_spend: float = Field(ge=0.0, le=10_000.0)
    support_calls: int = Field(ge=0, le=100)


class DriftRequest(BaseModel):
    recent_monthly_spends: list[float] = Field(min_length=20)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: ChurnRequest) -> dict[str, float | bool]:
    features = np.array([[req.tenure_months, req.monthly_spend, float(req.support_calls)]])
    prob = float(STATE["pipeline"].predict_proba(features)[0])
    return {"churn_probability": round(prob, 4), "churn_alert": prob >= 0.5}


@app.post("/drift")
def check_drift(req: DriftRequest) -> dict[str, float | bool]:
    psi_val = compute_psi(STATE["baseline_spend"], np.asarray(req.recent_monthly_spends, dtype=float))
    return {"psi": round(psi_val, 4), "retrain_required": psi_val >= 0.25}
