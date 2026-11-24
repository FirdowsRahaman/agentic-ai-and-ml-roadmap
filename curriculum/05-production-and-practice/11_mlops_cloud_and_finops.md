# Module 11: Production MLOps, AI Engineering, Drift & Cloud FinOps

> **Estimated Study Time:** 8–10 Hours | **Prerequisites:** [Module 03](../02-core-ml-and-dl/03_classical_ml_and_recsys.md), [Module 10](../04-agentic-ai/10_agent_evals_and_safety.md) | **Starter Repo:** [Local MLOps Starter Template](../../projects/starter-local-mlops/)

[← Previous: Module 10 — Agent Evaluation & Safety](../04-agentic-ai/10_agent_evals_and_safety.md) | [Home](../../README.md) | [Next: Module 12 — Practical Troubleshooting Scenarios →](./12_system_troubleshooting_scenarios.md)

---

## Mental Model: Notebook Code Is Only 10% of Production ML

A model scoring `0.94 AUC` in a local Jupyter notebook delivers **zero business value** until it is safely serialized, containerized, versioned alongside its training data, served behind low-latency health-checked APIs, protected from Training-Serving Skew, monitored for statistical drift, and guarded against runaway GPU cloud bills.

```mermaid
flowchart LR
    DATA["Versioned Data (DVC + Feast)"] --> TRAIN["Tracked Training (MLflow + Optuna)"]
    TRAIN --> PKG["Safe Packaging (safetensors / ONNX + Docker)"]
    PKG --> DEP["Safe Rollout (Shadow -> Canary -> A/B)"]
    DEP --> MON["Drift & Infra FinOps (PSI + Spot + Budgets)"]
    MON -->|"PSI >= 0.25 Retrain Trigger"| TRAIN

    classDef stage fill:#E0F2FE,stroke:#0284C7,stroke-width:1.5px,color:#0369A1;
    classDef prod fill:#EDE9FE,stroke:#7C3AED,stroke-width:2px,color:#4C1D95;
    classDef guard fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;

    class DATA,TRAIN stage;
    class PKG,DEP prod;
    class MON guard;
```

---

## Tier 1: Fresher Core (Serialization, `FastAPI`, `Docker`, `MLflow` & `DVC`)

### 1. Safe Model Serialization & Serving via `FastAPI`

#### Q1: Why is Python `pickle` dangerous for ML model files, and when should you use `safetensors`, `ONNX`, or `joblib`?
**Answer:**
- **Security Trap (`pickle` / `torch.load` default):** `pickle` executes arbitrary Python bytecode during deserialization — loading an untrusted `.pkl` file can compromise your production server!
- **`safetensors` (Hugging Face / PyTorch standard):** Zero-copy, memory-mapped tensor format that stores **pure weights + JSON header** with **zero arbitrary code execution** and 2x–4x faster GPU loading.
- **`ONNX` (Open Neural Network Exchange):** Framework-agnostic computation graph enabling graph fusion and quantized CPU/GPU inference via **ONNX Runtime** (`2x–5x` speedup over eager PyTorch).
- **`joblib`:** Efficient NumPy buffer compression for trusted internal `scikit-learn` pipelines.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model ONCE at startup (never inside the request handler!)
    models["churn"] = joblib.load("artifacts/churn_pipeline.joblib")
    yield
    models.clear()

app = FastAPI(title="Churn Inference Service", lifespan=lifespan)

class ChurnFeatures(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_spend: float = Field(ge=0.0)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}

@app.post("/predict")
def predict_churn(payload: ChurnFeatures) -> dict[str, float | bool]:
    X = [[payload.tenure_months, payload.monthly_spend]]
    prob = float(models["churn"].predict_proba(X)[0, 1])
    return {"churn_probability": round(prob, 4), "high_risk": prob >= 0.5}
```

---

### 2. Containerization (`Docker`) & Experiment/Data Versioning (`MLflow` + `DVC`)

#### Q2: Docker Image vs. Container, and what makes a `Dockerfile` production-ready?
**Answer:**
- **Image:** Immutable layered filesystem blueprint. **Container:** Isolated running process instantiated from an image.
- **Production Dockerfile Best Practices:** Pin a slim base image (`python:3.12-slim`), copy `requirements.txt`/`pyproject.toml` *before* source code to maximize Docker layer caching, drop root privileges (`USER appuser`), and expose a `/health` probe.

```dockerfile
FROM python:3.12-slim
WORKDIR /app

# 1. Cache dependencies separately from fast-changing application code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Run as non-root user for container security
RUN useradd --create-home appuser
COPY --chown=appuser:appuser src/ ./src/
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

#### Q3: How do `Git`, `DVC`, and `MLflow` work together for 100% ML Reproducibility?
**Answer:**
- **`Git`:** Versions source code, configs, and small `.dvc` pointer files.
- **`DVC` (Data Version Control):** Stores 50 GB Parquet/image datasets in S3/GCS content-addressed storage while committing tiny SHA-256 pointer files into Git.
- **`MLflow`:** Logs hyperparameters, validation metrics (`PR-AUC`, `p99` latency), Git commit SHA, and versioned model artifacts in a central **Model Registry** (`Staging -> Champion`).

---

## Tier 2: MNC Production Depth (Serving Architectures, `Feast`, `PSI` Drift, Rollouts & Infra FinOps)

### 3. Classical ML Serving vs. LLM Serving, and Feature Stores (`Feast`)

#### Q4: How does deploying an LLM differ architecturally from deploying an `XGBoost` microservice?
**Answer:**

| Dimension | Classical ML (`XGBoost` / `LightGBM`) | LLM Serving (`vLLM` / `TensorRT-LLM`) |
|---|---|---|
| **Compute & State** | Stateless CPU pods (`<500 MB` RAM), `5ms` inference | Stateful `KV Cache` on high-VRAM GPUs (`24–80 GB`), streaming `SSE` tokens |
| **Batching** | Simple request or micro-batching | **Continuous iteration-level batching** + `PagedAttention` |
| **Autoscaling Signal** | CPU utilization / RPS | **KV Cache VRAM % utilization** + waiting queue depth (CPU % stays near 0 on GPU pods!) |

---

#### Q5: What is Training-Serving Skew, and how does a Feature Store (`Feast`) eliminate it?
**Answer:**
**Training-Serving Skew** occurs when offline training features (computed in batch SQL/Pandas) subtly diverge from live inference features (re-implemented in Go/Python microservices with different null handling or window boundaries).
- **`Feast` Dual-Store Architecture:** Defines feature transformations **once**:
  1. **Offline Store (`BigQuery` / `Snowflake` / `Parquet`):** Generates historical training sets using **point-in-time joins** (preventing future label leakage).
  2. **Online Store (`Redis` / `DynamoDB`):** Materializes the exact same latest feature values for `<5ms` real-time inference lookups.

---

### 4. Statistical Drift (`PSI`), Safe Deployments & Cloud GPU FinOps

#### Q6: Covariate Drift vs. Concept Drift, and how `Population Stability Index (PSI)` triggers retraining
**Answer:**
- **Data / Covariate Drift ($P(X)$ shifts):** Input distribution changes (e.g., inflation shifts transaction amounts upward). Quantified via **Kolmogorov-Smirnov (K-S) test** or **PSI**:
  $$\text{PSI} = \sum_{i=1}^{B} (p_{\text{prod}, i} - p_{\text{train}, i}) \cdot \ln\left(\frac{p_{\text{prod}, i}}{p_{\text{train}, i}}\right)$$
  - `PSI < 0.10`: Stable | `0.10 <= PSI < 0.25`: Investigate | **`PSI >= 0.25`: Significant drift -> trigger automated CI/CD retraining**.
- **Concept Drift ($P(Y \mid X)$ shifts):** Input histograms look identical (`PSI < 0.05`), yet fraudsters changed tactics so precision collapses — detected via delayed ground-truth label monitoring and business KPIs.

---

#### Q7: Shadow Deployment vs. Canary Release vs. Online A/B Testing
**Answer:**
1. **Shadow (Dark Traffic):** Mirror 100% of live requests to the Challenger model asynchronously; discard Challenger outputs from user responses while verifying latency, memory, and prediction distributions at **zero user risk**.
2. **Canary Rollout:** Route `1% -> 10% -> 50% -> 100%` of live traffic with automated rollback if `5xx` error rates or `p99` latency breach guardrails.
3. **Online A/B Test:** Randomize by `user_id` (`50/50`), measuring **Primary Business KPI** statistical lift ($p < 0.05$) alongside strict **Guardrail Metrics** (latency, unsubscribe rate).

---

#### Q8: How do ML Teams slash Cloud GPU Training Costs by 70% and prevent Orphaned GPU Billing Disasters?
**Answer:**
1. **Spot / Preemptible GPUs + Emergency Checkpointing:** Rent spare cloud A100/H100 capacity at a **60–80% discount**. Save checkpoints to S3/GCS every $N$ steps and trap the cloud `SIGTERM` / 2-minute preemption notice to flush optimizer + model state cleanly so replacement nodes resume automatically.
2. **Automated Cloud Infrastructure Kill-Switches:**
   - Cap Kubernetes autoscalers (`maxReplicas`), enforce idle-GPU auto-shutdown (`GPU utilization < 5%` for 30 mins), and wire **AWS Budgets -> SNS -> Lambda** (or **GCP Billing Budgets -> Pub/Sub -> Cloud Function**) to automatically scale non-prod GPU groups to zero when spend crosses threshold.

---

## Mini Hands-On Lab (15-Minute Exercise)

Implement **Population Stability Index (`PSI`)** in NumPy to detect production covariate shift:

```python
import numpy as np

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

rng = np.random.default_rng(42)
train_dist = rng.normal(loc=50.0, scale=10.0, size=5000)
stable_prod = rng.normal(loc=50.5, scale=10.0, size=5000)
drifted_prod = rng.normal(loc=65.0, scale=14.0, size=5000)

assert compute_psi(train_dist, stable_prod) < 0.10
assert compute_psi(train_dist, drifted_prod) >= 0.25
print(f"Lab 11 Passed! Stable PSI={compute_psi(train_dist, stable_prod):.3f}, Drifted PSI={compute_psi(train_dist, drifted_prod):.3f}")
```

---

## Key Takeaways for Interviews
- Load models **once inside `FastAPI` `lifespan`** (never per request), prefer **`safetensors` / `ONNX`** over unsafe `pickle`, and run Docker containers as a non-root user.
- Autoscale LLM GPU pods on **KV Cache VRAM utilization & queue depth**, not CPU %.
- Prevent Training-Serving Skew with **`Feast`**, detect covariate shift with **`PSI >= 0.25`**, roll out via **Shadow -> Canary -> A/B**, and train on **Spot GPUs with preemption checkpointing**.

---

[← Previous: Module 10 — Agent Evaluation & Safety](../04-agentic-ai/10_agent_evals_and_safety.md) | [Home](../../README.md) | [Next: Module 12 — Practical Troubleshooting Scenarios →](./12_system_troubleshooting_scenarios.md)
