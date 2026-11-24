# Module 11: Production MLOps, AI Engineering, Drift & Cloud FinOps

> **Estimated Study Time:** 8–10 Hours | **Prerequisites:** [Module 03](../02-core-ml-and-dl/03_classical_ml_and_recsys.md), [Module 10](../04-agentic-ai/10_agent_evals_and_safety.md) | **Starter Repo:** [Local MLOps Starter Template](../../projects/starter-local-mlops/)

[← Previous: Module 10 — Agent Evaluation & Safety](../04-agentic-ai/10_agent_evals_and_safety.md) | [Home](../../README.md) | [Next: Module 12 — Practical Troubleshooting Scenarios →](./12_system_troubleshooting_scenarios.md)

---

## Mental Model: ML Modeling Code Is Only ~5%–10% of a Production System

Across [Modules 01–10](../04-agentic-ai/10_agent_evals_and_safety.md), you learned how to train, align, ground, and evaluate AI models. Now comes the reality check that separates portfolio demos from real engineering: as Google researchers famously documented in *[Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)* (Sculley et al., NeurIPS 2015), core ML training code makes up **only ~5%–10%** of a live production system. A `0.94 AUC` model or smart agent delivers zero business value until surrounded by safe packaging, `FastAPI` endpoints, `Docker`, feature stores, drift monitors, and cloud GPU FinOps:

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

## Tier 1: Core Foundations (Serialization, `FastAPI`, `Docker`, `MLflow` & `DVC`)

### 1. Safe Model Serialization & Serving via `FastAPI`

> **Two Core Terms (`Serialization` & `FastAPI`):**
> - **Serialization:** Saving a trained model from your computer's live RAM onto disk as a file so another server can load it tomorrow without retraining.
> - **`FastAPI`:** Python's standard web framework for wrapping a trained model behind a live URL (`POST /predict`) so mobile apps, websites, or teammates can send JSON inputs and get predictions back in milliseconds.

#### 1.1 Safe Model Serialization (`safetensors`, `ONNX`) & Startup-Loaded `FastAPI` Services
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

### 2. Containerization (`Docker`) & Experiment/Data Versioning (`MLflow` + `DVC`)

Your `FastAPI` endpoint runs smoothly on your laptop—now how do you make sure it runs identically on a Linux cloud server, and track *which exact dataset* trained *which model version*?

> **Three Everyday MLOps Tools in Plain English:**
> - **`Docker` (Sealed Shipping Container):** Solves *"It worked on my laptop, why did it crash on the server?"* by packing your code + exact Python version + system libraries into one portable **Image** (blueprint) that runs identically anywhere as a **Container** (live process).
> - **`DVC` ("Git for Giant Datasets"):** `Git` chokes if you commit a 50 GB dataset; `DVC` keeps heavy files in S3/cloud storage while saving a tiny 5-line ID card (`.dvc`) inside Git so you always know *which exact dataset* trained *which model*.
> - **`MLflow` (Automated Lab Notebook):** Automatically records every training run's settings (`learning_rate`), test scores (`PR-AUC`), and saved model file so you never lose track of which experiment won.

#### 2.1 Layered Container Caching & Non-Root `Docker` Hardening
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

#### 2.2 Synchronizing Code, Datasets & Artifacts with `Git`, `DVC` & `MLflow`
Before building that `Docker` image in CI/CD, these three versioning tools keep your code, data, and trained weights locked in sync:
- **`Git`:** Versions source code, configs, and small `.dvc` pointer files.
- **`DVC` (Data Version Control):** Stores 50 GB Parquet/image datasets in S3/GCS content-addressed storage while committing tiny SHA-256 pointer files into Git.
- **`MLflow`:** Logs hyperparameters, validation metrics (`PR-AUC`, `p99` latency), Git commit SHA, and versioned model artifacts in a central **Model Registry** (`Staging -> Champion`).

---

## Tier 2: Production Architecture & Scale (Serving Architectures, `Feast`, `PSI` Drift, Rollouts & Infra FinOps)

Shipping Version 1 in a container is **Day 1**; **Tier 2** covers **Day 2 and beyond**—scaling GPU vs. CPU pods, keeping live features identical to training features (`Feast`), streaming real-time agent tokens (`SSE`), catching data drift (`PSI`), and slashing cloud GPU bills.

### 3. Classical ML Serving vs. LLM Serving, and Feature Stores (`Feast`)

#### 3.1 Stateless CPU Model Serving vs. Stateful GPU `KV Cache` Serving
First, notice how differently Kubernetes must autoscale a classical `XGBoost` microservice versus an LLM `vLLM` server:

| Dimension | Classical ML (`XGBoost` / `LightGBM`) | LLM Serving (`vLLM` / `TensorRT-LLM`) |
|---|---|---|
| **Compute & State** | Stateless CPU pods (`<500 MB` RAM), `5ms` inference | Stateful `KV Cache` on high-VRAM GPUs (`24–80 GB`), streaming `SSE` tokens |
| **Batching** | Simple request or micro-batching | **Continuous iteration-level batching** + `PagedAttention` |
| **Autoscaling Signal** | CPU utilization / RPS | **KV Cache VRAM % utilization** + waiting queue depth (CPU % stays near 0 on GPU pods!) |

#### 3.2 Eliminating Training-Serving Skew with Feature Stores (`Feast`)
On the classical ML side, how do you guarantee the live `FastAPI` server feeds `XGBoost` the exact same feature math used during training?
- **In Plain English:** Imagine a Data Scientist calculates `"user_30d_spend"` in Pandas (including tax, UTC time) to train a model, and a Backend Engineer rewrites `"user_30d_spend"` in Go (excluding tax, local time) for the live app. That tiny mismatch (**Training-Serving Skew**) silently ruins live predictions! A **Feature Store (`Feast`)** computes the feature **once** and serves it to both:
  1. **Offline Store (`BigQuery` / `Snowflake` / `Parquet`):** Historical tables with **point-in-time joins** for training.
  2. **Online Store (`Redis` / `DynamoDB`):** Latest precomputed values in `<5ms` for live API requests.

#### 3.3 Real-Time `SSE` Token/Tool Streaming (`FastAPI`) & User Feedback Data Flywheels
On the GenAI & Agent side, users won't stare at a frozen spinner for 12 seconds—so instead of returning one blocking JSON blob, production APIs stream live **Server-Sent Events (`SSE`)** and wire user feedback straight back into training:

```python
import json
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

stream_app = FastAPI()

async def agent_event_stream(session_id: str, prompt: str) -> AsyncIterator[str]:
    # 1. Emit live tool-status frame for UI progress indicator
    yield f"event: status\ndata: {json.dumps({'step': 'qdrant_hybrid_search'})}\n\n"
    # 2. Stream incremental answer tokens as they decode
    for tok in ("Refund ", "approved ", "under ", "Policy ", "[POL-101]."):
        yield f"event: token\ndata: {json.dumps({'text': tok})}\n\n"
    # 3. Emit trace_id so frontend thumbs-up/down binds to the exact OpenTelemetry span!
    yield f"event: done\ndata: {json.dumps({'trace_id': f'tr_{session_id}'})}\n\n"

@stream_app.get("/chat/stream")
async def chat_stream(session_id: str, q: str) -> StreamingResponse:
    return StreamingResponse(agent_event_stream(session_id, q), media_type="text/event-stream")
```
- **Closing the Data Flywheel:** Binding frontend thumbs-up/down, copy events, and human-edited replies to `trace_id` automatically populates (a) **hard-negative regression tests** for your `EDD` Golden Set (`Module 10`) and (b) `(prompt, human_edited_chosen, model_rejected)` preference pairs for **`DPO` fine-tuning** (`Module 06`).

### 4. Statistical Drift (`PSI`), Safe Deployments & Cloud GPU FinOps

Even a perfectly deployed model decays quietly as the outside world changes—so we close the production loop with **automated drift alarms**, **safe staged rollouts**, and **Spot GPU cost controls**:

#### 4.1 Detecting Data Drift (`PSI`) vs. Concept Drift
- **In Plain English:**
  - **Data / Covariate Drift ($P(X)$ changes):** *The customers walking into your store changed*—e.g., your loan model was trained on `\$50k` average salaries, and suddenly `\$120k` applicants flood in. Measured with **`PSI`** (`< 0.10` safe, `>= 0.25` retrain!).
  - **Concept Drift ($P(Y \mid X)$ changes):** *The same customers changed what their behavior means*—e.g., input salaries look identical (`PSI = 0.02`), yet a recession makes people default twice as often! Caught only by tracking real outcomes and accuracy over time.

<details>
<summary><b>Deep Dive: Population Stability Index (`PSI`) Equation & Thresholds</b></summary>

$$\text{PSI} = \sum_{i=1}^{B} (p_{\text{prod}, i} - p_{\text{train}, i}) \cdot \ln\left(\frac{p_{\text{prod}, i}}{p_{\text{train}, i}}\right)$$

Standard MLOps & credit-risk thresholds (*Siddiqi, Credit Risk Scorecards*): `PSI < 0.10` = Stable | `0.10 <= PSI < 0.25` = Moderate shift (investigate) | **`PSI >= 0.25` = Significant shift -> trigger automated CI/CD retraining**. Pair with the non-parametric **Kolmogorov-Smirnov (K-S)** test on continuous columns.

</details>

#### 4.2 Progressive Delivery: Shadow Traffic, Canary Rollouts & Online A/B Tests
When `PSI >= 0.25` triggers a newly retrained **Challenger** model, never swap 100% of live user traffic onto it overnight—promote it safely through three progressive gates:
1. **Shadow (Dark Traffic):** Mirror 100% of live requests to the Challenger model asynchronously; discard Challenger outputs from user responses while verifying latency, memory, and prediction distributions at **zero user risk**.
2. **Canary Rollout:** Route `1% -> 10% -> 50% -> 100%` of live traffic with automated rollback if `5xx` error rates or `p99` latency breach guardrails.
3. **Online A/B Test:** Randomize by `user_id` (`50/50`), measuring **Primary Business KPI** statistical lift ($p < 0.05$) alongside strict **Guardrail Metrics** (latency, unsubscribe rate).

#### 4.3 Preemptible Spot GPU Checkpointing & Automated Cloud Cost Kill-Switches
Running those retraining jobs and GPU clusters on full-price On-Demand cloud instances gets expensive fast—here is how production infra teams slash GPU bills safely:
1. **Spot / Preemptible GPUs + Emergency Checkpointing:** Rent spare cloud GPU capacity at **up to 60%–91% discount** compared to On-Demand rates ([AWS EC2 Spot](https://aws.amazon.com/ec2/spot/); [Google Cloud Spot VMs](https://cloud.google.com/spot-vms)). Save checkpoints to S3/GCS every $N$ steps and trap the cloud `SIGTERM` preemption signal to flush optimizer + model state cleanly so replacement nodes resume automatically.
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

## Key Engineering Takeaways
- Load models **once inside `FastAPI` `lifespan`** (never per request), prefer **`safetensors` / `ONNX`** over unsafe `pickle`, and run Docker containers as a non-root user.
- Autoscale LLM GPU pods on **KV Cache VRAM utilization & queue depth**, not CPU %.
- Prevent Training-Serving Skew with **`Feast`**, detect covariate shift with **`PSI >= 0.25`**, roll out via **Shadow -> Canary -> A/B**, and train on **Spot GPUs with preemption checkpointing**.

> **Next in the Journey:** Your models and agents are now live in production! What happens at 2:00 AM when shadow AUC mysteriously drops to `0.54`, a `24 GB` GPU throws `CUDA out of memory`, or a RAG bot invents fake policy numbers? Let's sharpen your real-world incident instincts in **[Module 12: Real-World ML & GenAI System Troubleshooting Scenarios](./12_system_troubleshooting_scenarios.md)**.

---

[← Previous: Module 10 — Agent Evaluation & Safety](../04-agentic-ai/10_agent_evals_and_safety.md) | [Home](../../README.md) | [Next: Module 12 — Practical Troubleshooting Scenarios →](./12_system_troubleshooting_scenarios.md)
