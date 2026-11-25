---
module: "01"
title: "Python, Software Engineering, SQL & Modern Data Stack"
difficulty: "core"
estimated_hours: 18
prerequisites:
  mandatory: []
  recommended: ["basic-terminal-and-variables"]
learning_levels:
  - foundation
  - applied
  - production
topics:
  - python-software-engineering
  - git-and-linux-shell
  - pytest-and-mocking
  - oop-dependency-injection
  - numpy-vectorization
  - pandas-cow-and-polars
  - sql-window-functions
  - async-pydantic-apis
---

# Module 01: Python, Software Engineering, SQL & Modern Data Engineering Stack

[← Back to Portal](../../README.md) | [Phase 1 Index](./README.md) | [Next: Module 02: Math & Statistics →](./02_math_and_statistics.md)

---

## 1. Overview

Every machine learning model and autonomous AI agent begins life as raw, messy data sitting inside a database, file system, or streaming log. Before touching neural networks or LLMs, engineers must master the fundamental software engineering and data stack: managing codebases with Git, processing raw files via Linux shell pipelines, writing test-driven Python with Dependency Injection, extracting clean rows with SQL window functions, transforming feature matrices at compiled speeds with NumPy, and serving concurrent pipelines with type-safe asynchronous HTTP.

```mermaid
flowchart LR
    DEV["Linux Shell + Git + uv"] -->|"Clean Software Architecture"| CODE["Python OOP + Pytest + Logs"]
    CODE -->|"SQL (CTEs & Windows)"| RAW["Relational Data / Event Streams"]
    RAW -->|"Pandas CoW / Polars"| CLEAN["Clean Feature DataFrames"]
    CLEAN -->|"NumPy Vectorization"| MAT["Tensors (X, y)"]
    MAT -->|"Pydantic + Async HTTP"| API["Production ML & Agent APIs"]

    classDef dev fill:#F1F5F9,stroke:#64748B,stroke-width:1.5px,color:#0F172A;
    classDef store fill:#E0F2FE,stroke:#0284C7,stroke-width:1.5px,color:#0369A1;
    classDef proc fill:#EDE9FE,stroke:#7C3AED,stroke-width:1.5px,color:#4C1D95;
    classDef out fill:#DCFCE7,stroke:#16A34A,stroke-width:1.5px,color:#15803D;

    class DEV,CODE dev;
    class RAW store;
    class CLEAN,MAT proc;
    class API out;
```

---

## 2. Audience

- **Aspiring AI/ML Engineers** transitioning from theoretical scripts or ad-hoc Jupyter notebooks into production-grade engineering practices.
- **Backend & Data Engineers** seeking to master vectorized numeric computation, modern Pandas Copy-on-Write, and async AI API patterns.
- **Data Scientists** wanting to eliminate data leakage, unmaintainable monkey-patching, and $O(N)$ execution bottlenecks.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Operate production Git workflows**: Use atomic feature branching, interactive rebasing, and squash merges without dirty commit histories.
- **Harness Linux pipelines for big data triage**: Use `awk`, `sed`, `grep`, `xargs`, and background job controls to parse gigabyte-scale logs without GUI tools.
- **Package Python ML libraries**: Formulate modern `pyproject.toml` configurations with strict dependency pins, entry points, and editable installs.
- **Apply OOP & Dependency Injection**: Decouple data loaders from models using abstract base classes (`typing.Protocol`) to enable seamless mocking.
- **Write robust ML unit tests**: Leverage `pytest`, fixtures, and `monkeypatch` to validate data invariants, model shapes, and API client failure modes.
- **Instrument structured logging**: Replace unformatted `print()` statements with structured JSON logging and contextual metadata.
- **Formulate vectorized tensor transformations**: Broadcast multidimensional arrays in `NumPy` without slow Python interpreter loops.
- **Execute leak-free data cleaning**: Enforce Pandas 2.0+ Copy-on-Write (CoW) semantics to prevent silent data frame mutations.
- **Write complex analytical SQL**: Structure Common Table Expressions (CTEs), window functions (`ROW_NUMBER`, `LAG`, `LEAD`), and partitioned aggregations.
- **Construct concurrent async pipelines**: Utilize `asyncio.gather` with `Pydantic v2` validation and exponential backoff retry for AI inference endpoints.

---

## 4. Prerequisites

- **Mandatory:** Basic programming intuition (variables, control flow, functions) in any language.
- **Recommended:** Familiarity with opening a terminal emulator and running basic commands.

---

## 5. Learning Levels

1. **Foundations (Level 1):** Python data structures, memory layout, Linux shell navigation, Git fundamentals.
2. **Applied (Level 2):** NumPy vector broadcasting, Pandas CoW feature engineering, SQL analytical window functions, pytest fixtures.
3. **Production (Level 3):** Dependency Injection architectures, structured JSON logging, async HTTP client connection pooling with retry backoff.

---

## 6. Core Concepts

### 6.1 The Steering Wheel Model: Python vs Native Speed
Python is slower than C++ or Rust for CPU-bound computations, yet it dominates AI. Python acts as the **steering wheel of a supercar**: you write expressive, readable orchestration in Python, while the underlying math engine (`NumPy`, `PyTorch`, `Polars`) executes compiled C++, CUDA, or Rust instructions at native machine speeds.
> **Rule:** Never iterate raw Python loops over large numerical arrays when a vectorized SIMD operation is available.

### 6.2 Data Structure Complexity in ML Pipelines
| Structure | Syntax | Mutability | Lookup | Memory Overhead | ML Application |
|---|---|---|---|---|---|
| **List** | `[1, 2, 3]` | Mutable | $O(N)$ | High (pointer array) | Dynamic token batches, append-only logs |
| **Tuple** | `(32, 3, 224, 224)` | Immutable | $O(N)$ | Low (compact) | Tensor shapes, dictionary composite keys |
| **Set** | `{"cat", "dog"}` | Mutable | $O(1)$ avg | Medium (hash table) | Vocabulary deduplication, stop-word filters |
| **Dict** | `{"token_id": 42}` | Mutable | $O(1)$ avg | Medium (hash table) | Vocabularies, JSON payloads, feature maps |

### 6.3 Modern Dependency Management with `uv`
Avoid global environment contamination. Modern Python ML projects standardize on Rust-powered package resolvers such as `uv`:
```bash
# Initialize project and create virtual environment in milliseconds
uv init ml-pipeline && cd ml-pipeline
uv venv --python 3.12
source .venv/bin/activate

# Add strictly pinned dependencies
uv add numpy pandas polars pydantic httpx pytest
```

---

## 7. Technical Deep Dive

### 7.1 Linux Shell Mastery for ML Data Triage
When an uncompressed 20 GB training dataset lands on an EC2 instance, opening it in Python immediately exhausts RAM. Master the Linux toolkit:
```bash
# 1. Preview top 5 JSONL records formatted
head -n 5 training_data.jsonl | jq .

# 2. Count distinct label occurrences using awk, sort, and uniq
awk -F',' '{print $3}' raw_data.csv | sort | uniq -c | sort -nr

# 3. Parallelize batch conversion of 1000 compressed chunks across 8 CPU cores
find ./chunks -name "*.gz" | xargs -P 8 -I {} gzip -d {}

# 4. Monitor GPU processes and memory usage without GUI tools
nvidia-smi --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
```

### 7.2 Git Collaboration & Branching Standards
In production AI engineering, dirty histories break bisecting and automated deployment:
1. **Branch Naming:** `feat/feature-name`, `fix/issue-description`, `experiment/model-variant`.
2. **Interactive Rebasing:** Squash exploratory checkpoint commits before opening PRs:
   ```bash
   git checkout feat/rag-chunking
   git rebase -i HEAD~4   # squash WIP commits into one clear semantic commit
   git push origin feat/rag-chunking --force-with-lease
   ```
3. **Avoid Merge Commits in Feature Branches:** Rebase against `origin/main` rather than pulling with merge commits to keep history linear.

### 7.3 Object-Oriented Architecture & Dependency Injection
Decouple ML components so that models, feature extractors, and data loaders can be tested in complete isolation:

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol defining the contract for any feature extraction strategy."""
    def extract(self, texts: list[str]) -> np.ndarray: ...

class BagOfWordsExtractor:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def extract(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                if word in self.vocab:
                    matrix[i, self.vocab[word]] += 1.0
        return matrix

class MLClassifier:
    """Classifier utilizing Dependency Injection for its feature extractor."""
    def __init__(self, extractor: FeatureExtractor) -> None:
        self.extractor = extractor
        self.weights = np.ones(3, dtype=np.float32)

    def predict(self, texts: list[str]) -> np.ndarray:
        features = self.extractor.extract(texts)
        return (features @ self.weights > 0).astype(int)
```

### 7.4 Robust Testing & Mocking with `pytest`
Never deploy un-tested ML preprocessing pipelines. Test invariants and mock expensive model calls:

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def sample_vocab() -> dict[str, int]:
    return {"ai": 0, "machine": 1, "learning": 2}

def test_feature_extractor_shape(sample_vocab):
    extractor = BagOfWordsExtractor(sample_vocab)
    result = extractor.extract(["AI machine learning", "unknown text"])
    assert result.shape == (2, 3)
    assert np.allclose(result[0], [1.0, 1.0, 1.0])
    assert np.allclose(result[1], [0.0, 0.0, 0.0])

def test_classifier_with_mock():
    mock_extractor = MagicMock(spec=FeatureExtractor)
    mock_extractor.extract.return_value = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    classifier = MLClassifier(extractor=mock_extractor)
    preds = classifier.predict(["any text"])
    assert preds.tolist() == [1]
    mock_extractor.extract.assert_called_once_with(["any text"])
```

### 7.5 Structured JSON Logging
Replace unstructured `print()` outputs with machine-parsable logs that include latencies and sample IDs:

```python
import json
import logging
import sys
import time

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra"):
            log_payload.update(record.extra)
        return json.dumps(log_payload)

logger = logging.getLogger("ml_inference")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Production usage
start = time.perf_counter()
# simulated inference
duration_ms = (time.perf_counter() - start) * 1000
logger.info("Inference completed", extra={"extra": {"batch_size": 32, "latency_ms": round(duration_ms, 2)}})
```

### 7.6 Vectorized NumPy Broadcasting & Memory Strides
A NumPy array consists of a raw pointer to a contiguous byte block and a tuple of **strides** (the number of bytes to step in each dimension):

```python
import numpy as np

# Feature matrix X (4 samples, 3 features)
X = np.array([
    [10.0, 20.0, 30.0],
    [15.0, 25.0, 35.0],
    [20.0, 30.0, 40.0],
    [25.0, 35.0, 45.0],
], dtype=np.float32)

# Vectorized Z-Score Normalization (zero Python loops)
means = X.mean(axis=0)       # shape (3,) broadcast over (4, 3)
stds = X.std(axis=0) + 1e-8
X_normalized = (X - means) / stds
assert X_normalized.shape == (4, 3)
assert np.allclose(X_normalized.mean(axis=0), 0.0, atol=1e-6)
```

### 7.7 Modern Pandas (2.0+) Copy-on-Write (CoW)
Under Copy-on-Write, creating a slice or modifying a column never unexpectedly mutates the parent DataFrame:

```python
import pandas as pd
pd.options.mode.copy_on_write = True

df = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "spend": [50.0, 0.0, 120.0]})
subset = df[df["spend"] > 0]
subset["spend_log"] = np.log1p(subset["spend"])  # Safe! Does not mutate parent df
assert "spend_log" not in df.columns
```

### 7.8 Analytical SQL with Window Functions
Window functions calculate rolling metrics and deduplicate rows without expensive self-joins:

```sql
WITH RankedTransactions AS (
    SELECT
        transaction_id,
        user_id,
        amount,
        created_at,
        ROW_NUMBER() OVER (
            PARTITION BY user_id 
            ORDER BY created_at DESC
        ) AS recency_rank,
        LAG(amount, 1, 0.0) OVER (
            PARTITION BY user_id 
            ORDER BY created_at ASC
        ) AS previous_amount,
        AVG(amount) OVER (
            PARTITION BY user_id 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_spend
    FROM transactions
)
SELECT *
FROM RankedTransactions
WHERE recency_rank = 1;
```

---

## 8. Hands-on Implementation

Build an end-to-end, resilient asynchronous batch processor with schema validation and exponential backoff retry:

```python
import asyncio
import io
import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pydantic import BaseModel, Field, ValidationError

class UserEvent(BaseModel):
    user_id: str = Field(..., min_length=2)
    engagement_score: float = Field(..., ge=0.0, le=1.0)
    features: list[float]

def stream_batches(stream: Iterable[str], batch_size: int = 64) -> Iterator[list[dict]]:
    """Genuine streaming generator that yields deserialized JSON batches lazily."""
    iterator = iter(stream)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            break
        yield [json.loads(line) for line in chunk if line.strip()]

async def process_batch_with_retry(batch: list[dict], max_retries: int = 3) -> list[UserEvent]:
    valid_events: list[UserEvent] = []
    for item in batch:
        try:
            valid_events.append(UserEvent(**item))
        except ValidationError:
            continue  # Drop malformed records defensively

    for attempt in range(1, max_retries + 1):
        try:
            # Simulate non-blocking asynchronous remote embedding / model inference
            await asyncio.sleep(0.01)
            return valid_events
        except Exception:
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** attempt * 0.05)
    return valid_events

# Demonstration verification
async def main():
    stream_data = io.StringIO("\n".join([
        json.dumps({"user_id": f"u_{i}", "engagement_score": 0.85, "features": [1.0, 2.0]})
        for i in range(100)
    ]))
    batches = list(stream_batches(stream_data, batch_size=25))
    results = await asyncio.gather(*(process_batch_with_retry(b) for b in batches))
    total_processed = sum(len(r) for r in results)
    assert total_processed == 100
    print(f"✅ Successfully processed {total_processed} events across {len(batches)} batches.")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Practical Exercises

### Exercise 1: Multi-Axis Rolling Statistics in Pure NumPy
Given a matrix $X \in \mathbb{R}^{T \times D}$ representing $T$ time steps across $D$ features:
1. Write a function `rolling_zscore(X: np.ndarray, window: int = 5) -> np.ndarray` using sliding window views (`np.lib.stride_tricks.sliding_window_view`).
2. Guarantee that no Python `for` loop iterates over the $D$ feature columns.

### Exercise 2: Dependency-Injected Data Pipeline with Pytest
1. Define a `StorageBackend` Protocol with `save_model(name: str, weights: np.ndarray)` and `load_model(name: str) -> np.ndarray`.
2. Implement a `LocalDiskStorage` and an in-memory `MockStorage`.
3. Write a pytest test suite verifying that an ML pipeline correctly invokes storage without writing to disk.

---

## 10. Mini Project

### Production Event Ingestion & Feature Engineering Service
Construct a modular Python package adhering to modern standards:
- **Project Structure:** Include `pyproject.toml`, `src/pipeline/`, and `tests/`.
- **Ingestion:** Stream a simulated 100,000-line JSONL file using a lazy generator.
- **SQL Analytics:** Ingest into SQLite and execute window queries calculating user-level rolling conversion velocities.
- **Validation:** Enforce strict Pydantic v2 validation and serialize the cleaned feature matrix into `.npz` format.
- **Testing:** Include a comprehensive pytest suite with 100% test coverage on edge cases (null inputs, division by zero, invalid JSON).

---

## 11. Production Considerations

1. **Connection Pooling in Async AI Clients:** When making calls to OpenAI, Anthropic, or internal model servers, always share a single `httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50, max_connections=200))` instance. Creating a new client per request incurs TLS handshake overhead on every query.
2. **Deterministic NumPy Randomness:** Always instantiate `rng = np.random.default_rng(seed=42)` instead of legacy global `np.random.seed(42)`. Global seeds cause hidden cross-thread coupling in multi-worker environments.
3. **Memory Leaks from Closure References:** In PyTorch and Pandas pipelines, holding a reference to a slice or intermediate tensor inside an exception traceback or dictionary retains the entire parent array in memory. Always explicitly slice `.copy()` before caching.

---

## 12. Common Mistakes & Gotchas

1. **Mutable Default Arguments:** Writing `def train_batch(data, cache=[])` creates a single shared list across all function invocations. Always use `cache: list | None = None`.
2. **Chained Indexing in Pandas:** Writing `df[df['a'] > 0]['b'] = 1` causes silent failure or unpredictable copies. Always use `df.loc[df['a'] > 0, 'b'] = 1`.
3. **Filtering Window Functions with `WHERE`:** Trying to write `WHERE ROW_NUMBER() OVER (...) = 1` causes a syntax error. Window functions execute *after* `WHERE`; always wrap window logic in a CTE or subquery.
4. **Unpinned Python Dependencies:** Writing `pandas` instead of `pandas>=2.2.0,<3.0.0` in package definitions causes silent breakage when major breaking releases occur.

---

## 13. Interview Questions & System Design

### Q1: What is the computational difference between Python's `range()` and a NumPy array?
`range()` is a generator-like object that yields Python integer objects on demand, consuming $O(1)$ memory but incurring Python interpreter overhead on each iteration. A NumPy array preallocates contiguous memory for fixed-width integers (e.g. `int64`), enabling direct SIMD hardware operations at compile speed.

### Q2: Why is Dependency Injection essential when building production ML systems?
ML systems interface with external, volatile dependencies: remote object storage (S3), feature stores (Feast), and database connections. Dependency Injection replaces hardcoded client initializations with abstract protocols (`FeatureStoreProtocol`). This allows testing training logic with fast in-memory mock fixtures without needing live cloud connections.

### Q3: How does DuckDB achieve sub-second analytics over 50 GB CSV/Parquet files without loading everything into RAM?
DuckDB uses a vectorized columnar query execution engine with lazy execution and memory-mapped I/O. It reads only the required columns (projection pushdown) and filter rows (predicate pushdown) in small memory-sized chunks (vectors of 1024 or 2048 tuples), keeping RAM usage bounded.

---

## 14. Recommended Resources

- **Book:** *Designing Machine Learning Systems* by Chip Huyen (Chapter 3: Data Engineering Fundamentals).
- **Book:** *Architecture Patterns with Python* by Harry Percival & Bob Gregory (Domain-Driven Design and Dependency Injection).
- **Documentation:** [NumPy Broadcasting Guide](https://numpy.org/doc/stable/user/basics.broadcasting.html).
- **Documentation:** [Pandas Copy-on-Write User Guide](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).
- **Tutorial:** [SQL Window Functions Guide (Mode Analytics)](https://mode.com/sql-tutorial/sql-window-functions/).

---

## 15. Completion Checklist

- [ ] Configure a modern Python virtual environment using `uv` with strict dependency pinning.
- [ ] Implement an interactive Git rebase squashing exploratory commits into clean semantic milestones.
- [ ] Parse and summarize a simulated CSV dataset using `awk` and `sort` in the Linux terminal.
- [ ] Create a Protocol-based `FeatureExtractor` and test it with a mocked `pytest` fixture.
- [ ] Vectorize a multi-feature Z-score standardization in pure NumPy without loops.
- [ ] Write an analytical SQL CTE utilizing `ROW_NUMBER()` and `LAG()` to compute sequential deltas.
- [ ] Build a validated `Pydantic` schema and execute concurrent async queries using `asyncio.gather`.
- [ ] Pass all assertions in [`01_numpy_pandas_sql_lab.ipynb`](../../notebooks/foundations/01_numpy_pandas_sql_lab.ipynb).

---

## 16. Further Exploration

- **Apache Arrow & IPC Zero-Copy:** Explore PyArrow memory-mapped buffers enabling Python and C++ to read shared memory arrays with zero serialization cost.
- **Polars Query Optimizer:** Inspect `.explain()` plans in Polars to examine filter and projection pushdown transformations on disk.
- **SIMD Vector Registers:** Read Intel AVX-512 and ARM Neon architecture specifications to understand 512-bit vector registers performing parallel floating-point operations.

---

[← Back to Portal](../../README.md) | [Phase 1 Index](./README.md) | [Next: Module 02: Math & Statistics →](./02_math_and_statistics.md)
