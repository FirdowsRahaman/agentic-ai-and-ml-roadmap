# Intermediate Portfolio Projects (100% Local Runnable Starter Kits)

[← Back to Projects Catalog](../README.md) | [Back to Roadmap](../../README.md)

---

## Overview
These intermediate projects come with **complete, runnable source code, pytest suites, and zero cloud API dependencies**. You can clone and run them on consumer hardware immediately.

---

### Starter Kit 1: Air-Gapped Desktop Hybrid RAG & SQL Router Agent
- **Directory:** [`./starter-local-rag-agent`](./starter-local-rag-agent/README.md)
- **Stack:** Python 3.12+, Pure NumPy, SQLite in-memory, BM25 + Dense Vector Cosine
- **Architecture:** Okapi BM25 sparse lexical search combined with dense semantic vectors via Reciprocal Rank Fusion (RRF), coupled with a guarded SQL query engine.
- **Runnable Command:**
  ```bash
  python projects/intermediate/starter-local-rag-agent/app.py
  pytest projects/intermediate/starter-local-rag-agent
  ```

---

### Starter Kit 2: Full-Lifecycle Production MLOps Engine
- **Directory:** [`./starter-local-mlops`](./starter-local-mlops/README.md)
- **Stack:** FastAPI, Uvicorn, Scikit-Learn, NumPy, Docker, Pytest
- **Architecture:** Zero-pickle `.npz` model serialization, leak-free data scaling, production FastAPI inference API with health checks, automated Population Stability Index (PSI) drift detection, and non-root Dockerfile.
- **Runnable Command:**
  ```bash
  python projects/intermediate/starter-local-mlops/src/train.py
  PYTHONPATH=projects/intermediate/starter-local-mlops pytest projects/intermediate/starter-local-mlops/tests
  ```
