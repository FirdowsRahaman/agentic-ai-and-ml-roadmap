# Starter Kit 1: Local Air-Gapped Hybrid RAG & SQL Router Agent ($0 API Cost)

[← Back to Portfolio Projects](../README.md)

A minimal, **100% runnable, zero-cloud-cost** starter implementation of **Local Project 1** combining:
1. **Deterministic Intent Routing** (classifying analytical SQL queries vs. unstructured policy document queries)
2. **Hybrid Search (`BM25` + Dense Vector `RRF`)** for policy grounding
3. **Read-Only In-Memory `SQLite` / `DuckDB`-compatible Analytical SQL Tool** with mutation blocking (`DROP` / `DELETE` / `UPDATE` rejected)
4. **Optional Local `Ollama` Generation** (falls back gracefully to deterministic grounded synthesis when Ollama is offline so unit tests pass anywhere!)

## Quickstart

```bash
cd projects/intermediate/starter-local-rag-agent
pip install -r requirements.txt
pytest -q
python app.py
```

## Extend This into Your Portfolio Flagship
- Swap the zero-dependency `HybridRetriever` with **`qdrant-client`** (`QdrantClient(path="./qdrant_db")`) using named Dense (`Qwen/Qwen3-Embedding-0.6B` or `BAAI/bge-m3`) + Sparse (`BM25`/`SPLADE`) vectors and `FusionQuery(fusion=Fusion.RRF)`.
- Point `ReadOnlySQLTool` to `duckdb.connect()` over a 1M-row Parquet dataset.
- Add a `Streamlit` chat UI displaying retrieved chunk IDs, Qdrant RRF scores, and executed SQL queries side by side.
