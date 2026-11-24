# Modern AI/ML Production Tools & Frameworks

[← Back to Resources](./recommended_resources.md) | [Back to Roadmap](../README.md)

---

## 1. High-Performance Inference & Serving
- **[vLLM](https://github.com/vllm-project/vllm):** High-throughput, low-latency LLM serving with PagedAttention, continuous batching, and prefix caching.
- **[Ollama](https://github.com/ollama/ollama):** Frictionless local LLM runtime for running GGUF-quantized models ($0 cloud cost).
- **[ONNX Runtime](https://onnxruntime.ai/):** Cross-platform, high-performance scoring engine for ML models and edge vision systems.

## 2. Vector Stores & Search Engines
- **[Qdrant](https://github.com/qdrant/qdrant):** Production-grade vector database with native hybrid search (dense + sparse BM25), payload filtering, and fast local disk embedding.
- **[DuckDB](https://duckdb.org/):** High-performance in-process analytical SQL OLAP database for tabular data exploration and agent SQL tools.

## 3. Agent Frameworks & Protocols
- **[LangGraph](https://github.com/langchain-ai/langgraph):** Stateful multi-actor orchestration engine with cyclical graphs, persistent state checkpointing, and human-in-the-loop (HITL) gates.
- **[Google ADK](https://github.com/google/adk-python):** Google Agent Development Kit for building, testing, and visualizing multi-agent applications.
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/):** Standardized JSON-RPC protocol enabling AI models to access secure external tools, resources, and contextual services.

## 4. MLOps, Observability & Data Engineering
- **[Polars](https://pola-rs.github.io/polars/):** Blazing fast multithreaded DataFrame library written in Rust with Apache Arrow memory layout.
- **[MLflow](https://mlflow.org/):** Open-source platform for tracking experiments, packaging code, and deploying models.
- **[Feast](https://feast.dev/):** Operational feature store for managing point-in-time correct training-serving feature pipelines.
- **[Evidently AI](https://www.evidentlyai.com/):** Production monitoring suite for data drift, target drift, and model metric degradation.
