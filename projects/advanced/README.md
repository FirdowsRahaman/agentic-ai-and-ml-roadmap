# Advanced Production & Cloud Projects

[← Back to Projects Catalog](../README.md) | [Back to Roadmap](../../README.md)

---

## Overview
These advanced projects reflect architectures built by senior engineers at Google, AWS, Meta, and Stripe, emphasizing scale, reliability, and modern protocols.

---

### Project 1: Autonomous Operations Copilot with Google ADK & MCP
- **Stack:** Google Agent Development Kit (`google-adk`), Gemini 3.1 Pro / Flash, FastMCP, Cloud Run
- **Architecture:** Supervisor root agent coordinating specialized subagents, isolated tool execution over JSON-RPC 2.0 (Model Context Protocol), and human-in-the-loop (HITL) authorization gates for sensitive actions.
- **Companion Module:** [`curriculum/05-agentic-ai/08_tools_langgraph_and_adk.md`](../../curriculum/05-agentic-ai/08_tools_langgraph_and_adk.md)

---

### Project 2: Two-Stage Recommendation & Ranking Engine (Meta / Amazon Scale)
- **Stack:** PyTorch Two-Tower Encoders, FAISS HNSW, LightGBM LambdaMART, Redis, FastAPI
- **Architecture:** Sub-10ms Stage 1 ANN candidate retrieval (Recall@300) combined with ~12ms Stage 2 point-in-time ranking with rich real-time cross-features, followed by MMR diversity filtering.
- **Companion Module:** [`curriculum/02-machine-learning/03_classical_ml_and_recsys.md`](../../curriculum/02-machine-learning/03_classical_ml_and_recsys.md)

---

### Project 3: Real-Time Fraud Prevention with Feast Feature Store & Drift CI/CD (FinTech)
- **Stack:** Feast (Parquet offline + Redis online), XGBoost, Evidently AI, MLflow, GitHub Actions
- **Architecture:** Eliminating training-serving skew through point-in-time correct joins, cost-matrix threshold optimization under 0.1% class imbalance, and automated PSI drift canary retraining pipelines.
- **Companion Module:** [`curriculum/06-production-ai/11_mlops_cloud_and_finops.md`](../../curriculum/06-production-ai/11_mlops_cloud_and_finops.md)

---

### Project 4: High-Throughput LLM Serving with vLLM & Semantic Caching
- **Stack:** vLLM (PagedAttention), Redis Semantic Caching, OpenTelemetry, Arize Phoenix
- **Architecture:** KV cache paging for 2x–4x throughput gains, prefix caching for system prompt reuse, sub-15ms semantic cache hits ($\cos \ge 0.96$), and distributed span tracing for TTFT/TPOT latency observability.
- **Companion Module:** [`curriculum/04-generative-ai/06_llms_finetuning_and_rag.md`](../../curriculum/04-generative-ai/06_llms_finetuning_and_rag.md)
