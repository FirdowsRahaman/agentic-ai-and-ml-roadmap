# AI/ML & Agent Systems Interview Kits

[← Back to Roadmap](../README.md) | [Explore Projects](../projects/README.md)

---

## Overview

Landing a senior AI/ML, GenAI, or Agent Engineer role requires demonstrating end-to-end systems thinking, mathematical grounding, coding fluency, and architectural trade-off analysis. This kit provides targeted preparation modules structured by domain:

### 1. [Machine Learning & Classical RecSys](./machine-learning/classical_ml_questions.md)
- Algorithmic deep dives (gradient boosting, tree pruning, PCA).
- Leak-free preprocessing pipelines and cross-validation strategies.
- Handling extreme class imbalance ($0.1\%$ fraud, cost matrices, PR-AUC).
- Recommendation systems: Two-Tower retrieval, candidate ranking, and diversity reranking.

### 2. [Deep Learning & PyTorch Systems](./deep-learning/deep_learning_interview_prep.md)
- Backpropagation mechanics, vanishing/exploding gradient proofs.
- Modern architectures (ResNets, Transformers, LayerNorm, SwiGLU).
- KV cache memory math, PagedAttention, and FlashAttention.
- Silent PyTorch bug detection, tensor broadcasting pitfalls, and single-batch overfitting protocols.

### 3. [Generative AI, LLMs & Multi-Agent Systems](./generative-ai/genai_and_agents_questions.md)
- Transformer self-attention mechanics and positional encodings (RoPE).
- Hybrid RAG architectures (Dense Cosine + Sparse BM25 via Reciprocal Rank Fusion).
- Fine-tuning trade-offs: LoRA / QLoRA memory calculation vs. DPO preference alignment.
- Autonomous agent loops: ReAct reasoning, stateful graph cycles, and Model Context Protocol (MCP).

### 4. [ML & GenAI System Design Framework](./agentic-ai/ml_system_design_framework.md)
- The 7-step production system design protocol.
- Clarifying requirements, functional/non-functional SLAs (latency budgets, throughput, cold starts).
- High-level architecture, data ingestion, feature stores, and offline-online consistency.
- Model selection, inference optimization (quantization, speculative decoding, caching).
- Evaluation, monitoring (PSI drift, canary rollouts), and failure mode mitigation.

### 5. [Live Coding & SQL Screenings](./machine-learning/sql_and_coding_screenings.md)
- Python NumPy vectorization and memory-efficient data processing.
- Analytical SQL window functions (`ROW_NUMBER()`, `LAG()`, rolling metrics).
- First-principles algorithm implementations (Softmax, Top-K Cosine, Gradient Descent).

### 6. [Resume CAR Playbook & Flashcards](./resume_and_flashcards.md)
- Context-Action-Result (CAR) bullet formula for high-impact resumes.
- Take-home assignment evaluation rubric.
- High-frequency revision flashcards for rapid interview prep.
