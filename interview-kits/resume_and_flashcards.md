# Fresher AI/ML Resume Playbook, Take-Home Rubric & Quick Flashcards

[← SQL & Coding Screenings](./sql_and_coding_screenings.md) | [Home](../README.md) | [Portfolio Projects →](../projects/README.md)

---

## 1. The High-Signal 1-Page AI/ML Fresher Resume Blueprint

Recruiters and hiring managers scan a fresher resume in **8–12 seconds**. Avoid generic "Built Titanic/Iris notebook" bullets and follow the **CAR (Context → Action → Result)** engineering formula:

```text
[Weak Bullet]:  Built a PDF RAG chatbot using LangChain and OpenAI.
[Hired Bullet]: Built an air-gapped Hybrid RAG agent (BM25 + ChromaDB + BGE Reranker + Ollama 4-bit)
                over 2,500 policy PDFs, improving RAGAS Faithfulness from 0.68 to 0.93 at $0 API cost.
```

### Take-Home ML Assignment Checklist (What Senior Engineers Actually Grade)
1. **One-Command Reproducibility:** Clean `pyproject.toml` / `requirements.txt`, `Dockerfile`, and exact commands in `README.md`.
2. **Zero Data Leakage:** Chronological or grouped validation splits wrapped inside pipelines, evaluated against a trivial baseline first.
3. **Engineering Hygiene:** Type hints, `Pydantic` validation, `pytest` unit tests for edge cases (`NaN`, empty input, rate limits), and no hardcoded API keys.
4. **Error Analysis & Trade-offs:** A 10-line section analyzing *where* your model still fails, `p95` latency, and what you would improve next with 1 more week.

---

## 2. How to Structure Any Technical Interview Answer

1. **Direct 1-Sentence Definition:** Answer the exact question immediately without rambling.
2. **Intuition / Contrast:** Contrast against its sibling concept (e.g., *"Unlike Random Forest which averages parallel deep trees to cut variance, XGBoost..."*).
3. **Production Trade-off / Metric:** Mention latency, memory, failure mode, or metric (`PR-AUC`, `TTFT`, `PSI`).
4. **Honest Depth Calibration:** If asked about something you haven't run at petabyte scale, state the architectural principle clearly and anchor it to your local/cloud project implementation.

### Top 10 Interview Red Flags to Avoid
- Reporting *"99% accuracy"* without checking class imbalance (`PR-AUC` / Confusion Matrix).
- Fitting `StandardScaler`, imputers, or target encoders **before** `train_test_split`.
- Jumping to a 70B LLM or deep neural net before establishing a simple logistic/GBDT/regex baseline.
- Confusing model **parameters** (learned weights) with **hyperparameters** (`lr`, `max_depth`, `LoRA` rank $r$).
- Confusing discrete **subword tokens** with continuous **embedding vectors**.
- Claiming **RAG completely eliminates hallucinations** (it reduces and grounds them; retrieval/generation errors still occur).
- Claiming **Fine-Tuning** is the right way to inject daily changing facts (use RAG for dynamic facts, `LoRA` for behavior/format/distillation).
- Using Python `eval()` on LLM outputs or loading untrusted `.pkl` files instead of `safetensors`/`ONNX`.
- Forgetting `model.eval()` and `torch.no_grad()` during PyTorch inference.
- Designing ML systems without asking clarifying questions on latency SLA, QPS, label delay, and false-positive vs. false-negative business cost.

---

## 3. Quick Revision Flashcards (36 Essential Terms)

| Term | Crisp 1-Line Interview Definition |
|---|---|
| **Bias–Variance Tradeoff** | Balancing underfitting (oversimplified high-bias models) against overfitting (noise-memorizing high-variance models). |
| **`XGBoost` / `LightGBM`** | Gradient-boosted decision tree ensembles where each shallow tree sequentially predicts the negative gradient residuals of prior trees. |
| **PR-AUC** | Area under the Precision–Recall curve; the gold-standard metric for heavily imbalanced rare-event classification (`<1%` positives). |
| **Data Leakage** | When target proxies, future events, or test-fold statistics contaminate training transformations, causing live Day-1 collapse. |
| **Two-Tower Model** | Dual user/item neural encoders mapping into a shared vector space for `<10ms` ANN candidate retrieval (`FAISS` / `HNSW`). |
| **`NDCG@K`** | Normalized Discounted Cumulative Gain: ranking metric rewarding relevant items placed near rank #1 with logarithmic position decay. |
| **LayerNorm / RMSNorm** | Per-token feature-axis normalization used in Transformers, invariant to batch size and sequence padding. |
| **`AdamW`** | Adaptive momentum optimizer with **decoupled weight decay**, standard for training ViTs and LLMs. |
| **Self-Attention ($Q,K,V$)** | $\text{softmax}(QK^\top / \sqrt{d_k})V$ — computes dynamic contextual token representations across full sequences in parallel. |
| **`RoPE`** | Rotary Positional Embeddings: encodes relative token distance by rotating Query and Key vectors in complex/2D planes. |
| **`CLIP`** | Contrastive vision-language model aligning image and text encoders into a shared embedding space via symmetric InfoNCE loss. |
| **Token (`BPE`)** | Subword text unit (~3–4 chars) produced by Byte-Pair Encoding, avoiding Out-Of-Vocabulary failures and governing LLM billing/context. |
| **Embedding** | Dense continuous vector capturing semantic similarity such that related concepts have high cosine/dot-product alignment. |
| **Structured Outputs** | Grammar-constrained token decoding guaranteeing LLM responses conform 100% to a strict JSON / `Pydantic` schema. |
| **Hybrid Search + `RRF`** | Fusing lexical `BM25` keyword hits with dense semantic `HNSW` vectors via Reciprocal Rank Fusion. |
| **Cross-Encoder Reranker** | Joint query+chunk transformer (`BGE`/`Cohere`) scoring top retrieved candidates with full cross-attention before LLM generation. |
| **`RAGAS` Triad** | RAG evaluation framework isolating **Context Precision/Recall**, **Faithfulness** (groundedness), and **Answer Relevance**. |
| **`LoRA` / `QLoRA`** | Parameter-Efficient Fine-Tuning freezing base weights (optionally 4-bit `NF4` quantized) while training tiny low-rank adapter matrices $BA$. |
| **`DPO` & `GRPO`** | Post-training alignment (`DPO` on preference pairs) and verifiable reasoning RL (`GRPO` group relative rewards without a critic model). |
| **`TTFT` vs. `TPOT`** | Compute-bound prefill latency to first token (`TTFT`) vs. memory-bandwidth-bound decode latency per output token (`TPOT`). |
| **`KV Cache`** | Cached past Key/Value tensors across Transformer layers avoiding $O(T^2)$ recomputation during autoregressive decoding. |
| **`vLLM` (`PagedAttention`)** | High-throughput LLM serving engine storing `KV Cache` in non-contiguous memory pages alongside continuous iteration-level batching. |
| **AI Agent vs. Workflow** | Workflows follow developer-coded paths; Agents dynamically select tools, inspect observations, and control their own loop (`ReAct`). |
| **`LangGraph` / `Google ADK`** | Production frameworks for stateful cyclical agent graphs, durable `HITL` checkpointing, visual trace debugging, and subagent delegation. |
| **`MCP`** | Model Context Protocol: open JSON-RPC 2.0 standard (`stdio` / `Streamable HTTP`) connecting AI hosts to `Tools`, `Resources`, and `Prompts`. |
| **Indirect Prompt Injection** | Adversarial instructions hidden inside untrusted external data (PDFs, emails, web pages) read by an agent's tools. |
| **Human-in-the-Loop (`HITL`)** | Persisting agent graph state to a database checkpoint and pausing execution for human sign-off before irreversible actions. |
| **`safetensors` / `ONNX`** | Safe, zero-code-execution tensor serialization (`safetensors`) and portable optimized inference graph format (`ONNX`). |
| **`DVC` & `MLflow`** | Git-synced large dataset/artifact pointer versioning (`DVC`) paired with experiment tracking and model registry (`MLflow`). |
| **Feature Store (`Feast`)** | Single-definition feature platform serving point-in-time offline training joins and sub-5ms online Redis features to prevent skew. |
| **Covariate vs. Concept Drift** | Shift in input distribution $P(X)$ (measured by `PSI >= 0.25`) vs. shift in input-to-label mapping $P(Y \mid X)$. |
| **Shadow vs. Canary** | Mirroring live traffic silently to test a challenger (`Shadow`) vs. routing 1%->10%->100% live user traffic with auto-rollback (`Canary`). |
| **Model Cascading** | Routing 80%+ routine queries to fast/cheap small models (or semantic cache) and escalating only hard queries to frontier models. |

---

[← SQL & Coding Screenings](./sql_and_coding_screenings.md) | [Home](../README.md) | [Portfolio Projects →](../projects/README.md)
