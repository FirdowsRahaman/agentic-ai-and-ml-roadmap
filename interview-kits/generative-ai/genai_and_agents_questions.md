# Generative AI, LLMs, RAG & Autonomous Agents Interview Questions

[← Classical ML Questions](../machine-learning/classical_ml_questions.md) | [Home](../README.md) | [Next: SQL & Coding Screenings →](../machine-learning/sql_and_coding_screenings.md)

---

## 1. Transformer Internals & Post-Training

**Q1: Why does Scaled Dot-Product Attention divide $QK^\top$ by $\sqrt{d_k}$, and why do modern LLMs use `LayerNorm`/`RMSNorm` + `RoPE`?**
*Answer:*
- If components of $q, k \in \mathbb{R}^{d_k}$ have unit variance, their dot product $q \cdot k$ has variance $d_k$. Without dividing by $\sqrt{d_k}$, large logits push `softmax` into flat saturation where gradients vanish.
- `LayerNorm` / `RMSNorm` normalizes per token across hidden features independently of batch size or padding, while **`RoPE` (Rotary Positional Embeddings)** rotates $Q, K$ vectors by position angles so attention scores naturally decay and generalize across relative token distances.

**Q2: Walk through how a base pretrained LLM becomes an aligned reasoning model (`SFT`, `LoRA`/`QLoRA`, `DPO`, `GRPO`).**
*Answer:*
1. **Pre-training:** Next-token prediction over trillions of tokens (broad knowledge, zero instruction following).
2. **`SFT` (Supervised Fine-Tuning):** Instruction-response pairs; **`LoRA`** freezes $W_0$ and trains low-rank matrices $BA$ ($r \ll d$), while **`QLoRA`** quantizes $W_0$ to 4-bit `NF4` to fit on consumer GPUs without catastrophic forgetting.
3. **Preference Alignment (`DPO`):** Directly optimizes policy likelihood ratios on `(chosen, rejected)` human preference pairs without training a separate `PPO` reward model.
4. **Verifiable Reasoning RL (`GRPO`):** Samples groups of reasoning chains and rewards verified math/unit-test correctness, teaching test-time self-correction.

---

## 2. Production RAG & Evaluation

**Q3: Why does pure Dense Vector Search fail in enterprise RAG, and how do you architect + evaluate the fix?**
*Answer:*
Dense embeddings blur exact identifiers (error codes, SKUs, clause numbers, names).
- **Architecture:** Contextual/layout-aware chunking -> **Hybrid Search (`BM25` + Dense `HNSW`)** merged via **Reciprocal Rank Fusion (`RRF`)** -> **Cross-Encoder Reranker (`Top 50 -> Top 5`)** -> Constrained JSON generation with verified citations.
- **Evaluation (`RAGAS` Triad):** Separate retrieval quality (**Context Precision & Context Recall**) from generation quality (**Faithfulness** to context and **Answer Relevance** to query).

**Q4: Why do naive Pre-Filtering and Post-Filtering break vector search in multi-tenant RAG, and how does `Qdrant` solve it?**
*Answer:*
- **Post-Filtering** retrieves global `Top-K` vectors first and filters by `tenant_id` afterward: if a small tenant owns `0.2%` of chunks, `Top-20` often yields **0 valid results**.
- **Naive Pre-Filtering** masks unauthorized nodes out of an `HNSW` graph, severing navigation edges and degrading recall.
- **`Qdrant` Filterable HNSW** builds payload-aware subgraph edges directly into the `HNSW` index (and dynamically falls back to indexed scan at very low cardinality), maintaining sub-10ms high-recall filtered search alongside 4x-32x Scalar/Binary Quantization.

**Q5: Does RAG completely eliminate hallucinations? When would you Fine-Tune instead of using RAG?**
*Answer:*
**No!** RAG reduces hallucinations by grounding context, but noisy retrieval, conflicting chunks, or weak prompts still produce fabrications. Use **RAG** for dynamic/private facts requiring citations and ACL filtering; use **Fine-Tuning (`LoRA`)** to instill consistent output schemas, domain DSLs, tone, or latency/cost distillation into a smaller model.

---

## 3. Inference Optimization, Agents & Security

**Q6: Explain `TTFT` vs. `TPOT`, `KV Cache` fragmentation, and why `vLLM` achieves 2x-4x higher throughput.**
*Answer:*
- **`TTFT` (Prefill)** processes all input tokens in parallel (**compute-bound**; speed up via Prefix Caching).
- **`TPOT` (Decode)** generates 1 token at a time (**memory-bandwidth bound**; speed up via `FP8`/`AWQ` quantization & Speculative Decoding). Streaming improves *perceived* UX latency without lowering total generation wall-time.
- **`vLLM` `PagedAttention` + Continuous Batching** stores `KV Cache` in non-contiguous fixed pages (eliminating 60%+ contiguous VRAM waste) and swaps completed sequences per token step.

**Q7: When should you build a Deterministic Workflow instead of an Autonomous Agent, and how do `MCP` and `A2A` differ?**
*Answer:*
Use **Workflows** (routing, prompt chaining, `asyncio.gather` parallelization) whenever step structure is predictable: lower latency, lower cost, deterministic testing. Reserve **Autonomous Agents (`LangGraph` / `Google ADK`)** for open-ended multi-step loops requiring dynamic recovery.
- **`MCP` (Model Context Protocol: Vertical):** Connects a single agent to deterministic **`Tools`, `Resources`, and `Prompts`** over JSON-RPC 2.0 (`stdio` locally, `Streamable HTTP` + OAuth 2.1 remotely).
- **`A2A` (Agent-to-Agent Protocol: Horizontal):** Linux Foundation / Google standard enabling peer agents across different frameworks/vendors to discover capabilities via `/.well-known/agent-card.json` and delegate opaque multi-turn tasks without exposing internal prompts or database credentials.

**Q8: What is Indirect Prompt Injection, and how do you secure tool-calling agents in production?**
*Answer:*
Malicious instructions embedded inside untrusted external data retrieved by tools (emails, PDFs, web pages, poisoned MCP tool descriptions). Defend via **Defense-in-Depth**: read-only least-privilege DB credentials, isolated Docker/WASM sandboxes, strict `Pydantic` output validation, session token/iteration circuit breakers, and durable **`HITL` approval interrupts** before any external write/financial action.

---

[← Classical ML Questions](../machine-learning/classical_ml_questions.md) | [Home](../README.md) | [Next: SQL & Coding Screenings →](../machine-learning/sql_and_coding_screenings.md)
