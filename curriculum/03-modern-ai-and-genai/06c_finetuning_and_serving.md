---
module: "06C"
title: "Fine-Tuning, Serving & LLMOps"
difficulty: "advanced"
estimated_hours: 16
prerequisites:
  mandatory:
    - "04A"
    - "06A"
  recommended:
    - "04B"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - lora-qlora
  - parameter-efficient-finetuning
  - quantization-awq-gptq
  - vllm-serving
  - continuous-batching
  - latency-benchmarks
  - semantic-cache
---

# Module 06C: Fine-Tuning, Serving & LLMOps

[← Previous: Module 06B](./06b_rag_engineering.md) | [Phase 3 Index](./README.md) | [Next: Phase 4 / Module 07 →](../04-agentic-ai/07_agent_fundamentals.md)

---

## 1. Overview

Deploying generative models in enterprise environments requires balancing task-specific customization with production inference efficiency. This unit covers Parameter-Efficient Fine-Tuning (LoRA & QLoRA), weight quantization (AWQ, GPTQ, INT4), modern inference serving engines (vLLM, PagedAttention), latency metrics (TTFT, TPOT), and semantic caching.

---

## 2. Who Should Study This?

- **Generative AI Engineers** customizing open-weights models for specialized domains or domain-specific grammar formats.
- **LLM Infrastructure Specialists** deploying high-throughput, memory-efficient inference servers.
- **FinOps Engineers** optimizing inference token costs and GPU utilization.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Calculate trainable parameter reductions under Low-Rank Adaptation (LoRA)**, explaining the mathematical decomposition $\Delta W = B \cdot A$.
- **Compare quantization techniques** (Post-Training Quantization, AWQ, GPTQ, BitsAndBytes 4-bit NormalFloat).
- **Configure high-throughput inference engines (vLLM)** leveraging continuous batching and PagedAttention.
- **Measure latency benchmarks**: Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT).
- **Implement a Semantic Cache** to intercept redundant queries and cut token expenditures.

---

## 4. Prerequisites

- **Matrix Decomposition & SVD:** Low-rank approximations from [Module 02](../01-foundations/02_math_and_statistics.md).
- **LLM Architecture & KV Cache:** From [Module 06A](./06a_llm_architecture_and_inference.md).

---

## 5. Learning Levels

### Foundation
- Distinguish between Pretraining, Supervised Fine-Tuning (SFT), and Post-Training Alignment (DPO/RLHF).
- Explain why full parameter fine-tuning is impractical for enterprise teams.

### Applied Engineering
- Configure a LoRA adapter with rank $r=16, \alpha=32$ on Query and Value projection layers.
- Deploy a model with vLLM using an OpenAI-compatible REST API endpoint.

### Production Engineering
- Benchmark TTFT and TPOT under varying concurrency loads using load testing tools.
- Implement an in-memory or Redis-based semantic vector cache with cosine thresholding.

---

## 6. Core Concepts

### 6.1 Low-Rank Adaptation (LoRA)
Instead of updating the full weight matrix $W_0 \in \mathbb{R}^{d \times k}$ ($d \times k$ parameters), LoRA freezes $W_0$ and parameterizes the update $\Delta W$ as the product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Where $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise and $B \in \mathbb{R}^{d \times r}$ is initialized with zeros, ensuring $\Delta W = 0$ at the start of training. For rank $r=16$ on a $4096 \times 4096$ matrix, trainable parameters drop by **$99.6\%$** ($131,072$ vs. $16,777,216$ parameters).

### 6.2 Serving Latency: TTFT vs. TPOT
- **Time-To-First-Token (TTFT):** The latency from request submission until the first output token is generated. Dominated by prefill phase (processing prompt tokens in parallel).
- **Time-Per-Output-Token (TPOT):** The time to generate each subsequent token during autoregressive decoding. Dominated by memory bandwidth and KV cache fetching.

---

## 7. Technical Deep Dive

### 7.1 vLLM & PagedAttention
Traditional serving frameworks pre-allocate contiguous memory for each request's maximum context length ($4096+$ tokens), causing up to 80% memory waste from fragmentation and unused slots. **PagedAttention** (inspired by virtual memory paging in OS design) divides KV cache into non-contiguous physical memory blocks. This allows vLLM to achieve **$2\times-4\times$ higher throughput** via dynamic block allocation and continuous request batching.

---

## 8. Practical Implementation

Below is a complete LoRA parameter calculation and verification script:

```python
def calculate_lora_parameters(d_in: int, d_out: int, rank: int) -> tuple[int, int, float]:
    """Calculates baseline vs. LoRA adapter parameter count and reduction ratio."""
    base_params = d_in * d_out
    # Matrix A: (rank x d_in), Matrix B: (d_out x rank)
    lora_params = (d_in * rank) + (d_out * rank)
    reduction = (1.0 - (lora_params / base_params)) * 100.0
    return base_params, lora_params, reduction

# Example: LLaMA-7B Linear Projection Layer (4096 x 4096)
d_model = 4096
r = 16

base, lora, pct_saved = calculate_lora_parameters(d_model, d_model, rank=r)
print(f"Base Parameters: {base:,}")
print(f"LoRA Parameters: {lora:,}")
print(f"Parameter Reduction: {pct_saved:.2f}%")

assert pct_saved > 99.0
print("LoRA Math Validated Successfully!")
```

---

## 9. Hands-on Exercises

### Beginner
Calculate the total trainable parameters when applying LoRA with rank $r=8$ to all Query, Key, Value, and Output projection layers of a 32-layer 7B model.

### Intermediate
Start a local vLLM instance serving a small model (e.g., `Qwen2.5-Coder-1.5B`) and submit asynchronous concurrent inference requests using `httpx`.

### Advanced
Build a semantic cache using an embedding model: if an incoming query has cosine similarity $\ge 0.96$ with a cached query, return the cached completion instantly without invoking the LLM.

---

## 10. Mini Project: Local Model Serving Benchmark

Build an inference benchmarking suite:
1. Spin up a local model server (Ollama or vLLM).
2. Measure TTFT and TPOT across 50 simulated user queries with concurrent client workers.
3. Generate a latency distribution report plotting mean, P95, and P99 latency percentiles.

---

## 11. Production Considerations

- **Adapter Merging:** For high-throughput single-task inference, merge LoRA weights back into the base weights ($W = W_0 + \frac{\alpha}{r} BA$) to eliminate adapter lookup overhead.
- **Continuous Batching:** Never use static batching for generative inference, as short requests are held hostage waiting for long requests to finish. Always enable continuous iteration-level batching.

---

## 12. Common Mistakes & Gotchas

1. **Quantizing Embeddings and LayerNorm to INT4:** Normalization and embedding layers are highly sensitive to numerical precision; only quantize large feed-forward and projection weight matrices.
2. **Ignoring Pre-fill Latency in RAG:** Stuffing 10,000 tokens of retrieved documentation into prompt context drastically spikes TTFT, making user interfaces feel sluggish.

---

## 13. Interview Questions & System Design

1. **Q:** How does PagedAttention eliminate memory fragmentation in LLM inference servers?  
   **A:** By storing KV cache states in non-contiguous, fixed-size virtual memory pages rather than pre-allocated contiguous arrays, PagedAttention allocates memory dynamically token-by-token, bringing memory waste down to under 4%.
2. **System Design:** Design a multi-model serving gateway routing between low-cost fast models and expensive reasoning models based on query complexity.

---

## 14. Recommended Resources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al.
- [vLLM: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180) — Kwon et al.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al.

---

## 15. Completion Checklist

- [ ] Understand LoRA low-rank decomposition math.
- [ ] Understand TTFT vs. TPOT latency metrics.
- [ ] Understand PagedAttention and continuous batching in vLLM.
- [ ] Can design and implement a semantic cache.

---

## 16. Further Exploration

- Explore AWQ (Activation-aware Weight Quantization) for 4-bit inference.
- Investigate speculative decoding with small draft models.
