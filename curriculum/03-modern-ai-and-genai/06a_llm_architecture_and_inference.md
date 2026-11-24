---
module: "06A"
title: "LLM Architecture & Inference Engines"
difficulty: "advanced"
estimated_hours: 16
prerequisites:
  - "04A"
  - "05A"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - transformer-decoder
  - rope-embeddings
  - kv-cache
  - decoding-strategies
  - structured-outputs
  - constrained-decoding
---

# Module 06A: LLM Architecture & Inference Engines

[← Previous: Module 05B](./05b_computer_vision_and_multimodal.md) | [Phase 3 Index](./README.md) | [Next: Module 06B →](./06b_rag_engineering.md)

---

## 1. Overview

Generative Large Language Models (LLMs) are predominantly autoregressive decoder-only Transformers. This unit dissects the internal mechanics of LLM inference: Rotary Position Embeddings (RoPE), KV cache memory scaling, sampling strategies (temperature, top-k, top-p, min-p), and constrained grammar decoding for deterministic structured JSON output.

---

## 2. Who Should Study This?

- **LLM Application Engineers** building high-reliability applications requiring deterministic structured JSON outputs.
- **Inference Optimization Engineers** needing to understand KV cache memory scaling and decoding bottlenecks.
- **Backend Developers** integrating open-weights models into production services.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Explain Rotary Position Embedding (RoPE) mechanics**, showing how relative token distances are preserved through complex rotations.
- **Calculate exact KV Cache memory footprints** during multi-turn generation.
- **Implement sampling algorithms** (Temperature scaling, Top-P Nucleus, Top-K) in NumPy.
- **Enforce structured outputs via constrained decoding**, understanding how context-free grammars (CFGs) mask invalid token logits during autoregressive generation.

---

## 4. Prerequisites

- **Attention Mechanism & Self-Attention:** From [Module 05A](./05a_nlp_and_language_models.md).
- **Matrix Calculus & Vector Geometry:** From [Module 02](../01-foundations/02_math_and_statistics.md).

---

## 5. Learning Levels

### Foundation
- Distinguish between Encoder-Only (BERT), Encoder-Decoder (T5), and Decoder-Only (GPT, LLaMA) architectures.
- Understand the autoregressive token-by-token generation loop.

### Applied Engineering
- Implement KV cache caching in a decoder loop to avoid recomputing past key-value states.
- Configure sampling parameters (temperature, top-p, repetition penalty) for specific tasks.

### Production Engineering
- Calculate VRAM requirements for concurrent multi-user serving under long contexts ($32k+$).
- Implement grammar-constrained decoding ensuring 100% syntactically valid JSON responses.

---

## 6. Core Concepts

### 6.1 Rotary Position Embedding (RoPE)
Instead of adding fixed coordinate vectors to embeddings, RoPE applies a 2D orthogonal rotation matrix to Query and Key vector pairs in the complex domain:
$$\mathbf{R}_{\Theta, m}^d \mathbf{x}_m$$
This enables the inner product $\mathbf{q}_m^T \mathbf{k}_n$ to encode relative distance $m - n$ naturally, enabling superior length generalization compared to absolute positional encodings.

### 6.2 The KV Cache Memory Formula
During autoregressive generation, past tokens are frozen; recomputing their $K$ and $V$ vectors at each step wastes $O(N^2)$ FLOPs. Caching $K$ and $V$ tensors turns step computation into $O(N)$.

$$\text{KV Cache Size (Bytes)} = 2 \times 2 \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times L_{\text{context}} \times \text{Batch Size} \times \text{Bytes per Float}$$

---

## 7. Technical Deep Dive

### 7.1 Constrained Grammar Decoding
Traditional prompt engineering asks the LLM to "respond in JSON format", which frequently produces markdown fences or invalid trailing commas that break downstream parsers. **Constrained decoding** intercepts the model's raw output logits before the Softmax layer, using a Context-Free Grammar (CFG) or JSON Schema state machine to set logits of illegal tokens to $-\infty$. This guarantees with mathematical certainty that every generated token satisfies the target schema.

---

## 8. Practical Implementation

Below is a complete implementation of temperature scaling and Top-P (Nucleus) sampling in pure NumPy:

```python
import numpy as np

def sample_next_token(
    logits: np.ndarray, temperature: float = 0.7, top_p: float = 0.9
) -> int:
    """Applies temperature scaling, top-p nucleus filtering, and categorical sampling."""
    # 1. Apply Temperature Scaling
    scaled_logits = logits / max(temperature, 1e-5)
    
    # 2. Compute Softmax Probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # 3. Sort Probabilities in Descending Order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # 4. Cumulative Sum for Nucleus (Top-P) Filtering
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to always keep at least one token
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
    sorted_indices_to_remove[0] = False
    
    # Zero out removed token probabilities and re-normalize
    sorted_probs[sorted_indices_to_remove] = 0.0
    filtered_probs = sorted_probs / np.sum(sorted_probs)
    
    # 5. Sample Next Token from Filtered Distribution
    chosen_idx = np.random.choice(len(filtered_probs), p=filtered_probs)
    return int(sorted_indices[chosen_idx])

# Validation: 10 candidate vocabulary logits
np.random.seed(42)
test_logits = np.array([1.2, 0.5, 4.8, 0.1, 3.2, -1.0, 0.2, 2.1, 0.0, 1.5])
sampled_token = sample_next_token(test_logits, temperature=0.5, top_p=0.85)
print("Sampled Token ID:", sampled_token)
assert sampled_token in [2, 4], "Must sample from top probability mass"
```

---

## 9. Hands-on Exercises

### Beginner
Calculate the KV cache memory size (in Gigabytes) for a 7B LLaMA model ($N_{\text{layers}}=32, N_{\text{heads}}=32, d_{\text{head}}=128$) running a batch size of 16 with a 4,096 token context in FP16.

### Intermediate
Implement greedy decoding, temperature sampling, and top-k filtering in Python and compare their output diversity on a fixed distribution of logits.

### Advanced
Build a lightweight JSON grammar state machine that enforces opening and closing braces `{` and `}` during mock autoregressive generation.

---

## 10. Mini Project: KV Cache Decoder Simulator

Build a simulator of autoregressive decoding:
1. Define a mock Transformer layer producing Query, Key, and Value vectors.
2. Implement two modes: naive recomputation vs. incremental KV cache updates.
3. Benchmark latency and FLOP scaling across generation lengths from 1 to 256 tokens, verifying the $O(N)$ vs. $O(N^2)$ theoretical speedup.

---

## 11. Production Considerations

- **Memory Fragmentation:** Dynamic allocations for KV caches cause GPU memory fragmentation. In production, utilize PagedAttention (vLLM) to allocate non-contiguous physical memory blocks.
- **Speculative Decoding:** Accelerate inference by drafting tokens with a tiny auxiliary model and verifying them in parallel with the flagship model.

---

## 12. Common Mistakes & Gotchas

1. **Allocating KV Cache in Contiguous Blocks:** Pre-allocating maximum context length for all concurrent requests wastes 60–80% of VRAM on idle padding.
2. **Temperature = 0 with Random Sampling:** If deterministic greedy output is required, use `argmax(logits)` directly rather than stochastic sampling.

---

## 13. Interview Questions & System Design

1. **Q:** Why does inference latency during autoregressive token generation depend primarily on GPU memory bandwidth rather than GPU FLOPs?  
   **A:** During generation (batch size 1), each step requires transferring all model weights and KV cache tensors from High-Bandwidth Memory (HBM) to SRAM to compute a single token vector product. The GPU compute units remain starved waiting for memory transfers, making the step memory-bandwidth bound.
2. **System Design:** Design an inference gateway serving 1,000 queries per second with a Time-To-First-Token (TTFT) budget under 200ms.

---

## 14. Recommended Resources

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al.
- [Outlines: Structured Text Generation with Regular Expressions and CFGs](https://github.com/dottxt-ai/outlines).
- [The Illustrated Transformer Decoder](http://jalammar.github.io/illustrated-gpt2/) — Jay Alammar.

---

## 15. Completion Checklist

- [ ] Understand RoPE relative coordinate rotations.
- [ ] Can calculate KV Cache memory consumption from model dimensions.
- [ ] Understand temperature scaling and Top-P sampling.
- [ ] Understand how constrained decoding guarantees valid JSON output.

---

## 16. Further Exploration

- Study Grouped-Query Attention (GQA) mechanics and memory savings.
- Investigate Medusa and speculative decoding techniques.
