---
module: "05A"
title: "NLP & Language Modeling Foundations"
difficulty: "intermediate"
estimated_hours: 14
prerequisites:
  - "02"
  - "04A"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - tokenization-bpe
  - word-embeddings
  - tf-idf-bm25
  - attention-mechanism
  - transformers
---

# Module 05A: NLP & Language Modeling Foundations

[← Previous: Phase 2 / Module 04B](../02-core-ml-and-dl/04b_advanced_dl_and_rl.md) | [Phase 3 Index](./README.md) | [Next: Module 05B →](./05b_computer_vision_and_multimodal.md)

---

## 1. Overview

Natural Language Processing (NLP) bridges raw human communication and computational vector spaces. This unit teaches the foundational progression from lexical statistical techniques (TF-IDF, BM25) and subword tokenization (Byte-Pair Encoding, WordPiece) to the core mathematical engine of modern AI: **Scaled Dot-Product Attention** and the Transformer architecture.

---

## 2. Who Should Study This?

- **Software Engineers** entering Generative AI who need to understand why tokenization quirks (numbers, whitespace, code indentation) affect model behavior.
- **Data Scientists** wanting to implement Scaled Dot-Product Attention from mathematical first principles in NumPy.
- **AI Developers** implementing sparse lexical retrieval (BM25) for hybrid search systems.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Implement Scaled Dot-Product Attention** with causal masking in pure NumPy, proving why the $\frac{1}{\sqrt{d_k}}$ scaling factor prevents softmax gradient saturation.
- **Compare subword tokenization algorithms** (BPE, WordPiece, Tiktoken) against character and whitespace baselines.
- **Implement Okapi BM25 ranking** with term frequency saturation and document length normalization.
- **Trace the forward flow of multi-head self-attention** through Query, Key, and Value linear projections.

---

## 4. Prerequisites

- **Linear Algebra & Matrix Operations:** Dot products, matrix multiplication, and projections from [Module 02](../01-foundations/02_math_and_statistics.md).
- **Core Deep Learning:** Computational graphs and PyTorch training loops from [Module 04A](../02-core-ml-and-dl/04a_deep_learning_and_pytorch.md).

---

## 5. Learning Levels

### Foundation
- Understand tokens, vocabulary size, vocabulary out-of-bounds (`<UNK>`), and basic word embeddings (Word2Vec, GloVe).
- Explain term frequency, inverse document frequency (TF-IDF), and document vector spaces.

### Applied Engineering
- Implement tokenization pipelines with HuggingFace Tokenizers and Tiktoken.
- Compute BM25 ranking scores over a corpus of documents using Python and NumPy.

### Production Engineering
- Optimize memory usage in attention matrices, understanding $O(N^2)$ quadratic context length constraints.
- Implement numerically stable softmax functions that avoid overflow under large attention logits.

---

## 6. Core Concepts

### 6.1 Subword Byte-Pair Encoding (BPE)
BPE merges the most frequent pairs of bytes or characters into a fixed vocabulary (50,000–256,000 tokens). This guarantees that every conceivable character sequence (including code, emojis, and foreign scripts) decomposes into subwords without throwing out-of-vocabulary errors.

### 6.2 Scaled Dot-Product Attention Mechanics
Given input sequences projected into Query ($Q$), Key ($K$), and Value ($V$) matrices:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where $M$ is an optional causal mask setting future token interactions to $-\infty$.

---

## 7. Technical Deep Dive

### 7.1 Why Scale by $1 / \sqrt{d_k}$?
Assume elements of $q$ and $k$ are independent random variables with zero mean and unit variance. Their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$. For large dimensions ($d_k = 128$), the dot products grow large in magnitude, pushing the Softmax function into regions with near-zero gradients. Dividing by $\sqrt{d_k}$ rescales variance back to $1.0$, preserving healthy gradient flow during backpropagation.

---

## 8. Practical Implementation

Below is a complete implementation of Scaled Dot-Product Attention in pure NumPy:

```python
import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Computes scaled dot-product attention in pure NumPy."""
    d_k = Q.shape[-1]
    # 1. Compute raw affinity scores (Seq_len x Seq_len)
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # 2. Apply causal mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    # 3. Numerically stable Softmax across last axis
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 4. Context aggregation
    context = weights @ V
    return context, weights

# Verification: Sequence of 3 tokens, dimension 4
Q_test = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
K_test = Q_test.copy()
V_test = np.array([[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]])

# Causal Mask (lower triangular)
causal_mask = np.tril(np.ones((3, 3)))
context, weights = scaled_dot_product_attention(Q_test, K_test, V_test, mask=causal_mask)

assert context.shape == (3, 2)
assert np.allclose(weights.sum(axis=-1), 1.0)
print("Attention Implementation Validated! Context shape:", context.shape)
```

---

## 9. Hands-on Exercises

### Beginner
Tokenize a code snippet using Python's `tiktoken` library and observe how indentation spaces and variable names are grouped into token IDs.

### Intermediate
Implement Okapi BM25 scoring from scratch in Python, calculating IDF scores and ranking 5 sample documents against a query string.

### Advanced
Implement Multi-Head Attention by splitting $Q, K, V$ across $H$ heads, executing parallel scaled dot-product attention, and concatenating results through an output projection $W_O$.

---

## 10. Mini Project: In-Memory BM25 Search Engine

Build an in-memory document search engine:
1. Tokenize document strings with case-folding and punctuation stripping.
2. Build an inverted index mapping tokens to document IDs and term frequencies.
3. Compute BM25 scores with $k_1 = 1.5, b = 0.75$, returning Top-5 ranked documents for any input search query.

---

## 11. Production Considerations

- **Quadratic Context Cost:** Self-attention scales as $O(N^2)$ in compute and memory with respect to sequence length $N$. For long contexts ($N > 8192$), utilize FlashAttention or linear attention approximations.
- **Tokenizer Alignment:** Always ensure the tokenizer used during inference exactly matches the tokenizer used during model pretraining. Mismatched tokenizers silently corrupt outputs.

---

## 12. Common Mistakes & Gotchas

1. **Unstable Softmax in Attention:** Computing `np.exp(scores)` without subtracting `np.max(scores)` causes immediate `NaN` overflow when attention scores exceed 700.
2. **Ignoring Causal Masking in Autoregressive Models:** Failing to mask future positions allows generative models to "cheat" during training by looking ahead, causing generation to collapse.

---

## 13. Interview Questions & System Design

1. **Q:** Why do modern LLMs use Byte-Pair Encoding (BPE) instead of character-level or word-level tokenization?  
   **A:** Character tokenization results in excessively long sequences that balloon $O(N^2)$ attention costs. Word tokenization produces enormous vocabularies with frequent `<UNK>` out-of-vocabulary failures on code, URLs, and typos. BPE strikes the optimal balance with compact subwords and 100% vocabulary coverage.
2. **System Design:** Design an inverted index system for 10 million documents that returns BM25 search results with sub-15ms latency.

---

## 14. Recommended Resources

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — Jay Alammar.
- [HuggingFace Tokenizers Library](https://github.com/huggingface/tokenizers).

---

## 15. Completion Checklist

- [ ] Can implement Scaled Dot-Product Attention in NumPy.
- [ ] Understand why attention scores are scaled by $1/\sqrt{d_k}$.
- [ ] Understand BM25 term frequency saturation and document length normalization.
- [ ] Can explain the subword merge process in BPE.

---

## 16. Further Exploration

- Explore Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) used in modern LLMs to reduce KV cache memory.
- Investigate FlashAttention-2 tiling optimizations.
