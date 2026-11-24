---
module: "13A"
title: "Machine Learning & Numerical Algorithms from Scratch"
difficulty: "advanced"
estimated_hours: 16
prerequisites:
  - "01"
  - "02"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - stable-softmax
  - gradient-descent
  - top-k-cosine-retrieval
  - autograd-backprop
  - causal-attention-numpy
---

# Module 13A: Machine Learning & Numerical Algorithms from Scratch

[Module 13 Index](./README.md) | [Phase 5 Index](../README.md) | [Next: Module 13B →](./13b_system_patterns.md)

---

## 1. Overview

Relying exclusively on high-level framework wrappers (`import torch`, `from sklearn import ...`) leaves engineers vulnerable when numerical precision bugs, memory leaks, or silent mathematical degradations occur. This practice track implements the mathematical core of machine learning from first principles in pure NumPy.

---

## 2. Who Should Study This?

- **AI Engineers & Interview Candidates** preparing for top-tier algorithmic screenings and ML coding rounds.
- **Systems Developers** optimizing low-level inference runtimes and custom C++/CUDA kernels.
- **Researchers** requiring complete mathematical visibility into gradient dynamics.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Implement numerically stable Softmax** preventing floating-point overflow and underflow.
- **Implement Batch & Mini-Batch Gradient Descent** with momentum from scratch.
- **Implement Top-K Cosine Similarity Search** in $O(N + K \log K)$ using `np.argpartition`.
- **Implement a reverse-mode Autograd engine** tracking computational graphs and backpropagating loss.

---

## 4. Prerequisites

- **NumPy Vectorization:** Array indexing and memory strides from [Module 01](../../01-foundations/01_python_sql_and_data_stack.md).
- **Matrix Calculus:** Gradients, partial derivatives, and MSE derivations from [Module 02](../../01-foundations/02_math_and_statistics.md).

---

## 5. Learning Levels

### Foundation
- Implement scalar linear regression via ordinary least squares (OLS) closed-form solution:
  $$\mathbf{w} = (X^T X)^{-1} X^T y$$

### Applied Engineering
- Implement vectorized gradient descent with $L_2$ weight regularization penalties.
- Implement high-performance Top-K retrieval using partial partition algorithms.

### Production Engineering
- Build a micro-autograd computational graph engine supporting scalar addition, multiplication, and backpropagation.

---

## 6. Core Concepts & Practical Implementation

### 6.1 Numerically Stable Softmax
Direct evaluation of $e^{z_i} / \sum e^{z_j}$ overflows float64 when $z_i > 709$. Subtracting the maximum coordinate $\max(\mathbf{z})$ eliminates overflow while preserving mathematical equivalence:

```python
import numpy as np

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax across the last dimension."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

# Test with extreme values that would overflow naive exp()
extreme_logits = np.array([1000.0, 1001.0, 1002.0])
probs = stable_softmax(extreme_logits)
assert np.allclose(np.sum(probs), 1.0)
print("Stable Softmax Probabilities:", probs)
```

### 6.2 Top-K Cosine Retrieval via Partial Partition
Sorting an entire database of $N$ embeddings costs $O(N \log N)$. In production search engines, `np.argpartition` identifies the top $K$ items in linear $O(N)$ time:

```python
def top_k_cosine_search(
    query: np.ndarray, corpus: np.ndarray, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Finds top-K most similar corpus vectors in O(N + K log K) time."""
    # Assume query and corpus are L2-normalized
    sims = corpus @ query  # Vector-matrix dot products
    
    # Partition top K elements (unsorted top-K partition)
    partitioned_indices = np.argpartition(sims, -k)[-k:]
    # Sort only the small K-element slice
    sorted_top_k = partitioned_indices[np.argsort(-sims[partitioned_indices])]
    return sorted_top_k, sims[sorted_top_k]

# Validation
np.random.seed(42)
corpus_vecs = np.random.randn(10000, 64)
corpus_norm = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
query_vec = corpus_norm[42]  # Query is identical to item 42

top_indices, top_scores = top_k_cosine_search(query_vec, corpus_norm, k=3)
assert top_indices[0] == 42
assert np.isclose(top_scores[0], 1.0)
print("Top-K Search Validated! Best Match Index:", top_indices[0])
```

---

## 7. Hands-on Exercises

### Beginner
Implement a 1D linear regression gradient descent loop from scratch in pure Python without importing any external libraries.

### Intermediate
Implement K-Means clustering from scratch in pure NumPy, updating centroids iteratively until convergence within $\epsilon = 1e-4$.

### Advanced
Build a scalar `Value` autograd class supporting addition, multiplication, ReLU, and `.backward()`.

---

## 8. Completion Checklist

- [ ] Can write numerically stable softmax from scratch without overflow.
- [ ] Understand why `np.argpartition` achieves $O(N)$ Top-K retrieval.
- [ ] Can derive and implement linear regression gradients.
