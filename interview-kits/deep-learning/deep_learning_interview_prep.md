# Deep Learning & Neural Network Interview Kit

[← Back to Interview Kits](../README.md) | [Back to Roadmap](../../README.md)

---

## 1. Core Neural Network Architecture & Mechanics

### Q1: Why does vanishing/exploding gradient happen in deep networks, and how do modern architectures mitigate it?
- **Root Cause:** In standard backpropagation with the chain rule, $\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial a_L} \prod_{l=2}^L \frac{\partial a_l}{\partial a_{l-1}} \frac{\partial a_1}{\partial W_1}$. If the Jacobian eigenvalues are $< 1$ (e.g. saturated sigmoid where $\sigma'(z) \le 0.25$), gradients shrink exponentially with depth $L$. If eigenvalues are $> 1$, gradients explode.
- **Modern Mitigations:**
  1. **Residual Connections (Skip connections):** $a_{l+1} = a_l + \mathcal{F}(a_l)$. The gradient $\frac{\partial a_{l+1}}{\partial a_l} = I + \frac{\partial \mathcal{F}}{\partial a_l}$, creating an identity gradient highway where signal flows without exponential decay.
  2. **Layer Normalization & RMSNorm:** Stabilizes activations across features within a single sample, avoiding covariate shift.
  3. **Non-saturating Activations:** SwiGLU, GeLU, and LeakyReLU maintain non-zero derivatives across wide input domains.

---

### Q2: Explain the computational and memory complexity of standard Self-Attention vs. PagedAttention.
- **Standard Self-Attention:** Computing $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ over sequence length $N$ requires $O(N^2 \cdot d)$ computation and $O(N^2)$ memory for the attention matrix.
- **KV Cache Memory Footprint:** At inference time, storing keys and values for each layer and head requires:
  $$\text{Memory} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times N_{\text{seq}} \times \text{sizeof(dtype)}$$
- **PagedAttention Solution:** Standard KV caches suffer from 60%–80% memory fragmentation due to contiguous memory allocation. PagedAttention divides the KV cache into fixed-size physical blocks (e.g. 16 tokens), mapping virtual tokens to non-contiguous physical pages via a block table, nearly eliminating internal and external fragmentation and enabling 2x–4x higher serving concurrency.

---

### Q3: How do you identify and debug a silent gradient bug in a custom PyTorch training loop?
1. **Check for In-Place Operations:** Mutating tensors in-place (`x += y` instead of `x = x + y`) can overwrite values needed for backward pass computation, triggering runtime errors or silent incorrect gradients.
2. **Track Gradient Norms:** Log `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to detect zero or exploding gradients early.
3. **Overfit on a Single Batch:** Train the model on 2 to 4 samples. If the loss does not drop to near zero, there is an algorithmic or tensor broadcasting defect in the loss or architecture.
4. **Inspect `param.grad` directly:** Iterate `for name, p in model.named_parameters(): if p.grad is None or torch.isnan(p.grad).any(): ...` to locate disconnected compute graph branches.
