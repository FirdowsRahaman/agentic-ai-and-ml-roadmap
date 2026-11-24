# Module 05: NLP, Transformers, Computer Vision & Multimodal AI

> **Estimated Study Time:** 9–11 Hours | **Prerequisites:** [Module 04](../02-core-ml-and-dl/04_deep_learning_and_rl.md) | **Lab:** [Mini-Transformer Attention Notebook](../../notebooks/03_mini_transformer_attention.ipynb) | **[Open in Colab](https://colab.research.google.com/github/FirdowsRahaman/ai-ml-fresher-roadmap/blob/main/notebooks/03_mini_transformer_attention.ipynb)**

[← Previous: Module 04 — Deep Learning & RL](../02-core-ml-and-dl/04_deep_learning_and_rl.md) | [Home](../../README.md) | [Next: Module 06 — LLMs, Fine-Tuning & Advanced RAG →](./06_llms_finetuning_and_rag.md)

---

## Mental Model: Convergence of Language, Vision & Audio

Historically, NLP used recurrent networks, Vision used CNNs, and Audio used signal-processing pipelines. Modern AI unifies **all three modalities** by converting raw signals into sequences of dense patch/subword/frame tokens projected into a shared **Transformer Self-Attention** backbone.

```mermaid
flowchart LR
    TXT["Text -> Subword BPE Tokens"] --> PROJ["Linear Projection + Positional Encoding (RoPE)"]
    IMG["Image -> 16x16 Pixel Patches"] --> PROJ
    AUD["Audio -> Log-Mel Spectrogram Frames"] --> PROJ
    PROJ --> ATTN["Multi-Head Self-Attention + SwiGLU MLP"]
    ATTN --> OUT["Unified Multimodal Understanding & Generation"]

    classDef mod fill:#E0F2FE,stroke:#0284C7,stroke-width:1.5px,color:#0369A1;
    classDef core fill:#EDE9FE,stroke:#7C3AED,stroke-width:2px,color:#4C1D95;
    classDef final fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;

    class TXT,IMG,AUD mod;
    class PROJ,ATTN core;
    class OUT final;
```

---

## Tier 1: Core Foundations (NLP Foundations, Transformers & CNNs)

### 1. Classical NLP & Subword Tokenization

#### 1.1 Lexical Representations (`BoW`, `TF-IDF`) & Subword Byte-Pair Encoding (`BPE`)
- **Bag-of-Words (BoW):** Counts word frequencies while discarding word order (`"not good, very bad"` == `"not bad, very good"`). Produces ultra-sparse vectors.
- **TF-IDF:** Multiplies local term frequency by corpus rarity:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
  Still lexical (zero semantic overlap between `"car"` and `"automobile"`), but remains a fast baseline alongside **BM25**.
- **Byte-Pair Encoding (`BPE`):** Modern LLMs do **not** strip stopwords or stem words. Instead, `BPE` iteratively merges frequent character/byte pairs into a fixed vocabulary (`~100k–200k` tokens), handling rare words, code, and typos with **zero Out-Of-Vocabulary (`<UNK>`) failures**.

---

#### 1.2 Static Word Vectors vs. Contextualized Token Representations
- **Static (`Word2Vec` CBOW / Skip-gram):** Assigns **one fixed vector per word** regardless of context — `"bank"` has the exact same vector in *"river bank"* and *"investment bank"*.
- **Contextual (`BERT`):** Computes each token's representation dynamically via bidirectional self-attention over its surrounding sentence, resolving polysemy.

---

### 2. Transformer Architecture & Self-Attention Mechanics

#### 2.1 Scaled Dot-Product Attention ($Q, K, V$), Causal Masking & `RoPE`
Given input token embeddings $X \in \mathbb{R}^{T \times d_{\text{model}}}$, we project each token into three learned subspaces:
- **Query ($Q = X W_Q$):** *"What information am I looking for?"*
- **Key ($K = X W_K$):** *"What information do I contain?"*
- **Value ($V = X W_V$):** *"What payload do I pass forward if attended to?"*

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

1. **Why divide by $\sqrt{d_k}$?** For large head dimension $d_k$, dot products $q \cdot k$ grow in variance ($\propto d_k$), pushing `softmax` into flat saturation regions where gradients vanish. Scaling by $\frac{1}{\sqrt{d_k}}$ restores unit variance!
2. **Causal Mask ($M$):** Decoder LLMs set future positions ($j > i$) to $-\infty$ before `softmax` so token $i$ cannot cheat by peeking at future tokens during parallel training.
3. **Multi-Head Attention (`MHA`) & `RoPE`:** Splits $d_{\text{model}}$ into $h$ parallel heads (capturing syntax, coreference, and long-range semantics simultaneously). Because attention is permutation-invariant, modern LLMs rotate $Q$ and $K$ by token index angles using **Rotary Positional Embeddings (`RoPE`)** to encode relative token distance.

---

#### 2.2 Encoder-Only vs. Decoder-Only vs. Encoder-Decoder Topologies

| Architecture | Attention Visibility | Representative Models | Primary Production Use Cases |
|---|---|---|---|
| **Encoder-Only** | Bidirectional (sees full text) | `BERT`, `DeBERTa`, `BGE`, `ModernBERT` | Dense embeddings, Cross-Encoder reranking, NER, fast classification |
| **Decoder-Only** | Causal / Autoregressive (left-to-right) | `GPT`, `Gemini`, `Claude`, `Llama`, `Qwen` | Generative chat, code generation, reasoning, autonomous agents |
| **Encoder-Decoder** | Bidirectional input + Causal output | `T5`, `BART`, `Whisper` (Audio->Text) | Sequence-to-sequence translation, speech recognition (`Whisper`) |

---

### 3. Computer Vision: CNNs, ResNet & Transfer Learning

#### 3.1 Spatial Convolutions, Residual Identity Highways & Transfer Learning
Flattening a `1000×1000×3` image into a 1,000-unit dense layer creates **3 billion weights** in layer 1 alone while destroying 2D spatial locality. **CNNs** use small sliding kernels (local receptive fields + weight sharing) with spatial output size:
$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1$$

**ResNet Skip Connections (He et al., 2015):** Learn residual $\mathcal{F}(x)$ added to identity shortcut $x$:

```mermaid
flowchart LR
    X["Input Tensor (x)"] --> W1["Conv 3x3 + BatchNorm + ReLU"]
    W1 --> W2["Conv 3x3 + BatchNorm"]
    W2 --> ADD(("Add: F(x) + x"))
    X -->|"Identity Skip Highway (+1 Gradient)"| ADD
    ADD --> ACT["ReLU Output"]

    classDef inp fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#0369A1;
    classDef conv fill:#EDE9FE,stroke:#7C3AED,stroke-width:1.5px,color:#4C1D95;
    classDef sum fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;

    class X inp;
    class W1,W2 conv;
    class ADD,ACT sum;
```

Because $\frac{\partial (x + \mathcal{F}(x))}{\partial x} = 1 + \frac{\partial \mathcal{F}}{\partial x}$, gradients flow backward unimpeded through the `+1` identity path even across 150+ layers.

```python
import torch
import torch.nn as nn
from torchvision import models

# Production Transfer Learning: Freeze pretrained backbone, train lightweight head
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
)
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
```

---

## Tier 2: Production Architecture & Scale (Detection, ViT, CLIP & Audio AI)

### 4. Vision Hierarchy (`IoU`, `mAP`, `ViT`) & Multimodal Alignment (`CLIP`)

#### 4.1 Detection/Segmentation Metrics (`IoU`, `mAP`), Vision Transformers (`ViT`) & `CLIP`
- **Classification** (`ConvNeXt`, `ViT`) -> single image label; evaluated via Accuracy / Top-5.
- **Object Detection** (`YOLOv11`, `RT-DETR`) -> `[x, y, w, h, class, conf]` boxes; evaluated via **IoU** ($\frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$) and **mAP@0.5:0.95**.
- **Semantic vs. Instance Segmentation** (`SegFormer`, `SAM 2`) -> pixel-level class mask (`mIoU`) vs. distinct per-object instance masks (`Mask AP`).
- **`CNN` vs. `Vision Transformer (ViT)`:** CNNs bake in spatial locality and translation equivariance (better on `<100k` images); `ViT` splits images into `16×16` patches with global self-attention from layer 1 (dominates at massive pretraining scale).
- **`CLIP` Contrastive Alignment:** Trains an Image Encoder and Text Encoder jointly with **InfoNCE loss** over $N \times N$ batch pairs — maximizing diagonal cosine similarity of matched `(image_i, text_i)` pairs while pushing apart $N^2 - N$ mismatched pairs, unlocking zero-shot classification and cross-modal image search.

---

### 5. Audio & Speech AI (`Mel Spectrograms`, `Whisper` & Native Audio Models)

#### 5.1 Log-Mel Spectrograms, `Whisper` ASR & Native Neural Audio Codecs
1. **Waveform to Log-Mel Spectrogram:** Raw 16 kHz 1D audio waveforms are windowed via Short-Time Fourier Transform (STFT) into 2D **Log-Mel Spectrograms** (`80` or `128` Mel frequency bins × time frames, matching human pitch perception).
2. **`Whisper` Architecture (OpenAI):** Passes the Log-Mel spectrogram through 2 `1D Conv` layers + a **Transformer Encoder**, then decodes timestamps, language IDs, and transcribed/translated text autoregressively via a **Transformer Decoder**.
3. **Native Multimodal Audio Tokens:** Frontier voice models (`Gemini Live`, `GPT-4o Realtime`) discretize audio directly into neural codec tokens (`SoundStream` / `EnCodec`), enabling low-latency speech-in, speech-out reasoning while preserving emotion, tone, and speaker prosody without a brittle `ASR -> Text LLM -> TTS` cascade.

---

## Mini Hands-On Lab (15-Minute Exercise)

Implement **Causal Scaled Dot-Product Self-Attention** in pure NumPy and verify that changing future tokens (`t = 3`) has **zero** effect on earlier token outputs (`t = 0, 1`):

```python
import numpy as np

def causal_self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    seq_len, d_k = Q.shape
    scores = (Q @ K.T) / np.sqrt(d_k)
    # Upper-triangular mask (above main diagonal) set to -inf
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal_mask, -1e9, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    return weights @ V

rng = np.random.default_rng(7)
Q1 = rng.normal(size=(4, 8))
K1 = rng.normal(size=(4, 8))
V1 = rng.normal(size=(4, 8))
out1 = causal_self_attention(Q1, K1, V1)

# Mutate the last future token (index 3) completely:
Q2, K2, V2 = Q1.copy(), K1.copy(), V1.copy()
K2[3] *= 999.0
V2[3] *= -999.0
out2 = causal_self_attention(Q2, K2, V2)

assert np.allclose(out1[:3], out2[:3]), "Causal mask leaked future tokens!"
print("Lab 05 Passed! Causal attention verified with zero future leakage.")
```

---

## Key Engineering Takeaways
- Know the **Scaled Dot-Product Attention** equation $\text{softmax}(QK^\top / \sqrt{d_k} + M)V$ cold — especially why $\frac{1}{\sqrt{d_k}}$ prevents softmax gradient saturation and how causal masking prevents future token leakage.
- Contrast **Encoder (`BERT` — embeddings/reranking)** vs. **Decoder (`GPT`/`Llama` — generation)** vs. **Encoder-Decoder (`Whisper`/`T5`)**.
- Explain **ResNet skip connections** via the `+1` identity gradient term, **`CLIP`** via symmetric InfoNCE contrastive alignment, and **Audio AI** via Log-Mel Spectrograms vs. native neural audio codecs.

---

[← Previous: Module 04 — Deep Learning & RL](../02-core-ml-and-dl/04_deep_learning_and_rl.md) | [Home](../../README.md) | [Next: Module 06 — LLMs, Fine-Tuning & Advanced RAG →](./06_llms_finetuning_and_rag.md)
