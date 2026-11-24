# Module 05: NLP, Transformers, Computer Vision & Multimodal AI

> **Estimated Study Time:** 9-11 Hours | **Prerequisites:** [Module 04](../02-core-ml-and-dl/04_deep_learning_and_rl.md) | **Lab:** [Mini-Transformer Attention Notebook](../../notebooks/03_mini_transformer_attention.ipynb) | **[Open in Colab](https://colab.research.google.com/github/FirdowsRahaman/agentic-ai-and-ml-roadmap/blob/main/notebooks/03_mini_transformer_attention.ipynb)**

[← Previous: Module 04: Deep Learning & RL](../02-core-ml-and-dl/04_deep_learning_and_rl.md) | [Home](../../README.md) | [Next: Module 06: LLMs, Fine-Tuning & Advanced RAG →](./06_llms_finetuning_and_rag.md)

---

## Mental Model: Convergence of Language, Vision & Audio

In [Module 04](../02-core-ml-and-dl/04_deep_learning_and_rl.md), you learned how `PyTorch` trains deep layers of tensors. Historically, text, images, and audio each required completely separate neural network families. Today, modern AI unifies **all three modalities** with one elegant trick: chop text into subwords, chop images into pixel squares, and chop sound into tiny spectrogram slices, then feed all of them as token vectors into the exact same **Transformer Self-Attention** engine!

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

### 1. Subword Tokenization, Lexical Baselines & Contextual Embeddings

> **Two Foundational Terms (`Token` vs. `Embedding`):** Computers cannot multiply raw letters. First, a **Tokenizer** chops text into common word-pieces called **Tokens** and looks up each piece's ID number (`"playing" -> ["play", "ing"] -> [1402, 289]`). Next, an **Embedding table** turns each ID into a list of ~768+ GPS coordinates (`[0.14, -0.62, ...]`) so words with similar meanings sit right next to each other in space!

#### 1.1 Subword Byte-Pair Encoding (`BPE`) vs. Lexical Weighting (`TF-IDF` / `BM25`)
> **2026 Scope Note:** Legacy preprocessing pipelines (manual regex stemming, stopword stripping, `CountVectorizer` Bag-of-Words, and static 2013 `Word2Vec` CBOW/Skip-gram tables) are obsolete in modern AI engineering. Focus on **subword tokenizers (`BPE`)** for neural models and **`TF-IDF` / `BM25`** for sparse hybrid search.

- **Byte-Pair Encoding (`BPE` / `Tiktoken`):** Modern LLMs preserve exact casing, punctuation, and syntax: iteratively merging frequent byte/character pairs into a fixed vocabulary (`~100k-200k` subword tokens) to represent arbitrary text, code, and multilingual scripts with **zero Out-Of-Vocabulary (`<UNK>`) failures**.
- **Sparse Lexical Scoring (`TF-IDF` & `BM25`):** Multiplies local term frequency by inverse corpus rarity:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
  While purely lexical (zero semantic overlap between `"car"` and `"automobile"`), **Okapi `BM25`** (which adds saturating term frequency and document-length normalization) remains indispensable as the sparse half of **Hybrid RAG** (`Module 06`).
- **Contextual vs. Static Embeddings:** Unlike legacy static word tables that assigned one fixed vector to `"bank"` everywhere, modern Transformer encoders (`ModernBERT`, `Qwen3-Embedding`, `BGE-M3`) compute dynamic, context-sensitive token vectors via self-attention over the entire sequence.

### 2. Transformer Architecture & Self-Attention Mechanics

How does a Transformer actually turn `"bank"` in *"river bank"* vs. *"bank account"* into two totally different meaning vectors? By letting every token look around the sentence and pull context from relevant neighbors via **Self-Attention**:

#### 2.1 Scaled Dot-Product Attention ($Q, K, V$), Causal Masking & `RoPE`
- **In Plain English (The Library Analogy):** When reading *"The server crashed because **it** ran out of RAM"*, how does the word **`"it"`** figure out it refers to `"server"` and not `"RAM"`? Every token creates three vectors:
  - **Query ($Q$):** A sticky note asking *"What am I looking for?"* (e.g., `"it"` looking for a singular noun subject).
  - **Key ($K$):** The label on every word's spine announcing *"Here is what I am!"*
  - **Value ($V$):** The actual meaning inside each word passed forward when its Key matches your Query.
- **Causal Masking (No Peeking Ahead):** During training, an LLM reads the whole sentence at once for speed, so we grey out future words (`-inf`) to stop word #3 from cheating by peeking at word #4.
- **`RoPE` (Word Order):** Since pure attention treats a sentence like an unordered bag, **Rotary Positional Embeddings (`RoPE`)** slightly rotate each word's Query/Key angle by its position (`1st, 2nd, 3rd...`) so the model feels exact word distance.

<p align="center">
  <img src="../../assets/animations/self_attention_qkv_motion.svg" width="100%" alt="Animated Transformer Self-Attention showing Query, Key, Value matching and Causal Masking" />
</p>

<details>
<summary><b>Deep Dive: Attention Equation & Why We Divide by $\sqrt{d_k}$</b></summary>

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

Why divide by $\sqrt{d_k}$? When head dimension $d_k$ is large (e.g., `64` or `128`), raw dot products $q \cdot k$ grow large in variance ($\propto d_k$), pushing `softmax` into extreme `0.9999 / 0.0001` saturation where gradients vanish to zero. Dividing by $\sqrt{d_k}$ keeps scores well-behaved with unit variance. **Multi-Head Attention (`MHA`)** runs $h$ independent $(Q,K,V)$ heads in parallel so one head tracks grammar while another tracks long-range facts.

</details>

#### 2.2 Encoder-Only vs. Decoder-Only vs. Encoder-Decoder Topologies
Depending on whether we leave that Causal Mask **off** (letting every token read both left and right) or **on** (generating strictly left-to-right), the exact same Self-Attention block gives us three distinct Transformer families:

| Architecture | Attention Visibility | Representative 2026 Models | Primary Production Use Cases |
|---|---|---|---|
| **Encoder-Only** | Bidirectional (sees full text) | `ModernBERT`, `Qwen3-Embedding`, `BGE-M3` *(descended from `BERT`)* | Dense embeddings, Cross-Encoder reranking, NER, fast classification |
| **Decoder-Only** | Causal / Autoregressive (left-to-right) | `Gemini 3/3.1`, `GPT-5`, `Claude 4.5/4.6`, `Llama 4`, `Qwen 3.5` | Generative chat, code generation, reasoning, autonomous agents |
| **Encoder-Decoder** | Bidirectional input + Causal output | `Whisper-large-v3-turbo` (Audio->Text), `T5` family | Speech recognition (`Whisper`) & sequence-to-sequence tasks |

### 3. Computer Vision: CNNs, ResNet & Transfer Learning

Now let's switch from words to **images**. Before chopping images into Transformer patches, let's understand the classic spatial workhorse of computer vision, and why its skip-connection highway also lives inside every Transformer layer!

#### 3.1 Spatial Convolutions (`CNNs`), Residual Skip Highways & Transfer Learning
- **In Plain English (The Sliding Flashlight):** Connecting every pixel of a `1000×1000` color photo to 1,000 neurons creates **3 billion weights** in layer 1 alone! Instead, a **Convolutional Neural Network (`CNN`)** slides a tiny `3×3` "flashlight" (**kernel/filter**) across the image, reusing those exact 9 weights everywhere to spot edges in layer 1, textures in layer 2, and whole objects in deeper layers.

<details>
<summary><b>Deep Dive: Convolution Output Size Formula</b></summary>

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1$$

Where $K$ = kernel size (e.g., `3`), $P$ = zero-padding border (e.g., `1` keeps width unchanged when stride is `1`), and $S$ = stride step size.

</details>

**ResNet Skip Connections (He et al., 2015):** Learn a small correction $\mathcal{F}(x)$ added to the original input shortcut $x$:

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

In Tier 1, we handled text tokens and single-label photos separately. **Tier 2** brings everything together, locating multiple objects inside busy scenes (`IoU`/`mAP`), merging vision and language into one shared space (`ViT` & `CLIP`), and extending the exact same Transformer pipeline to live speech and audio (`Whisper`).

### 4. Vision Hierarchy (`IoU`, `mAP`, `ViT`) & Multimodal Alignment (`CLIP`)

#### 4.1 Detection/Segmentation Metrics (`IoU`, `mAP`), Vision Transformers (`ViT`) & `CLIP`
- **Classification** (`ConvNeXt`, `ViT`) -> single image label; evaluated via Accuracy / Top-5.
- **Object Detection** (`YOLO26`: native NMS-free edge flagship, `YOLO11`, `RT-DETR`) -> `[x, y, w, h, class, conf]` boxes; evaluated via **IoU** ($\frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$) and **mAP@0.5:0.95**.
- **Semantic vs. Instance Segmentation** (`SegFormer`, `SAM 2`) -> pixel-level class mask (`mIoU`) vs. distinct per-object instance masks (`Mask AP`).
- **`CNN` vs. `Vision Transformer (ViT)`:** CNNs have built-in "neighboring pixels belong together" intuition (win on smaller `<100k` image datasets); `ViT` chops an image into `16×16` square tiles like words in a sentence (wins at internet scale).
- **`CLIP` / `SigLIP 2` (Connecting Images & Text in One Map):** Imagine a memory-matching game with 32 photos and 32 captions on a table. Contrastive vision-language models (`CLIP`, Google's `SigLIP 2`) train an Image Encoder and a Text Encoder together so Photo #1's vector lands right on top of Caption #1's vector while pushing all 31 wrong captions away, unlocking instant image search and zero-shot labeling without retraining!

<details>
<summary><b>Deep Dive: `IoU` Formula & `CLIP` Symmetric `InfoNCE` Loss</b></summary>

- **Intersection over Union (`IoU`):** $\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$. Predictions with $\text{IoU} \ge 0.5$ (or swept across `0.50:0.95` for `mAP`) count as True Positives.
- **`InfoNCE` Contrastive Objective:** Computes the $N \times N$ cosine similarity matrix across a batch of $N$ `(image, text)` pairs and applies symmetric cross-entropy along rows and columns to maximize the $N$ diagonal matches against $N^2 - N$ negatives.

</details>

### 5. Audio & Speech AI (`Mel Spectrograms`, `Whisper` & Native Audio Models)

Having connected text and images, only one major human sense remains: **sound**. Remarkably, we don't even need a brand-new architecture: we convert sound waves into a 2D visual heatmap and hand it straight to the Transformer:

#### 5.1 Log-Mel Spectrograms, `Whisper` ASR & Native Neural Audio Codecs
- **In Plain English (Turning Sound Into a Picture):** How do you feed sound into a Transformer? First, slice the sound wave into tiny 25-millisecond windows and plot a **Log-Mel Spectrogram**: a "sheet-music heatmap" where **left-to-right is time**, **bottom-to-top is pitch (bass to treble)**, and **brightness is loudness**! Once sound is a 2D picture, vision/sequence Transformers can read it directly:
1. **Waveform to Log-Mel Spectrogram:** Raw 16 kHz audio converted via Short-Time Fourier Transform (STFT) into `80` or `128` human-ear pitch bands (`Mel` bins).
2. **`Whisper` Architecture (OpenAI):** Passes the Log-Mel spectrogram through 2 `1D Conv` layers + a **Transformer Encoder**, then decodes timestamps, language IDs, and transcribed/translated text autoregressively via a **Transformer Decoder**.
3. **Native Multimodal Audio Tokens:** Frontier voice models (`Gemini Live` / `gemini-3.1-flash-live`, OpenAI `gpt-realtime`) discretize audio directly into neural codec tokens (`SoundStream` / `EnCodec`), enabling low-latency speech-in, speech-out reasoning while preserving emotion, tone, and speaker prosody without a brittle `ASR -> Text LLM -> TTS` cascade.

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
- Know the **Scaled Dot-Product Attention** equation $\text{softmax}(QK^\top / \sqrt{d_k} + M)V$ cold, especially why $\frac{1}{\sqrt{d_k}}$ prevents softmax gradient saturation and how causal masking prevents future token leakage.
- Contrast **Encoder (`ModernBERT`/`Qwen3-Embedding`: embeddings/reranking)** vs. **Decoder (`Gemini 3`/`GPT-5`/`Llama 4`: generation)** vs. **Encoder-Decoder (`Whisper`)**.
- Explain **ResNet skip connections** via the `+1` identity gradient term, **`CLIP`** via symmetric InfoNCE contrastive alignment, and **Audio AI** via Log-Mel Spectrograms vs. native neural audio codecs.

> **Next in the Journey:** Now that you know how the **Decoder-Only Transformer** predicts tokens under the hood, how do we transform a raw internet-autocomplete LLM into an accurate, cited, fast enterprise system using **Context Engineering, `Qdrant` Hybrid RAG, `LoRA`/`DPO`, and `vLLM`**? Let's flow into **[Module 06: Generative AI, Fine-Tuning, `Qdrant` & Production RAG](./06_llms_finetuning_and_rag.md)**.

---

[← Previous: Module 04: Deep Learning & RL](../02-core-ml-and-dl/04_deep_learning_and_rl.md) | [Home](../../README.md) | [Next: Module 06: LLMs, Fine-Tuning & Advanced RAG →](./06_llms_finetuning_and_rag.md)
