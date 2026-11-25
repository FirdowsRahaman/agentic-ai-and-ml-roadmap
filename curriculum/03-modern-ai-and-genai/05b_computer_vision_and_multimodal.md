---
module: "05B"
title: "Computer Vision & Multimodal AI"
difficulty: "intermediate"
estimated_hours: 16
prerequisites:
  mandatory:
    - "04A"
  recommended:
    - "05A"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - cnn-resnet
  - vision-transformers-vit
  - edge-detection-yolo
  - clip-multimodal-alignment
  - vision-language-models
  - audio-spectrograms
---

# Module 05B: Computer Vision & Multimodal AI

[← Previous: Module 05A](./05a_nlp_and_language_models.md) | [Phase 3 Index](./README.md) | [Next: Module 06A →](./06a_llm_architecture_and_inference.md)

---

## 1. Overview

Computer vision and audio processing have converged with NLP under the Transformer paradigm. This unit explores modern visual representation learning: Convolutional Neural Networks (ResNet), Vision Transformers (ViT), real-time edge detection (YOLOv10, YOLO11, RT-DETR), contrastive joint text-image alignment (CLIP), Vision-Language Models (VLMs), and audio spectrogram representations.

---

## 2. Who Should Study This?

- **Computer Vision Engineers** updating classical CNN pipelines to modern Vision Transformers and edge detection backbones.
- **Multimodal AI Engineers** building visual search, document OCR, and image-text retrieval engines.
- **Edge AI Practitioners** deploying low-latency vision models to embedded hardware or edge containers.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Formulate the Vision Transformer (ViT) architecture**, tracking how 2D images are patched, flattened, projected, and processed as sequence tokens.
- **Explain contrastive multimodal alignment (CLIP)** via symmetric InfoNCE loss across joint image-text vector spaces.
- **Deploy modern NMS-free edge object detection models** (YOLOv10, RT-DETR) for sub-10ms inference.
- **Process audio signals into Log-Mel spectrograms** for automatic speech recognition (Whisper).

---

## 4. Prerequisites

- **Deep Learning Mechanics:** Computational graphs and training loops from [Module 04A](../02-core-ml-and-dl/04a_deep_learning_and_pytorch.md).
- **Self-Attention Mechanics:** Scaled dot-product attention from [Module 05A](./05a_nlp_and_language_models.md).

---

## 5. Learning Levels

### Foundation
- Understand 2D image representations (height, width, RGB channels) and convolution operations.
- Explain residual skip connections in ResNet.

### Applied Engineering
- Fine-tune a pretrained Vision Transformer (`vit_b_16`) using PyTorch transfer learning.
- Implement zero-shot image classification using OpenAI CLIP embeddings.

### Production Engineering
- Export edge vision models to ONNX runtimes for optimized CPU/GPU deployment.
- Implement audio preprocessing pipelines extracting 80-bin Log-Mel spectrogram features from audio waveforms.

---

## 6. Core Concepts

### 6.1 Vision Transformers (ViT)
An image $X \in \mathbb{R}^{H \times W \times C}$ is sliced into non-overlapping $P \times P$ pixel patches (e.g. $16 \times 16$). Each patch is flattened into a 1D vector of dimension $P^2 C$, linearly projected into embedding dimension $D$, prepended with a `[CLS]` token, and processed through standard Transformer encoder blocks.

### 6.2 Contrastive Language-Image Pretraining (CLIP)
CLIP trains an image encoder and text encoder simultaneously using a symmetric cross-entropy loss over cosine similarities:
$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{image}\to\text{text}} + \mathcal{L}_{\text{text}\to\text{image}} \right)$$
This aligns images and natural language text into a shared metric embedding space, enabling zero-shot classification and semantic visual search.

---

## 7. Technical Deep Dive

### 7.1 Real-Time Edge Vision: Eliminating NMS Bottlenecks
Traditional object detectors (YOLOv5/v8) generate dense overlapping bounding boxes requiring Non-Maximum Suppression (NMS) during post-processing, introducing latency jitter on edge CPUs. Modern architectures (**YOLOv10**, **RT-DETR**) employ dual label assignments during training:
- **One-to-Many Branch:** Maximizes gradient signals during training.
- **One-to-One Branch:** Enables direct, NMS-free end-to-end inference with deterministic execution latency.

---

## 8. Practical Implementation

Below is a complete transfer learning implementation fine-tuning a pretrained Vision Transformer:

```python
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 1. Load Pretrained ViT Weights
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)

# 2. Freeze Pretrained Backbone Parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace Final Classification Head for 5 Custom Classes
num_classes = 5
in_features = model.heads.head.in_features
model.heads.head = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.GELU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, num_classes)
)

# 4. Verify Parameter Gradients
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Frozen Backbone Params: {frozen_params:,}")
print(f"Trainable Head Params:  {trainable_params:,}")

# 5. Forward Pass with Synthetic Image Batch (Batch=2, Channels=3, 224x224)
dummy_image_batch = torch.randn(2, 3, 224, 224)
output_logits = model(dummy_image_batch)
assert output_logits.shape == (2, 5)
print("ViT Forward Pass Successful! Logits shape:", output_logits.shape)
```

---

## 9. Hands-on Exercises

### Beginner
Calculate the exact number of $16 \times 16$ image patches produced by an input image of dimension $384 \times 384 \times 3$, including the `[CLS]` token.

### Intermediate
Implement zero-shot image classification using HuggingFace `transformers` CLIP model: classify a sample image against 3 candidate text labels.

### Advanced
Write an audio preprocessing function in Python extracting an 80-bin Log-Mel spectrogram from a 16kHz audio waveform using `librosa` or `torchaudio`.

---

## 10. Mini Project: Zero-Shot Visual Semantic Search Engine

Build a semantic image retrieval engine:
1. Embed 100 sample images using a pretrained CLIP image encoder.
2. Index image embeddings in an in-memory normalized cosine similarity matrix.
3. Accept arbitrary natural language text queries (e.g., "a red sports car parked in rain"), encode them with the CLIP text encoder, and return Top-3 matching images.

---

## 11. Production Considerations

- **Resolution Constraints:** Vision Transformers are sensitive to image resolution. Resizing non-square images without preserving aspect ratio introduces distortion; always use letterbox padding.
- **Edge Deployment:** When deploying vision models to mobile or edge devices, quantize weights to INT8 and convert architectures to ONNX or TensorRT.

---

## 12. Common Mistakes & Gotchas

1. **Ignoring Input Normalization:** Feeding unnormalized $[0, 255]$ pixel values into a ViT trained on ImageNet mean/std causes complete failure of attention activations.
2. **Fine-Tuning All ViT Layers on Tiny Datasets:** Training all 86M parameters on a 500-image dataset causes immediate, severe overfitting. Always freeze the backbone first.

---

## 13. Interview Questions & System Design

1. **Q:** Why do Vision Transformers require more pretraining data than Convolutional Neural Networks to achieve peak performance?  
   **A:** CNNs possess hard-coded inductive biases: translation equivariance and locality (pixels close together are related). Vision Transformers possess minimal spatial inductive bias, meaning they must learn 2D spatial relationships purely from data across millions of images.
2. **System Design:** Design an enterprise document analysis pipeline that processes 50,000 PDF invoices per hour, extracting text and tabular structures using a Vision-Language Model.

---

## 14. Recommended Resources

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al.
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al.
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/).

---

## 15. Completion Checklist

- [ ] Can explain how 2D images are converted to 1D patch tokens in ViT.
- [ ] Understand symmetric InfoNCE loss in CLIP.
- [ ] Understand NMS-free edge detection in YOLOv10 and RT-DETR.
- [ ] Understand audio waveform to Log-Mel spectrogram conversion.

---

## 16. Further Exploration

- Study Vision-Language Models (VLM) architectures such as LLaVA and PaliGemma.
- Explore Whisper ASR encoder-decoder architecture.
