# Phase 3: Modern AI & Generative AI

[← Back to Portal](../../README.md) | [Roadmap](../../ROADMAP.md) | [Diagnostic Assessment](../../docs/assessment-guide.md)

---

## Overview

Phase 3 is the generative and representation learning powerhouse of the curriculum. It unifies natural language processing, computer vision, foundation models, retrieval systems, and parameter-efficient serving into 5 specialized engineering units:
- **05A (NLP & Language Models):** Subword BPE tokenizers, dense embeddings, BM25 lexical retrieval, and Transformer self-attention.
- **05B (Computer Vision & Multimodal AI):** Vision Transformers (ViT), edge detection (YOLOv10/11, RT-DETR), CLIP multimodal contrastive alignment, and audio spectrograms.
- **06A (LLM Architecture & Inference):** Transformer decoder internals, RoPE relative embeddings, KV cache memory scaling, decoding strategies, and constrained grammar generation.
- **06B (RAG Engineering):** Enterprise hybrid retrieval, Qdrant vector indexing, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and retrieval evaluation (Recall@K, MRR).
- **06C (Fine-Tuning, Serving & LLMOps):** LoRA/QLoRA parameter math, vLLM PagedAttention continuous batching, TTFT/TPOT latency benchmarking, and semantic caching.

---

## Learning Units in Phase 3

| Unit ID | Title | Difficulty | Est. Hours | Key Hands-on Lab |
|---|---|---|---|---|
| **[Module 05A](./05a_nlp_and_language_models.md)** | NLP & Language Modeling Foundations | Intermediate | 14 Hours | Scaled Dot-Product Attention in pure NumPy |
| **[Module 05B](./05b_computer_vision_and_multimodal.md)** | Computer Vision & Multimodal AI | Intermediate | 16 Hours | CLIP joint text-image embedding alignment & ViT fine-tuning |
| **[Module 06A](./06a_llm_architecture_and_inference.md)** | LLM Architecture & Inference Engines | Advanced | 16 Hours | Autoregressive KV cache decoder & structured JSON grammar parsing |
| **[Module 06B](./06b_rag_engineering.md)** | Retrieval-Augmented Generation (RAG) Engineering | Advanced | 18 Hours | Air-Gapped Hybrid RAG (Qdrant BM25 + Dense Cosine via RRF) |
| **[Module 06C](./06c_finetuning_and_serving.md)** | Fine-Tuning, Serving & LLMOps | Advanced | 16 Hours | LoRA rank parameter adapter math & vLLM TTFT/TPOT benchmark |

---

## Capstone Project for Phase 3
- **[Air-Gapped Hybrid RAG & SQL Router Agent](../../projects/intermediate/starter-local-rag-agent/README.md):** Combines local BM25 lexical search with dense vector embeddings and isolated read-only SQLite routing.
