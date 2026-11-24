# Landmark Research Papers

[← Back to Resources](./recommended_resources.md) | [Back to Roadmap](../README.md)

---

## 1. Architectural Foundations
1. **Transformers:** *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* (Vaswani et al., 2017)  
   Introduced the pure self-attention mechanism, discarding recurrent and convolutional layers for sequence modeling.
2. **Residual Networks:** *[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)* (He et al., 2015)  
   Solved vanishing gradients in deep architectures using skip/identity connections.
3. **Multimodal Alignment:** *[Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)* (Radford et al., 2021)  
   Introduced contrastive image-text representation learning.

## 2. Generative AI, RAG & Adaptation
4. **Retrieval-Augmented Generation:** *[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)* (Lewis et al., 2020)  
   Combined pre-trained parametric memory with non-parametric dense vector index retrieval.
5. **Parameter-Efficient Tuning (LoRA):** *[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)* (Hu et al., 2021)  
   Froze pre-trained weights and injected trainable rank decomposition matrices into transformer layers.
6. **Direct Preference Optimization:** *[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)* (Rafailov et al., 2023)  
   Eliminated separate reward model training in RLHF by optimizing language model likelihood directly on preference pairs.

## 3. Agentic Systems & High-Throughput Serving
7. **Reasoning & Action (ReAct):** *[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)* (Yao et al., 2022)  
   Interleaved reasoning traces and task-specific actions for tool invocation.
8. **PagedAttention & vLLM:** *[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)* (Kwon et al., 2023)  
   Virtual memory paging applied to KV caches, increasing serving throughput 2x–4x.
9. **Agentic Workflows:** *[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)* (Anthropic, 2024)  
   Pragmatic design principles for deterministic workflows versus autonomous agent loops.
