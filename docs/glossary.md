# Production AI & Agent Systems Technical Glossary

[← Back to Documentation](./index.md) | [Roadmap](../ROADMAP.md)

---

## A
- **A2A (Agent-to-Agent):** Communication protocols allowing autonomous software agents to negotiate, delegate tasks, and exchange structured artifacts.
- **AST (Abstract Syntax Tree):** A tree representation of source code structure used in this curriculum to parse and evaluate mathematical expressions safely without allowing arbitrary Python code execution.

## B
- **BM25 (Best Matching 25):** A probabilistic ranking function used in information retrieval that estimates the relevance of documents to a given search query based on term frequency and document length normalization.
- **BPE (Byte-Pair Encoding):** A subword tokenization algorithm that iteratively replaces the most frequent pairs of bytes in a text with a single, unused byte.

## C
- **CAR (Context-Action-Result):** A structured framework for framing technical achievements on engineering resumes and in behavioral interviews.
- **CLIP (Contrastive Language-Image Pre-Training):** A multimodal neural network trained on image-caption pairs to embed visual and textual representations into a shared semantic space.

## D
- **DPO (Direct Preference Optimization):** A stable, non-RL alternative to RLHF that optimizes foundation language models directly on human preference pairs using implicit reward formulation.

## H
- **HITL (Human-in-the-Loop):** A governance pattern where an autonomous agent workflow suspends execution and persists state until an authorized human approves or edits the proposed action.
- **HNSW (Hierarchical Navigable Small World):** A multi-layer graph-based algorithm for fast approximate nearest neighbor (ANN) vector search.

## L
- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning technique that freezes base model weights and injects trainable low-rank rank decomposition matrices into the multi-head attention projection layers.

## M
- **MCP (Model Context Protocol):** An open standard JSON-RPC 2.0 protocol introduced by Anthropic that allows foundation models to interact with isolated, modular tools, prompts, and resources.

## P
- **PagedAttention:** A memory management algorithm designed by vLLM that partitions the Key-Value (KV) cache into non-contiguous physical memory blocks, reducing fragmentation and increasing batching throughput.
- **PSI (Population Stability Index):** A statistical metric measuring the shift between a baseline reference distribution and a production inference distribution:
  $$PSI = \sum_{i=1}^k (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$
  - $PSI < 0.10$: No significant change.
  - $0.10 \le PSI < 0.25$: Moderate drift (investigate).
  - $PSI \ge 0.25$: Significant population drift (trigger automatic retraining).

## R
- **ReAct (Reasoning + Acting):** An agentic prompting pattern that interleaves thought generation ("Reasoning") with tool invocation ("Acting") and environment feedback observation.
- **RRF (Reciprocal Rank Fusion):** A rank aggregation algorithm combining disparate retrieval lists (e.g. sparse BM25 and dense cosine) into a unified scored list:
  $$RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$

## T
- **TPOT (Time Per Output Token):** The average time taken to generate each subsequent token during autoregressive LLM decoding.
- **TTFT (Time to First Token):** The latency duration between sending an input prompt to an inference server and receiving the very first output token.
