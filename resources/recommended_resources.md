# Curated High-Signal Resources for AI/ML Engineers

[← Back to Roadmap](../README.md) | [Start Module 01 →](../curriculum/01-foundations/01_python_sql_and_data_stack.md) | [Explore Projects →](../projects/README.md)

---

## 1. Industry Trends and Recommended Skills

Recent industry shifts (as highlighted in reports like the [Stanford AI Index](https://aiindex.stanford.edu/report/) and [O'Reilly AI Adoption Reports](https://www.oreilly.com/radar/topics/ai/)) have redefined the expectations for AI and Machine Learning practitioners. Generalist scripting is giving way to disciplined, end-to-end systems engineering.

| Traditional / De-Emphasized Focus | High-Signal Industry Focus | Supporting References & Standards |
|---|---|---|
| Memorizing legacy architectures (Word2Vec, Bag-of-Words, basic RNNs) | **Subword Tokenizers (BPE), Hybrid Retrieval (Dense + BM25 RRF), Cross-Encoders, & Vision-Language Transfer** | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), [Radford et al. (CLIP, 2021)](https://arxiv.org/abs/2103.00020) |
| Standalone prompt tweaking & single-turn chat wrappers | **Context Engineering: KV prefix-caching, structured schema enforcement (Pydantic / JSON schema), and token budgeting** | [Anthropic: Prompt Engineering Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) |
| Standalone Jupyter notebooks without tests, APIs, or containerization | **Production Deployment: Type-safe APIs (FastAPI), data drift monitoring (PSI), non-root Docker, and CI/CD validation** | [Chip Huyen: Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) |
| Unverified agent loops without test harnesses | **Eval-Driven Development (EDD) with golden evaluation sets, stateful workflows, and standard protocols (MCP)** | [Model Context Protocol Specification](https://modelcontextprotocol.io/), [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) |
| Unreviewed code generation | **Rigorous code auditing for silent data leakage, async bottlenecks, and security vulnerabilities** | [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) |

---

## 2. Essential Textbooks

- **[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** & **[AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)** by Chip Huyen: Production MLOps, feature stores, drift detection, RAG architecture, evaluation, and system design.
- **[Hands-On Machine Learning with Scikit-Learn, Keras, and PyTorch](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** by Aurélien Géron: Practical implementation reference for classical ML, ensembles, and neural networks.
- **[Understanding Deep Learning](https://udlbook.github.io/udlbook/)** by Simon J.D. Prince *(Free PDF)*: Mathematical and visual treatment of Transformers, Diffusion, Residual Networks, and Reinforcement Learning.

---

## 3. Video Courses & Visual Intuition

- **[Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**: Building micrograd, makemore, GPT-2, and tokenizers from scratch in Python/PyTorch.
- **[3Blue1Brown: Essence of Linear Algebra & Deep Learning](https://www.3blue1brown.com/)**: Geometric and visual intuition for matrices, backpropagation, and Transformer attention.
- **[StatQuest with Josh Starmer](https://statquest.org/)**: Visual step-by-step breakdowns of XGBoost, PCA, ROC/PR curves, and fundamental statistics.
- **[Made With ML](https://madewithml.com/)** by Goku Mohandas: Production engineering lifecycle: packaging, testing, tracking, serving, and CI/CD.

---

## 4. Landmark Papers Every Engineer Should Know

1. **Transformers:** *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* (Vaswani et al., 2017)
2. **Residual Networks:** *[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)* (He et al., 2015)
3. **Multimodal Alignment (CLIP):** *[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)* (Radford et al., 2021)
4. **Parameter-Efficient Fine-Tuning (LoRA):** *[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)* (Hu et al., 2021)
5. **Retrieval-Augmented Generation:** *[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)* (Lewis et al., 2020)
6. **Agentic Reasoning (ReAct):** *[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)* (Yao et al., 2022)
7. **High-Throughput LLM Serving:** *[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)* (Kwon et al., 2023)
8. **Agent Architecture & Interoperability:** *[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)* & *[Model Context Protocol](https://modelcontextprotocol.io/)*

---

## 5. High-Impact Open-Source Codebases

- **Inference & Vector Search:** [vllm-project/vllm](https://github.com/vllm-project/vllm) • [qdrant/qdrant](https://github.com/qdrant/qdrant) • [ollama/ollama](https://github.com/ollama/ollama)
- **Agent Frameworks & Protocols:** [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) • [google/adk-python](https://github.com/google/adk-python) • [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
- **Data Engineering & MLOps:** [pola-rs/polars](https://github.com/pola-rs/polars) • [duckdb/duckdb](https://github.com/duckdb/duckdb) • [mlflow/mlflow](https://github.com/mlflow/mlflow) • [feast-dev/feast](https://github.com/feast-dev/feast)

---

[← Back to Roadmap](../README.md) | [Start Module 01 →](../curriculum/01-foundations/01_python_sql_and_data_stack.md)
