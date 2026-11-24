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

## 2. Granular Resource Catalogs

Explore our curated resource collections categorized by topic:

- **[Essential Textbooks & Reference Books](./books.md):** Production ML systems, AI engineering, classical ML, and deep learning texts.
- **[High-Signal Courses & Video Lectures](./courses.md):** Andrej Karpathy's Zero to Hero, 3Blue1Brown, Stanford CS229/CS224N/CS25, and StatQuest.
- **[Landmark Research Papers](./research-papers.md):** Must-read foundational papers across Transformers, ResNets, CLIP, LoRA, RAG, ReAct, and vLLM.
- **[Modern Production Tools & Frameworks](./tools.md):** Open-source infrastructure for inference (vLLM, Ollama), vector databases (Qdrant), agent protocols (LangGraph, ADK, MCP), and MLOps (MLflow, Feast, Evidently).

---

[← Back to Roadmap](../README.md) | [Start Module 01 →](../curriculum/01-foundations/01_python_sql_and_data_stack.md)
