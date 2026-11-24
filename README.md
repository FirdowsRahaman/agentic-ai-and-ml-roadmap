<div align="center">

<img src="assets/icons/ai.svg" width="76" height="76" alt="Agentic AI and ML Engineering Roadmap Logo" />

# Agentic AI & ML Engineering Roadmap

**From Mathematical & Algorithmic Foundations to Production-Grade Agentic Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Validation: Passing](https://img.shields.io/badge/CI%2FCD-Passing-brightgreen.svg)](https://github.com/FirdowsRahaman/agentic-ai-and-ml-roadmap/actions/workflows/ci-verify.yml)
[![Docs: MkDocs](https://img.shields.io/badge/Docs-MkDocs%20Material-526CFE.svg?logo=materialformkdocs&logoColor=white)](docs/index.md)
[![Curriculum Modules](https://img.shields.io/badge/Curriculum-13%20Modules-8A2BE2.svg)](ROADMAP.md)
[![GitHub Stars](https://img.shields.io/github/stars/FirdowsRahaman/agentic-ai-and-ml-roadmap?style=flat&color=yellow)](https://github.com/FirdowsRahaman/agentic-ai-and-ml-roadmap)
[![GitHub Forks](https://img.shields.io/github/forks/FirdowsRahaman/agentic-ai-and-ml-roadmap?style=flat&color=blue)](https://github.com/FirdowsRahaman/agentic-ai-and-ml-roadmap/network/members)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<br/>

<img src="assets/banner.webp" alt="Agentic AI and ML Engineering Roadmap Hero Banner" width="100%" />

<br/>

> **"From mathematical and algorithmic first principles to production-grade distributed Agentic AI systems."**

A disciplined, production-first engineering curriculum designed to build reliable, high-performance Machine Learning and Agentic AI systems.

<br/>

[Explore Roadmap](./ROADMAP.md) • [Getting Started](./docs/getting-started.md) • [Learning Paths](./docs/learning-paths.md) • [Portfolio Projects](./projects/README.md) • [Interview Kits](./interview-kits/README.md)

</div>

---

## <img src="assets/icons/toc.svg" width="22" height="22" align="absmiddle" /> Table of Contents <a id="table-of-contents"></a>

- <img src="assets/icons/compass.svg" width="16" height="16" align="absmiddle" /> [Who This Roadmap Is For](#who-this-roadmap-is-for)
- <img src="assets/icons/target.svg" width="16" height="16" align="absmiddle" /> [What You Will Build & Master](#what-you-will-build--master)
- <img src="assets/icons/roadmap.svg" width="16" height="16" align="absmiddle" /> [High-Level Learning Roadmap](#high-level-learning-roadmap)
- <img src="assets/icons/index.svg" width="16" height="16" align="absmiddle" /> [Quick Navigation](#quick-navigation)
- <img src="assets/icons/apps.svg" width="16" height="16" align="absmiddle" /> [Project Highlights ($0 API Cost Starters)](#project-highlights-0-api-cost-starters)
- <img src="assets/icons/rocket.svg" width="16" height="16" align="absmiddle" /> [How to Get Started in 3 Steps](#how-to-get-started-in-3-steps)
- <img src="assets/icons/contribute.svg" width="16" height="16" align="absmiddle" /> [Contributing](#contributing)
- <img src="assets/icons/license.svg" width="16" height="16" align="absmiddle" /> [License](#license)

---

## <img src="assets/icons/compass.svg" width="22" height="22" align="absmiddle" /> Who This Roadmap Is For <a id="who-this-roadmap-is-for"></a>

- **Software & Backend Engineers** seeking to transition into AI/ML Engineering without getting lost in shallow API wrapper hype.
- **Data Scientists & Classical ML Engineers** advancing to Generative AI, RAG systems, and Autonomous Agent orchestration.
- **AI Engineers & Architects** designing enterprise multi-agent platforms, Model Context Protocol (MCP) tool servers, and production MLOps pipelines.

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/target.svg" width="22" height="22" align="absmiddle" /> What You Will Build & Master <a id="what-you-will-build--master"></a>

- **Algorithmic Mastery:** Implement Autograd, Backpropagation, Scaled Dot-Product Attention, and Numerically Stable Softmax from scratch in pure NumPy.
- **Enterprise Retrieval (RAG):** Design hybrid vector search using dense cosine embeddings and sparse BM25 fused via Reciprocal Rank Fusion (RRF) in Qdrant.
- **Autonomous Systems:** Construct stateful cyclical multi-agent workflows using LangGraph and Google Agent Development Kit (ADK) with durable human-in-the-loop (HITL) checkpoints.
- **Distributed Tool Protocols:** Build decoupled, secure tool backends using the open Model Context Protocol (MCP) JSON-RPC 2.0 standard.
- **Production AI Engineering:** Eliminate pickle security vulnerabilities with safe `.npz` serialization, serve low-latency FastAPI endpoints, and monitor data drift with automated Population Stability Index (PSI) calculations.

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/roadmap.svg" width="22" height="22" align="absmiddle" /> High-Level Learning Roadmap <a id="high-level-learning-roadmap"></a>

The curriculum is structured into 6 progressive tiers. Each module contains measurable learning objectives, production code, failure modes, and hands-on exercises:

```mermaid
flowchart TD
    T1["1. Engineering Foundations<br/>(Core: Python 3.12, Vectorized NumPy, Modern SQL, Linear Algebra & Probability)"]
    T2["2. Classical Machine Learning<br/>(Core: Leak-Free Preprocessing, Ensembles, Imbalance, Two-Tower RecSys)"]
    T3["3. Deep Learning & Neural Systems<br/>(Intermediate: PyTorch Loops, Autograd Backprop, Normalization & Optimization)"]
    T4["4. Generative AI & Foundation Models<br/>(Intermediate: Transformers, LoRA/DPO, Qdrant Hybrid RAG & vLLM PagedAttention)"]
    T5["5. Autonomous Agents & MCP Protocols<br/>(Advanced: ReAct Loops, Cyclical LangGraph, FastMCP Servers & Safety Guardrails)"]
    T6["6. Production AI Engineering & MLOps<br/>(Advanced: FastAPI Serving, PSI Drift, HITL Governance & Cloud FinOps)"]

    T1 --> T2 --> T3 --> T4 --> T5 --> T6

    classDef c fill:#E0F2FE,stroke:#0284C7,stroke-width:1.5px,color:#0369A1;
    classDef i fill:#EDE9FE,stroke:#7C3AED,stroke-width:1.5px,color:#4C1D95;
    classDef a fill:#FEF3C7,stroke:#D97706,stroke-width:1.5px,color:#92400E;

    class T1,T2 c;
    class T3,T4 i;
    class T5,T6 a;
```

> 📖 **[View the Complete Module-by-Module Sequence & Prerequisite Graph in ROADMAP.md →](./ROADMAP.md)**

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/index.svg" width="22" height="22" align="absmiddle" /> Quick Navigation <a id="quick-navigation"></a>

| Resource | Purpose | Direct Link |
|---|---|---|
| <img src="assets/icons/roadmap.svg" width="16" height="16" align="absmiddle" /> **Curriculum Roadmap** | Detailed breakdown of all 13 modules across 6 tiers | [ROADMAP.md](./ROADMAP.md) |
| <img src="assets/icons/compass.svg" width="16" height="16" align="absmiddle" /> **Learning Paths** | Personalized tracks (Foundations, GenAI, Agents, MLOps) | [docs/learning-paths.md](./docs/learning-paths.md) |
| <img src="assets/icons/layers.svg" width="16" height="16" align="absmiddle" /> **Prerequisites Guide** | Self-assessment checklist and math/programming refresher | [docs/prerequisites.md](./docs/prerequisites.md) |
| <img src="assets/icons/apps.svg" width="16" height="16" align="absmiddle" /> **Portfolio Projects** | 10 real-world project blueprints (4 local, 6 enterprise cloud) | [projects/README.md](./projects/README.md) |
| <img src="assets/icons/tools.svg" width="16" height="16" align="absmiddle" /> **Runnable Starters** | Fully working, local $0 API cost codebases with pytest suites | [projects/intermediate/](./projects/intermediate/README.md) |
| <img src="assets/icons/brain.svg" width="16" height="16" align="absmiddle" /> **Interactive Labs** | Jupyter Notebooks for vectorized math, ML, and transformers | [notebooks/foundations/](./notebooks/foundations/01_numpy_pandas_sql_lab.ipynb) |
| <img src="assets/icons/business.svg" width="16" height="16" align="absmiddle" /> **Interview Preparation** | System design, technical screenings, and resume CAR guide | [interview-kits/README.md](./interview-kits/README.md) |
| <img src="assets/icons/api.svg" width="16" height="16" align="absmiddle" /> **Curated Resources** | Textbooks, courses, landmark research papers, and open-source tools | [resources/recommended_resources.md](./resources/recommended_resources.md) |
| <img src="assets/icons/index.svg" width="16" height="16" align="absmiddle" /> **Technical Glossary** | Production terminology (RRF, PSI, HITL, MCP, PagedAttention) | [docs/glossary.md](./docs/glossary.md) |

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/apps.svg" width="22" height="22" align="absmiddle" /> Project Highlights ($0 API Cost Starters) <a id="project-highlights-0-api-cost-starters"></a>

You can clone and run complete production-grade systems immediately on your laptop without registering credit cards or generating paid API tokens:

- **[Starter 1: Air-Gapped Desktop Hybrid RAG & SQL Router Agent](./projects/intermediate/starter-local-rag-agent/README.md)**  
  *Stack:* Python 3.12+, Pure NumPy, SQLite in-memory, BM25 + Dense Cosine Vector Fusion  
  Combines local lexical and semantic document search with read-only SQL execution in an isolated offline environment.

- **[Starter 2: Full-Lifecycle Production MLOps & PSI Drift Engine](./projects/intermediate/starter-local-mlops/README.md)**  
  *Stack:* FastAPI, Uvicorn, Scikit-Learn, NumPy, Docker, Pytest  
  Implements zero-pickle `.npz` model persistence, leak-free training, REST prediction endpoints, automated Population Stability Index (PSI) drift monitoring, and non-root container deployment.

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/rocket.svg" width="22" height="22" align="absmiddle" /> How to Get Started in 3 Steps <a id="how-to-get-started-in-3-steps"></a>

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FirdowsRahaman/agentic-ai-and-ml-roadmap.git
   cd agentic-ai-and-ml-roadmap
   ```

2. **Select your learning path:**
   Review the [Learning Paths Guide](./docs/learning-paths.md) to pick the track suited to your goals (e.g. Applied ML, Generative AI, or Agent Systems).

3. **Verify the environment and run your first lab:**
   ```bash
   python scripts/validate_structure.py
   python scripts/validate_links.py
   ```

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/contribute.svg" width="22" height="22" align="absmiddle" /> Contributing <a id="contributing"></a>

We welcome contributions from engineers, researchers, and educators! Please read our [Contribution Guidelines](./CONTRIBUTING.md) for details on code style, link validation, notebook requirements, and the pull request process.

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)

---

## <img src="assets/icons/license.svg" width="22" height="22" align="absmiddle" /> License <a id="license"></a>

This project is open-source under the [MIT License](./LICENSE).

[<img src="assets/icons/arrow-up.svg" width="14" height="14" align="absmiddle" /> Back to Top](#table-of-contents)
