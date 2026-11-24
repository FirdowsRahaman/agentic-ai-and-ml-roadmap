# Implementation Backlog & 18-Unit Curriculum Architecture

**Target Repository:** `FirdowsRahaman/agentic-ai-and-ml-roadmap`  
**Target Milestone:** Release v1.1.0 Architecture & Multi-Track Learning Platform  
**Architecture:** 5 Progressive Phases · 18 Specialized Learning Units · Dual Entry Routes

---

## 1. Executive Summary & Curriculum Positioning

The **Agentic AI & ML Engineering Roadmap** is positioned as a practical, engineering-first AI/ML curriculum that transitions learners from programming and mathematical foundations to building, evaluating, deploying, and maintaining production-grade ML, GenAI, and distributed Agentic AI systems.

### Dual Audience & Two Entry Routes
The curriculum provides two distinct entry pathways through a standardized diagnostic assessment ([`docs/assessment-guide.md`](../docs/assessment-guide.md)) without requiring divergent, unmaintainable repository copies:
1. **Beginner Route (Guided Journey, 22–28 Weeks):** Linear progression from Python basics, linear algebra geometry, and SQL through to deep learning and production agent deployment.
2. **Experienced Developer Route (Accelerated Track, 12–16 Weeks):** Diagnostic bypass of foundations directly into applied ML pipelines, LoRA fine-tuning, LangGraph state machines, FastMCP tool servers, and production MLOps.

---

## 2. 5-Phase & 18-Unit Architecture Matrix

| Phase | Directory | Unit ID | Module Title | Difficulty | Est. Hours |
|---|---|---|---|---|---|
| **Phase 1: Foundations** | `curriculum/01-foundations/` | `01` | [Python, Software Engineering, SQL & Data](../curriculum/01-foundations/01_python_sql_and_data_stack.md) | Core | 14 |
| | | `02` | [Mathematics & Statistics for AI Engineers](../curriculum/01-foundations/02_math_and_statistics.md) | Core | 16 |
| **Phase 2: Core ML & DL** | `curriculum/02-core-ml-and-dl/` | `03` | [Classical Machine Learning & RecSys](../curriculum/02-core-ml-and-dl/03_classical_ml_and_recsys.md) | Intermediate | 18 |
| | | `04A` | [Core Deep Learning & PyTorch Loops](../curriculum/02-core-ml-and-dl/04a_deep_learning_and_pytorch.md) | Intermediate | 16 |
| | | `04B` | [Advanced Deep Learning & RL](../curriculum/02-core-ml-and-dl/04b_advanced_dl_and_rl.md) | Advanced | 16 |
| **Phase 3: Modern AI & GenAI** | `curriculum/03-modern-ai-and-genai/` | `05A` | [NLP & Language Modeling Foundations](../curriculum/03-modern-ai-and-genai/05a_nlp_and_language_models.md) | Intermediate | 14 |
| | | `05B` | [Computer Vision & Multimodal AI](../curriculum/03-modern-ai-and-genai/05b_computer_vision_and_multimodal.md) | Intermediate | 16 |
| | | `06A` | [LLM Architecture & Inference Engines](../curriculum/03-modern-ai-and-genai/06a_llm_architecture_and_inference.md) | Advanced | 16 |
| | | `06B` | [Retrieval-Augmented Generation (RAG) Engineering](../curriculum/03-modern-ai-and-genai/06b_rag_engineering.md) | Advanced | 18 |
| | | `06C` | [Fine-Tuning, Serving & LLMOps](../curriculum/03-modern-ai-and-genai/06c_finetuning_and_serving.md) | Advanced | 16 |
| **Phase 4: Agentic AI** | `curriculum/04-agentic-ai/` | `07` | [AI Agent Fundamentals & Architecture](../curriculum/04-agentic-ai/07_agent_fundamentals.md) | Intermediate | 16 |
| | | `08` | [Agent Orchestration & Workflow Engineering](../curriculum/04-agentic-ai/08_agent_orchestration.md) | Advanced | 20 |
| | | `09` | [Multi-Agent Systems, MCP & A2A](../curriculum/04-agentic-ai/09_multi_agent_mcp_a2a.md) | Advanced | 20 |
| | | `10` | [Agent Evals, Safety, Security & FinOps](../curriculum/04-agentic-ai/10_agent_evaluation_and_safety.md) | Advanced | 18 |
| **Phase 5: Production & Practice** | `curriculum/05-production-and-practice/` | `11` | [Production ML, GenAI & AgentOps](../curriculum/05-production-and-practice/11_production_ai_engineering.md) | Advanced | 22 |
| | | `12` | [Incident Troubleshooting & Diagnostic Playbooks](../curriculum/05-production-and-practice/12_incident_and_troubleshooting_labs.md) | Advanced | 16 |
| | | `13A` | [Machine Learning Algorithms from Scratch](../curriculum/05-production-and-practice/13_algorithms_and_patterns_from_scratch/13a_ml_algorithms.md) | Advanced | 16 |
| | | `13B` | [AI Systems Engineering Patterns from Scratch](../curriculum/05-production-and-practice/13_algorithms_and_patterns_from_scratch/13b_system_patterns.md) | Advanced | 14 |

---

## 3. Milestones & Task Backlog (CUR-001 to CUR-052)

### Milestone 1 — Fix Correctness & Broken Navigation (P0)
- [x] **CUR-001:** Verify all README module links and validate zero 404 responses.
- [x] **CUR-002:** Fix Module 01 streaming example with genuine chunked iterator / file stream (`io.StringIO` / file handle).
- [x] **CUR-003:** Fix Module 01 SQL schema example with complete DDL (`tx_id`, `user_id`, `amount_usd`, `created_at`, `updated_at`).
- [x] **CUR-004:** Fix Module 02 P99 tail latency example with exact quantile methodology and outlier distribution.
- [x] **CUR-005:** Add comprehensive Calculus & Gradients coverage to Module 02 (derivatives, partials, chain rule, Jacobians).
- [x] **CUR-006:** Fix A/B testing edge cases with defensive input validation, division-by-zero guards, and power sizing.
- [x] **CUR-007:** Review README market claims ensuring evidence-based, evergreen engineering terminology.
- [x] **CUR-008:** Audit and categorize all external resource links.

### Milestone 2 — Complete Foundation Content (P1)
- [x] **CUR-009:** Expand Python engineering in Module 01 (OOP, typing, dataclasses, context managers).
- [x] **CUR-010:** Integrate Developer Tooling lessons (Git branching, shell scripting, Linux process isolation).
- [x] **CUR-011:** Expand Data Engineering foundations (Parquet columnar storage, Arrow zero-copy memory, DuckDB).
- [x] **CUR-012:** Expand Linear Algebra in Module 02 (eigenvectors, SVD decomposition, PCA projections).
- [x] **CUR-013:** Expand Probability distributions (Multivariate Normal, Poisson, Bernoulli, Bayesian updates).
- [x] **CUR-014:** Expand Statistical Inference (confidence intervals, two-tailed Z-tests, power analysis).
- [x] **CUR-015:** Expand Optimization mathematics (SGD, Momentum, Adam, L1/L2 regularization penalties).
- [x] **CUR-016:** Improve Module 03 Classical ML (leak-free pipelines, XGBoost/LightGBM, cost-sensitive matrix).
- [x] **CUR-017:** Partition Module 04 into 04A (Core PyTorch) and 04B (Advanced DL & RL).

### Milestone 3 — Restructure Modern AI & GenAI (P1)
- [x] **CUR-018:** Partition Module 05 into 05A (NLP Foundations) and 05B (Computer Vision & Multimodal).
- [x] **CUR-019:** Author dedicated Module 06A on LLM Architecture, RoPE, KV cache, and decoding.
- [x] **CUR-020:** Author dedicated Module 06B on RAG Engineering, Hybrid Qdrant, RRF, and chunking.
- [x] **CUR-021:** Author dedicated Module 06C on Fine-Tuning (LoRA/QLoRA), vLLM serving, and TTFT/TPOT.
- [x] **CUR-022:** Add retrieval evaluation framework (Recall@K, MRR, NDCG).
- [x] **CUR-023:** Add inference latency & throughput benchmarking lab.
- [x] **CUR-024:** Add reproducible parameter-efficient fine-tuning lab.

### Milestone 4 — Strengthen Agentic AI (P1)
- [x] **CUR-025:** Expand Agent Architecture patterns in Module 07 (ReAct loops, planning, execution limits).
- [x] **CUR-026:** Build bounded agent loop lab with AST sandbox security.
- [x] **CUR-027:** Expand Module 08 on cyclical LangGraph state machines and checkpoint persistence.
- [x] **CUR-028:** Add durable workflow recovery and Human-In-The-Loop (HITL) approval gates.
- [x] **CUR-029:** Expand Google Agent Development Kit (ADK) session management and multi-tool orchestration.
- [x] **CUR-030:** Author Module 09 FastMCP JSON-RPC 2.0 client/server decoupling.
- [x] **CUR-031:** Implement Agent-to-Agent (A2A) distributed collaboration protocol.
- [x] **CUR-032:** Add multi-agent trajectory evaluation and benchmarking harness.
- [x] **CUR-033:** Expand Module 10 security against indirect prompt injection and tool poisoning.
- [x] **CUR-034:** Establish golden evaluation datasets and deterministic regression suites.

### Milestone 5 — Production Engineering & Practice (P1)
- [x] **CUR-035:** Author Module 11 Production APIs with FastAPI, Pydantic v2, and non-root Docker containers.
- [x] **CUR-036:** Implement zero-pickle model serialization with `.npz` arrays.
- [x] **CUR-037:** Build CI/CD verification workflows with automated pytest test runners.
- [x] **CUR-038:** Expand real-time Population Stability Index (PSI) drift monitoring and retraining triggers.
- [x] **CUR-039:** Implement LLMOps token budget monitoring, semantic caching, and routing.
- [x] **CUR-040:** Implement AgentOps execution replay and trace visualization.
- [x] **CUR-041:** Incorporate production infrastructure (Redis cache, DuckDB, async job queues).
- [x] **CUR-042:** Reorganize Module 12 into 15 real-world production incident triage trees.
- [x] **CUR-043:** Split Module 13 into 13A (ML Algorithms from Scratch) and 13B (Systems Patterns from Scratch).
- [x] **CUR-044:** Specify end-to-end Enterprise AI Research & Operations Capstone application.

### Milestone 6 — Documentation, Standardization & Release (P2)
- [x] **CUR-045:** Standardize all 18 learning units to the unified 16-section pedagogical template.
- [x] **CUR-046:** Add beginner and experienced developer diagnostic assessment in `docs/assessment-guide.md`.
- [x] **CUR-047:** Construct portfolio project progression ladder (Beginner, Intermediate, Advanced, Capstone).
- [x] **CUR-048:** Enforce standardized YAML topic metadata frontmatter across all curriculum units.
- [x] **CUR-049:** Validate all Jupyter Notebooks with isolated cell runners.
- [x] **CUR-050:** Validate documentation build with MkDocs Material.
- [x] **CUR-051:** Document curriculum maintenance and version release policy.
- [x] **CUR-052:** Tag and publish curriculum v1.1.0.

---

## 4. Module Quality Scorecard Standard

Every learning unit must achieve a minimum score of **80 / 100** before being certified:

| Evaluation Dimension | Weight | Acceptance Criteria |
|---|---|---|
| **Conceptual Correctness** | 20% | Mathematically rigorous definitions, accurate architectural trade-offs, zero pseudo-scientific claims. |
| **Learning Progression** | 15% | Explicit prerequisites, Bloom's taxonomy objectives, three-tier difficulty ramp. |
| **Practical Implementation** | 20% | Executable production Python 3.12+ code, zero placeholder ellipses (`...`), zero synthetic list-as-stream mocks. |
| **Engineering Quality** | 15% | Type annotations, error handling, defensive input validation, zero-pickle safety. |
| **Assessment & Exercises** | 15% | Tiered exercises (Beginner, Intermediate, Advanced), system design interview questions. |
| **Resources & Freshness** | 10% | Validated external documentation, landmark research papers, verified open-source libraries. |
| **Documentation Consistency**| 5% | Strict 16-section template, topic metadata frontmatter, clean Markdown formatting. |

---

## 5. 30-Day Execution Cycle Tracker

```
[============================================================] 100% (24 / 24 Core Tasks Completed)
```

- **Days 1–3 (Audit & Correctness):** Completed CUR-001 through CUR-008.
- **Days 4–8 (Foundations Expansion):** Completed CUR-009 through CUR-015.
- **Days 9–13 (Core ML & Deep Learning Split):** Completed CUR-016 through CUR-017.
- **Days 14–19 (Modern AI & GenAI Split):** Completed CUR-018 through CUR-024.
- **Days 20–25 (Agentic AI Specialization):** Completed CUR-025 through CUR-034.
- **Days 26–30 (Production Systems & Practice):** Completed CUR-035 through CUR-052.
