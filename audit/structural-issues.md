# Structural Issues & Architecture Audit

**Repository:** `agentic-ai-and-ml-roadmap`  
**Phase:** Phase 0 — Repository Audit  
**Status:** Complete  

---

## 1. High-Level Architecture Assessment

The repository contains valuable, high-signal technical content (e.g., zero-pickle architectures, Reciprocal Rank Fusion, LangGraph, Model Context Protocol, and PSI drift calculations). However, the directory layout reflects organic, ad-hoc growth rather than a deliberate, scalable information architecture.

### Current Directory Layout vs. Target Target Architecture

```
CURRENT STRUCTURE                                TARGET TARGET ARCHITECTURE
─────────────────                                ──────────────────────────
agentic-ai-and-ml-roadmap/                       agentic-ai-and-ml-roadmap/
├── README.md (Monolithic, 15KB)                 ├── README.md (Clean portal, 30s orientation)
├── CONTRIBUTING.md                              ├── ROADMAP.md (Comprehensive sequence)
├── LICENSE                                      ├── CONTRIBUTING.md (Standard dev & testing guide)
├── mkdocs.yml (docs_dir: .)                     ├── CHANGELOG.md (Release history)
├── .gitignore                                   ├── mkdocs.yml (docs_dir: docs)
├── interview/ (EMPTY DIRECTORY)                 ├── requirements-docs.txt (Pinned site deps)
├── interview-kits/ (Flat 5 files)               │
├── curriculum/                                  ├── docs/ (Site pages & reference guides)
│   ├── 01-foundations/                          │   ├── index.md, getting-started.md
│   ├── 02-core-ml-and-dl/                       │   ├── learning-paths.md, prerequisites.md
│   ├── 03-modern-ai-and-genai/                  │   └── glossary.md, faq.md
│   ├── 04-agentic-ai/                           │
│   └── 05-production-and-practice/              ├── curriculum/ (Standardized 6-tier sequence)
├── notebooks/ (Flat 3 files)                    │   ├── 01-foundations/
├── projects/                                    │   ├── 02-machine-learning/
│   ├── README.md (10 mixed blueprints)          │   ├── 03-deep-learning/
│   ├── end_to_end_lifecycle_playbook.md         │   ├── 04-generative-ai/
│   ├── starter-local-mlops/                     │   ├── 05-agentic-ai/
│   └── starter-local-rag-agent/                 │   └── 06-production-ai/
├── resources/                                   │
│   └── recommended_resources.md (Single file)   ├── notebooks/ (Partitioned by curriculum tier)
└── assets/                                      │   ├── foundations/, machine-learning/
    ├── banner.webp                              │   ├── deep-learning/, generative-ai/
    ├── social-preview.png                       │   └── agentic-ai/
    └── animations/*.svg                         │
                                                 ├── projects/ (Categorized by difficulty)
                                                 │   ├── README.md
                                                 │   ├── beginner/, intermediate/, advanced/
                                                 │   └── starters/
                                                 │
                                                 ├── interview-kits/ (Categorized by track)
                                                 │   ├── machine-learning/, deep-learning/
                                                 │   ├── generative-ai/, agentic-ai/
                                                 │   └── frameworks/
                                                 │
                                                 ├── resources/ (Modular domain guides)
                                                 │   ├── books.md, courses.md
                                                 │   ├── research-papers.md, tools.md
                                                 │
                                                 ├── assets/ (diagrams/, images/, animations/)
                                                 ├── scripts/ (Automated link & structure validators)
                                                 └── tests/ (Root test runner & fixtures)
```

---

## 2. Key Structural Deficiencies Identified

### 1. Root Directory Pollution & Over-Responsibility
- **Problem:** `README.md` acts as curriculum guide, project catalog, track selector, and module index all at once (over 15,000 bytes). A first-time learner takes several minutes to understand how to begin.
- **Remediation:** Transform `README.md` into a lean, 30-second landing page with clear badges, persona paths, and direct links. Move detailed learning tracks to `ROADMAP.md` and `docs/learning-paths.md`.

### 2. MkDocs Misconfiguration (`docs_dir: .`)
- **Problem:** `mkdocs.yml` is configured with `docs_dir: .`. This forces MkDocs to treat the root repository directory as the documentation source, resulting in Python files, Dockerfiles, and cache directories leaking into documentation builds unless explicitly excluded.
- **Remediation:** Introduce a dedicated `docs/` folder for general documentation pages (`index.md`, `getting-started.md`, `learning-paths.md`, `prerequisites.md`, `glossary.md`, `faq.md`), and update `mkdocs.yml` navigation structure.

### 3. Curriculum Folder Grouping Inconsistencies
- **Problem:** Currently, classical Machine Learning and Deep Learning are conflated into `02-core-ml-and-dl/`, and Computer Vision/NLP is combined with Generative AI in `03-modern-ai-and-genai/`. This blurs the distinct pedagogical boundaries between statistical ML, neural representation learning, and modern LLM/Agent architectures.
- **Remediation:** Re-tier the curriculum into 6 distinct, logical folders:
  - `01-foundations` (Python, SQL, Math, Statistics)
  - `02-machine-learning` (Classical ML, RecSys, Tabular)
  - `03-deep-learning` (PyTorch, Representation Learning, RL)
  - `04-generative-ai` (Transformers, Attention, Fine-Tuning, RAG)
  - `05-agentic-ai` (ReAct, Multi-Agent Systems, LangGraph, MCP, Evals)
  - `06-production-ai` (MLOps, Cloud FinOps, Troubleshooting, Scratch Algorithms)

### 4. Orphaned Directory (`./interview`)
- **Problem:** An empty folder `./interview` exists on disk alongside the populated `interview-kits/` folder. This is an artifact of an aborted folder rename or partial migration.
- **Remediation:** Delete `./interview` immediately to prevent confusion.

### 5. Flat Organization of Supporting Materials
- **Notebooks:** 3 notebooks sit flat in `notebooks/` without explicit alignment to the 6 curriculum tiers.
- **Interview Kits:** 5 markdown files sit flat in `interview-kits/` rather than organized by domain (ML, GenAI, Agents, System Design).
- **Resources:** A single monolithic `recommended_resources.md` (14.5 KB) combines books, academic papers, tools, industry reports, and courses without modular discoverability.
- **Projects:** Flat directory mixing documentation (`README.md`, `end_to_end_lifecycle_playbook.md`) and code packages (`starter-local-rag-agent`, `starter-local-mlops`) without categorization by learner level (beginner, intermediate, advanced).

### 6. Missing Automation & Validation Scripts
- **Problem:** All link checks, SVG validations, and notebook tests are inlined directly into `.github/workflows/ci-verify.yml`. Contributors cannot run link validation or structure checks locally with simple commands like `python scripts/validate_links.py`.
- **Remediation:** Extract CI logic into standalone, reusable Python scripts under `scripts/`.

---

## 3. Structural Severity Classification

| Issue ID | Component | Description | Severity | Impact |
|---|---|---|---|---|
| **STR-01** | `mkdocs.yml` | `docs_dir: .` leaks repository root into static site builds | **High** | Broken doc builds, security risks |
| **STR-02** | `./interview` | Orphaned empty directory | **Low** | Codebase cleanliness |
| **STR-03** | `README.md` | Monolithic content overload | **High** | High bounce rate for learners |
| **STR-04** | `curriculum/` | Conflation of ML/DL and GenAI/Multimodal | **Medium** | Curriculum progression friction |
| **STR-05** | `resources/` | Monolithic resource file | **Medium** | Poor discoverability |
| **STR-06** | `scripts/` | Missing local developer validation tools | **Medium** | Broken CI loops for contributors |
| **STR-07** | `.gitignore` | Missing cache rules (`.pytest_cache`, `.pyc`) | **High** | Pollutes Git commits |
