# Master Refactoring & Migration Plan

**Repository:** `agentic-ai-and-ml-roadmap`  
**Execution Strategy:** Incremental Phased Refactor  
**Target Version:** v1.0.0 Release  
**Working Branch:** `refactor/repository-cleanup`  

---

## 1. Migration Principles & Guiding Rules

1. **Preserve Educational Integrity:** Never delete valuable domain knowledge simply because it does not match a directory structure. Relocate, enrich, or merge existing content.
2. **Reviewable, Atomic Git Commits:** Each phase and sub-milestone must be committed independently with clear, conventional commit messages.
3. **Continuous Verification:** Every file move and link change must pass local validation (`scripts/validate_links.py`, `scripts/validate_structure.py`, and test suites) before moving to subsequent phases.
4. **No Direct Commits to Main:** All work proceeds on `refactor/repository-cleanup` until full audit and verification gates pass.

---

## 2. File-by-File Action Matrix

| Source Path | Proposed Target Path | Action | Rationale |
|---|---|---|---|
| `README.md` | `README.md` | **REWRITE** | Convert into concise 30-second portal; offload detailed tracks |
| *New* | `ROADMAP.md` | **CREATE** | Dedicated comprehensive curriculum sequence & milestone guide |
| `CONTRIBUTING.md` | `CONTRIBUTING.md` | **REWRITE** | Update development, linting, testing, and PR standards |
| *New* | `CHANGELOG.md` | **CREATE** | Track v1.0.0 release notes and version history |
| *New* | `requirements-docs.txt` | **CREATE** | Pinned dependencies for MkDocs Material & plugins |
| `.gitignore` | `.gitignore` | **REWRITE** | Add modern Python, pytest, site, and IDE ignore patterns |
| `./interview` | *Deleted* | **REMOVE** | Clean up orphaned empty directory |
| `curriculum/01-foundations/01_python_sql_and_data_stack.md` | `curriculum/01-foundations/01_python_sql_and_data_stack.md` | **REWRITE** | Standardize to 13-section pedagogical template |
| `curriculum/01-foundations/02_math_and_statistics.md` | `curriculum/01-foundations/02_math_and_statistics.md` | **REWRITE** | Standardize to 13-section pedagogical template |
| `curriculum/02-core-ml-and-dl/03_classical_ml_and_recsys.md` | `curriculum/02-machine-learning/03_classical_ml_and_recsys.md` | **MOVE/REWRITE** | Isolate classical ML into dedicated domain folder |
| `curriculum/02-core-ml-and-dl/04_deep_learning_and_rl.md` | `curriculum/03-deep-learning/04_deep_learning_and_rl.md` | **MOVE/REWRITE** | Isolate neural representation learning & RL |
| `curriculum/03-modern-ai-and-genai/05_nlp_cv_and_multimodal.md` | `curriculum/04-generative-ai/05_nlp_cv_and_multimodal.md` | **MOVE/REWRITE** | Fix `YOLO26` anomaly, re-structure into 13-sections |
| `curriculum/03-modern-ai-and-genai/06_llms_finetuning_and_rag.md` | `curriculum/04-generative-ai/06_llms_finetuning_and_rag.md` | **MOVE/REWRITE** | Standardize LLM, LoRA, and RAG sections |
| `curriculum/04-agentic-ai/07_ai_agents_fundamentals.md` | `curriculum/05-agentic-ai/07_ai_agents_fundamentals.md` | **MOVE/REWRITE** | Standardize ReAct loops and AST security tools |
| `curriculum/04-agentic-ai/08_tools_langgraph_and_adk.md` | `curriculum/05-agentic-ai/08_tools_langgraph_and_adk.md` | **MOVE/REWRITE** | Standardize LangGraph and ADK state machines |
| `curriculum/04-agentic-ai/09_multi_agent_and_mcp.md` | `curriculum/05-agentic-ai/09_multi_agent_and_mcp.md` | **MOVE/REWRITE** | Standardize MCP JSON-RPC protocols |
| `curriculum/04-agentic-ai/10_agent_evals_and_safety.md` | `curriculum/05-agentic-ai/10_agent_evals_and_safety.md` | **MOVE/REWRITE** | Standardize evaluation harnesses and safety guardrails |
| `curriculum/05-production-and-practice/11_mlops_cloud_and_finops.md` | `curriculum/06-production-ai/11_mlops_cloud_and_finops.md` | **MOVE/REWRITE** | Standardize MLOps and FinOps modules |
| `curriculum/05-production-and-practice/12_system_troubleshooting_scenarios.md` | `curriculum/06-production-ai/12_system_troubleshooting_scenarios.md` | **MOVE/REWRITE** | Standardize troubleshooting diagnostic trees |
| `curriculum/05-production-and-practice/13_algorithms_from_scratch.md` | `curriculum/06-production-ai/13_algorithms_from_scratch.md` | **MOVE/REWRITE** | Standardize scratch algorithms to template |
| `notebooks/01_numpy_pandas_sql_lab.ipynb` | `notebooks/foundations/01_numpy_pandas_sql_lab.ipynb` | **MOVE/REWRITE** | Enrich with problem statement, objectives, and exercises |
| `notebooks/02_ml_from_scratch_challenges.ipynb` | `notebooks/machine-learning/02_ml_from_scratch_challenges.ipynb` | **MOVE/REWRITE** | Enrich with challenge problems and tuning tasks |
| `notebooks/03_mini_transformer_attention.ipynb` | `notebooks/deep-learning/03_mini_transformer_attention.ipynb` | **MOVE/REWRITE** | Enrich with causal masking and attention exercises |
| *New* | `docs/index.md` | **CREATE** | Main documentation site homepage |
| *New* | `docs/getting-started.md` | **CREATE** | Quickstart setup for environment and dependencies |
| *New* | `docs/learning-paths.md` | **CREATE** | Persona tracks (Beginner, Applied ML, Agent Engineer) |
| *New* | `docs/prerequisites.md` | **CREATE** | Detailed prerequisite map and diagnostics |
| *New* | `docs/glossary.md` | **CREATE** | Production terminology and definitions |
| *New* | `docs/faq.md` | **CREATE** | Frequently asked technical questions |
| `resources/recommended_resources.md` | `resources/recommended_resources.md` | **REWRITE/SPLIT** | Split into `books.md`, `courses.md`, `research-papers.md`, `tools.md` |
| `interview-kits/*.md` | `interview-kits/*/*.md` | **MOVE/REFINE** | Organize into domains (machine-learning, deep-learning, generative-ai, agentic-ai) |
| `projects/README.md` | `projects/README.md` | **REWRITE** | Reorganize blueprints into beginner, intermediate, advanced tiers |
| *New* | `scripts/validate_links.py` | **CREATE** | Standalone Python link & anchor validator |
| *New* | `scripts/validate_structure.py` | **CREATE** | Automated structure & convention validator |
| `.github/workflows/ci-verify.yml` | `.github/workflows/ci-verify.yml` | **REWRITE** | Modular CI calling standalone scripts and safe notebook tests |

---

## 3. Phased Implementation Roadmap

### Phase 1: Information Architecture (Target Directory Setup)
- Establish the new directory hierarchy:
  - `docs/`
  - `curriculum/01-foundations/` through `06-production-ai/`
  - `notebooks/foundations/`, `machine-learning/`, etc.
  - `projects/beginner/`, `intermediate/`, `advanced/`
  - `interview-kits/machine-learning/`, `deep-learning/`, etc.
  - `resources/books.md`, `courses.md`, `research-papers.md`, `tools.md`
  - `scripts/`, `tests/`
- Migrate files to their target locations safely with git moves.
- Update internal relative links to avoid link degradation.

### Phase 2: Documentation Redesign
- Redesign root `README.md` (compact, professional, 30s orientation).
- Create `ROADMAP.md` with complete pedagogical tracks.
- Populate `docs/` (`getting-started.md`, `learning-paths.md`, `prerequisites.md`, `glossary.md`, `faq.md`).
- Standardize `CONTRIBUTING.md` and create `CHANGELOG.md`.

### Phase 3: Curriculum Quality & Technical Review
- Refactor all 13 modules to adhere to the standardized 13-section template.
- Correct factual anomalies (replace `YOLO26` with `YOLOv10 / YOLO11`).
- Embed prerequisite chains, explicit Bloom's taxonomy objectives, and quantitative completion checklists.

### Phase 4: Code & Notebook Engineering
- Enrich notebooks (`01`, `02`, `03`) with full educational framing, markdown context, and exercises.
- Structure starter projects with standard metadata and unit tests.
- Ensure all Python examples run reproducibly without external API requirements ($0 barrier).

### Phase 5: CI/CD & Automation
- Build `scripts/validate_links.py` and `scripts/validate_structure.py`.
- Modernize `.github/workflows/ci-verify.yml` with isolated, secure notebook execution (eliminating unsafe `exec()`).
- Add linters (`ruff`) and testing checks.

### Phase 6: Documentation Website (MkDocs Material)
- Reconfigure `mkdocs.yml` with `docs_dir: docs` or appropriate site mapping.
- Enable search, code copy, Mermaid diagrams, KaTeX math extensions, and navigation tabs.
- Verify local documentation build (`mkdocs build`).

### Phase 7: Final Release (v1.0.0)
- End-to-end verification of all links, tests, and documentation.
- Prepare v1.0.0 release notes in `CHANGELOG.md`.
- Merge `refactor/repository-cleanup` into `main`.
