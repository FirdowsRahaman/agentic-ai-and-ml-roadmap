# Repository Inventory Audit

**Project:** Agentic AI & ML Roadmap  
**Repository:** `agentic-ai-and-ml-roadmap`  
**Audit Date:** November 2026  
**Auditor:** Automated Engineering Audit Engine  
**Branch:** `refactor/repository-cleanup`

---

## 1. Executive Summary

This inventory accounts for every file, directory, and asset currently residing in the repository. The audit covers tracked Git files as well as untracked artifacts, cache files, and orphaned folders.

### Summary Metrics
| Metric | Count | Details |
|---|---|---|
| **Total Tracked Git Files** | 46 | Version controlled files |
| **Total Non-Git Files on Disk** | 65 | Includes untracked `.pyc` and `.pytest_cache` artifacts |
| **Curriculum Modules** | 13 | Markdown guides covering Foundations to Production |
| **Project Blueprints** | 10 | 4 Local ($0 API cost), 6 Enterprise Cloud blueprints |
| **Runnable Starter Projects** | 2 | `starter-local-rag-agent` and `starter-local-mlops` |
| **Jupyter Notebooks** | 3 | NumPy, ML From Scratch, Transformer Self-Attention |
| **Interview Kits** | 5 | System Design, Classical ML, GenAI/Agents, Screenings, Resumes |
| **Visual Assets / Animations** | 14 | 12 SVG animations, 1 WebP banner, 1 PNG social preview |
| **Empty Directories** | 1 | `./interview` (orphaned legacy folder) |
| **Cache / Stale Files** | 7 | 6 `.pyc` files + `.pytest_cache/` directory |

---

## 2. File-by-File Detailed Inventory

### A. Root & Core Configuration Files
| File Path | Size (Bytes) | Lines | Category | Disposition | Notes / Proposed Action |
|---|---|---|---|---|---|
| `README.md` | 15,046 | 320 | Documentation | **REWRITE** | Monolithic entry point; extract detailed tracks into `docs/` and `ROADMAP.md` |
| `CONTRIBUTING.md` | 1,370 | 48 | Documentation | **REWRITE** | Expand with standard dev setup, linting, testing, and PR guidelines |
| `LICENSE` | 1,072 | 21 | Legal | **KEEP** | Standard MIT License |
| `mkdocs.yml` | 3,172 | 92 | Configuration | **REWRITE** | Update `docs_dir` to `docs/` and align navigation with refactored hierarchy |
| `.gitignore` | 167 | 15 | Configuration | **REWRITE** | Missing rules for `.pytest_cache`, `.venv`, `.mypy_cache`, and `site/` |
| `.github/workflows/ci-verify.yml` | 2,750 | 85 | CI/CD | **REWRITE** | Replace unsafe `exec()` notebook check with `nbclient` & modular test scripts |
| `.github/ISSUE_TEMPLATE/content_or_bug.md` | 560 | 25 | Community | **KEEP** | Refine labels and checklists |

### B. Curriculum Modules (`curriculum/`)
| File Path | Size (Bytes) | Lines | Current Sections | Disposition | Target Path & Notes |
|---|---|---|---|---|---|
| `curriculum/01-foundations/01_python_sql_and_data_stack.md` | 13,858 | 313 | 5 | **REWRITE** | `curriculum/01-foundations/01_python_sql_and_data_stack.md` (Standardize to 13-section template) |
| `curriculum/01-foundations/02_math_and_statistics.md` | 10,210 | 220 | 5 | **REWRITE** | `curriculum/01-foundations/02_math_and_statistics.md` (Standardize to 13-section template) |
| `curriculum/02-core-ml-and-dl/03_classical_ml_and_recsys.md` | 11,452 | 223 | 5 | **MOVE/REWRITE** | `curriculum/02-machine-learning/03_classical_ml_and_recsys.md` |
| `curriculum/02-core-ml-and-dl/04_deep_learning_and_rl.md` | 12,890 | 255 | 5 | **MOVE/REWRITE** | `curriculum/03-deep-learning/04_deep_learning_and_rl.md` |
| `curriculum/03-modern-ai-and-genai/05_nlp_cv_and_multimodal.md` | 10,980 | 217 | 5 | **MOVE/REWRITE** | `curriculum/04-generative-ai/05_nlp_cv_and_multimodal.md` (Fix `YOLO26` anomaly) |
| `curriculum/03-modern-ai-and-genai/06_llms_finetuning_and_rag.md` | 13,420 | 268 | 5 | **MOVE/REWRITE** | `curriculum/04-generative-ai/06_llms_finetuning_and_rag.md` |
| `curriculum/04-agentic-ai/07_ai_agents_fundamentals.md` | 9,840 | 195 | 5 | **MOVE/REWRITE** | `curriculum/05-agentic-ai/07_ai_agents_fundamentals.md` |
| `curriculum/04-agentic-ai/08_tools_langgraph_and_adk.md` | 11,200 | 216 | 5 | **MOVE/REWRITE** | `curriculum/05-agentic-ai/08_tools_langgraph_and_adk.md` |
| `curriculum/04-agentic-ai/09_multi_agent_and_mcp.md` | 10,650 | 206 | 5 | **MOVE/REWRITE** | `curriculum/05-agentic-ai/09_multi_agent_and_mcp.md` |
| `curriculum/04-agentic-ai/10_agent_evals_and_safety.md` | 10,480 | 206 | 5 | **MOVE/REWRITE** | `curriculum/05-agentic-ai/10_agent_evals_and_safety.md` |
| `curriculum/05-production-and-practice/11_mlops_cloud_and_finops.md` | 12,100 | 239 | 5 | **MOVE/REWRITE** | `curriculum/06-production-ai/11_mlops_cloud_and_finops.md` |
| `curriculum/05-production-and-practice/12_system_troubleshooting_scenarios.md` | 8,950 | 153 | 5 | **MOVE/REWRITE** | `curriculum/06-production-ai/12_system_troubleshooting_scenarios.md` |
| `curriculum/05-production-and-practice/13_algorithms_from_scratch.md` | 14,800 | 315 | 4 | **MOVE/REWRITE** | `curriculum/06-production-ai/13_algorithms_from_scratch.md` |

### C. Notebooks (`notebooks/`)
| File Path | Size (Bytes) | Cells (Code/MD) | Current State | Disposition | Target Path & Notes |
|---|---|---|---|---|---|
| `notebooks/01_numpy_pandas_sql_lab.ipynb` | 4,210 | 3 / 1 | Functional script | **MOVE/REWRITE** | `notebooks/foundations/01_numpy_pandas_sql_lab.ipynb` (Add objectives, data spec, exercises) |
| `notebooks/02_ml_from_scratch_challenges.ipynb` | 4,890 | 3 / 1 | Functional script | **MOVE/REWRITE** | `notebooks/machine-learning/02_ml_from_scratch_challenges.ipynb` |
| `notebooks/03_mini_transformer_attention.ipynb` | 3,120 | 1 / 1 | Functional script | **MOVE/REWRITE** | `notebooks/deep-learning/03_mini_transformer_attention.ipynb` |

### D. Projects & Starter Kits (`projects/`)
| File Path | Size (Bytes) | Category | Disposition | Notes / Proposed Action |
|---|---|---|---|---|
| `projects/README.md` | 14,293 | Documentation | **REWRITE** | Portfolio catalog containing all 10 blueprints; reorganize by difficulty |
| `projects/end_to_end_lifecycle_playbook.md` | 11,200 | Playbook | **KEEP** | High-signal engineering guide |
| `projects/starter-local-rag-agent/README.md` | 4,320 | Documentation | **KEEP/REFINE** | Starter 1 guide |
| `projects/starter-local-rag-agent/app.py` | 5,120 | Python Implementation | **KEEP/REFINE** | BM25 + Vector RRF + SQL Router |
| `projects/starter-local-rag-agent/requirements.txt` | 32 | Dependencies | **KEEP** | `numpy>=1.26.0`, `pytest>=8.0.0` |
| `projects/starter-local-rag-agent/test_router.py` | 2,150 | Unit Tests | **KEEP** | Pytest test suite |
| `projects/starter-local-mlops/README.md` | 5,200 | Documentation | **KEEP/REFINE** | Starter 2 guide |
| `projects/starter-local-mlops/Dockerfile` | 890 | Containerization | **KEEP** | Non-root production Dockerfile |
| `projects/starter-local-mlops/requirements.txt` | 88 | Dependencies | **KEEP** | FastAPI, Uvicorn, Scikit-learn, etc. |
| `projects/starter-local-mlops/src/__init__.py` | 0 | Package Init | **KEEP** | Empty init file |
| `projects/starter-local-mlops/src/app.py` | 4,320 | Python API | **KEEP/REFINE** | FastAPI prediction & PSI drift endpoints |
| `projects/starter-local-mlops/src/train.py` | 3,840 | Model Training | **KEEP/REFINE** | Zero-pickle .npz training pipeline |
| `projects/starter-local-mlops/artifacts/churn_model.npz` | 2,140 | Model Artifact | **KEEP** | Pre-trained model weights |
| `projects/starter-local-mlops/tests/test_api.py` | 1,280 | Unit Tests | **KEEP** | Fast API test suite |

### E. Interview Kits (`interview-kits/` and `interview/`)
| File Path | Size (Bytes) | Lines | Disposition | Notes / Proposed Action |
|---|---|---|---|---|
| `./interview` | 0 | 0 | **REMOVE** | Empty directory to be deleted |
| `interview-kits/classical_ml_questions.md` | 8,900 | 180 | **MOVE/REFINE** | `interview-kits/machine-learning/classical_ml_questions.md` |
| `interview-kits/genai_and_agents_questions.md` | 10,400 | 210 | **MOVE/REFINE** | `interview-kits/generative-ai/genai_and_agents_questions.md` |
| `interview-kits/ml_system_design_framework.md` | 11,200 | 225 | **MOVE/REFINE** | `interview-kits/agentic-ai/ml_system_design_framework.md` |
| `interview-kits/resume_and_flashcards.md` | 7,800 | 160 | **KEEP/REFINE** | `interview-kits/resume_and_flashcards.md` |
| `interview-kits/sql_and_coding_screenings.md` | 8,100 | 170 | **MOVE/REFINE** | `interview-kits/machine-learning/sql_and_coding_screenings.md` |

### F. Resources (`resources/`)
| File Path | Size (Bytes) | Lines | Disposition | Notes / Proposed Action |
|---|---|---|---|---|
| `resources/recommended_resources.md` | 14,500 | 290 | **SPLIT/RESTRUCTURE** | Split into modular files: `books.md`, `courses.md`, `research-papers.md`, `tools.md` |

### G. Assets (`assets/`)
| File Path | Type | Disposition | Notes |
|---|---|---|---|
| `assets/banner.webp` | Image | **KEEP** | Hero banner for README |
| `assets/social-preview.png` | Image | **KEEP** | Social card preview |
| `assets/animations/*.svg` (12 files) | SVG Vector | **KEEP** | Motion architecture SVGs embedded in curriculum modules |

### H. Untracked Artifacts & Cache Files
| File Path | Category | Disposition | Notes |
|---|---|---|---|
| `projects/starter-local-mlops/src/__pycache__/*` (3 files) | Stale Bytecode | **REMOVE** | Clean up and ignore |
| `projects/starter-local-mlops/tests/__pycache__/*` (1 file) | Stale Bytecode | **REMOVE** | Clean up and ignore |
| `projects/starter-local-rag-agent/__pycache__/*` (2 files) | Stale Bytecode | **REMOVE** | Clean up and ignore |
| `.pytest_cache/*` (4 files) | Test Cache | **REMOVE** | Clean up and add to `.gitignore` |

---

## 3. Inventory Reconciliation Summary
- **Total Tracked Files to Retain/Refactor:** 46
- **Files to Move/Restructure:** 16
- **Files to Rewrite/Standardize:** 15
- **Files/Directories to Purge:** 1 empty folder (`./interview`), 10 cache files/dirs
- **New Files to Introduce:** `ROADMAP.md`, `CHANGELOG.md`, `requirements-docs.txt`, `docs/*.md`, `scripts/*.py`
