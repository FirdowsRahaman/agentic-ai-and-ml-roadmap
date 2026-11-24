# Technical Debt & Defect Register

**Repository:** `agentic-ai-and-ml-roadmap`  
**Phase:** Phase 0 — Repository Audit  
**Status:** Complete  

---

## 1. Defect Classification Summary

| Priority | Level | Description | Count | Action Required |
|---|---|---|---|---|
| **P0** | **Critical** | Insecure execution patterns, broken link vulnerabilities, invalid dependencies | 3 | Immediate fix in Phase 0 / Phase 1 |
| **P1** | **High** | Missing architectural files, monolithic README, unstandardized curriculum templates | 4 | Remediate in Phase 1 & Phase 2 |
| **P2** | **Medium** | Unvalidated notebooks, lack of static linting/typechecking, flat folder structures | 5 | Remediate in Phase 4 & Phase 5 |
| **P3** | **Low** | Badges, visual refinements, advanced search parameters, minor typo fixes | 3 | Address in Phase 6 & Phase 7 |

---

## 2. P0: Critical Severity Defects

### TECH-01: Insecure Arbitrary Notebook Execution via `exec()` in CI
- **Location:** `.github/workflows/ci-verify.yml` (lines 70–82)
- **Description:** The CI workflow currently validates Jupyter notebooks using unrestricted Python `exec()` inside an unisolated namespace:
  ```python
  for nb_path in sorted(glob.glob("notebooks/*.ipynb")):
      with open(nb_path, encoding="utf-8") as fp:
          nb = json.load(fp)
      ns = {}
      for cell in nb["cells"]:
          if cell["cell_type"] == "code":
              exec("".join(cell["source"]), ns)
  ```
- **Risk:**
  - Arbitrary code execution without security sandboxing.
  - State leaks between cells and between notebook runs.
  - Zero execution timeout limits; an infinite loop blocks CI indefinitely.
  - Does not validate that notebooks run cleanly in a fresh kernel session.
- **Remediation:** Replace with `nbclient` or `pytest --nbval` running in isolated virtual subprocesses with explicit timeouts.

### TECH-02: Missing Cache Rules in `.gitignore`
- **Location:** `.gitignore`
- **Description:** `.gitignore` only ignores `.DS_Store`, `*.egg-info/`, `dist/`, and `build/`. It fails to ignore:
  - `.pytest_cache/`
  - `__pycache__/` and `*.pyc`
  - `.mypy_cache/` and `.ruff_cache/`
  - `site/` (MkDocs build artifacts)
  - `.venv/` and `venv/`
- **Risk:** Untracked bytecode files (`.pyc`) and test cache files currently clutter the workspace and risk accidental commits into Git history.
- **Remediation:** Overhaul `.gitignore` with standard modern Python, pytest, and documentation ignore rules.

### TECH-03: Fictional/Erroneous Library References (`YOLO26`)
- **Location:** `curriculum/03-modern-ai-and-genai/05_nlp_cv_and_multimodal.md`, `projects/README.md`
- **Description:** Mentions `YOLO26` as the "native NMS-free edge flagship". In reality, NMS-free edge architectures are pioneered by `YOLOv10` and `RT-DETR`, while the latest mainstream release from Ultralytics is `YOLO11`.
- **Risk:** Severely impacts technical credibility for senior ML/AI engineers and hiring managers reading the curriculum.
- **Remediation:** Correct references to `YOLOv10 (NMS-Free)` and `YOLO11`.

---

## 3. P1: High Severity Technical Debt

### TECH-04: Monolithic Root Documentation
- **Location:** `README.md` (15,046 bytes)
- **Description:** The root README attempts to do everything: overview, curriculum roadmap, project catalog, module tables, and resources.
- **Risk:** High cognitive load for newcomers; prevents quick orientation within 30 seconds.
- **Remediation:** Break down into focused entry documents (`README.md`, `ROADMAP.md`, `docs/learning-paths.md`, `docs/getting-started.md`).

### TECH-05: Non-Standardized Curriculum Modules
- **Location:** `curriculum/**/*.md` (all 13 modules)
- **Description:** Modules only feature 4–5 sections without learning objectives, prerequisites, common mistakes, or completion checklists.
- **Risk:** Inconsistent learner experience; difficult to assess competence.
- **Remediation:** Standardize every module to the 13-section pedagogical template.

### TECH-06: Inline CI Validation Scripts
- **Location:** `.github/workflows/ci-verify.yml`
- **Description:** 60+ lines of Python code for verifying Markdown anchors, relative links, and SVG syntax are embedded directly into bash heredocs in the YAML workflow file.
- **Risk:** Untested CI code; local developers cannot run link or structure checks before committing or opening a PR.
- **Remediation:** Extract validation logic into dedicated scripts under `scripts/validate_links.py` and `scripts/validate_structure.py`.

### TECH-07: MkDocs Document Root Misalignment
- **Location:** `mkdocs.yml` (`docs_dir: .`)
- **Description:** MkDocs uses the entire repository root as its document source.
- **Risk:** Non-documentation files, test fixtures, and scripts are indexed, causing bloated builds and potential link resolution errors.
- **Remediation:** Migrate documentation pages into a dedicated `docs/` folder and point `docs_dir: docs`.

---

## 4. P2: Medium Severity Technical Debt

### TECH-08: Incomplete Notebook Structure
- **Location:** `notebooks/*.ipynb`
- **Description:** Notebooks are bare execution snippets without challenge descriptions, expected outputs, or exercises.
- **Remediation:** Restructure notebooks to include standard pedagogical sections: Problem Statement, Learning Objectives, Required Dependencies, Dataset Description, Implementation, Results & Observations, and Exercises.

### TECH-09: Absence of Static Analysis & Linting
- **Location:** Entire codebase
- **Description:** No formatting (`ruff format`), linting (`ruff check`), or type verification (`mypy`) is currently enforced either locally via pre-commit or in CI.
- **Remediation:** Add `ruff` and `mypy` configurations and include them in CI pull request validation.

### TECH-10: Lack of Top-Level Test Runner
- **Location:** Root repository
- **Description:** Running `pytest` from the root directory fails because tests are distributed inside individual sub-packages with distinct `PYTHONPATH` requirements.
- **Remediation:** Introduce a root `tests/` directory or a consolidated `pyproject.toml` configuring test discovery paths.

### TECH-11: Monolithic Resources Guide
- **Location:** `resources/recommended_resources.md`
- **Description:** Combines books, courses, papers, and tools into one long markdown document.
- **Remediation:** Decompose into `resources/books.md`, `resources/courses.md`, `resources/research-papers.md`, and `resources/tools.md`.

### TECH-12: Unpinned CI Dependency Versions
- **Location:** `.github/workflows/ci-verify.yml`
- **Description:** CI installs packages using unpinned `pip install` commands, risking sudden CI breakage when upstream library APIs update.
- **Remediation:** Pin tooling versions or create a `requirements-ci.txt`.

---

## 5. P3: Low Severity Debt

### TECH-13: Missing Visual Badges & Social Metadata
- Missing CI status badge, license badge, and Python version compatibility badges in `README.md`.

### TECH-14: Orphaned `./interview` Directory
- Empty folder left behind on disk.

### TECH-15: Missing LaTeX KaTeX Support Configuration in Documentation Site
- Mathematical equations in markdown need explicit delimiters and KaTeX rendering support in MkDocs Material configuration.
