# Changelog

All notable changes to the **Agentic AI & ML Roadmap** curriculum will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-25

### Added
- **Curriculum Architecture:** Refactored 13 comprehensive modules into a structured 6-tier hierarchy (`01-foundations`, `02-machine-learning`, `03-deep-learning`, `04-generative-ai`, `05-agentic-ai`, `06-production-ai`).
- **Standard 13-Section Pedagogical Template:** Standardized all curriculum modules with measurable Bloom's taxonomy learning objectives, prerequisites, hands-on implementations, failure modes, and completion checklists.
- **Dedicated Documentation Hub (`docs/`):** Introduced structured docs for Getting Started, Learning Paths, Prerequisites, Technical Glossary, and FAQ.
- **Modular Resource Guides:** Partitioned external resources into `books.md`, `courses.md`, `research-papers.md`, and `tools.md`.
- **Validation Engine (`scripts/`):** Added automated link, anchor, and repository architecture validation scripts (`validate_links.py`, `validate_structure.py`).
- **Standardized Notebooks:** Partitioned and enriched notebooks into `foundations/`, `machine-learning/`, and `deep-learning/` with problem statements, datasets, and practical exercises.
- **Project Catalog Scaffolding:** Categorized all 10 portfolio blueprints into Beginner ($0 API spend), Intermediate (Runnable Starters), and Advanced (Enterprise Cloud) tiers.

### Changed
- **Root README:** Redesigned into a fast, navigable 30-second entry portal; decoupled in-depth roadmaps into `ROADMAP.md` and `docs/learning-paths.md`.
- **MkDocs Material Configuration:** Updated `mkdocs.yml` to support modern theme features, search, tabbed navigation, code copying, and KaTeX math formulas.
- **CI/CD Quality Gates:** Eliminated insecure `exec()` notebook execution in GitHub Actions, replacing it with isolated validation harnesses and pinned dependencies.

### Fixed
- Fixed factual model version anomaly in Module 05 and Projects catalog (standardized to `YOLOv10 / YOLO11`).
- Cleaned up orphaned `./interview` directory.
- Added comprehensive `.gitignore` rules preventing `.pytest_cache`, `__pycache__`, and build artifact pollution.
