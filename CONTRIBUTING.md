# Contributing to Agentic AI & ML Roadmap

Thank you for your interest in improving the **Agentic AI & ML Roadmap**! This repository is an open educational platform dedicated to engineering rigor, mathematical correctness, and production reliability.

---

## 1. Development Principles

1. **Production-First Rigor:** Code examples should reflect real-world engineering standards (proper error handling, explicit typing, clean separation of concerns). Avoid toy snippets that would fail in production.
2. **First-Principles Understanding:** Prioritize mathematical and algorithmic clarity over opaque framework wrappers.
3. **Reproducibility ($0 Barrier):** Foundational examples, notebooks, and starter kits must remain runnable locally without requiring paid third-party API subscriptions.
4. **No Secrets in Code:** Never commit API keys, personal access tokens, or sensitive credentials.

---

## 2. Standard Curriculum Module Format

All modules in `curriculum/` must adhere to the **13-Section Pedagogical Standard**:

```markdown
# Module XX: Module Name

[Navigation bar]

---

## 1. Overview
## 2. Learning Objectives
## 3. Prerequisites
## 4. Core Concepts
## 5. Technical Deep Dive
## 6. Hands-on Implementation
## 7. Practical Exercises
## 8. Mini Project
## 9. Advanced Topics
## 10. Common Mistakes & Gotchas
## 11. Interview Questions & System Design
## 12. Recommended Resources
## 13. Completion Checklist
```

---

## 3. Local Development & Quality Gates

Before submitting a Pull Request, run the local validation suite to ensure that your changes pass all quality checks:

```bash
# 1. Validate internal Markdown links, anchors, and images
python scripts/validate_links.py

# 2. Validate repository architecture and conventions
python scripts/validate_structure.py

# 3. Run the repository test suite
pytest tests/
```

---

## 4. Pull Request Checklist

When opening a Pull Request, verify the following:

- [ ] All new or moved Markdown links are verified using `python scripts/validate_links.py`.
- [ ] No temporary files (`.DS_Store`, `__pycache__`, `.pytest_cache`, `.pyc`) are committed.
- [ ] Code samples follow PEP 8 style standards with type hints where appropriate.
- [ ] Explanations are factually verified against current, real-world library releases.
- [ ] Conventional commit messages are used (e.g. `docs:`, `fix:`, `refactor:`, `test:`).

---

## 5. Licensing

By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](./LICENSE).
