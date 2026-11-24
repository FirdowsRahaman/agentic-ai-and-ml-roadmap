# Contributing to Agentic AI & ML Roadmap

Thank you for helping improve this open-source learning path for aspiring AI/ML engineers!

## Quality Standards

1. **Follow the Two-Tier Handbook Structure (`curriculum/`):**
   - **Tier 1 (Core Foundations):** Clear first-principles intuition, zero unexplained jargon, runnable PEP-8 Python (4-space indentation).
   - **Tier 2 (Production Architecture & Scale):** Latency, memory, security, scale trade-offs, and failure modes.
   - Keep `curriculum/` written as a concept-first **Engineering Handbook** (`#### 1.1 ...`, `Key Engineering Takeaways`), and place explicit `Q&A` mock questions inside `interview-kits/`.
2. **Code Must Run Cleanly:**
   - Avoid deprecated Pandas chained `inplace=True` patterns, raw `eval()`, or untested pseudo-code.
   - Include `assert` checks on self-contained snippets whenever practical.
3. **Diagrams & Formatting:**
   - Use styled **Mermaid** diagrams (`flowchart LR` / `flowchart TD`) with quoted labels (`A["Step Name"]`) instead of ASCII art boxes.
   - Ensure every relative file link and `#heading-anchor` resolves cleanly.

## Local Verification Before Opening a PR

```bash
# 1. Run starter kit unit tests
pytest projects/intermediate/starter-local-rag-agent projects/intermediate/starter-local-mlops

# 2. Preview optional MkDocs documentation site locally
pip install mkdocs-material
mkdocs serve
```
