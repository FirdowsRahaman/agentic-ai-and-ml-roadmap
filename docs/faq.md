# Frequently Asked Questions (FAQ)

[← Back to Documentation](./index.md) | [Getting Started](./getting-started.md) | [Roadmap](../ROADMAP.md)

---

## General & Learning Questions

### Q1: Is this roadmap designed for beginners or experienced engineers?
This curriculum is designed for **both**. Beginners with basic Python scripting ability can follow **Tier 1 (Foundations)** and work their way up systematically. Experienced software engineers, backend developers, or data scientists looking to pivot to AI Engineering or Agentic Systems can jump straight into **Tier 4 (Generative AI)** and **Tier 5 (Autonomous Agents)**.

### Q2: Do I need expensive cloud GPUs or paid API keys to learn?
**No.** All core modules, local starter projects (`starter-local-rag-agent` and `starter-local-mlops`), and notebooks are designed for **$0 spend**. They run completely offline on consumer hardware (macOS Apple Silicon, Linux, or Windows) using open-weight quantized models (Ollama), embedded vector stores (Qdrant Local), in-memory databases (SQLite/DuckDB), and pure NumPy implementations.

### Q3: How long does it take to complete the curriculum?
- **Accelerated Pace (Full-time):** 4 to 6 weeks (15–20 hours/week).
- **Practical Engineering Pace (Part-time):** 10 to 14 weeks (5–8 hours/week).

---

## Technical & Architecture Questions

### Q4: Why does the curriculum avoid Python `pickle` for model serialization?
Standard `pickle` allows arbitrary code execution upon deserialization (`pickle.load`), creating severe Remote Code Execution (RCE) vulnerabilities when loading unverified model weights. In Module 11 and our MLOps starter kit, we teach safe, structured weight storage using NumPy `.npz` arrays (with `allow_pickle=False`), SafeTensors, or ONNX runtimes.

### Q5: What is the difference between an Agentic Workflow and an Autonomous Agent?
- **Agentic Workflow:** A deterministic, directed graph where the sequence of tool execution and branches is predetermined by engineering logic (e.g., prompt chaining, router patterns).
- **Autonomous Agent:** A system where the foundation model dynamically determines the sequence of actions, loops, and termination conditions based on ongoing environmental observations (e.g., ReAct loops).

### Q6: How do I run tests and validate internal links locally?
You can validate the repository structure and relative links with the built-in validation scripts:
```bash
python scripts/validate_links.py
python scripts/validate_structure.py
```
