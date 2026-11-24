# Getting Started Guide

[← Back to Documentation](./index.md) | [Learning Paths](./learning-paths.md) | [Roadmap](../ROADMAP.md)

---

## 1. Prerequisites & Environment Setup

To follow the roadmap and run all practical labs locally, you will need:
- **Operating System:** macOS (Apple Silicon / Intel), Linux (Ubuntu 22.04+), or Windows with WSL2.
- **Python:** Python 3.11, 3.12, or 3.13.
- **Git:** Git 2.30+.
- **Container Runtime (Optional for MLOps):** Docker Desktop or OrbStack.

### Clone the Repository
```bash
git clone https://github.com/FirdowsRahaman/agentic-ai-and-ml-roadmap.git
cd agentic-ai-and-ml-roadmap
```

---

## 2. Setting Up a Virtual Environment

Create and activate an isolated Python virtual environment:

```bash
# Using standard venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

---

## 3. Running Local Starter Projects

The repository includes two self-contained, 100% local starter projects requiring zero API keys.

### Starter 1: Air-Gapped Hybrid RAG & SQL Router Agent
```bash
cd projects/intermediate/starter-local-rag-agent
pip install -r requirements.txt

# Run the hybrid search & SQL agent demo
python app.py

# Run the test suite
pytest test_router.py
```

### Starter 2: Production MLOps & PSI Drift API
```bash
cd ../intermediate/starter-local-mlops
pip install -r requirements.txt

# Train the zero-pickle model
python src/train.py

# Run test suite
pytest tests/test_api.py
```

---

## 4. Validating Repository Links & Architecture
You can run automated repository integrity checks locally at any time:
```bash
python scripts/validate_links.py
python scripts/validate_structure.py
```
