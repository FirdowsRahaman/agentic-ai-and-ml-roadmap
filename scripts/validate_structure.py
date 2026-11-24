#!/usr/bin/env python3
"""Validates the repository information architecture, naming conventions, and required structure."""

from __future__ import annotations

import os
import sys

REQUIRED_ROOT_FILES = [
    "README.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "CHANGELOG.md",
    "mkdocs.yml",
    "requirements-docs.txt",
    ".gitignore",
]

REQUIRED_CURRICULUM_DIRS = [
    "curriculum/01-foundations",
    "curriculum/02-machine-learning",
    "curriculum/03-deep-learning",
    "curriculum/04-generative-ai",
    "curriculum/05-agentic-ai",
    "curriculum/06-production-ai",
]

EXPECTED_MODULES = [
    "curriculum/01-foundations/01_python_sql_and_data_stack.md",
    "curriculum/01-foundations/02_math_and_statistics.md",
    "curriculum/02-machine-learning/03_classical_ml_and_recsys.md",
    "curriculum/03-deep-learning/04_deep_learning_and_rl.md",
    "curriculum/04-generative-ai/05_nlp_cv_and_multimodal.md",
    "curriculum/04-generative-ai/06_llms_finetuning_and_rag.md",
    "curriculum/05-agentic-ai/07_ai_agents_fundamentals.md",
    "curriculum/05-agentic-ai/08_tools_langgraph_and_adk.md",
    "curriculum/05-agentic-ai/09_multi_agent_and_mcp.md",
    "curriculum/05-agentic-ai/10_agent_evals_and_safety.md",
    "curriculum/06-production-ai/11_mlops_cloud_and_finops.md",
    "curriculum/06-production-ai/12_system_troubleshooting_scenarios.md",
    "curriculum/06-production-ai/13_algorithms_from_scratch.md",
]

REQUIRED_NOTEBOOK_DIRS = [
    "notebooks/foundations",
    "notebooks/machine-learning",
    "notebooks/deep-learning",
    "notebooks/generative-ai",
    "notebooks/agentic-ai",
]

REQUIRED_RESOURCE_FILES = [
    "resources/books.md",
    "resources/courses.md",
    "resources/research-papers.md",
    "resources/tools.md",
]

FORBIDDEN_PATTERNS = [
    ".pytest_cache",
    "__pycache__",
]


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    errors: list[str] = []

    # 1. Check required root files
    for rf in REQUIRED_ROOT_FILES:
        if not os.path.isfile(rf):
            errors.append(f"Missing required root file: {rf}")

    # 2. Check required curriculum directories and modules
    for cd in REQUIRED_CURRICULUM_DIRS:
        if not os.path.isdir(cd):
            errors.append(f"Missing curriculum directory: {cd}")

    for em in EXPECTED_MODULES:
        if not os.path.isfile(em):
            errors.append(f"Missing curriculum module: {em}")

    # 3. Check notebook directories
    for nd in REQUIRED_NOTEBOOK_DIRS:
        if not os.path.isdir(nd):
            errors.append(f"Missing notebook directory: {nd}")

    # 4. Check resources files
    for res in REQUIRED_RESOURCE_FILES:
        if not os.path.isfile(res):
            errors.append(f"Missing resource file: {res}")

    # 5. Check for forbidden cache artifacts
    for root, dirs, files in os.walk("."):
        if ".git" in root:
            continue
        for d in dirs:
            if d in FORBIDDEN_PATTERNS:
                errors.append(f"Forbidden cache directory found: {os.path.join(root, d)}")
        for f in files:
            if f.endswith(".pyc") or f == ".DS_Store":
                errors.append(f"Forbidden artifact file found: {os.path.join(root, f)}")

    if errors:
        print(f"Structure validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  ❌ {err}")
        return 1

    print("Structure validation PASSED! All directories and standard conventions verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
