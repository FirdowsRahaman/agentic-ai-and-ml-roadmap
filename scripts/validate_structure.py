#!/usr/bin/env python3
"""Validates the repository information architecture, naming conventions, and required structure."""

from __future__ import annotations

import os
import subprocess
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
    "curriculum/02-core-ml-and-dl",
    "curriculum/03-modern-ai-and-genai",
    "curriculum/04-agentic-ai",
    "curriculum/05-production-and-practice",
    "curriculum/05-production-and-practice/13_algorithms_and_patterns_from_scratch",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges",
]

EXPECTED_MODULES = [
    # Phase 1: Foundations
    "curriculum/01-foundations/01_python_sql_and_data_stack.md",
    "curriculum/01-foundations/02_math_and_statistics.md",
    # Phase 2: Core ML & Deep Learning
    "curriculum/02-core-ml-and-dl/03_classical_ml_and_recsys.md",
    "curriculum/02-core-ml-and-dl/04a_deep_learning_and_pytorch.md",
    "curriculum/02-core-ml-and-dl/04b_advanced_dl_and_rl.md",
    # Phase 3: Modern AI & GenAI
    "curriculum/03-modern-ai-and-genai/05a_nlp_and_language_models.md",
    "curriculum/03-modern-ai-and-genai/05b_computer_vision_and_multimodal.md",
    "curriculum/03-modern-ai-and-genai/06a_llm_architecture_and_inference.md",
    "curriculum/03-modern-ai-and-genai/06b_rag_engineering.md",
    "curriculum/03-modern-ai-and-genai/06c_finetuning_and_serving.md",
    # Phase 4: Agentic AI
    "curriculum/04-agentic-ai/07_agent_fundamentals.md",
    "curriculum/04-agentic-ai/08_agent_orchestration.md",
    "curriculum/04-agentic-ai/09_multi_agent_mcp_a2a.md",
    "curriculum/04-agentic-ai/10_agent_evaluation_and_safety.md",
    # Phase 5: Production & Practice
    "curriculum/05-production-and-practice/11_production_ai_engineering.md",
    "curriculum/05-production-and-practice/12_incident_and_troubleshooting_labs.md",
    "curriculum/05-production-and-practice/13_algorithms_and_patterns_from_scratch/13a_ml_algorithms.md",
    "curriculum/05-production-and-practice/13_algorithms_and_patterns_from_scratch/13b_system_patterns.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/README.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/01_production_ai_assistant.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/02_research_agent.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/03_production_rag_agent.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/04_multi_agent_research_team.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/05_customer_support_multi_agent.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/06_coding_agent.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/07_data_analyst_agent.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/08_enterprise_workflow_agent.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/09_reliability_and_resilience.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/10_deployment_and_observability.md",
    "curriculum/05-production-and-practice/14_production_ai_agent_challenges/11_capstone_multi_agent_system.md",
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

    # 5. Check that no forbidden cache or bytecode files are tracked by Git
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if res.returncode == 0:
        tracked_files = res.stdout.splitlines()
        for tf in tracked_files:
            if tf.endswith(".pyc") or ".pytest_cache" in tf or "__pycache__" in tf or tf.endswith(".DS_Store"):
                errors.append(f"Forbidden artifact tracked by Git: {tf}")

    if errors:
        print(f"Structure validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  ❌ {err}")
        return 1

    print("Structure validation PASSED! All directories, files, and Git conventions verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
