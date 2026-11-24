#!/usr/bin/env python3
"""Validates Jupyter notebooks using isolated execution and timeout limits."""

from __future__ import annotations

import glob
import json
import os
import sys
import time


def execute_notebook(nb_path: str, timeout_seconds: int = 60) -> tuple[bool, str]:
    """Executes code cells in an isolated namespace with elapsed time tracking."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    isolated_namespace: dict = {}
    start_time = time.time()

    for i, cell in enumerate(code_cells, start=1):
        if time.time() - start_time > timeout_seconds:
            return False, f"Execution timed out after {timeout_seconds}s at cell {i}"

        source = "".join(cell.get("source", []))
        try:
            exec(compile(source, f"{nb_path}:cell_{i}", "exec"), isolated_namespace)
        except Exception as exc:
            return False, f"Exception in cell {i}: {type(exc).__name__}: {exc}"

    return True, f"Successfully executed {len(code_cells)} code cells in {time.time() - start_time:.2f}s"


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    notebooks = sorted(glob.glob("notebooks/**/*.ipynb", recursive=True))
    if not notebooks:
        print("❌ No notebooks found to validate!")
        return 1

    print(f"Validating {len(notebooks)} Jupyter notebook(s) in isolated execution environments...")
    failed = False

    for nb in notebooks:
        rel_path = os.path.relpath(nb, repo_root).replace("\\", "/")
        success, message = execute_notebook(nb)
        if success:
            print(f"  ✅ {rel_path}: {message}")
        else:
            print(f"  ❌ {rel_path}: {message}")
            failed = True

    if failed:
        print("Notebook validation FAILED!")
        return 1

    print(f"Notebook validation PASSED! Verified {len(notebooks)} notebook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
