#!/usr/bin/env python3
"""Validates Jupyter notebooks using nbformat and nbclient in isolated execution environments."""

from __future__ import annotations

import glob
import os
import sys
import time

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError
except ImportError:
    print("❌ nbformat and nbclient are required to run notebook validation.")
    print("Install them via: pip install nbformat nbclient ipykernel")
    sys.exit(1)


def execute_notebook(nb_path: str, timeout_seconds: int = 120) -> tuple[bool, str]:
    """Executes code cells in a separate disposable Jupyter kernel process with hard timeouts."""
    start_time = time.time()
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
        if not code_cells:
            return True, "No code cells to execute"

        # Execute notebook via isolated Jupyter kernel process with per-cell timeout
        client = NotebookClient(
            nb,
            timeout=timeout_seconds,
            kernel_name="python3",
            record_timing=False,
        )
        client.execute()

        duration = time.time() - start_time
        return True, f"Successfully executed {len(code_cells)} code cell(s) via isolated kernel in {duration:.2f}s"

    except CellTimeoutError as timeout_err:
        return False, f"Execution timed out ({timeout_seconds}s limit exceeded): {timeout_err}"
    except CellExecutionError as exec_err:
        return False, f"Cell execution failed: {exec_err}"
    except Exception as exc:
        return False, f"Unexpected error during execution: {type(exc).__name__}: {exc}"


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    notebooks = sorted(glob.glob("notebooks/**/*.ipynb", recursive=True))
    if not notebooks:
        print("❌ No notebooks found to validate!")
        return 1

    print(f"Validating {len(notebooks)} Jupyter notebook(s) using nbclient in isolated kernel environments...")
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

    print(f"Notebook validation PASSED! Verified {len(notebooks)} notebook(s) successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
