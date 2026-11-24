"""Automated Pytest Suite verifying repository structure, link integrity, and notebooks."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_structure_validation():
    script = REPO_ROOT / "scripts" / "validate_structure.py"
    res = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"Structure validation failed:\n{res.stdout}\n{res.stderr}"


def test_link_validation():
    script = REPO_ROOT / "scripts" / "validate_links.py"
    res = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"Link validation failed:\n{res.stdout}\n{res.stderr}"


def test_notebooks_validation():
    script = REPO_ROOT / "scripts" / "validate_notebooks.py"
    res = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"Notebook validation failed:\n{res.stdout}\n{res.stderr}"
