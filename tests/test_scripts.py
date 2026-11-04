"""The repo's static checkers, run as tests so `pytest` is the single entry point.

None of these execute a notebook or call an LM. `api_audit.py` imports dspy, so it
is skipped when dspy is not installed (the CI job installs requirements.txt).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )


def test_validate_notebooks_passes():
    result = run("validate_notebooks.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed for 13 notebooks." in result.stdout


def test_check_data_passes():
    result = run("check_data.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All hard checks passed." in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("dspy") is None, reason="dspy not installed")
def test_api_audit_passes():
    result = run("api_audit.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All API spot-checks passed." in result.stdout


def test_toc_lists_every_notebook_once():
    """The Jupyter Book TOC must cover exactly the 13 course notebooks."""
    toc = (ROOT / "_toc.yml").read_text()
    listed = [line.split("file:", 1)[1].strip() for line in toc.splitlines() if "file:" in line]
    notebooks = sorted(p.stem for p in ROOT.glob("[01][0-9]-*.ipynb"))
    assert sorted(listed) == notebooks
    assert len(listed) == len(set(listed))
