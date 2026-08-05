#!/usr/bin/env python3
"""Static validation of the course notebooks (no execution, no LM calls).

Run from the repo root:  python scripts/validate_notebooks.py
Requires: nbformat (see requirements.txt).

Checks: nbformat validity; Python syntax of every code cell; no shell/cell
magics; byte-identity of the frozen config/boilerplate cells across notebooks;
presence and order of the course's standard sections; arXiv citations against
the verified allowlist; drift between notebook 03's search code and
scripts/check_data.py.
"""

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(ROOT.glob("[01][0-9]-*.ipynb"))

CONFIG_SENTINEL = "# === config (identical in every notebook) ==="
WORLD_SENTINEL = "# === boilerplate: aldervane micro-world — same code as notebook 03 ==="
EVAL_SENTINEL = "# === boilerplate: data + metric — same code as notebook 04 ==="
PHOENIX_SENTINEL = "# optional: Phoenix tracing (see notebook 11)"
MLFLOW_SENTINEL = "# optional: MLflow tracing (see notebook 11)"

VERIFIED_ARXIV = {
    "2310.03714",  # DSPy compiler paper (Khattab et al.)
    "2406.11695",  # MIPROv2 (Opsahl-Ong et al.)
    "2507.19457",  # GEPA (Agrawal et al.)
    "2407.10930",  # BetterTogether (Soylu et al.)
    "2210.03629",  # ReAct (Yao et al.)
    "2211.12588",  # Program of Thoughts (Chen et al.)
    "2309.03409",  # OPRO (Yang et al.)
    "2201.11903",  # Chain-of-Thought (Wei et al.)
    "2205.11916",  # zero-shot CoT (Kojima et al.)
}
REQUIRED_CITATIONS = {
    "03": {"2201.11903"},
    "05": {"2310.03714"},
    "06": {"2406.11695", "2309.03409"},
    "07": {"2507.19457"},
    "08": {"2210.03629", "2211.12588"},
    "09": {"2407.10930"},
}

failures = []


def fail(msg):
    failures.append(msg)


def code_cells(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def md_text(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


assert len(NOTEBOOKS) == 13, f"expected 13 notebooks, found {len(NOTEBOOKS)}"

frozen = {CONFIG_SENTINEL: {}, WORLD_SENTINEL: {}, EVAL_SENTINEL: {},
          PHOENIX_SENTINEL: {}, MLFLOW_SENTINEL: {}}

for path in NOTEBOOKS:
    name = path.name
    nb = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(nb)
    except Exception as e:
        fail(f"{name}: nbformat validation error: {e}")

    for i, src in enumerate(code_cells(nb)):
        for line in src.splitlines():
            if line.lstrip().startswith(("!", "%")):
                fail(f"{name}: cell {i} uses a shell/line magic: {line.strip()[:50]}")
        try:
            compile(src, f"<{name}:code-cell-{i}>", "exec")
        except SyntaxError as e:
            fail(f"{name}: syntax error in code cell {i}: {e}")

        first = src.splitlines()[0] if src.splitlines() else ""
        if first in frozen:
            frozen[first][name] = src

    md = md_text(nb)
    required = ["What you'll learn", "## Recap", "## Exercises"]
    pos = -1
    for section in required:
        p = md.find(section)
        if p == -1:
            fail(f"{name}: missing section {section!r}")
        elif p < pos:
            fail(f"{name}: section {section!r} out of order")
        else:
            pos = p
    if "**What you should see" not in md:
        fail(f"{name}: no 'What you should see' callout")
    if name[:2] != "12" and "Next up" not in md:
        fail(f"{name}: missing 'Next up' teaser")
    for ch in md:
        if 0x1F000 <= ord(ch) <= 0x1FAFF:
            fail(f"{name}: contains emoji {ch!r}")
            break

    cited = set(re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", md))
    unknown = cited - VERIFIED_ARXIV
    if unknown:
        fail(f"{name}: unverified arXiv ids {sorted(unknown)}")
    missing = REQUIRED_CITATIONS.get(name[:2], set()) - cited
    if missing:
        fail(f"{name}: missing required citations {sorted(missing)}")

# Frozen-block identity
if len(frozen[CONFIG_SENTINEL]) != 13:
    fail(f"config cell present in {sorted(frozen[CONFIG_SENTINEL])} — expected all 13 notebooks")
for sentinel, found in frozen.items():
    variants = set(found.values())
    if len(variants) > 1:
        fail(f"frozen block {sentinel!r} differs across {sorted(found)}")
for sentinel, found in frozen.items():
    label = sentinel.strip("# =").split("—")[0].strip()
    print(f"frozen block [{label}] in notebooks {sorted(n[:2] for n in found)}")

# Drift check: notebook 03's search code vs scripts/check_data.py
check_data_src = (ROOT / "scripts" / "check_data.py").read_text()
nb03 = nbformat.read(ROOT / "03-modules-and-the-graph.ipynb", as_version=4)
nb03_code = "\n".join(code_cells(nb03))
for marker in [
    'w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w',
    'key=lambda p: (-sum(IDF.get(t, 0.0) for t in q & DOC_TOKENS[p["id"]]), p["id"]),',
]:
    if marker not in nb03_code:
        fail(f"notebook 03 lost canonical search code: {marker[:50]}...")
    if marker not in check_data_src:
        fail(f"scripts/check_data.py drifted from notebook 03: {marker[:50]}...")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"\nAll checks passed for {len(NOTEBOOKS)} notebooks.")
