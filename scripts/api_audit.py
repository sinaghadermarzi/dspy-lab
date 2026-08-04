#!/usr/bin/env python3
"""Audit every dspy.* symbol used in the notebooks against the installed dspy.

Imports only — makes zero LM/network calls. Run from the repo root with dspy
installed:  python scripts/api_audit.py
"""

import inspect
import re
import sys
from pathlib import Path

import nbformat

import dspy

ROOT = Path(__file__).resolve().parent.parent
failures = []


def fail(msg):
    failures.append(msg)


# --- 1. every dotted dspy.* name used in code cells must resolve ------------
used = set()
for path in sorted(ROOT.glob("[01][0-9]-*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for match in re.findall(r"\bdspy(?:\.[A-Za-z_][A-Za-z_0-9]*)+", cell.source):
            used.add(match)

import importlib

for dotted in sorted(used):
    obj = dspy
    try:
        for part in dotted.split(".")[1:]:
            obj = getattr(obj, part)
    except AttributeError:
        try:
            importlib.import_module(dotted)   # e.g. `from dspy.datasets import ...`
        except ImportError:
            fail(f"{dotted} does not resolve on dspy {dspy.__version__}")
print(f"resolved {len(used)} distinct dspy.* names against dspy {dspy.__version__}")

# --- 2. load-bearing signature spot-checks ----------------------------------
def params(fn):
    return inspect.signature(fn).parameters


p = params(dspy.Evaluate.__init__)
if any(v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self" for k, v in p.items()):
    fail("Evaluate.__init__ is no longer keyword-only")

if "eval_kwargs" not in params(dspy.COPRO.compile):
    fail("COPRO.compile lost eval_kwargs")

for arg in ("trainset", "valset", "teacher"):
    if arg not in params(dspy.GEPA.compile):
        fail(f"GEPA.compile lost {arg}")
for arg in ("reflection_lm", "reflection_minibatch_size", "candidate_selection_strategy",
            "component_selector", "use_merge", "track_stats", "auto",
            "max_full_evals", "max_metric_calls"):
    if arg not in params(dspy.GEPA.__init__):
        fail(f"GEPA.__init__ lost {arg}")

for arg in ("trainset", "valset", "program_aware_proposer", "data_aware_proposer",
            "tip_aware_proposer", "minibatch"):
    if arg not in params(dspy.MIPROv2.compile):
        fail(f"MIPROv2.compile lost {arg}")
if params(dspy.MIPROv2.__init__)["auto"].default != "light":
    fail("MIPROv2 auto default changed")

for arg in ("metric", "max_bootstrapped_demos", "max_labeled_demos", "teacher_settings"):
    if arg not in params(dspy.BootstrapFewShot.__init__):
        fail(f"BootstrapFewShot.__init__ lost {arg}")
if "num_candidate_programs" not in params(dspy.BootstrapFewShotWithRandomSearch.__init__):
    fail("BootstrapFewShotWithRandomSearch lost num_candidate_programs")

for arg in ("module", "N", "reward_fn", "threshold"):
    if arg not in params(dspy.Refine.__init__):
        fail(f"Refine.__init__ lost {arg}")

for arg in ("tools", "max_iters"):
    if arg not in params(dspy.ReAct.__init__):
        fail(f"ReAct.__init__ lost {arg}")

lm = params(dspy.LM.__init__)
if lm["temperature"].default is not None or lm["max_tokens"].default is not None:
    fail("dspy.LM temperature/max_tokens defaults changed from None")

if "k" not in params(dspy.LabeledFewShot.__init__):
    fail("LabeledFewShot.__init__ lost k")

# GEPA enforces 5-positional-arg metrics at construction — keep the teaching cell honest.
try:
    dspy.GEPA(metric=lambda example, pred, trace=None: 0.0, auto="light",
              reflection_lm=object())
    fail("GEPA no longer rejects 2-arg metrics at construction")
except Exception:
    pass

# The deep import taught in notebook 04's datasets tour.
try:
    from dspy.datasets.gsm8k import GSM8K, gsm8k_metric  # noqa: F401
except ImportError as e:
    fail(f"deep import of GSM8K failed: {e}")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All API spot-checks passed.")
