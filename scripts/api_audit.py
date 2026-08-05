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
        # negative lookbehind: don't match `dspy` inside dotted paths like mlflow.dspy.autolog
        for match in re.findall(r"(?<![\w.])dspy(?:\.[A-Za-z_][A-Za-z_0-9]*)+", cell.source):
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

# --- 2b. LLM-layer spot-checks (bonus notebook 12) ---------------------------
# The bonus notebook teaches the fallback ladder from source and uses one private
# helper; both must survive a dspy bump or the notebook's cells go stale.
if "use_json_adapter_fallback" not in params(dspy.ChatAdapter.__init__):
    fail("ChatAdapter.__init__ lost use_json_adapter_fallback (notebook 12's ladder opt-out)")
try:
    from dspy.adapters.json_adapter import _get_structured_outputs_response_format
    # Exercise the exact call notebook 12 makes: one positional arg, then .model_json_schema().
    _sig = dspy.ChainOfThought("question -> answer").predict.signature
    _schema = _get_structured_outputs_response_format(_sig).model_json_schema()
    _props = _schema.get("properties", {})
    if "reasoning" not in _props or "answer" not in _props:
        fail(f"schema builder no longer emits the output fields (got {sorted(_props)})")
except ImportError as e:
    fail(f"private schema builder moved (notebook 12 pins it): {e}")
except Exception as e:
    fail(f"schema builder call as notebook 12 uses it failed: {type(e).__name__}: {e}")
for adapter in ("ChatAdapter", "JSONAdapter", "XMLAdapter", "TwoStepAdapter"):
    if not hasattr(dspy, adapter):
        fail(f"dspy.{adapter} missing (notebook 12 ladder/exercises)")

# --- 3. observability spot-checks (notebook 11) -----------------------------
from dspy.utils.callback import BaseCallback

hooks = [f"on_{kind}_{edge}"
         for kind in ("module", "lm", "adapter_format", "adapter_parse", "tool", "evaluate")
         for edge in ("start", "end")]
for h in hooks:
    if not hasattr(BaseCallback, h):
        fail(f"BaseCallback lost hook {h}")
if not hasattr(dspy, "Tool"):
    fail("dspy.Tool missing (notebook 11, exercise 2)")

try:
    import mlflow.dspy
    autolog_params = set(params(mlflow.dspy.autolog))
    if not {"log_compiles", "log_evals", "log_traces_from_compile", "disable"} <= autolog_params:
        fail(f"mlflow.dspy.autolog lost expected kwargs (has {sorted(autolog_params)})")
except ImportError:
    print("mlflow not installed — skipped autolog signature check")

try:
    from phoenix.otel import register
    if "auto_instrument" not in params(register):
        fail("phoenix.otel.register lost auto_instrument")
except ImportError:
    print("arize-phoenix not installed — skipped Phoenix register check")

# --- 4. the obs helper module used by the optional tracing cells ------------
sys.path.insert(0, str(ROOT))
try:
    import obs
    for fn in ("enable_mlflow", "mlflow_ui", "enable_phoenix"):
        if not callable(getattr(obs, fn, None)):
            fail(f"obs.{fn} missing or not callable")
except Exception as e:
    fail(f"import obs failed: {e}")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All API spot-checks passed.")
