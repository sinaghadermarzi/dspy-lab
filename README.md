# dspy-lab — learn DSPy layer by layer

[![CI](https://github.com/sinaghadermarzi/dspy-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/sinaghadermarzi/dspy-lab/actions/workflows/ci.yml)
[![Site](https://github.com/sinaghadermarzi/dspy-lab/actions/workflows/site.yml/badge.svg)](https://sinaghadermarzi.github.io/dspy-lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A course of twelve bite-sized Jupyter notebooks — plus a bonus deep dive — that teach [DSPy](https://dspy.ai)'s abstractions from the ground up: you build a multi-hop QA program over a fictional micro-world the model has never seen, define a metric, and then watch a series of increasingly clever optimizers improve the program — reading the learned prompts at every step. The finale reruns the entire workflow on real data (HotPotQA).

Written against **DSPy 3.3.0**. Each notebook is self-contained (open any of them cold), takes 15–40 minutes, and states its own cost up front.

## The mental model

The course leans on one analogy throughout — if you know PyTorch, you already know the shape of DSPy:

| PyTorch | DSPy |
|---|---|
| `nn.Module` / `forward()` | `dspy.Module` / `forward()` (define-by-run graph) |
| layer shapes | signatures — `"question -> answer"` (typed I/O contracts) |
| weights θ | per-predictor **instructions + few-shot demos** |
| loss function | `metric(example, prediction)` — plain Python you write |
| `optimizer.step()` | `teleprompter.compile(program, trainset=...)` |
| eval loop | `dspy.Evaluate(devset=..., metric=...)` |
| `state_dict()` | `program.save("prog.json")` — human-readable learned prompts |
| TensorBoard / profiler | MLflow tracing (or Arize Phoenix) — notebook 11 |

## Curriculum

| # | Notebook | You learn | Est. cost* |
|---|----------|-----------|-----------|
| 00 | [Hello, DSPy](00-hello-dspy.ipynb) | the big idea; `dspy.LM` / `configure` / `context`; caching | < $0.01 |
| 01 | [Signatures & Predict](01-signatures-and-predict.ipynb) | typed I/O contracts; `Prediction`; `inspect_history` | < $0.02 |
| 02 | [Under the hood](02-under-the-hood-parameters.ipynb) | adapters (signature → prompt); `Example`; **θ = instructions + demos**; `save()` | < $0.02 |
| 03 | [Modules & the graph](03-modules-and-the-graph.ipynb) | custom modules; LM + plain-Python computation graphs; `named_predictors` | ~$0.03 |
| 04 | [Data & evaluation](04-data-and-evaluation.ipynb) | splits; metrics & the `trace` contract; `Evaluate`; error analysis | ~$0.05 |
| 05 | [Bootstrap few-shot](05-bootstrap-fewshot.ipynb) | the program teaches itself demos; random search; DSPy-paper map | ~$0.15–0.30 |
| 06 | [Instruction optimization](06-instruction-optimization-miprov2.ipynb) | COPRO; MIPROv2 (grounded proposal + Bayesian search); paper maps | ~$0.30–0.60 |
| 07 | [Feedback & GEPA](07-feedback-and-gepa.ipynb) | feedback metrics; reflective evolution; Pareto pools; GEPA-paper map | ~$0.40–0.80 |
| 08 | [Agents & inference-time](08-agents-and-inference-time.ipynb) | ReAct + tools; dynamic graphs; `Refine` / `BestOfN` | ~$0.05 |
| 09 | [The optimizer landscape](09-optimizer-landscape.ipynb) | finetuning & BetterTogether; ensembles; **which optimizer when** | ~$0–0.05 |
| 10 | [Capstone: HotPotQA](10-capstone-hotpotqa.ipynb) | the whole workflow on real data, end to end | ~$1–2 |
| 11 | [Observability & tracing](11-observability-and-tracing.ipynb) | `BaseCallback` hooks; MLflow traces + optimizer runs; Arize Phoenix option | ~$0.05 |
| 12 | [Bonus: under the LLM layer](12-bonus-llm-layer-deep-dive.ipynb) | adapters on the wire; the format-fallback ladder; `response_format` protocol; constrained decoding | < $0.01 |

\* at the default model (`deepseek-v3.2` via OpenRouter). **Whole course ≈ $2.50–5** (typically less at the default). DSPy's disk cache means re-running cells and notebooks is free.

Run them in order — concepts build strictly layer by layer — but every notebook rebuilds its world from scratch, so returning to any one of them later works fine.

**Read it online:** the notebooks render as a book at [sinaghadermarzi.github.io/dspy-lab](https://sinaghadermarzi.github.io/dspy-lab/), built by `.github/workflows/site.yml` from the same files without executing them (build it locally with `pip install "jupyter-book<2" && jupyter-book build .`). The notebooks in git carry no outputs; run them with your own key to see the learned prompts and scores.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env: your API key, and (optionally) your models
jupyter lab
```

**Keys & models — one `.env` for the whole course.** Every notebook's config cell reads `.env` first, so the model choice is global: set `MODEL` once and all twelve notebooks follow (`STRONG_MODEL` is the pricier model a few notebooks use as a teacher/reflection LM). No `.env`? The config cell falls back to the defaults — `openrouter/deepseek/deepseek-v3.2` and `openrouter/deepseek/deepseek-v4-flash` — and prompts for the key. A per-notebook override stays one uncomment away in each config cell (code beats `.env`).

The defaults route through [OpenRouter](https://openrouter.ai) (one key, every major model — handy for the "swap the model, keep the program" experiments; create a key at openrouter.ai → Keys). Any [LiteLLM provider](https://docs.litellm.ai/docs/providers) string works instead (`openai/...`, `anthropic/...`, `gemini/...`, `ollama_chat/...`); model slugs current as of authoring — check [openrouter.ai/models](https://openrouter.ai/models) if one 404s.

**Caching.** DSPy caches every LM call in `~/.dspy_cache` (on by default). Interrupting and re-running notebooks is safe and free; only genuinely new prompts cost money.

**Observability (optional).** Notebook 11 teaches tracing; every other notebook carries two small optional sections right after its config cell — run the Phoenix cell or the MLflow cell (`obs.enable_phoenix()` / `obs.enable_mlflow()` + `obs.mlflow_ui()`) to trace that notebook, or skip both — either way the cell ends by printing the localhost link where the traces appear. The `obs.py` helper makes this safe to leave and return to: sessions are dated (`dspy-lab-YYYY-MM-DD`) so days never collide, the MLflow store is pinned inside the repo (gitignored) and falls back cleanly if a remnant file is unusable, and the UIs are found-or-started from a cell — reusing a healthy server, skipping ports held by strangers. No terminal, no accounts, no extra API keys; the only thing you do outside Jupyter is click the printed localhost link. Phoenix keeps its data in `~/.phoenix`.

## Repo map

```
00-…-12-….ipynb        the course (12 is an optional bonus deep dive)
.env.example           copy to .env: API key + global MODEL / STRONG_MODEL choice
obs.py                 optional tracing helpers: dated sessions, find-or-start UIs (notebook 11)
data/corpus.json       the Aldervane micro-world: 30 fictional passages (so the LM can't cheat)
data/qa.json           48 hand-written QA pairs — train 24 / dev 16 / test 8, with gold passage ids
artifacts/             created at runtime by program.save() (gitignored)
scripts/check_data.py           data integrity + retrieval-guarantee checks
scripts/validate_notebooks.py   structural/citation/frozen-block checks (no execution)
scripts/api_audit.py            audits every dspy.* symbol used against the installed dspy
tests/                          pytest: runs the three scripts + unit tests for obs.py
_config.yml, _toc.yml           Jupyter Book site definition (.github/workflows/site.yml)
```

## Checks

Nothing below executes a notebook or calls an LM, so it all runs without a key:

```bash
python scripts/validate_notebooks.py   # nbformat validity, cell syntax, frozen blocks, citations
python scripts/check_data.py           # micro-world invariants and retrieval guarantees
python scripts/api_audit.py            # every dspy.* name used still resolves on the pinned dspy
python -m pytest tests -q              # the three above, plus unit tests for obs.py
```

`.github/workflows/ci.yml` runs the same four steps on every push and pull request. Executing the notebooks end to end is deliberately not in CI: it needs an OpenRouter key and costs money.

## License

MIT — see [LICENSE](LICENSE). To cite the course, use [CITATION.cff](CITATION.cff).

## Coming from older DSPy tutorials?

Three breaking changes in DSPy 3.x that make pre-3.x snippets fail, all covered in the notebooks: per-provider clients (`dspy.OpenAI(...)`) are gone — use `dspy.LM("provider/model")`; `dspy.Assert` / `dspy.Suggest` are gone — use `dspy.Refine` / `dspy.BestOfN` (notebook 08); `TypedPredictor` is gone — plain `Predict` handles typed signatures natively.

## Papers mapped in the notebooks

Wherever a component comes from research, the notebook embeds a *paper concept ↔ code* table:

- DSPy compiler paper — Khattab et al., [arXiv:2310.03714](https://arxiv.org/abs/2310.03714) (notebook 05)
- MIPROv2 — Opsahl-Ong et al., [arXiv:2406.11695](https://arxiv.org/abs/2406.11695) (notebook 06)
- OPRO — Yang et al., [arXiv:2309.03409](https://arxiv.org/abs/2309.03409) (notebook 06)
- GEPA — Agrawal et al., [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) (notebook 07)
- Chain-of-Thought — Wei et al., [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) (notebook 03)
- ReAct — Yao et al., [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) (notebook 08)
- Program of Thoughts — Chen et al., [arXiv:2211.12588](https://arxiv.org/abs/2211.12588) (notebook 08)
- BetterTogether — Soylu et al., [arXiv:2407.10930](https://arxiv.org/abs/2407.10930) (notebook 09)

## Which optimizer, in one breath

Few labels → `LabeledFewShot`. Metric + ~20 examples → `BootstrapFewShot` (+`RandomSearch`). Want machine-written 0-shot prompts → `MIPROv2`. Your metric can *explain* failures in text → `GEPA`. Own the weights → `BootstrapFinetune` / `BetterTogether`. High stakes per call → wrap in `Refine` / `BestOfN` or `Ensemble`. The full decision guide is notebook 09.
