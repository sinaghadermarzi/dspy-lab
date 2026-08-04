# dspy-lab — learn DSPy layer by layer

A course of eleven bite-sized Jupyter notebooks that teach [DSPy](https://dspy.ai)'s abstractions from the ground up: you build a multi-hop QA program over a fictional micro-world the model has never seen, define a metric, and then watch a series of increasingly clever optimizers improve the program — reading the learned prompts at every step. The finale reruns the entire workflow on real data (HotPotQA).

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

\* at the default model (`gpt-4o-mini` via OpenRouter). **Whole course ≈ $2.50–5.** DSPy's disk cache means re-running cells and notebooks is free.

Run them in order — concepts build strictly layer by layer — but every notebook rebuilds its world from scratch, so returning to any one of them later works fine.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

**API key.** The notebooks default to [OpenRouter](https://openrouter.ai) (one key, every major model — handy for the "swap the model, keep the program" experiments). Create a key at openrouter.ai → Keys, then:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

or just run a notebook's config cell and paste the key when prompted. Not an OpenRouter user? Every notebook's config cell is one line away from any [LiteLLM provider](https://docs.litellm.ai/docs/providers) (`openai/...`, `anthropic/...`, `gemini/...`, `ollama_chat/...`); model slugs current as of authoring — check [openrouter.ai/models](https://openrouter.ai/models) if one 404s.

**Caching.** DSPy caches every LM call in `~/.dspy_cache` (on by default). Interrupting and re-running notebooks is safe and free; only genuinely new prompts cost money.

## Repo map

```
00-…-10-….ipynb        the course
data/corpus.json       the Aldervane micro-world: 30 fictional passages (so the LM can't cheat)
data/qa.json           48 hand-written QA pairs — train 24 / dev 16 / test 8, with gold passage ids
artifacts/             created at runtime by program.save() (gitignored)
scripts/check_data.py           data integrity + retrieval-guarantee checks
scripts/validate_notebooks.py   structural/citation/frozen-block checks (no execution)
scripts/api_audit.py            audits every dspy.* symbol used against the installed dspy
```

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
