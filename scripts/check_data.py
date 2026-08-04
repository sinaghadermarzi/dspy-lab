#!/usr/bin/env python3
"""Integrity checks for the Aldervane micro-world (data/corpus.json, data/qa.json).

Run from the repo root:  python3 scripts/check_data.py

The tokenize/search/norm code below is a copy of the code taught in notebook 03
(kept in sync by scripts/validate_notebooks.py). Checks marked HARD fail the
script; INFO lines are diagnostics for the course's "engineered headroom" claims.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- canonical course code (same as notebook 03) -----------------------------
# >>> world-code
CORPUS = json.loads((ROOT / "data" / "corpus.json").read_text())

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "did", "do", "does", "each", "every",
    "for", "from", "he", "her", "his", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "with",
}

def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    # naive stemming: "captains" -> "captain", "records" -> "record"
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
             for w in words]
    return [w for w in words if w not in STOPWORDS]

DOC_TOKENS = {p["id"]: set(tokenize(p["title"] + " " + p["text"])) for p in CORPUS}
IDF = Counter()
for tokens in DOC_TOKENS.values():
    IDF.update(tokens)
IDF = {t: math.log(len(CORPUS) / df) for t, df in IDF.items()}

def search(query, k=2):
    """Return the k corpus passages that best match the query (keyword + IDF)."""
    q = set(tokenize(query))
    ranked = sorted(
        CORPUS,
        key=lambda p: (-sum(IDF.get(t, 0.0) for t in q & DOC_TOKENS[p["id"]]), p["id"]),
    )
    return ranked[:k]

def norm(s):
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
# <<< world-code
# -----------------------------------------------------------------------------

failures = []

def hard(cond, msg):
    if not cond:
        failures.append(msg)

QA = json.loads((ROOT / "data" / "qa.json").read_text())
splits = {name: QA[name] for name in ("train", "dev", "test")}
all_rows = [r for rows in splits.values() for r in rows]

# Corpus shape
ids = [p["id"] for p in CORPUS]
hard(len(ids) == 30, f"corpus has {len(ids)} passages, expected 30")
hard(len(set(ids)) == len(ids), "duplicate passage ids")
for p in CORPUS:
    n_words = len(p["text"].split())
    hard(n_words <= 65, f"{p['id']} too long ({n_words} words)")

# QA shape
hard(len(splits["train"]) == 24, f"train size {len(splits['train'])} != 24")
hard(len(splits["dev"]) == 16, f"dev size {len(splits['dev'])} != 16")
hard(len(splits["test"]) == 8, f"test size {len(splits['test'])} != 8")
questions = [r["question"] for r in all_rows]
hard(len(set(questions)) == len(questions), "duplicate questions across splits")

id_set = set(ids)
for r in all_rows:
    q = r["question"]
    hard(r["hops"] in (1, 2), f"bad hops for: {q}")
    hard(len(r["gold_ids"]) == r["hops"], f"len(gold_ids) != hops for: {q}")
    hard(all(g in id_set for g in r["gold_ids"]), f"unknown gold id for: {q}")
    hard(bool(norm(r["answer"])), f"answer normalizes to empty for: {q}")
    gold_text = " ".join(
        p["title"] + " " + p["text"] for p in CORPUS if p["id"] in r["gold_ids"]
    )
    hard(norm(r["answer"]) in norm(gold_text), f"answer not in gold passages for: {q}")

# Hop mix
one_hop = sum(1 for r in all_rows if r["hops"] == 1)
frac = one_hop / len(all_rows)
hard(0.30 <= frac <= 0.50, f"1-hop fraction {frac:.2f} outside [0.30, 0.50]")

# Retrieval guarantees / headroom diagnostics
for name, rows in splits.items():
    for r in rows:
        top2 = {p["id"] for p in search(r["question"], k=2)}
        if r["hops"] == 1:
            hard(
                r["gold_ids"][0] in top2,
                f"1-hop gold {r['gold_ids'][0]} not in top-2 for: {r['question']} (got {sorted(top2)})",
            )

anchor_hits = answer_hits = both_hits = n2 = 0
for r in all_rows:
    if r["hops"] != 2:
        continue
    n2 += 1
    top2 = {p["id"] for p in search(r["question"], k=2)}
    anchor_hits += r["gold_ids"][0] in top2
    answer_hits += r["gold_ids"][1] in top2
    both_hits += set(r["gold_ids"]) <= top2

print(f"corpus: {len(ids)} passages | qa: {len(all_rows)} "
      f"(train {len(splits['train'])}/dev {len(splits['dev'])}/test {len(splits['test'])}), "
      f"{one_hop} one-hop / {n2} two-hop")
print(f"INFO 2-hop, naive search(question, k=2): anchor passage found {anchor_hits}/{n2}, "
      f"answer passage found {answer_hits}/{n2}, both found {both_hits}/{n2}")
print("INFO (low 'both found' = headroom for learned hop-2 queries)")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll hard checks passed.")
