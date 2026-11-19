"""Reconstructed verification of the paired bootstrap confidence intervals."""

import json
from pathlib import Path
import numpy as np

root = Path("rag/results")
archive = root / "candidate_pool_and_fusion"
rerank = root / "dense20_to5_rerankers"
fusion = archive / "rank_fusion"

def load(path):
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    scores = {
        row["query_id"]: float(row["normalised_answer_score"])
        for row in rows
    }
    assert len(rows) == len(scores) == 350, path
    assert all(0 <= s <= 1 for s in scores.values())
    return scores

dense = load(rerank / "dense/answer_judge_per_query.jsonl")

original_dense = load(
    root / "system_extension/retrieval_depth/repeat5_practice350/k5_per_query.jsonl"
)

assert dense == original_dense

ids = sorted(dense)
baseline = np.array([dense[q] for q in ids])

rng = np.random.default_rng(42)
indices = rng.integers(0, 350, size=(10000, 350))

configs = [
    (
        "BGE 20-to-5",
        rerank / "bge/answer_judge_per_query.jsonl",
        (-0.0238, 0.0281)
    ),
    (
        "Mixedbread 20-to-5",
        rerank / "mixedbread/answer_judge_per_query.jsonl",
        (-0.0028, 0.0521)
    ),
    (
        "Fusion BGE tie-break",
        fusion / "bge_tiebreak/judge/answer_judge_per_query.jsonl",
        (-0.0173, 0.0421)
    ),
    (
        "Fusion Mixedbread tie-break",
        fusion / "mixedbread_tiebreak/judge/answer_judge_per_query.jsonl",
        (-0.0215, 0.0362)
    )
]

print("Queries: 350")
print("Bootstrap repetitions: 10000")
print("Random seed: 42")
print(f"Dense mean: {baseline.mean():.6f}")

for name, path, reported in configs:
    scores = load(path)
    assert scores.keys() == dense.keys()

    values = np.array([scores[q] for q in ids])
    delta = values - baseline

    boot = delta[indices].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n{name}")
    print(f"Mean R5: {values.mean():.6f}")
    print(f"Delta R5: {delta.mean():+.6f}")
    print(f"Recomputed CI: [{lo:+.6f}, {hi:+.6f}]")
    print(f"Previously reported CI: {reported}")
    print(f"Rounded match: {tuple(round(x, 4) for x in (lo, hi)) == reported}")
    print(f"Contains zero: {lo <= 0 <= hi}")
