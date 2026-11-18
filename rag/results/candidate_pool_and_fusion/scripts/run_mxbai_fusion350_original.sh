#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/edin-mls-26-cxy"

PYTHONPATH=rag "$HOME/mls-rag-env/bin/python" -u - <<'PY'
import json
import os
from math import isfinite
from pathlib import Path
from statistics import mean, median
from time import perf_counter

print("Importing PyTorch and CrossEncoder...")
import torch
from sentence_transformers import CrossEncoder

from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k

root = Path("rag/results")
data = CourseData(Path("rag/data/rag_corpus"))
qrels = data.load_qrels("practice")
rows = [
    row
    for shard in range(7)
    for row in load_jsonl(root / f"bge100-full-3718533-{shard}.jsonl")
]
by_id = {row["query_id"]: row for row in rows}
transcripts = {
    row["query_id"]: row["transcript"]
    for row in load_jsonl(root / "baseline_all_cpu_350_v1.jsonl")
    if row["query_id"] in qrels
}
if len(rows) != 350 or len(qrels) != 350 or by_id.keys() != qrels.keys():
    raise SystemExit("BGE cache does not match the 350 practice queries")
if transcripts.keys() != qrels.keys():
    raise SystemExit("Missing practice transcripts")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable on the allocated node")

model_name = "mixedbread-ai/mxbai-rerank-large-v1"
print(f"Loading {model_name} on {torch.cuda.get_device_name(0)}...")
model = CrossEncoder(model_name, device="cuda")

methods = (
    "Dense Top-5",
    "BGE 100->5",
    "Mixedbread 100->5",
    "Dense+BGE Top-20 fusion",
    "BGE+Mixedbread Top-100 fusion",
)
recalls = {name: [] for name in methods}
fusion_name = methods[-1]
times = []
gains = {name: 0 for name in methods[:-1]}
losses = {name: 0 for name in methods[:-1]}

job_id = os.environ["SLURM_JOB_ID"]
output = root / f"mxbai-fusion350-{job_id}.jsonl"

with output.open("x", encoding="utf-8") as stream:
    for index, qid in enumerate(sorted(by_id), 1):
        row = by_id[qid]
        dense = row["candidate_record_ids"]
        bge_scores = row["candidate_scores"]
        if (len(dense) != 100 or len(set(dense)) != 100
                or len(bge_scores) != 100
                or not all(isfinite(score) for score in bge_scores)):
            raise SystemExit(f"Invalid BGE cache for {qid}")

        bge_by_id = dict(zip(dense, bge_scores))
        bge = sorted(dense, key=bge_by_id.__getitem__, reverse=True)
        bge_rank = {rid: rank for rank, rid in enumerate(bge, 1)}

        bge20 = set(bge[:20])
        dense_local_order = [rid for rid in dense if rid in bge20]
        dense_local_rank = {
            rid: rank for rank, rid in enumerate(dense_local_order, 1)
        }
        dense_bge = sorted(
            bge[:20],
            key=lambda rid: (dense_local_rank[rid] + bge_rank[rid], bge_rank[rid]),
        )

        pairs = [(transcripts[qid], data.record_text(rid)) for rid in dense]
        torch.cuda.synchronize()
        start = perf_counter()
        predictions = model.predict(pairs, batch_size=8, show_progress_bar=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        if getattr(predictions, "shape", None) != (100,):
            raise SystemExit(f"Expected 100 scalar Mixedbread scores for {qid}")
        mixed_scores = [float(score) for score in predictions]
        if not all(isfinite(score) for score in mixed_scores):
            raise SystemExit(f"Non-finite Mixedbread score for {qid}")
        times.append(elapsed)

        mixed_by_id = dict(zip(dense, mixed_scores))
        mixed = sorted(dense, key=mixed_by_id.__getitem__, reverse=True)
        mixed_rank = {rid: rank for rank, rid in enumerate(mixed, 1)}
        fused = sorted(
            dense,
            key=lambda rid: (bge_rank[rid] + mixed_rank[rid], bge_rank[rid]),
        )

        top5 = dict(zip(methods, (
            dense[:5], bge[:5], mixed[:5], dense_bge[:5], fused[:5]
        )))
        labels = set(qrels[qid])
        final_hits = labels.intersection(top5[fusion_name])
        for name, ranking in top5.items():
            recalls[name].append(recall_at_k(ranking, labels, 5))
            if name != fusion_name:
                baseline_hits = labels.intersection(ranking)
                gains[name] += len(final_hits - baseline_hits)
                losses[name] += len(baseline_hits - final_hits)

        print(json.dumps({
            "query_id": qid,
            "candidate_record_ids": dense,
            "mixedbread_scores": mixed_scores,
            "mixedbread_scoring_seconds": elapsed,
            "top5": top5,
        }), file=stream, flush=True)

        if index == 1 or index % 10 == 0:
            print(f"{index}/350 complete")

print("\n=== Same 350 practice queries ===")
for name in methods:
    print(f"{name:<32} {mean(recalls[name]):.2%}")

for name in methods[1:4]:
    differences = [
        after - before
        for after, before in zip(recalls[fusion_name], recalls[name])
    ]
    up = sum(delta > 0 for delta in differences)
    down = sum(delta < 0 for delta in differences)
    print(
        f"Fusion vs {name}: {100 * mean(differences):+.2f} pp | "
        f"{up} up, {len(differences) - up - down} same, {down} down | "
        f"marked hits gained/lost {gains[name]}/{losses[name]}"
    )

print(f"Median Mixedbread GPU scoring time: {median(times):.2f} s/query")
print(f"Saved per-query scores and rankings: {output}")
PY
