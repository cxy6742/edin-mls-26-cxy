import json
import os
from math import isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from sentence_transformers import CrossEncoder
from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k

root = Path("rag/results")
data = CourseData(Path("rag/data/rag_corpus"))
qrels = data.load_qrels("practice")
rows = []
for shard in range(7):
    rows.extend(load_jsonl(
    root / "candidate_pool_and_fusion" / "candidate_pool"
    / f"bge100-full-3718533-{shard}.jsonl"
))

if len(rows) != 350 or {r["query_id"] for r in rows} != set(qrels) or len(qrels) != 350:
    raise ValueError("Expected all 350 practice queries exactly once")

baseline = root / "dense20_to5_rerankers/dense/answers.jsonl"
baseline_rows = load_jsonl(baseline)
transcripts = {r["query_id"]: r["transcript"] for r in baseline_rows}
if len(baseline_rows) != 350 or set(transcripts) != set(qrels):
    raise ValueError("Baseline transcripts do not match practice queries")
for row in rows:
    row["transcript"] = transcripts[row["query_id"]]

for row in rows:
    ids = row["candidate_record_ids"]
    if len(ids) != 100 or len(set(ids)) != 100 or not row["transcript"].strip():
        raise ValueError(f"Invalid input for {row['query_id']}")

device = torch.cuda.get_device_name(0)
print(f"GPU: {device}", flush=True)
if "H200" not in device or "1g.18gb" not in device:
    raise RuntimeError(f"Unexpected GPU: {device}")

model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device="cuda")
print(f"Batch size: 16 | Max length: {model.max_length}", flush=True)
output = root / f"mxbai100-h200-timing-{os.environ['SLURM_JOB_ID']}.jsonl"
recalls = []
scoring_seconds = 0.0

print("Starting timed reranking", flush=True)
torch.cuda.synchronize()
start = perf_counter()
with output.open("x", encoding="utf-8") as stream:
    for index, row in enumerate(rows, 1):
        qid = row["query_id"]
        ids = row["candidate_record_ids"]
        pairs = [(row["transcript"], data.record_text(rid)) for rid in ids]

        torch.cuda.synchronize()
        score_start = perf_counter()
        predictions = model.predict(pairs, batch_size=16, show_progress_bar=False)
        torch.cuda.synchronize()
        scoring_seconds += perf_counter() - score_start

        scores = [float(value) for value in predictions]
        if len(scores) != 100 or not all(map(isfinite, scores)):
            raise ValueError(f"Invalid scores for {qid}")
        ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
        top5 = [rid for rid, _ in ranked[:5]]
        recall = recall_at_k(top5, set(qrels[qid]), 5)
        recalls.append(recall)
        print(json.dumps({"query_id": qid, "retrieved_record_ids": top5,
                          "recall_at_5": recall}), file=stream, flush=True)
        if index % 50 == 0:
            print(f"Mixedbread 100->5: {index}/350", flush=True)

torch.cuda.synchronize()
elapsed = perf_counter() - start
print(f"Mixedbread scoring time: {scoring_seconds:.3f} s")
print(f"Mixedbread 100->5 GPU reranking wall time: {elapsed:.3f} s")
print(f"Mixedbread 100->5 GPU Recall@5: {mean(recalls):.6f} ({mean(recalls):.2%})")
print(f"Saved results: {output}")
