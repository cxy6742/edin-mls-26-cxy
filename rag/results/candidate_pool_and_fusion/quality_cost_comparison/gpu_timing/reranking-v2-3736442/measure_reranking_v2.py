import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter

print("[startup] importing PyTorch", flush=True)
import torch
import sentence_transformers
from sentence_transformers import CrossEncoder
from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["bge", "mixedbread"], required=True)
parser.add_argument("--depth", type=int, choices=[20, 100], required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

root = Path("rag/results")
models = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "mixedbread": "mixedbread-ai/mxbai-rerank-large-v1",
}

def indexed(path):
    rows = list(load_jsonl(path))
    result = {row["query_id"]: row for row in rows}
    if len(rows) != 350 or len(result) != 350:
        raise ValueError(f"Expected 350 unique queries: {path}")
    return result

print("[startup] reading inputs", flush=True)
baseline_path = root / "dense20_to5_rerankers/dense/answers.jsonl"
baseline = indexed(baseline_path)
depth20_path = root / "depth_k20_practice350.jsonl"
depth20 = indexed(depth20_path)

shard_paths = [
    root / "candidate_pool_and_fusion" / "candidate_pool"
    / f"bge100-full-3718533-{i}.jsonl"
    for i in range(7)
]
rows100 = [
    row for path in shard_paths for row in load_jsonl(path)
]
depth100 = {row["query_id"]: row for row in rows100}
if len(rows100) != 350 or len(depth100) != 350:
    raise ValueError("Expected 350 unique Top-100 queries")

data = CourseData(Path("rag/data/rag_corpus"))
qrels = data.load_qrels("practice")
if not (baseline.keys() == depth20.keys() == depth100.keys() == qrels.keys()):
    raise ValueError("Query IDs differ between inputs")

for qid, row in baseline.items():
    ids20 = depth20[qid]["retrieved_record_ids"]
    ids100 = depth100[qid]["candidate_record_ids"]
    if len(ids20) != 20 or len(set(ids20)) != 20:
        raise ValueError(f"Invalid Top-20 candidates: {qid}")
    if len(ids100) != 100 or len(set(ids100)) != 100:
        raise ValueError(f"Invalid Top-100 candidates: {qid}")
    if ids20[:5] != row["retrieved_record_ids"]:
        raise ValueError(f"Baseline Top-5 differs: {qid}")
    if not row["transcript"].strip():
        raise ValueError(f"Empty transcript: {qid}")
    if row["transcript"] != depth20[qid]["transcript"]:
        raise ValueError(f"Transcript mismatch: {qid}")

reference = None
if args.depth == 20:
    reference = indexed(
        root / "dense20_to5_rerankers" / args.model / "answers.jsonl"
    )
    if reference.keys() != baseline.keys():
        raise ValueError("Reference answer query IDs differ")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
gpu = torch.cuda.get_device_name(0)
if "H200" not in gpu or "1g.18gb" not in gpu:
    raise RuntimeError(f"Unexpected GPU: {gpu}")

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
args.out.mkdir(parents=True, exist_ok=True)
output = args.out / f"{args.model}{args.depth}.jsonl"
summary_path = args.out / f"{args.model}{args.depth}_summary.json"
if output.exists() or summary_path.exists():
    raise FileExistsError("Output already exists")

metadata = {
    "model": models[args.model],
    "depth": args.depth,
    "batch_size": 16,
    "max_length": 512,
    "gpu": gpu,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "cuda": torch.version.cuda,
    "input_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [baseline_path, depth20_path, *shard_paths]
    },
}
print(json.dumps(metadata, indent=2), flush=True)
print("[startup] loading model", flush=True)
model = CrossEncoder(models[args.model], device="cuda", max_length=512)

recalls = []
selected_by_query = {}
print("Starting timed reranking", flush=True)
torch.cuda.synchronize()
start = perf_counter()

with output.open("x", encoding="utf-8") as stream:
    for number, (qid, row) in enumerate(baseline.items(), 1):
        ids = (
            depth20[qid]["retrieved_record_ids"]
            if args.depth == 20
            else depth100[qid]["candidate_record_ids"]
        )
        pairs = [(row["transcript"], data.record_text(rid)) for rid in ids]
        scores = [
            float(value) for value in model.predict(
                pairs, batch_size=16, show_progress_bar=False
            )
        ]
        if len(scores) != len(ids) or not all(map(math.isfinite, scores)):
            raise ValueError(f"Invalid scores: {qid}")
        order = sorted(range(len(ids)), key=lambda i: -scores[i])
        selected = [ids[i] for i in order[:5]]
        recall = recall_at_k(selected, qrels[qid], 5)
        recalls.append(recall)
        selected_by_query[qid] = selected
        stream.write(json.dumps({
            "query_id": qid,
            "candidate_record_ids": ids,
            "candidate_scores": scores,
            "retrieved_record_ids": selected,
            "recall_at_5": recall,
        }) + "\n")
        stream.flush()
        if number % 50 == 0:
            print(f"{args.model} {args.depth}->5: {number}/350", flush=True)

torch.cuda.synchronize()
elapsed = perf_counter() - start

metadata.update({
    "queries": len(recalls),
    "reranking_wall_seconds": elapsed,
    "mean_recall_at_5": mean(recalls),
})
if reference is not None:
    mismatches = [
        qid for qid in baseline
        if selected_by_query[qid] != reference[qid]["retrieved_record_ids"]
    ]
    metadata["ordered_top5_matches"] = 350 - len(mismatches)
    metadata["mismatched_query_ids"] = mismatches

summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
print(f"GPU reranking wall time: {elapsed:.3f} s", flush=True)
print(f"Recall@5: {mean(recalls):.6f} ({mean(recalls):.2%})", flush=True)
if reference is not None:
    print(f"Ordered Top-5 matches saved answers: {350-len(mismatches)}/350")
print(f"Saved results: {output}", flush=True)
print(f"Saved summary: {summary_path}", flush=True)
