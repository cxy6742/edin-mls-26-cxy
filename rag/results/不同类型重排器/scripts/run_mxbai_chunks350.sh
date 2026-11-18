#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/edin-mls-26-cxy"

PYTHONPATH=rag "$HOME/mls-rag-env/bin/python" -u - <<'PY'
import json
import os
from pathlib import Path
from random import Random
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
    raise SystemExit("CUDA unavailable")

sample = sorted(by_id)
model_name = "mixedbread-ai/mxbai-rerank-large-v1"
max_length = 512
print(f"Loading {model_name} on {torch.cuda.get_device_name(0)}...")
model = CrossEncoder(model_name, device="cuda", max_length=max_length)
tokenizer = model.tokenizer
if not tokenizer.is_fast:
    raise SystemExit("A fast tokenizer is needed for source-text offsets")

special_tokens = tokenizer.num_special_tokens_to_add(pair=True)


def split_document(text, offsets, window):
    if len(offsets) <= window:
        return [text]
    overlap = min(64, window // 4)
    step = window - overlap
    chunks = []
    for start in range(0, len(offsets), step):
        end = min(start + window, len(offsets))
        chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
        if end == len(offsets):
            break
    return chunks


def score(question, texts):
    pairs = [(question, text) for text in texts]
    torch.cuda.synchronize()
    start = perf_counter()
    values = model.predict(pairs, batch_size=8, show_progress_bar=False)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    if getattr(values, "shape", None) != (len(texts),):
        raise SystemExit("Unexpected number of model scores")
    if not all(float("-inf") < float(value) < float("inf") for value in values):
        raise SystemExit("Model returned a non-finite score")
    return [float(value) for value in values], elapsed


whole_recalls, chunk_recalls = [], []
whole_times, chunk_times = [], []
truncated_docs = gained = lost = 0
chunk_counts = []
output = root / f"mxbai-chunks350-{os.environ['SLURM_JOB_ID']}.jsonl"

with output.open("x", encoding="utf-8") as stream:
    for index, qid in enumerate(sample, 1):
        ids = by_id[qid]["candidate_record_ids"][:20]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise SystemExit(f"Expected 20 distinct documents for {qid}")
        question = transcripts[qid]
        texts = [data.record_text(rid) for rid in ids]
        query_tokens = len(tokenizer(
            question, add_special_tokens=False, truncation=False
        )["input_ids"])
        available = max_length - query_tokens - special_tokens
        if available < 32:
            raise SystemExit(f"Question leaves too little document space: {qid}")

        chunks_by_doc = []
        truncations = 0
        for text in texts:
            offsets = tokenizer(
                text, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            truncations += len(offsets) > available
            chunks_by_doc.append(split_document(text, offsets, min(256, available)))

        chunks = [chunk for group in chunks_by_doc for chunk in group]
        # Alternate scoring order to reduce warm-up effects in the timings.
        if index % 2:
            whole_scores, whole_time = score(question, texts)
            chunk_scores, chunk_time = score(question, chunks)
        else:
            chunk_scores, chunk_time = score(question, chunks)
            whole_scores, whole_time = score(question, texts)

        offset = 0
        chunk_doc_scores = []
        for group in chunks_by_doc:
            end = offset + len(group)
            chunk_doc_scores.append(max(chunk_scores[offset:end]))
            offset = end

        whole_top5 = sorted(ids, key=dict(zip(ids, whole_scores)).__getitem__,
                            reverse=True)[:5]
        chunk_top5 = sorted(ids, key=dict(zip(ids, chunk_doc_scores)).__getitem__,
                            reverse=True)[:5]
        labels = set(qrels[qid])
        whole_hits = labels.intersection(whole_top5)
        chunk_hits = labels.intersection(chunk_top5)
        before = recall_at_k(whole_top5, labels, 5)
        after = recall_at_k(chunk_top5, labels, 5)
        whole_recalls.append(before)
        chunk_recalls.append(after)
        whole_times.append(whole_time)
        chunk_times.append(chunk_time)
        chunk_counts.append(len(chunks))
        truncated_docs += truncations
        gained += len(chunk_hits - whole_hits)
        lost += len(whole_hits - chunk_hits)

        print(json.dumps({
            "query_id": qid,
            "candidate_record_ids": ids,
            "whole_scores": whole_scores,
            "chunk_doc_scores": chunk_doc_scores,
            "whole_top5": whole_top5,
            "chunk_top5": chunk_top5,
            "whole_seconds": whole_time,
            "chunk_seconds": chunk_time,
            "chunk_count": len(chunks),
            "whole_pairs_over_512": truncations,
        }), file=stream, flush=True)
        print(f"{index}/{len(sample)} {qid} | {before:.2%} -> {after:.2%} | {len(chunks)} chunks")

deltas = [after - before for before, after in zip(whole_recalls, chunk_recalls)]
up = sum(delta > 0 for delta in deltas)
down = sum(delta < 0 for delta in deltas)
print(f"\n=== Same {len(sample)} queries, same Dense Top-20 ===")
print(f"Whole-document Recall@5: {mean(whole_recalls):.2%}")
print(f"Chunked-document Recall@5: {mean(chunk_recalls):.2%}")
print(f"Chunking change: {100 * mean(deltas):+.2f} pp")
print(f"Queries up/same/down: {up}/{len(deltas) - up - down}/{down}")
print(f"Marked hits gained/lost: {gained}/{lost}")
print(f"Whole-document pairs over 512 tokens: {truncated_docs}/{20 * len(sample)}")
print(f"Mean chunks scored per query: {mean(chunk_counts):.1f}")
print(f"Median scoring seconds per query, whole/chunked: "
      f"{median(whole_times):.3f}/{median(chunk_times):.3f}")
print(f"Saved: {output}")
PY
