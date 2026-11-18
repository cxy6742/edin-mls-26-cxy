"""Reconstructed reproduction script for BGE job 3718533.

The original submission body is unavailable.
The model settings and candidate inputs follow the
historical experiment records.
"""

import argparse
import json
from math import isfinite
from pathlib import Path

from src.data import CourseData, load_jsonl
from sentence_transformers import CrossEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, choices=range(7), required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

archive = Path(__file__).resolve().parents[1]

rows = load_jsonl(
    archive / "inputs/depth_k100_practice350.jsonl"
)

data = CourseData(Path("rag/data/rag_corpus"))
qrels = data.load_qrels("practice")

assert len(rows) == 350
assert len({r["query_id"] for r in rows}) == 350
assert {r["query_id"] for r in rows} == qrels.keys()

for row in rows:
    ids = row["retrieved_record_ids"]
    assert len(ids) == len(set(ids)) == 100
    assert row["split"] == "practice"
    assert row["transcript"].strip()

selected = rows[args.shard * 50:(args.shard + 1) * 50]

print(f"Task {args.shard}: loading BGE for {len(selected)} queries",
      flush=True)

model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=512,
    device="cpu",
)

args.output_dir.mkdir(parents=True, exist_ok=True)

output = args.output_dir / f"bge100-reproduced-{args.shard}.jsonl"

with output.open("x", encoding="utf-8") as stream:
    for index, row in enumerate(selected, 1):
        qid = row["query_id"]
        ids = row["retrieved_record_ids"]

        pairs = [
            (row["transcript"], data.record_text(rid))
            for rid in ids
        ]

        scores = [
            float(score)
            for score in model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
        ]

        assert len(scores) == 100
        assert all(isfinite(s) for s in scores)

        result = {
            "query_id": qid,
            "candidate_record_ids": ids,
            "candidate_scores": scores,
        }

        print(json.dumps(result), file=stream, flush=True)
        print(f"{index}/{len(selected)} {qid}", flush=True)

print(f"Saved: {output}")
