from pathlib import Path
from statistics import mean, median
from time import perf_counter
from math import isfinite

from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k

root = Path("rag/results")

print("Loading saved scores...", flush=True)

data = CourseData(Path("rag/data/rag_corpus"))
qrels = data.load_qrels("practice")

bge_rows = [
    row
    for shard in range(7)
    for row in load_jsonl(
        root / f"bge100-full-3718533-{shard}.jsonl"
    )
]

mixed_rows = list(
    load_jsonl(root / "mxbai-fusion350-3720050.jsonl")
)

bge_by_query = {
    row["query_id"]: row for row in bge_rows
}

mixed_by_query = {
    row["query_id"]: row for row in mixed_rows
}

assert len(bge_rows) == 350
assert len(mixed_rows) == 350
assert len(qrels) == 350
assert bge_by_query.keys() == mixed_by_query.keys() == qrels.keys()

prepared = []

for qid in sorted(qrels):
    b = bge_by_query[qid]
    m = mixed_by_query[qid]

    dense = b["candidate_record_ids"]

    assert len(dense) == len(set(dense)) == 100
    assert dense == m["candidate_record_ids"]
    assert len(b["candidate_scores"]) == 100
    assert len(m["mixedbread_scores"]) == 100
    assert all(isfinite(s) for s in b["candidate_scores"])
    assert all(isfinite(s) for s in m["mixedbread_scores"])

    bge_scores = dict(zip(dense, b["candidate_scores"]))
    mixed_scores = dict(zip(dense, m["mixedbread_scores"]))

    bge = sorted(
        dense, key=bge_scores.__getitem__, reverse=True
    )

    mixed = sorted(
        dense, key=mixed_scores.__getitem__, reverse=True
    )

    bge_rank = {
        rid: rank for rank, rid in enumerate(bge, 1)
    }

    mixed_rank = {
        rid: rank for rank, rid in enumerate(mixed, 1)
    }

    prepared.append(
        (qid, dense, bge_rank, mixed_rank)
    )

print("Validated 350 queries and 100 candidates each.", flush=True)


def fuse_all(tie_break):
    results = {}

    for qid, dense, bge_rank, mixed_rank in prepared:
        tie_rank = (
            bge_rank if tie_break == "BGE" else mixed_rank
        )

        fused = sorted(
            dense,
            key=lambda rid: (
                bge_rank[rid] + mixed_rank[rid],
                tie_rank[rid],
            ),
        )

        results[qid] = fused[:5]

    return results


for tie_break in ("BGE", "Mixedbread"):
    results = fuse_all(tie_break)

    recall = mean(
        recall_at_k(results[qid], set(qrels[qid]), 5)
        for qid in results
    )

    times = []

    for _ in range(21):
        start = perf_counter()
        fuse_all(tie_break)
        times.append(perf_counter() - start)

    fusion_time = median(times)
    estimated_total = 1386.130 + 584.511 + fusion_time

    print(f"\n=== {tie_break} tie-break ===")
    print(f"Queries: {len(results)}")
    print(f"Recall@5: {recall:.6f} ({recall:.2%})")
    print(f"Fusion CPU wall time: {fusion_time:.6f} s")
    print(f"Estimated combined time: {estimated_total:.6f} s")

