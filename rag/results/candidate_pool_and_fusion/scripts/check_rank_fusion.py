"""Verify saved rank fusion and answer quality results."""

import json
from math import isfinite
from pathlib import Path
from statistics import mean

from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k

archive = Path(__file__).resolve().parents[1]
pool = archive / "candidate_pool"
fusion = archive / "rank_fusion"

qrels = CourseData(
    Path("rag/data/rag_corpus")
).load_qrels("practice")


def read_queries(path):
    rows = load_jsonl(path)
    result = {row["query_id"]: row for row in rows}
    assert len(rows) == len(result) == 350, path
    assert result.keys() == qrels.keys(), path
    return result


bge_rows = [
    row
    for shard in range(7)
    for row in load_jsonl(
        pool / f"bge100-full-3718533-{shard}.jsonl"
    )
]

bge = {row["query_id"]: row for row in bge_rows}

assert len(bge_rows) == len(bge) == 350
assert bge.keys() == qrels.keys()

mixed = read_queries(
    fusion / "scoring/mxbai-fusion350-3720050.jsonl"
)

mixed_tiebreak = read_queries(
    fusion / "scoring/mixedbread-tiebreak-fusion350.jsonl"
)

inputs = read_queries(
    archive / "inputs/depth_k100_practice350.jsonl"
)

configs = [
    ("BGE tie-break", "bge_tiebreak", "49.77%", "0.358714"),
    ("Mixedbread tie-break", "mixedbread_tiebreak",
     "50.49%", "0.353714"),
]

for name, folder, expected_recall, expected_r5 in configs:

    answers = read_queries(fusion / folder / "answers.jsonl")

    judge = read_queries(
        fusion / folder / "judge/answer_judge_per_query.jsonl"
    )

    summary = json.loads(
        (fusion / folder / "judge/answer_judge_summary.json").read_text()
    )

    recalls = []

    for qid in qrels:

        ids = bge[qid]["candidate_record_ids"]
        bs = bge[qid]["candidate_scores"]
        ms = mixed[qid]["mixedbread_scores"]

        assert len(ids) == len(set(ids)) == 100
        assert len(bs) == len(ms) == 100
        assert all(isfinite(s) for s in bs)
        assert all(isfinite(s) for s in ms)

        assert ids == mixed[qid]["candidate_record_ids"]
        assert ids == mixed_tiebreak[qid]["candidate_record_ids"]
        assert ids == inputs[qid]["retrieved_record_ids"]

        bge_scores = dict(zip(ids, bs))
        mixed_scores = dict(zip(ids, ms))

        bge_order = sorted(
            ids, key=bge_scores.__getitem__, reverse=True
        )

        mixed_order = sorted(
            ids, key=mixed_scores.__getitem__, reverse=True
        )

        bge_rank = {
            rid: rank for rank, rid in enumerate(bge_order, 1)
        }

        mixed_rank = {
            rid: rank for rank, rid in enumerate(mixed_order, 1)
        }

        if folder == "bge_tiebreak":
            tie_rank = bge_rank
            saved = mixed[qid]["top5"]["BGE+Mixedbread Top-100 fusion"]
        else:
            tie_rank = mixed_rank
            saved = mixed_tiebreak[qid]["whole_top5"]

        reproduced = sorted(
            ids,
            key=lambda rid: (
                bge_rank[rid] + mixed_rank[rid],
                tie_rank[rid],
            )
        )[:5]

        assert reproduced == saved, (name, qid)
        assert answers[qid]["retrieved_record_ids"] == saved
        assert answers[qid]["transcript"] == inputs[qid]["transcript"]
        assert answers[qid]["answer"].strip()

        recalls.append(recall_at_k(reproduced, qrels[qid], 5))

    recall = f"{mean(recalls):.2%}"

    scores = [
        row["normalised_answer_score"]
        for row in judge.values()
    ]

    assert all(isfinite(s) and 0 <= s <= 1 for s in scores)

    computed_r5 = mean(scores)
    r5 = f"{computed_r5:.6f}"

    assert recall == expected_recall
    assert r5 == expected_r5

    assert abs(
        computed_r5 - summary["mean_normalised_answer_score"]
    ) < 1e-10

    assert summary["queries"] == 350
    assert summary["repeats"] == 5
    assert summary["coverage"] == 1.0
    assert summary["question_mode"] == "transcript"

    print(f"\n=== {name} ===")
    print("Candidate order matches: 350/350")
    print("Reproduced fusion Top-5 matches: 350/350")
    print("Answer Top-5 matches: 350/350")
    print("Transcript matches: 350/350")
    print(f"Recall@5: {recall}")
    print(f"Recomputed Mean R5: {r5}")
    print("Validation: PASSED")
