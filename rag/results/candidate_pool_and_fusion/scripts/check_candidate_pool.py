import json
from math import isfinite
from pathlib import Path
from statistics import mean

from src.data import CourseData
from src.metrics import recall_at_k

pool = Path(__file__).resolve().parents[1] / "candidate_pool"
top20 = pool.parent.parent / "dense20_to5_rerankers"
qrels = CourseData(Path("rag/data/rag_corpus")).load_qrels("practice")


def load_rows(paths):
    rows = {}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    qid = row["query_id"]
                    if qid in rows:
                        raise ValueError(f"Duplicate query in {path}: {qid}")
                    rows[qid] = row
    if len(rows) != 350 or rows.keys() != qrels.keys():
        raise ValueError("Expected exactly the 350 practice queries")
    return rows


runs = [
    ("BGE", "bge", [pool / f"bge100-full-3718533-{i}.jsonl" for i in range(7)],
     "candidate_scores", ["47.18%", "46.72%", "45.58%"]),
    ("Mixedbread", "mixedbread", [pool / "mxbai-chunks100-350-3720102.jsonl"],
     "whole_scores", ["47.32%", "45.88%", "46.34%"]),
]

reference_ids = {}
print("Model\tCandidates\tQueries\tRecall@5")
for model, folder, paths, field, expected in runs:
    answers = load_rows([top20 / folder / "answers.jsonl"])
    rows = load_rows(paths)
    recalls = {20: [], 50: [], 100: []}
    for qid, row in rows.items():
        ids = row["candidate_record_ids"]
        scores = row[field]
        if len(ids) != 100 or len(set(ids)) != 100 or len(scores) != 100:
            raise ValueError(f"Invalid candidate or score count: {qid}")
        if not all(isfinite(score) for score in scores):
            raise ValueError(f"Invalid score: {qid}")
        if model == "BGE":
            reference_ids[qid] = ids
        elif ids != reference_ids[qid]:
            raise ValueError(f"Candidate order differs: {qid}")

        selected = answers[qid]["retrieved_record_ids"][:5]
        if len(set(selected)) != 5:
            raise ValueError(f"Invalid saved Top-5: {qid}")
        recalls[20].append(recall_at_k(selected, qrels[qid], 5))
        for depth in (50, 100):
            order = sorted(range(depth), key=lambda i: -scores[i])
            selected = [ids[i] for i in order[:5]]
            recalls[depth].append(recall_at_k(selected, qrels[qid], 5))

    for depth, target in zip((20, 50, 100), expected):
        actual = f"{mean(recalls[depth]):.2%}"
        if actual != target:
            raise ValueError(f"{model} {depth}->5: {actual}, expected {target}")
        print(f"{model}\t{depth}\t{len(rows)}\t{actual}")
