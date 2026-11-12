"""Compare paired k=5 and k=20 practice results and prepare blinded case review."""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("correctness", "grounding", "completeness", "normalised_answer_score")
EPSILON = 1e-9


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def practice_rows(path):
    rows = {}
    for row in jsonl(path):
        qid = row["query_id"]
        if qid in rows or row["split"] != "practice":
            raise ValueError(f"{path}: duplicate or non-practice query {qid}")
        rows[qid] = row
    if len(rows) != 350:
        raise ValueError(f"{path}: expected 350 queries, got {len(rows)}")
    return rows


def load_judged_run(folder):
    summary = json.loads((folder / "answer_judge_summary.json").read_text(encoding="utf-8"))
    if (summary["split"], summary["queries"], summary["repeats"]) != ("practice", 350, 5):
        raise ValueError(f"{folder}: expected 350 practice queries scored five times")
    rows = practice_rows(folder / "answer_judge_per_query.jsonl")
    for qid, row in rows.items():
        if len(row["judge_explanations"]) != 5 or row["question_mode"] != summary["question_mode"]:
            raise ValueError(f"{folder}: unexpected judging settings for {qid}")
        for metric in METRICS:
            upper = 1 if metric == "normalised_answer_score" else 4
            if not 0 <= row[metric] <= upper:
                raise ValueError(f"{folder}: invalid {metric} for {qid}")
        if not math.isclose(sum(row[m] for m in METRICS[:3]) / 12,
                            row[METRICS[-1]], abs_tol=EPSILON):
            raise ValueError(f"{folder}: inconsistent combined score for {qid}")
    for metric in METRICS:
        if not math.isclose(mean(row[metric] for row in rows.values()),
                            summary[f"mean_{metric}"], abs_tol=EPSILON):
            raise ValueError(f"{folder}: per-query {metric} does not match summary")
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = Path("rag/results")
    data = Path("rag/data/rag_corpus")
    answers = {
        5: practice_rows(results / "baseline_all_cpu_350_v1.jsonl"),
        20: practice_rows(results / "depth_k20_practice350.jsonl"),
    }
    judge5, scores5 = load_judged_run(results / "baseline_all_cpu_350_judge_v1")
    judge20, scores20 = load_judged_run(results / "depth_k20_practice350_judge")
    settings = ("judge_model", "question_mode", "rubric_version", "script_version")
    if any(judge5[key] != judge20[key] for key in settings):
        raise ValueError("The runs use different judge settings")

    qrels = defaultdict(set)
    with (data / "qrels_practice.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if float(row.get("score", 1)) > 0:
                qrels[row["query_id"]].add(row["record_id"])
    ids = answers[5].keys()
    if any(source.keys() != ids for source in (answers[20], scores5, scores20, qrels)):
        raise ValueError("The answers, judgments and qrels must cover the same queries")

    paired = []
    for qid in sorted(ids):
        a5, a20 = answers[5][qid], answers[20][qid]
        top5, top20 = a5["retrieved_record_ids"], a20["retrieved_record_ids"]
        if a5["transcript"] != a20["transcript"] or top5 != top20[:5]:
            raise ValueError(f"Transcript or ranking prefix mismatch: {qid}")
        if len(top5) != 5 or len(top20) != 20 or len(set(top20)) != 20:
            raise ValueError(f"Unexpected number of distinct documents: {qid}")
        relevant = qrels[qid]
        hits5 = len(relevant.intersection(top5))
        hits20 = len(relevant.intersection(top20))
        delta = scores20[qid][METRICS[-1]] - scores5[qid][METRICS[-1]]
        change = "down" if delta < -EPSILON else "up" if delta > EPSILON else "tie"
        row = {
            "query_id": qid,
            "recall_5": hits5 / len(relevant),
            "recall_20": hits20 / len(relevant),
            "delta_recall": (hits20 - hits5) / len(relevant),
            "added_relevant_6_10": len(relevant.intersection(top20[5:10])),
            "added_relevant_11_20": len(relevant.intersection(top20[10:])),
            "added_relevant": hits20 - hits5,
            "score_5": scores5[qid][METRICS[-1]],
            "score_20": scores20[qid][METRICS[-1]],
            "delta_score": delta,
            "score_change": change,
        }
        for metric in METRICS[:3]:
            row[f"delta_{metric}"] = scores20[qid][metric] - scores5[qid][metric]
        paired.append(row)

    groups = [
        ("A", [r for r in paired if r["added_relevant"] and r["score_change"] == "down"]),
        ("B", [r for r in paired if not r["added_relevant"] and r["score_change"] == "down"]),
        ("C", [r for r in paired if r["added_relevant"] and r["score_change"] == "up"]),
    ]
    rng = random.Random(args.seed)
    selected = []
    for group, rows in groups:
        selected.extend((group, row["query_id"]) for row in rng.sample(rows, min(5, len(rows))))
    rng.shuffle(selected)

    wanted = set()
    for _, qid in selected:
        wanted.update(answers[20][qid]["retrieved_record_ids"])
        wanted.update(qrels[qid])
    texts = {
        str(doc["record_id"]): doc["text"] for doc in jsonl(data / "corpus.jsonl")
        if str(doc["record_id"]) in wanted
    }
    if missing := wanted - texts.keys():
        raise ValueError(f"Missing corpus records: {sorted(missing)[:10]}")

    deltas = [r["delta_score"] for r in paired]
    bootstrap_rng = random.Random(args.seed)
    resamples = sorted(mean(bootstrap_rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    lines = [
        f"Paired queries: {len(paired)} | score delta = k20 - k5",
        f"Recall: {mean(r['recall_5'] for r in paired):.6f} -> "
        f"{mean(r['recall_20'] for r in paired):.6f}",
        f"Answer score: {mean(r['score_5'] for r in paired):.6f} -> "
        f"{mean(r['score_20'] for r in paired):.6f}",
        f"Mean score delta: {mean(deltas):+.6f} "
        f"(paired bootstrap 95% CI [{resamples[250]:+.6f}, {resamples[9749]:+.6f}])",
        f"Score changes: {dict(Counter(r['score_change'] for r in paired))}",
    ]
    for metric in METRICS[:3]:
        lines.append(f"Mean delta {metric} (/4): {mean(r[f'delta_{metric}'] for r in paired):+.6f}")
    for added in (False, True):
        subset = [r for r in paired if bool(r["added_relevant"]) == added]
        counts = Counter(r["score_change"] for r in subset)
        avg = f"{mean(r['delta_score'] for r in subset):+.6f}" if subset else "n/a"
        lines.append(
            f"Added qrels evidence={added}: n={len(subset)}, "
            f"up={counts['up']}, tie={counts['tie']}, down={counts['down']}, mean delta={avg}"
        )
    for field in ("added_relevant_6_10", "added_relevant_11_20"):
        lines.append(f"{field}: {sum(r[field] > 0 for r in paired)} queries with >=1 marked document")
    lines.append(f"Review sample seed={args.seed}: " + ", ".join(
        f"{group}={sum(selected_group == group for selected_group, _ in selected)}"
        for group, _ in groups
    ))

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "paired_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    with (args.output / "blind_cases.txt").open("w", encoding="utf-8") as blind, (
        args.output / "case_key.jsonl"
    ).open("w", encoding="utf-8") as key:
        for number, (group, qid) in enumerate(selected, 1):
            a5, a20 = answers[5][qid], answers[20][qid]
            docs = list(enumerate(a20["retrieved_record_ids"], 1))
            rng.shuffle(docs)
            case = f"Case {number:02d}"
            print(f"\n=== {case} ===\nQuestion: {a5['transcript']}", file=blind)
            mapping = []
            for index, (rank, rid) in enumerate(docs, 1):
                label = f"D{index:02d}"
                print(f"\n[{label}]\n{texts[rid]}", file=blind)
                mapping.append({"label": label, "original_rank": rank, "record_id": rid,
                                "qrels_relevant": rid in qrels[qid]})
            key.write(json.dumps({
                "case": case, "group": group, "query_id": qid, "documents": mapping,
                "qrels": [{"record_id": rid, "text": texts[rid]} for rid in sorted(qrels[qid])],
                "answer_5": a5["answer"], "answer_20": a20["answer"],
                "judge_5": scores5[qid], "judge_20": scores20[qid],
            }, ensure_ascii=False) + "\n")

    summary = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

