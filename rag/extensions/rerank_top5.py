"""Rerank fixed dense Top-20 candidates and evaluate the selected Top-5."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData, load_jsonl
from src.metrics import ndcg_at_k, recall_at_k

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_inputs(baseline_path, candidates_path):
    baseline = load_jsonl(baseline_path)
    candidates = load_jsonl(candidates_path)
    by_query = {row["query_id"]: row for row in candidates}
    query_ids = {row["query_id"] for row in baseline}
    if len(baseline) != 350 or len(query_ids) != 350:
        raise ValueError("Expected 350 unique baseline queries")
    if len(candidates) != 350 or by_query.keys() != query_ids:
        raise ValueError("Expected the same 350 unique queries in the Top-20 file")

    for row in baseline:
        qid = row["query_id"]
        pool = by_query[qid]
        ids = pool["retrieved_record_ids"]
        if row["split"] != "practice" or pool["split"] != "practice":
            raise ValueError(f"Expected practice split: {qid}")
        if row["transcript"] != pool["transcript"] or not row["transcript"].strip():
            raise ValueError(f"Transcript mismatch or empty transcript: {qid}")
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError(f"Expected 20 distinct candidate IDs: {qid}")
        if ids[:5] != row["retrieved_record_ids"]:
            raise ValueError(f"Original Top-5 is not the Top-20 prefix: {qid}")
    return baseline, by_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "results/baseline_all_cpu_350_v1.jsonl")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "results/depth_k20_practice350.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, choices=range(1, 351),
                        metavar="1..350", default=350)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    baseline, candidates = load_inputs(args.baseline, args.candidates)
    data = CourseData(ROOT / "data/rag_corpus")
    qrels = data.load_qrels("practice")
    if qrels.keys() != candidates.keys():
        raise ValueError("Candidate queries do not match practice qrels")
    print(f"Validated inputs; loading {args.model} on {args.device}", flush=True)

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(args.model, max_length=512, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("x", encoding="utf-8") as stream:
        for row in baseline[:args.limit]:
            qid = row["query_id"]
            pool = candidates[qid]["retrieved_record_ids"]
            original = row["retrieved_record_ids"]
            pairs = [(row["transcript"], data.record_text(rid)) for rid in pool]
            start = time.perf_counter()
            scores = [float(score) for score in reranker.predict(
                pairs, batch_size=16, show_progress_bar=False
            )]
            if len(scores) != 20 or not all(map(math.isfinite, scores)):
                raise ValueError(f"Expected 20 finite reranker scores: {qid}")
            ranked = [rid for _, rid in sorted(
                zip(scores, pool), key=lambda pair: -pair[0]
            )]
            rerank_ms = (time.perf_counter() - start) * 1000
            selected = ranked[:5]
            relevant = qrels[qid]
            result = {
                "query_id": qid,
                "split": row["split"],
                "audio_file": row["audio_file"],
                "transcript": row["transcript"],
                "reranker": args.model,
                "candidate_record_ids": pool,
                "candidate_scores": scores,
                "original_record_ids": original,
                "retrieved_record_ids": selected,
                "original_recall_at_5": recall_at_k(original, relevant, 5),
                "reranked_recall_at_5": recall_at_k(selected, relevant, 5),
                "original_ndcg_at_5": ndcg_at_k(original, relevant, 5),
                "reranked_ndcg_at_5": ndcg_at_k(selected, relevant, 5),
                "rerank_ms": rerank_ms,
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(result)
            print(f"{len(results)}/{args.limit} {qid}", flush=True)

    print(f"Queries: {len(results)}")
    for metric in ("recall_at_5", "ndcg_at_5"):
        before = mean(row[f"original_{metric}"] for row in results)
        after = mean(row[f"reranked_{metric}"] for row in results)
        print(f"{metric}: original={before:.6f}, reranked={after:.6f}, delta={after-before:+.6f}")
    print(f"Median rerank latency: {median(row['rerank_ms'] for row in results):.2f} ms")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
