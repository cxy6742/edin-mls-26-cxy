"""Compare dense order with a reranking of that same top-n set.

Retrieval still returns exactly k records. The reranker may change their order,
and therefore MRR, nDCG, and Recall at cutoffs below k. Recall at k stays the
same because the candidate set does not change.

Transcripts come from the saved ASR baseline, so this comparison does not rerun
ASR. Answer generation is separate: see generate_reranked_answers.py.
"""
import argparse
import json
import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))

import numpy as np

from extensions.rerank_core import build_reranker
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.retriever import ExactDenseRetriever


CUTOFFS = (1, 2, 5, 10, 20)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ranking_metrics(ranked_ids, relevance: dict, cutoffs) -> dict[str, float]:
    return {
        f"{name}@{cutoff}": function(ranked_ids, relevance, cutoff)
        for cutoff in cutoffs
        for name, function in (
            ("Recall", recall_at_k),
            ("MRR", mrr_at_k),
            ("NDCG", ndcg_at_k),
        )
    }


def first_relevant_rank(ranked_ids, relevance: dict):
    relevant = set(relevance)
    for rank, record_id in enumerate(ranked_ids, start=1):
        if record_id in relevant:
            return rank
    return None


def mean_metric(rows, key):
    return float(np.mean([row[key] for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], default="practice")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=RAG_ROOT / "results/baseline_all_cpu_350_v1.jsonl",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--method", choices=["bm25", "cross-encoder"], default="bm25")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=RAG_ROOT / "results/system_extension/rerank",
    )
    args = parser.parse_args()

    depths = sorted(set(args.k))
    if not depths or depths[0] < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--k and --limit must be positive")
    if not args.baseline.is_file():
        parser.error(f"Baseline answers not found: {args.baseline}")

    rows = load_jsonl(args.baseline)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or len({row["query_id"] for row in rows}) != len(rows):
        raise SystemExit("Expected distinct baseline queries.")
    if any(row.get("split", args.split) != args.split for row in rows):
        raise SystemExit(f"Baseline file is not a {args.split} split.")

    data_dir = RAG_ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    qrels = data.load_qrels(args.split)
    missing = [row["query_id"] for row in rows if row["query_id"] not in qrels]
    if missing:
        raise SystemExit(f"Missing qrels for {missing[0]}")

    encoder = LocalEncoder(RAG_ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    reranker = build_reranker(
        args.method,
        data.records,
        device=args.device,
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    depth = max(depths)
    per_depth = {k: [] for k in depths}
    retrieval_ms = []

    for index, row in enumerate(rows, start=1):
        query_id = row["query_id"]
        transcript = row["transcript"]
        relevance = qrels[query_id]
        # Encode one transcript at a time. Batched encoding does not reproduce
        # the saved baseline order for this float16 checkpoint.
        vector = encoder.encode(transcript)
        retrieved = retriever.retrieve(vector, k=depth)
        ranked = list(retrieved["record_ids"])
        if len(ranked) != depth or len(set(ranked)) != depth:
            raise SystemExit(f"{query_id}: expected {depth} distinct candidates")
        retrieval_ms.append(retrieved["latency_ms"])

        baseline_ids = list(row["retrieved_record_ids"])
        shared = min(len(baseline_ids), depth)
        if ranked[:shared] != baseline_ids[:shared]:
            raise SystemExit(
                f"{query_id}: first-stage prefix does not match the saved baseline"
            )

        texts = {record_id: data.record_text(record_id) for record_id in ranked}
        for k in depths:
            candidates = ranked[:k]
            reranked = reranker.rerank(
                transcript,
                candidates,
                [texts[record_id] for record_id in candidates],
            )
            ordered = list(reranked["record_ids"])
            if set(ordered) != set(candidates) or len(ordered) != k:
                raise SystemExit(f"{query_id}: reranker changed the k={k} candidate set")
            cutoffs = [cutoff for cutoff in CUTOFFS if cutoff <= k]
            before = ranking_metrics(candidates, relevance, cutoffs)
            after = ranking_metrics(ordered, relevance, cutoffs)
            if before[f"Recall@{k}"] != after[f"Recall@{k}"]:
                raise SystemExit(f"{query_id}: Recall@{k} changed after reordering")
            per_depth[k].append({
                "query_id": query_id,
                "k": k,
                "order_changed": ordered != candidates,
                "first_stage_record_ids": candidates,
                "reranked_record_ids": ordered,
                "rerank_ms": reranked["latency_ms"],
                "first_relevant_rank_before": first_relevant_rank(candidates, relevance),
                "first_relevant_rank_after": first_relevant_rank(ordered, relevance),
                "before": before,
                "after": after,
            })
        print(f"[{index:>3}/{len(rows)}] {query_id}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.split}{len(rows)}"
    per_query_path = args.output / f"{stem}_per_query.jsonl"
    with per_query_path.open("w", encoding="utf-8") as dest:
        for k in depths:
            for item in per_depth[k]:
                dest.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "split": args.split,
        "queries": len(rows),
        "baseline": str(args.baseline),
        "method": args.method,
        "model": getattr(reranker, "model_name", None),
        "encoder": "enc_a",
        "device": args.device,
        "candidate_set": "fixed top-k from exact dense retrieval",
        "index_build_ms": getattr(reranker, "index_build_ms", None),
        "model_load_ms": getattr(reranker, "load_ms", None),
        "retrieval_depth_retrieved_once": depth,
        "mean_retrieval_ms_at_that_depth": float(np.mean(retrieval_ms)),
        "by_k": {},
    }
    for k in depths:
        items = per_depth[k]
        cutoffs = [cutoff for cutoff in CUTOFFS if cutoff <= k]
        hits = [item for item in items if item["first_relevant_rank_before"] is not None]
        block = {
            "queries": len(items),
            "order_changed": sum(item["order_changed"] for item in items) / len(items),
            "queries_with_relevant_candidate": len(hits),
            "mean_rerank_ms": mean_metric(items, "rerank_ms"),
            "mean_first_relevant_rank_before": (
                mean_metric(hits, "first_relevant_rank_before") if hits else None
            ),
            "mean_first_relevant_rank_after": (
                mean_metric(hits, "first_relevant_rank_after") if hits else None
            ),
            "first_relevant_improved": sum(
                item["first_relevant_rank_after"] < item["first_relevant_rank_before"]
                for item in hits
            ),
            "first_relevant_worsened": sum(
                item["first_relevant_rank_after"] > item["first_relevant_rank_before"]
                for item in hits
            ),
            "metrics": {},
        }
        for cutoff in cutoffs:
            for name in ("Recall", "MRR", "NDCG"):
                key = f"{name}@{cutoff}"
                before = mean_metric([item["before"] for item in items], key)
                after = mean_metric([item["after"] for item in items], key)
                block["metrics"][key] = {
                    "first_stage": before,
                    "reranked": after,
                    "delta": after - before,
                }
        summary["by_k"][str(k)] = block

    summary_path = args.output / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
