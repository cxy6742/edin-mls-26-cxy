"""Generate answers after reordering a fixed top-k candidate set.

The ASR transcript and the dense-retrieval documents stay on the saved baseline
path. Only the evidence order changes before the prompt is built. Recall at k
is therefore unchanged. The existing answer judge can score the new answers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))

from extensions.rerank_core import build_reranker
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.retriever import ExactDenseRetriever


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, choices=[2, 5, 10, 20], required=True)
    parser.add_argument("--method", choices=["bm25", "cross-encoder"], default="bm25")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = RAG_ROOT / "results/baseline_all_cpu_350_v1.jsonl"
    baseline = read_rows(baseline_path)
    reference = {row["query_id"]: row for row in baseline}
    baseline_ids = [row["query_id"] for row in baseline]
    if len(baseline_ids) != 350 or len(set(baseline_ids)) != 350:
        raise ValueError("Expected 350 unique baseline queries")

    saved = read_rows(args.output) if args.output.exists() else []
    if len(saved) > len(baseline):
        raise ValueError("Output contains too many rows")
    for previous, original in zip(saved, baseline):
        depth = min(args.k, 5)
        if (
            previous["query_id"] != original["query_id"]
            or previous["transcript"] != original["transcript"]
            or previous.get("k") != args.k
            or previous.get("rerank_method") != args.method
            or len(previous["retrieved_record_ids"]) != args.k
            or len(previous["first_stage_record_ids"]) != args.k
            or set(previous["retrieved_record_ids"]) != set(previous["first_stage_record_ids"])
            or previous["first_stage_record_ids"][:depth]
            != reference[original["query_id"]]["retrieved_record_ids"][:depth]
            or not previous["answer"].strip()
        ):
            raise ValueError("Existing output failed resume validation")

    if len(saved) == len(baseline):
        print(f"k={args.k}: all 350 reranked answers already saved", flush=True)
        return

    data_dir = RAG_ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
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
    )
    builder = SimpleContextBuilder()
    generator = ELMGenerator()

    print(
        f"k={args.k}, method={args.method}, model={generator.model}, "
        f"temperature=0.1, encoder=enc_a, device={args.device}, "
        f"completed={len(saved)}, remaining={350 - len(saved)}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as dest:
        for index, row in enumerate(baseline[len(saved):], start=len(saved) + 1):
            query_id = row["query_id"]
            if row["transcript"] != reference[query_id]["transcript"]:
                raise ValueError(f"Transcript mismatch: {query_id}")

            start = time.perf_counter()
            vector = encoder.encode(row["transcript"])
            embedding_ms = (time.perf_counter() - start) * 1000.0
            retrieval = retriever.retrieve(vector, k=args.k)
            record_ids = list(retrieval["record_ids"])
            depth = min(args.k, 5)
            if (
                len(record_ids) != args.k
                or record_ids[:depth]
                != reference[query_id]["retrieved_record_ids"][:depth]
            ):
                raise ValueError(f"Retrieval prefix mismatch: {query_id}")

            texts = [data.record_text(record_id) for record_id in record_ids]
            reranked = reranker.rerank(row["transcript"], record_ids, texts)
            ordered_ids = list(reranked["record_ids"])
            if set(ordered_ids) != set(record_ids) or len(ordered_ids) != args.k:
                raise ValueError(f"{query_id}: reranker changed the candidate set")

            built = builder.build(row["transcript"], ordered_ids, data)
            generation = generator.generate(built["prompt"])
            answer = generation["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty answer: {query_id}")

            result = {
                **row,
                "k": args.k,
                "rerank_method": args.method,
                "answer": answer,
                "first_stage_record_ids": record_ids,
                "retrieved_record_ids": ordered_ids,
                "prompt_chars": len(built["prompt"]),
                "generation_config": {
                    "model": generator.model,
                    "base_url": str(generator.client.base_url),
                    "temperature": 0.1,
                },
                "timing": {
                    "embedding_ms": embedding_ms,
                    "retrieval_ms": retrieval["latency_ms"],
                    "rerank_ms": reranked["latency_ms"],
                    "generation_ms": generation["latency_ms"],
                    "transcript_to_answer_ms": (time.perf_counter() - start) * 1000.0,
                },
            }
            dest.write(json.dumps(result, ensure_ascii=False) + "\n")
            dest.flush()
            print(f"k={args.k} {index}/350 {query_id} OK", flush=True)

    print(f"Saved 350 answers: {args.output}", flush=True)


if __name__ == "__main__":
    main()
