import argparse
import json
import sys
import time
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))

from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.retriever import ExactDenseRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--k and --limit must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    baseline = RAG_ROOT / "results/baseline_all_cpu_350_v1.jsonl"
    with baseline.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    rows = rows[:args.limit]

    if not rows or len({row["query_id"] for row in rows}) != len(rows):
        raise SystemExit("Expected distinct baseline queries.")

    data_dir = RAG_ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    encoder = LocalEncoder(RAG_ROOT / "models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    builder = SimpleContextBuilder()
    generator = ELMGenerator()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as dest:
        for row in rows:
            query_id = row["query_id"]
            transcript = row["transcript"]

            start = time.perf_counter()
            vector = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - start) * 1000

            retrieval = retriever.retrieve(vector, k=args.k)
            record_ids = list(retrieval["record_ids"])
            baseline_ids = row["retrieved_record_ids"]
            shared_depth = min(args.k, len(baseline_ids))

            if (
                len(record_ids) != args.k
                or record_ids[:shared_depth] != baseline_ids[:shared_depth]
            ):
                raise SystemExit(
                    f"Unexpected retrieval for {query_id}: "
                    "check the document count and baseline prefix."
                )

            built = builder.build(transcript, record_ids, data)
            generation = generator.generate(built["prompt"])
            total_ms = (time.perf_counter() - start) * 1000

            answer = generation["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise SystemExit(f"Empty answer: {query_id}")

            result = {
                **row,
                "k": args.k,
                "answer": answer,
                "retrieved_record_ids": record_ids,
                "prompt_chars": len(built["prompt"]),
                "timing": {
                    "embedding_ms": embedding_ms,
                    "retrieval_ms": retrieval["latency_ms"],
                    "generation_ms": generation["latency_ms"],
                    "transcript_to_answer_ms": total_ms,
                },
            }
            dest.write(json.dumps(result, ensure_ascii=False) + "\n")
            dest.flush()
            print(f"{query_id} OK")

    print(f"Saved {len(rows)} answers to {args.output}")


if __name__ == "__main__":
    main()
