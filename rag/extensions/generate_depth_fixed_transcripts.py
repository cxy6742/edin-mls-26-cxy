import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.retriever import ExactDenseRetriever


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, choices=[2, 5, 10, 20], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_rows(ROOT / "results/baseline_all_cpu_350_v1.jsonl")
    reference = {row["query_id"]: row for row in baseline}
    baseline_ids = [row["query_id"] for row in baseline]
    if len(baseline_ids) != 350 or len(set(baseline_ids)) != 350:
        raise ValueError("Expected 350 unique baseline queries")
    if set(reference) != set(baseline_ids):
        raise ValueError("Baseline query IDs differ")

    saved = read_rows(args.output) if args.output.exists() else []
    if len(saved) > len(baseline):
        raise ValueError("Output contains too many rows")
    for previous, original in zip(saved, baseline):
        depth = min(args.k, 5)
        if (
            previous["query_id"] != original["query_id"]
            or previous["transcript"] != original["transcript"]
            or previous.get("k") != args.k
            or len(previous["retrieved_record_ids"]) != args.k
            or previous["retrieved_record_ids"][:depth]
            != reference[original["query_id"]]["retrieved_record_ids"][:depth]
            or not previous["answer"].strip()
        ):
            raise ValueError("Existing output failed resume validation")

    if len(saved) == len(baseline):
        print(f"k={args.k}: all 350 answers already saved", flush=True)
        return

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    builder = SimpleContextBuilder()
    generator = ELMGenerator()

    print(
        f"k={args.k}, model={generator.model}, "
        f"temperature=0.1, encoder=enc_a, device=cpu, "
        f"completed={len(saved)}, remaining={350 - len(saved)}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as dest:
        for index, row in enumerate(baseline[len(saved):], start=len(saved) + 1):
            qid = row["query_id"]
            if row["transcript"] != reference[qid]["transcript"]:
                raise ValueError(f"Transcript mismatch: {qid}")

            start = time.perf_counter()
            vector = encoder.encode(row["transcript"])
            embedding_ms = (time.perf_counter() - start) * 1000
            retrieval = retriever.retrieve(vector, k=args.k)
            ids = list(retrieval["record_ids"])
            depth = min(args.k, 5)

            if (
                len(ids) != args.k
                or ids[:depth] != reference[qid]["retrieved_record_ids"][:depth]
            ):
                diagnostic10 = retriever.retrieve(vector, k=10)
                print("PREFIX MISMATCH", flush=True)
                print("Query ID:", qid, flush=True)
                print("Expected baseline Top-5:",
                      reference[qid]["retrieved_record_ids"], flush=True)
                print("Actual Top-k:", ids, flush=True)
                print("Actual scores:", retrieval["scores"], flush=True)
                print("Same-vector Top-10:",
                      diagnostic10["record_ids"], flush=True)
                print("Same-vector Top-10 scores:",
                      diagnostic10["scores"], flush=True)
                raise ValueError(f"Retrieval prefix mismatch: {qid}")

            built = builder.build(row["transcript"], ids, data)
            generation = generator.generate(built["prompt"])
            total_ms = (time.perf_counter() - start) * 1000
            answer = generation["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty answer: {qid}")

            result = {
                **row,
                "k": args.k,
                "answer": answer,
                "retrieved_record_ids": ids,
                "prompt_chars": len(built["prompt"]),
                "generation_config": {
                    "model": generator.model,
                    "base_url": str(generator.client.base_url),
                    "temperature": 0.1,
                },
                "timing": {
                    "embedding_ms": embedding_ms,
                    "retrieval_ms": retrieval["latency_ms"],
                    "generation_ms": generation["latency_ms"],
                    "transcript_to_answer_ms": total_ms,
                },
            }
            dest.write(json.dumps(result, ensure_ascii=False) + "\n")
            dest.flush()
            print(f"k={args.k} {index}/350 {qid} OK", flush=True)

    print(f"Saved 350 answers: {args.output}", flush=True)


if __name__ == "__main__":
    main()
