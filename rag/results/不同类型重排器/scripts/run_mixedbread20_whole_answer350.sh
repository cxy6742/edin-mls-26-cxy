#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/edin-mls-26-cxy"
export PYTHONPATH=rag
run_python="$HOME/mls-rag-env/bin/python"
run_dir="rag/results/mixedbread20-whole-answer350"
mkdir -p "$run_dir"

"$run_python" -u - <<'PY'
import json
from pathlib import Path

from dotenv import load_dotenv

from src.context_builder import SimpleContextBuilder
from src.data import CourseData, load_jsonl
from src.generator import ELMGenerator

root = Path("rag/results")
run_dir = root / "mixedbread20-whole-answer350"
out = run_dir / "answers.jsonl"
data = CourseData(Path("rag/data/rag_corpus"))
qrels = data.load_qrels("practice")

def read_practice(path):
    rows = load_jsonl(path)
    by_id = {row["query_id"]: row for row in rows}
    if len(rows) != 350 or len(by_id) != 350 or set(by_id) != set(qrels):
        raise SystemExit(f"Expected exactly 350 matching practice queries: {path}")
    return by_id

original = read_practice(root / "baseline_all_cpu_350_v1.jsonl")
ranked = read_practice(root / "mxbai-chunks350-3720077.jsonl")

for qid, row in ranked.items():
    ids = row["whole_top5"]
    if (len(ids) != 5 or len(set(ids)) != 5
            or not set(ids).issubset(row["candidate_record_ids"])):
        raise SystemExit(f"Invalid Mixedbread whole-document Top-5: {qid}")
    for rid in ids:
        data.record_text(rid)

load_dotenv(Path("rag/.env"))
generator = ELMGenerator()
builder = SimpleContextBuilder()
config = {
    "model": generator.model,
    "temperature": 0.1,
    "context_builder": "SimpleContextBuilder",
}
saved = load_jsonl(out) if out.exists() else []
done = {row["query_id"] for row in saved}
if len(done) != len(saved) or not done.issubset(original):
    raise SystemExit("Existing answers contain duplicate or unknown queries")
for row in saved:
    qid = row["query_id"]
    if (row["transcript"] != original[qid]["transcript"]
            or row["retrieved_record_ids"] != ranked[qid]["whole_top5"]
            or row["generation_config"] != config
            or not row["answer"].strip()):
        raise SystemExit(f"Existing answer failed resume validation: {qid}")

print(f"Generation model: {generator.model}; completed: {len(done)}/350")
with out.open("a", encoding="utf-8") as stream:
    for qid, row in original.items():
        if qid in done:
            continue
        transcript = row["transcript"]
        ids = ranked[qid]["whole_top5"]
        prompt = builder.build(transcript, ids, data)["prompt"]
        result = generator.generate(prompt)
        if not result["answer"].strip():
            raise SystemExit(f"Empty answer: {qid}")
        print(json.dumps({
            "query_id": qid,
            "split": "practice",
            "transcript": transcript,
            "retrieved_record_ids": ids,
            "answer": result["answer"],
            "generation_ms": result["latency_ms"],
            "generation_config": config,
        }, ensure_ascii=False), file=stream, flush=True)
        done.add(qid)
        if len(done) == 1 or len(done) % 10 == 0:
            print(f"Generated {len(done)}/350 answers")
print(f"Saved 350 answers: {out}")
PY

judge_model=$("$run_python" - <<'JUDGE'
import json
from pathlib import Path

path = Path("rag/results/bge20-eval-3718056/judge/answer_judge_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
actual = tuple(summary[key] for key in ("split", "queries", "repeats", "question_mode"))
if actual != ("practice", 350, 5, "transcript"):
    raise SystemExit(f"Unexpected BGE scoring configuration: {actual}")
print(summary["judge_model"])
JUDGE
)

echo "Starting Repeat5 with Judge model: $judge_model"
"$run_python" -u rag/extensions/evaluate_answers_format_safe.py \
  --split practice \
  --candidates "$run_dir/answers.jsonl" \
  --out-dir "$run_dir/judge" \
  --model "$judge_model" \
  --repeats 5
