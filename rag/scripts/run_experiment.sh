#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON="${MLS_RAG_PYTHON:-$HOME/mls-rag-env/bin/python}"

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    out=$1
    asr=$2
    k=$3
    split=$4
    limit=$5
    runs=$6
    cache=$7
    repeat=$8
    publish=$9
    branch=${10}
    revision=${11}

    [[ "$(git rev-parse HEAD)" == "$revision" ]] ||
        die "The checkout changed after submission. Review the checkout before resuming."

    if [[ "$asr" != cached ]]; then
        export ASR_CUDA="$asr"
    fi
    export PYTHONPATH="$ROOT/rag${PYTHONPATH:+:$PYTHONPATH}"

    judge=rag/scripts/evaluate_answers.py
    for candidate in \
        rag/extensions/evaluate_answers_format_safe.py \
        rag/extensions/evaluate_answers_brace_safe.py
    do
        if [[ -f "$candidate" ]]; then
            judge=$candidate
            break
        fi
    done

    for ((run = 1; run <= runs; run++)); do
        printf -v name 'run%02d' "$run"
        run_dir="$out/$name"
        mkdir -p "$run_dir"
        answers="$run_dir/answers.jsonl"

        if [[ "$repeat" == judge && "$run" -gt 1 ]]; then
            cp -- "$out/run01/answers.jsonl" "$answers"
        else
            "$PYTHON" -u - "$asr" "$k" "$split" "$limit" "$cache" "$answers" <<'PY'
import json
import sys
import time
from pathlib import Path

from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.generator import ELMGenerator

mode, depth, split, limit, cache_path, output_path = sys.argv[1:]
k = int(depth)
output = Path(output_path)
rag = Path("rag")
data_dir = rag / "data/rag_corpus"
data = CourseData(data_dir)
items = list(data.audio_items(split))
if limit != "all":
    count = int(limit)
    if count > len(items):
        raise SystemExit(f"Requested {count} queries, but {split} has {len(items)}.")
    items = items[:count]
if not items:
    raise SystemExit("No queries selected.")

ids = [item["query_id"] for item in items]
if len(set(ids)) != len(ids):
    raise SystemExit("Duplicate query IDs in the selected split.")

completed = []
if output.exists():
    with output.open(encoding="utf-8") as stream:
        completed = [json.loads(line) for line in stream if line.strip()]
if [row["query_id"] for row in completed] != ids[:len(completed)]:
    raise SystemExit(f"Existing answers do not match the expected query prefix: {output}")
if len(completed) == len(items):
    print(f"Reusing {len(completed)} completed answers from {output}")
    raise SystemExit(0)

transcripts = {}
if mode == "cached":
    with Path(cache_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                qid = row["query_id"]
                if qid in transcripts:
                    raise SystemExit(f"Duplicate query in transcript cache: {qid}")
                transcripts[qid] = row["transcript"]
    missing = [qid for qid in ids if qid not in transcripts]
    if missing:
        raise SystemExit(f"Missing {len(missing)} cached transcripts. First: {missing[0]}")
else:
    from src.asr_adapter import ASRAdapter

    asr = ASRAdapter()
    asr.prepare()

if k:
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever

    encoder = LocalEncoder(rag / "models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )

builder = SimpleContextBuilder()
generator = ELMGenerator()

with output.open("a", encoding="utf-8") as stream:
    for position, item in enumerate(items[len(completed):], len(completed) + 1):
        qid = item["query_id"]
        started = time.perf_counter()
        timing = {}
        if mode == "cached":
            transcript = transcripts[qid]
        else:
            t0 = time.perf_counter()
            transcript = asr.transcribe(data_dir / item["audio_file"])
            timing["asr_ms"] = (time.perf_counter() - t0) * 1000

        record_ids = []
        if k:
            t0 = time.perf_counter()
            vector = encoder.encode(transcript)
            timing["embedding_ms"] = (time.perf_counter() - t0) * 1000
            retrieved = retriever.retrieve(vector, k=k)
            record_ids = list(retrieved["record_ids"])
            if len(record_ids) != k or len(set(record_ids)) != k:
                raise SystemExit(f"{qid}: expected {k} distinct retrieved documents.")
            timing["retrieval_ms"] = retrieved["latency_ms"]

        built = builder.build(transcript, record_ids, data)
        generation = generator.generate(built["prompt"])
        answer = generation["answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise SystemExit(f"{qid}: empty answer.")

        timing["generation_ms"] = generation["latency_ms"]
        timing["end_to_end_ms"] = (time.perf_counter() - started) * 1000
        row = {
            "query_id": qid,
            "split": split,
            "audio_file": str(item["audio_file"]),
            "transcript": transcript,
            "answer": answer,
            "k": k,
            "retrieved_record_ids": record_ids,
            "prompt_chars": len(built["prompt"]),
            "timing": timing,
        }
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        print(f"Generated {position}/{len(items)} {qid}")
PY
        fi

        if [[ ! -f "$run_dir/judge/answer_judge_summary.json" ]]; then
            judge_args=()
            [[ "$limit" == all ]] || judge_args+=(--allow-subset)
            "$PYTHON" -u "$judge" \
                --split "$split" --candidates "$answers" \
                --out-dir "$run_dir/judge" "${judge_args[@]}"
        fi
    done

    "$PYTHON" - "$out" "$asr" "$k" "$split" "$limit" "$runs" "$repeat" "$revision" "$judge" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

folder, asr, k, split, limit, runs, repeat, revision, judge = sys.argv[1:]
out = Path(folder)
rounds = []

for run in range(1, int(runs) + 1):
    directory = out / f"run{run:02d}"
    with (directory / "answers.jsonl").open(encoding="utf-8") as stream:
        answers = [json.loads(line) for line in stream if line.strip()]
    scores = json.loads(
        (directory / "judge/answer_judge_summary.json").read_text(encoding="utf-8")
    )
    if not answers or scores["queries"] != len(answers):
        raise SystemExit(f"Answer and score counts differ in {directory}")
    rounds.append({
        "run": run,
        "queries": len(answers),
        "correctness": scores["mean_correctness"],
        "grounding": scores["mean_grounding"],
        "completeness": scores["mean_completeness"],
        "answer_score": scores["mean_normalised_answer_score"],
        "generation_s": mean(r["timing"]["generation_ms"] for r in answers) / 1000,
        "prompt_chars": mean(r["prompt_chars"] for r in answers),
        "judge_model": scores["judge_model"],
    })

metrics = ("correctness", "grounding", "completeness", "answer_score",
           "generation_s", "prompt_chars")
averages = {metric: mean(r[metric] for r in rounds) for metric in metrics}
deviations = (
    {metric: pstdev(r[metric] for r in rounds) for metric in metrics}
    if len(rounds) > 1 else {}
)
summary = {
    "asr": asr, "k": int(k), "split": split, "limit": limit,
    "runs": int(runs), "repeat": repeat, "source_commit": revision,
    "judge_entry": judge, "rounds": rounds, "mean": averages,
    "population_sd": deviations,
}
(out / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# RAG experiment",
    "",
    f"- ASR mode: `{asr}`",
    f"- Retrieval depth: {k}",
    f"- Split: `{split}`",
    f"- Queries per round: {rounds[0]['queries']}",
    f"- Rounds: {runs} (`{repeat}`)",
    f"- Source commit: `{revision}`",
    f"- Judge entry: `{judge}`",
    "",
    "| Round | Correctness | Grounding | Completeness | Answer score | Generation (s) | Prompt characters |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rounds:
    lines.append(
        f"| {row['run']} | " + " | ".join(f"{row[m]:.4f}" for m in metrics) + " |"
    )
lines.append("| Mean | " + " | ".join(f"{averages[m]:.4f}" for m in metrics) + " |")
if deviations:
    lines.append("| Population SD | " + " | ".join(f"{deviations[m]:.4f}" for m in metrics) + " |")

lines.extend([
    "",
    "Answer scores are automated estimates. Generation latency covers the "
    "generation request, including API round-trip time. Prompt length is measured "
    "in characters. In judge-only mode, latency and prompt size come from the "
    "same saved generation round.",
])
report = "\n".join(lines) + "\n"
(out / "report.md").write_text(report, encoding="utf-8")
print(report)
PY

    if [[ "$publish" == yes ]]; then
        [[ "$(git branch --show-current)" == "$branch" ]] ||
            die "The branch changed. The report is available at $out/report.md."
        [[ "$(git rev-parse HEAD)" == "$revision" ]] ||
            die "The branch advanced. Publish the saved report manually."
        git diff --cached --quiet ||
            die "The Git index already contains staged changes. Publish manually."

        files=("$out/job.sh" "$out/summary.json" "$out/report.md")
        for ((run = 1; run <= runs; run++)); do
            printf -v name 'run%02d' "$run"
            files+=(
                "$out/$name/answers.jsonl"
                "$out/$name/judge/answer_judge_per_query.jsonl"
                "$out/$name/judge/answer_judge_summary.json"
            )
        done
        git add -f -- "${files[@]}"
        git commit --only -m "Record RAG experiment k=$k ($split, $runs runs)" -- "${files[@]}"
        GIT_TERMINAL_PROMPT=0 git push origin "HEAD:refs/heads/$branch"
    fi
    exit 0
fi

asr=cached
k=5
split=practice
limit=10
runs=1
repeat=full
cache=
publish=no
update=yes
walltime=03:00:00

while (( $# )); do
    case "$1" in
        --asr|--k|--split|--limit|--runs|--repeat|--cache|--time)
            (( $# >= 2 )) || die "Missing value for $1"
            case "$1" in
                --asr) asr=$2 ;;
                --k) k=$2 ;;
                --split) split=$2 ;;
                --limit) limit=$2 ;;
                --runs) runs=$2 ;;
                --repeat) repeat=$2 ;;
                --cache) cache=$2 ;;
                --time) walltime=$2 ;;
            esac
            shift 2 ;;
        --push) publish=yes; shift ;;
        --no-update) update=no; shift ;;
        --help)
            printf '%s\n' \
              'Usage: bash rag/scripts/run_experiment.sh' \
              '  [--asr cached|all|none] [--k N] [--split practice|dev]' \
              '  [--limit N|all] [--runs N] [--repeat full|judge]' \
              '  [--cache PATH] [--time HH:MM:SS] [--push] [--no-update]'
            exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$asr" =~ ^(cached|all|none)$ ]] || die "Invalid --asr."
[[ "$split" =~ ^(practice|dev)$ ]] || die "Invalid --split."
[[ "$k" =~ ^(0|[1-9][0-9]*)$ ]] || die "Invalid --k."
[[ "$limit" =~ ^(all|[1-9][0-9]*)$ ]] || die "Invalid --limit."
[[ "$runs" =~ ^[1-9][0-9]*$ ]] || die "Invalid --runs."
[[ "$repeat" =~ ^(full|judge)$ ]] || die "Invalid --repeat."
[[ -x "$PYTHON" ]] || die "Python environment not found: $PYTHON"
branch=$(git branch --show-current)
[[ -n "$branch" ]] || die "Check out a branch first."

if [[ "$update" == yes ]]; then
    git pull --ff-only origin "$branch"
fi
revision=$(git rev-parse HEAD)

if [[ "$asr" == cached ]]; then
    if [[ -z "$cache" ]]; then
        [[ "$split" == practice ]] || die "Supply a dev transcript JSONL with --cache."
        cache=rag/results/baseline_all_cpu_350_v1.jsonl
    fi
    [[ -f "$cache" ]] || die "Transcript cache not found: $cache"
fi

stamp=$(date +%Y%m%d_%H%M%S)
out="rag/results/depth_k${k}_${split}_runs${runs}_${stamp}"
mkdir -p rag/results
mkdir "$out" || die "Experiment directory already exists: $out"

resources=()
[[ "$asr" == cached ]] || resources+=(--gres=gpu:1)

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$ROOT"
    printf 'export MLS_RAG_PYTHON=%q\n' "$PYTHON"
    printf 'exec bash %q --worker' "$ROOT/rag/scripts/run_experiment.sh"
    printf ' %q' "$out" "$asr" "$k" "$split" "$limit" "$runs" \
        "$cache" "$repeat" "$publish" "$branch" "$revision"
    printf '\n'
} > "$out/job.sh"

job_id=$(sbatch --parsable -p Teaching \
    --job-name="ljq-depth-k${k}" \
    --cpus-per-task=2 --mem=16G --time="$walltime" \
    "${resources[@]}" --output="$ROOT/$out/slurm-%j.log" "$out/job.sh")
job_id=${job_id%%;*}

printf 'Job ID: %s\nResults: %s\n' "$job_id" "$out"
printf 'Watch: tail -n 20 -F %q\n' "$out/slurm-$job_id.log"
printf 'Resume after a failed job: '
printf '%q ' sbatch -p Teaching --cpus-per-task=2 --mem=16G --time="$walltime" \
    "${resources[@]}" --output="$ROOT/$out/slurm-%j.log" "$out/job.sh"
printf '\n'