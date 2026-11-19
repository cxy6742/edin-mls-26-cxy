# Retrieval Depth Experiment (350 Practice Queries)

This directory contains the results of the retrieval-depth experiment (Table 4).

The experiment compares k = 2, 5, 10, and 20 using the same 350 practice queries and saved ASR transcripts.

## 1. Experimental Results

| k | Mean Score (1 Repeat) | Mean Score (5 Repeats) |
|---|---:|---:|
| 2 | 0.326429 | 0.319667 |
| 5 | 0.336190 | 0.346286 |
| 10 | 0.321905 | 0.327476 |
| 20 | 0.311905 | 0.314524 |

The score is the normalised answer-quality score based on correctness, grounding and completeness.

The number of repeats refers to answer-judge evaluations, not independent answer-generation runs.

## 2. Result Files

### repeat1_practice350/

Contains summary files for k = 2, 5, 10, and 20.

Each summary records the evaluation of 350 practice queries with one judge repeat.

### repeat5_practice350/

Contains summary files for k = 2, 5, 10, and 20.

Each summary records the evaluation of 350 practice queries with five judge repeats.

Query-level results are also available for k = 5 and k = 10:

- k5_per_query.jsonl
- k10_per_query.jsonl

## 3. Related Scripts

The original scripts remain in rag/extensions/.

- [generate_depth_fixed_transcripts.py](../../../extensions/generate_depth_fixed_transcripts.py) - Generate answers at different retrieval depths using fixed ASR transcripts.
- [run_retrieval_depth.py](../../../extensions/run_retrieval_depth.py) - Run retrieval-depth experiments.
- [evaluate_answers_format_safe.py](../../../extensions/evaluate_answers_format_safe.py) - Evaluate generated answers using the LLM judge.
- [diagnose_retrieval_depth.py](../../../extensions/diagnose_retrieval_depth.py) - Analyse paired retrieval-depth results.

The scripts are linked rather than duplicated.

## 4. Reproduction

Run from the rag/ directory after configuring the course environment and ELM credentials.

For example:

```bash
OUT="$(mktemp -d)"

python extensions/generate_depth_fixed_transcripts.py \
  --k 10 --output "$OUT/k10_answers.jsonl"

python extensions/evaluate_answers_format_safe.py \
  --split practice \
  --candidates "$OUT/k10_answers.jsonl" \
  --repeats 5 \
  --out-dir "$OUT/k10_judge"
```

The example saves new results to a temporary directory without modifying the original experimental evidence.

New runs may produce different scores because answer generation and LLM-based evaluation are not fully deterministic.
