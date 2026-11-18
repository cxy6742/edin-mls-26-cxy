# Table 6 · Comparing rerankers on Dense Top-20 candidates

All four configurations use the same 350 practice queries and saved ASR transcripts. Dense Top-5 passes the first five documents directly to the generator. BGE, MiniLM and Mixedbread each select five documents from the same Dense Top-20 candidate pool.

| Configuration | Recall@5 | Mean R5 | Original run log |
|---|---:|---:|---|
| Dense Top-5 | 42.13% | 0.346286 | [Dense retrieval](dense/retrieval_all_cpu_course_v1.log) |
| BGE 20→5 | 47.18% | 0.348857 | [BGE reranking](bge/bge20-full-3718021.log) |
| MiniLM 20→5 | 41.11% | 0.337810 | [MiniLM reranking](minilm/rerank350-3715796.log) |
| Mixedbread 20→5 | 47.32% | 0.371190 | [Mixedbread scoring](mixedbread/mxbai-chunks350-3720077.log) |

## 1. Check the published results

Each method folder contains three files used to verify its row.

- `answers.jsonl` contains the generated answer and the five selected document IDs for each query. Compute Recall@5 against the practice qrels, then average across the 350 queries.
- `answer_judge_per_query.jsonl` contains the normalised answer score for each query. Average these scores to recover mean R5.
- `answer_judge_summary.json` records the aggregate score, query count and five judge repeats.

## 2. Rerun BGE or MiniLM

Follow the [course RAG setup](../../README.md) to install the environment and practice corpus. Run these commands from the repository root. The reranker checks that each baseline Top-5 matches the first five documents in the saved Top-20.

```bash
T=$(dirname "$(grep -l '^# Table 6 · Comparing rerankers' rag/results/*/README.md)")
OUT=$(mktemp -d)
for model in BAAI/bge-reranker-v2-m3 cross-encoder/ms-marco-MiniLM-L6-v2; do
  python "$T/scripts/rerank_top5.py" --model "$model" --device cpu \
    --baseline "$T/dense/answers.jsonl" \
    --candidates "$T/inputs/depth_k20_practice350.jsonl" \
    --output "$OUT/${model##*/}.jsonl" || break
done
```

## 3. Reproduce the Mixedbread whole-document result

The original GPU scoring script is `scripts/run_mxbai_chunks350.sh`. Table 6 uses the `whole_top5` field in `mixedbread/mxbai-chunks350-3720077.jsonl`. The script `scripts/run_mixedbread20_whole_answer350.sh` uses that ranking for answer generation and evaluation. Check the original cluster paths in both scripts before rerunning them.

## 4. Re-evaluate the saved answers

From the repository root, activate the course environment and configure the ELM credentials. Choose `dense`, `bge`, `minilm` or `mixedbread` for `method`.

```bash
T=$(dirname "$(grep -l '^# Table 6 · Comparing rerankers' rag/results/*/README.md)")
method=mixedbread
python "$T/scripts/evaluate_answers_format_safe.py" --split practice \
  --candidates "$T/$method/answers.jsonl" --repeats 5 \
  --out-dir "$(mktemp -d)/judge"
```
