# Dense–BM25 on a fixed dense candidate set

The reported run uses the same 350 practice queries and saved ASR transcripts as Dense Top-5. Dense retrieval still returns five documents. Dense–BM25 only reorders that set. Alpha 0.75 is the weight on the dense score. Recall@5 cannot change because the candidate set does not change.

| Configuration | Recall@5 | Mean R5 |
|---|---:|---:|
| Dense–BM25 α=0.75, k=5 | 42.13% | 0.351143 |

Judge model: utter-project/EuroLLM-22B-Instruct-2512

Judge repeats: 5

Question mode: transcript

## Files

- `answers.jsonl` contains the generated answer and the five document IDs for each query.
- `answer_judge_summary.json` records the aggregate score, query count and five judge repeats.
- `ranking/` contains the earlier ordering comparison for BM25, MiniLM, and Dense–BM25 at alpha 0.25, 0.50 and 0.75. These files do not contain answer scores.

The reranking code is `rag/extensions/rerank_core.py`. The answer run is `rag/extensions/generate_reranked_answers.py`. The ordering comparison is `rag/extensions/evaluate_rerank.py`.
