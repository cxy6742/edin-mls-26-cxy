# What we changed

This repository starts from the course ASR and RAG framework
(`ed-aisys/edin-mls-26-autumn`). The supplied pipeline is unchanged:
16 kHz audio, ASR, the `enc_a` dense retriever, and EuroLLM-22B-Instruct.
Our work adds CUDA kernels for ASR, then two RAG studies on the same
350 saved practice transcripts. Later RAG timings do not include ASR.

## ASR

We replaced four PyTorch operations inside GLM-ASR-Nano with CUDA kernels:
RoPE, reduction, tiled GEMM, and attention.

| Item | Path |
|---|---|
| Kernel sources | `hw1-asr/csrc/task1_rope.cu`, `task2_reduce.cu`, `task3_gemm.cu`, `task4_attention.cu` |
| Python binding | `hw1-asr/glm_asr_cuda/` |
| Tests | `hw1-asr/tests/` |
| Measured run | `hw1-asr/results/asr_cuda_profile.txt` |

All 33 kernel tests pass. The largest representative error is 3.72e-5.
On the 3.50 s clip, PyTorch and the all-CUDA build both score 100%.
PyTorch takes 1337.7 ms and 8.90 GiB. All CUDA takes 6094.5 ms and 8.56 GiB.
Linear layers account for 81.4% of CUDA kernel time.

## RAG: why later runs keep five documents

Dense retrieval recall rises with depth, but the five-repeat answer score
peaks at five documents. Those runs are the reason both methods below still
pass five documents to the generator.

| Item | Path |
|---|---|
| Retrieval and answer generation at k = 2, 5, 10, 20 | `rag/extensions/run_retrieval_depth.py` |
| Same comparison from saved transcripts | `rag/extensions/generate_depth_fixed_transcripts.py` |
| Paired k = 5 and k = 20 diagnosis | `rag/extensions/diagnose_retrieval_depth.py` |
| Repeat-5 answer scores | `rag/results/system_extension/retrieval_depth/repeat5_practice350/` |
| Saved Dense Top-20 candidates | `rag/results/depth_k20_practice350.jsonl` |

Recall@5 / @10 / @20 / @100 is 42.13% / 47.96% / 57.85% / 76.99%.
Mean R5 is 0.319667, 0.346286, 0.327476, and 0.314524 for k = 2, 5, 10, and 20.

## Method 1: reorder the same five documents

The dense top five stay fixed. Dense–BM25 with α = 0.75 uses 0.75 times the
dense score plus 0.25 times BM25. BM25 alone and MiniLM on these five lowered
MRR@5, so only α = 0.75 was sent to the generator.

| Item | Path |
|---|---|
| BM25, MiniLM, and Dense–BM25 | `rag/extensions/rerank_core.py` |
| Order metrics on the same set | `rag/extensions/evaluate_rerank.py` |
| Answer generation after reordering | `rag/extensions/generate_reranked_answers.py` |
| Tests | `rag/tests/test_rerank_extension.py` |
| Answers and judge summary | `rag/results/system_extension/rerank/dense_bm25/` |
| Ordering summaries, including α = 0.25 and 0.50 | `rag/results/system_extension/rerank/dense_bm25/ranking/` |
| Short result note | `rag/results/system_extension/rerank/dense_bm25/README.md` |

For α = 0.75 and k = 5, Recall@5 stays 42.13%. ΔMRR@5 is −0.003 and
ΔnDCG@5 is −0.002. Mean R5 is 0.351143 (ΔR5 +0.0049). The paired bootstrap
95% CI is [−0.0170, +0.0270]. Extra time is 1 minute 26 seconds.

## Method 2: select five documents from a larger pool

Cross-encoders rescore a dense candidate pool and keep five documents.
MiniLM, BGE, and Mixedbread each cut the Dense Top-20 to five. BGE and
Mixedbread are also run on larger pools, and their rankings are fused over
the Dense Top-100.

| Item | Path |
|---|---|
| MiniLM on the saved Dense Top-20 | `rag/extensions/rerank_top5.py` |
| MiniLM selected ids | `rag/results/dense20_to5_rerankers/minilm/reranked.jsonl` |
| MiniLM answers and judge file | `rag/results/dense20_to5_rerankers/minilm/` |
| BGE Top-20 scores | `rag/results/dense20_to5_rerankers/bge/bge20-full-3718021.jsonl` |
| BGE answers and judge file | `rag/results/dense20_to5_rerankers/bge/` |
| Mixedbread scoring script | `rag/results/dense20_to5_rerankers/scripts/run_mxbai_chunks350.sh` |
| Mixedbread scores | `rag/results/dense20_to5_rerankers/mixedbread/mxbai-chunks350-3720077.jsonl` |
| Mixedbread whole-document answers | `rag/results/dense20_to5_rerankers/scripts/run_mixedbread20_whole_answer350.sh` |
| Mixedbread answers and judge file | `rag/results/dense20_to5_rerankers/mixedbread/` |
| Reconstructed BGE Top-100 scorer | `rag/results/candidate_pool_and_fusion/scripts/run_bge100_reconstructed.py` |
| BGE Top-100 score shards | `rag/results/candidate_pool_and_fusion/candidate_pool/bge100-full-3718533-*.jsonl` |
| Mixedbread Top-100 script | `rag/results/candidate_pool_and_fusion/scripts/run_mxbai_chunks100_350.sh` |
| Rank fusion of BGE and Mixedbread | `rag/results/candidate_pool_and_fusion/scripts/run_mxbai_fusion350_original.sh` |
| Fusion scores | `rag/results/candidate_pool_and_fusion/rank_fusion/scoring/` |
| Fusion answers | `rag/results/candidate_pool_and_fusion/rank_fusion/bge_tiebreak/` and `mixedbread_tiebreak/` |
| Checks for pools and fusion | `rag/results/candidate_pool_and_fusion/scripts/check_candidate_pool.py`, `check_rank_fusion.py` |

The original BGE Top-20 scoring script is not in this repository. Only the
saved scores are. `run_bge100_reconstructed.py` rebuilds the Top-100 BGE run;
the original job body is unavailable.

On the same Top-20, MiniLM Recall@5 is 41.11% and mean R5 is 0.337810.
BGE is 47.18% and 0.348857. Mixedbread, scoring whole documents, is 47.32%
and 0.371190. Final Recall@5 does not rise when the pool grows from 20 to
50 or 100. Fusion reaches 49.77% or 50.49% Recall@5 in one shared run of
32 minutes 51 seconds. Every 95% CI for ΔR5 includes zero.

## Other added files

`rag/extensions/evaluate_answers_format_safe.py` and
`evaluate_answers_brace_safe.py` repair a small number of malformed judge
responses before scoring. They do not change retrieval or generation.
