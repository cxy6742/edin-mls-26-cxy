# Candidate Pool Size and Rank Fusion

## 1. Candidate Pool Experiments

All configurations use the same 350 practice queries.

| Reranker | 20 to 5 | 50 to 5 | 100 to 5 |
|---|---:|---:|---:|
| BGE | 47.18% | 46.72% | 45.58% |
| Mixedbread | 47.32% | 45.88% | 46.34% |

The Mixedbread results use whole-document scores.

## 2. Verify the Saved Results

Set up the course RAG environment and practice corpus.
Run from the repository root:

    PYTHONPATH=rag python rag/results/candidate_pool_and_fusion/scripts/check_candidate_pool.py

This script recalculates all six Recall@5 results.
All six results have been verified using the original
saved rankings and candidate scores.

No GPU inference is required for this verification.

The archived Dense Top-100 input was checked against
all seven BGE output shards. The candidate IDs and
their order matched for all 350 queries.

## 3. Reproduce BGE Scoring

The original BGE experiment used CPU inference across
seven shards of 50 queries each.

Original Slurm job: 3718533.

The original submission body is unavailable.
The reconstructed script uses the verified candidate
inputs and selected model settings.

Model: BAAI/bge-reranker-v2-m3
Batch size: 16
Maximum sequence length: 512

The reconstructed script has passed syntax checks.
Exact score reproduction has not been tested.
The original batch size and maximum sequence length
could not be independently confirmed from the original
submission body.

Run on a compute node from the repository root:

    T=rag/results/candidate_pool_and_fusion
    OUT=$(mktemp -d)

    for shard in 0 1 2 3 4 5 6; do
        PYTHONPATH=rag python "$T/scripts/run_bge100_reconstructed.py" --shard "$shard" --output-dir "$OUT" || break
    done

## 4. Reproduce Mixedbread Scoring

Original Slurm job: 3720102.

The original script is preserved in:

    scripts/run_mxbai_chunks100_350_original.sh

The portable version retains the original model and
scoring algorithm while adapting the file paths.

Model: mixedbread-ai/mxbai-rerank-large-v1
Device: CUDA
Batch size: 8
Maximum sequence length: 512

Run on a CUDA-enabled compute node:

    T=rag/results/candidate_pool_and_fusion
    OUT_DIR=$(mktemp -d) bash "$T/scripts/run_mxbai_chunks100_350.sh"

The portable script has passed Python and Shell syntax
checks. A new full GPU inference run has not been
performed.
