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

## 5. Rank Fusion

Rank Sum Fusion combines the BGE and Mixedbread rankings
over the same 100 dense candidates for each practice query.

Documents are ranked by the sum of their positions in
the two reranker rankings. Ties are resolved using either
the BGE rank or the Mixedbread rank.

| Configuration | Recall@5 | Mean R5 |
|---|---:|---:|
| BGE tie-break | 49.77% | 0.358714 |
| Mixedbread tie-break | 50.49% | 0.353714 |

Both configurations use 350 practice queries and five
final documents per query.

Judge model: utter-project/EuroLLM-22B-Instruct-2512

Judge repeats: 5

Question mode: transcript

### Verify the saved results

Run from the repository root:

    PYTHONPATH=rag python rag/results/candidate_pool_and_fusion/scripts/check_rank_fusion.py

The verification script independently reconstructs both
fusion rankings from the saved reranker scores.

It verifies candidate order, final Top-5 document order,
answer inputs and transcripts for all 350 queries in
each configuration.

Recall@5 and mean normalised answer scores are
independently recalculated from the saved results.

Both configurations passed all verification checks.

No GPU inference or ELM API calls are required.

### Original experiment scripts

The following original scripts are preserved without
modification:

- scripts/run_mxbai_fusion350_original.sh
- scripts/run_whole_fusion_answer350_original.sh
- scripts/run_mixedbread_tiebreak_fusion350_original.sh

These scripts retain the original cluster environment,
working directory and historical input and output paths.
Adjust these paths and configure the required environment
before rerunning them.

The original Mixedbread scoring job was 3720050.

The archived rank_fusion directory contains the original
scoring results, final rankings, generated answers,
per-query judge scores, summary results and logs.
