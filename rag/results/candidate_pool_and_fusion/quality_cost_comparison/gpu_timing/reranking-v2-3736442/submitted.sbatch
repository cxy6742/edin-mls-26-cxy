#!/usr/bin/env bash
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=reranking-v2
#SBATCH --output=rag/results/reranking-v2-%j.log

set -euo pipefail
cd "$HOME/edin-mls-26-cxy"
export PYTHONPATH=rag
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

out="rag/results/reranking-v2-${SLURM_JOB_ID}"
mkdir "$out"
cp rag/results/candidate_pool_and_fusion/quality_cost_comparison/gpu_timing/reranking-v2-3736442/measure_reranking_v2.py "$out/"
cp "$0" "$out/submitted.sbatch"

hostname
nvidia-smi -L
"$HOME/mls-rag-env/bin/python" -m pip freeze > "$out/environment.txt"

for config in "bge 20" "bge 100" "mixedbread 20"; do
    read -r model depth <<< "$config"
    "$HOME/mls-rag-env/bin/python" -u "$out/measure_reranking_v2.py" \
        --model "$model" --depth "$depth" --out "$out"
done
