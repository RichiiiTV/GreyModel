#!/usr/bin/env bash
#SBATCH --job-name=greymodel-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Equivalent cluster-specific submit pattern when your scheduler uses wrap syntax:
# sbatch -c 8 --mem=50G --gres=gpu:8 -p batch_gpu -q 3h --wrap="cd /path/to/GreyModel && export PYTHONPATH=$PWD/src && bash scripts/pretrain_8xa100_defect_spectrum.sh"

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

mkdir -p logs

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
export DATASET_DIR="${DATASET_DIR:-data/public_pretrain/defect_spectrum_full}"
export RUN_ROOT="${RUN_ROOT:-artifacts}"
export VARIANT="${VARIANT:-base}"
export EPOCHS="${EPOCHS:-200}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-200}"
export KEEP_LAST_K_CHECKPOINTS="${KEEP_LAST_K_CHECKPOINTS:-5}"

echo "[GreyModel] job_id=${SLURM_JOB_ID:-interactive}"
echo "[GreyModel] host=$(hostname)"
echo "[GreyModel] repo_root=${REPO_ROOT}"
echo "[GreyModel] dataset_dir=${DATASET_DIR}"
echo "[GreyModel] run_root=${RUN_ROOT}"
echo "[GreyModel] nproc_per_node=${NPROC_PER_NODE}"

bash scripts/pretrain_8xa100_defect_spectrum.sh
