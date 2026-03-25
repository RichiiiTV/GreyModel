#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH+:${PYTHONPATH}}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  TORCHRUN_BIN=".venv/bin/torchrun"
else
  PYTHON_BIN="python"
  TORCHRUN_BIN="torchrun"
fi

DATASET_DIR="${DATASET_DIR:-data/public_pretrain/defect_spectrum_full}"
RUN_ROOT="${RUN_ROOT:-artifacts}"
VARIANT="${VARIANT:-base}"
EPOCHS="${EPOCHS:-200}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-200}"
KEEP_LAST_K_CHECKPOINTS="${KEEP_LAST_K_CHECKPOINTS:-5}"
PRETRAIN_CROP_SIZE="${PRETRAIN_CROP_SIZE:-512}"
PRETRAIN_NUM_CROPS="${PRETRAIN_NUM_CROPS:-2}"
PRETRAIN_CROP_SCALES="${PRETRAIN_CROP_SCALES:-0.75 1.0 1.25}"
MAX_GLOBAL_FEATURE_GRID="${MAX_GLOBAL_FEATURE_GRID:-12}"
DIST_STRATEGY="${DIST_STRATEGY:-fsdp}"

echo "[GreyModel] Importing DefectSpectrum and converting to 8-bit grayscale at ${DATASET_DIR}"
"${PYTHON_BIN}" -m greymodel dataset build-hf \
  --dataset-preset "defect_spectrum_full" \
  --output-dir "${DATASET_DIR}" \
  --allow-rgb-conversion

echo "[GreyModel] Validating imported manifest"
"${PYTHON_BIN}" -m greymodel dataset validate "${DATASET_DIR}/manifest.jsonl"

echo "[GreyModel] Launching 8xA100 patch-based pretraining with ${DIST_STRATEGY}"
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" -m greymodel train pretrain \
  --manifest "${DATASET_DIR}/manifest.jsonl" \
  --index "${DATASET_DIR}/dataset_index.json" \
  --variant "${VARIANT}" \
  --run-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${PER_GPU_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --precision auto \
  --distributed-strategy "${DIST_STRATEGY}" \
  --activation-checkpointing \
  --memory-report \
  --pretrain-crop-size "${PRETRAIN_CROP_SIZE}" \
  --pretrain-num-crops "${PRETRAIN_NUM_CROPS}" \
  --pretrain-crop-scales ${PRETRAIN_CROP_SCALES} \
  --max-global-feature-grid "${MAX_GLOBAL_FEATURE_GRID}" \
  --channels-last \
  --log-every-n-steps "${LOG_EVERY_N_STEPS}" \
  --checkpoint-every-n-steps "${CHECKPOINT_EVERY_N_STEPS}" \
  --keep-last-k-checkpoints "${KEEP_LAST_K_CHECKPOINTS}"
