#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-graphec}"
PYTHON_BIN="${PYTHON_BIN:-conda run -n ${ENV_NAME} python}"
DATASET_ROOT="${DATASET_ROOT:-../../data/processed/datasets/enzyme_classification_dataset}"
CACHE_ROOT="${CACHE_ROOT:-emulator_bench/cache/enzyme_classification_dataset}"
SPOOLER_BIN="${SPOOLER_BIN:-ts}"
EXECUTION_MODE="${EXECUTION_MODE:-ts}"
EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CACHE_GPUS="${CACHE_GPUS:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_MODE="${EVAL_MODE:-direct}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
PROTTRANS_MODEL_PATH="${PROTTRANS_MODEL_PATH:-}"
CUDA_VISIBLE_DEVICES_ARG=()
SEED_VALUES=()
SEED_ARGS=()
MODEL_ARGS=()
SHOWING_HELP=0

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES_ARG=(--cuda-visible-devices "$CUDA_VISIBLE_DEVICES")
fi

if [[ -n "$PROTTRANS_MODEL_PATH" ]]; then
    MODEL_ARGS=(--prottrans-model-path "$PROTTRANS_MODEL_PATH")
fi

read -r -a SEED_VALUES <<< "${SEEDS:-0 1 2}"
for seed in "${SEED_VALUES[@]}"; do
    SEED_ARGS+=(--seed "$seed")
done

for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        SHOWING_HELP=1
    fi
done

if [[ "$SHOWING_HELP" -eq 0 ]]; then
    echo "Queuing GraphEC Leak-CURBER pipeline for multiple seeds..."
fi

# shellcheck disable=SC2086
$PYTHON_BIN -m emulator_bench.queue_pipeline \
    --dataset-root "$DATASET_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --split-group random_splits \
    --split-group enzyme_sequence_splits \
    --split-group enzyme_structure_splits \
    --split-group uniprot_time_splits \
    --split-group ec_hierarchy_splits/L1 \
    --split-group ec_hierarchy_splits/L2 \
    --split-group ec_hierarchy_splits/L3 \
    --split-group ec_hierarchy_splits/L4 \
    --env-name "$ENV_NAME" \
    --spooler-bin "$SPOOLER_BIN" \
    --execution-mode "$EXECUTION_MODE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --cache-gpus "$CACHE_GPUS" \
    --train-gpus "$TRAIN_GPUS" \
    --eval-gpus "$EVAL_GPUS" \
    --eval-mode "$EVAL_MODE" \
    --eval-cuda-device "$EVAL_CUDA_DEVICE" \
    "${MODEL_ARGS[@]}" \
    "${SEED_ARGS[@]}" \
    "${CUDA_VISIBLE_DEVICES_ARG[@]}" \
    "$@"

if [[ "$SHOWING_HELP" -eq 0 ]]; then
    echo "All GraphEC seeds queued."
fi
