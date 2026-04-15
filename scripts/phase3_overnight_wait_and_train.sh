#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_DIR="${DATASET_DIR:-data/training_v3_10000}"
DATASET_H5="${DATASET_H5:-$DATASET_DIR/training_data.h5}"
DATASET_SUMMARY="${DATASET_SUMMARY:-$DATASET_DIR/summary.json}"
RUN_NAME="${RUN_NAME:-phase3_offcenter_10k_2gpu}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "[$(date -Is)] Waiting for dataset artifacts:"
echo "  summary: $DATASET_SUMMARY"
echo "  h5:      $DATASET_H5"

while [[ ! -f "$DATASET_SUMMARY" || ! -f "$DATASET_H5" ]]; do
    echo "[$(date -Is)] Dataset not ready yet. Sleeping for ${WAIT_SECONDS}s."
    sleep "$WAIT_SECONDS"
done

echo "[$(date -Is)] Dataset detected. Starting two-GPU training."
echo "[$(date -Is)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec python -u scripts/phase3_train_unet.py \
    --data-h5 "$DATASET_H5" \
    --epochs 20 \
    --batch-size 16 \
    --num-workers 8 \
    --min-positive-amplitude 3e-5 \
    --threshold 0.92 \
    --device cuda \
    --gpu-ids 0,1 \
    --run-name "$RUN_NAME" \
    "$@"
