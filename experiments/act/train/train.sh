#!/usr/bin/env bash
# train.sh: Train an ACT policy on the merged right-arm lipbalm dataset.
#
# Usage:
#   bash train.sh                    # baseline: 54 train eps × 75k steps
#   bash train.sh --steps=15000      # forward extra flags to lerobot-train
#   TAG=aug bash train.sh --dataset.image_transforms.enable=true
set -euo pipefail

VENV="/home/tagi-runner1/train_test/lerobot/.venv"
LEROBOT_TRAIN="$VENV/bin/lerobot-train"
DATA_ROOT="${ACT_DATA_ROOT:-/home/tagi-runner1/train_test/mobileai-lipbalm-right-only}"
DST="${ACT_RUNS_DIR:-/home/tagi-runner1/train_test/logs}"
mkdir -p "$DST"
export TORCH_HOME="${TORCH_HOME:-$DST/.torch_cache}"

TAG="${TAG:-baseline-h}"
STEPS="${STEPS:-75000}"
# System-tuned defaults for RTX 5090 (32 GB) + 24 CPU cores + 62 GB RAM.
# Override via env if running on a different host.
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Default: train on eps 0..53, hold out 54..59. Override with TRAIN_EPS env.
TRAIN_EPS="${TRAIN_EPS:-[$(seq -s, 0 53)]}"

OUT="$DST/act_lipbalm_right_${TAG}"
LOG="$DST/act_lipbalm_right_${TAG}.log"

echo "=== $(date -Is) starting $TAG -> $OUT ==="
"$LEROBOT_TRAIN" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=mobileai/lipbalm_right_only \
  --dataset.root="$DATA_ROOT" \
  --dataset.episodes="$TRAIN_EPS" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=200 \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir="$OUT" \
  --job_name="act_lipbalm_right_${TAG}" \
  "$@" \
  >"$LOG" 2>&1
echo "=== $(date -Is) finished $TAG ==="
