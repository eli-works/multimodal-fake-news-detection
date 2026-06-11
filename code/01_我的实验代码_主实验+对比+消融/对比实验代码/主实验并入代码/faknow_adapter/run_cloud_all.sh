#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${SCRIPT_DIR}/artifacts}"
RUNS_DIR="${RUNS_DIR:-${SCRIPT_DIR}/runs}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"

CFND_ROOT="${CFND_ROOT:?Please set CFND_ROOT to your CFND dataset root.}"
GOSSIP_ROOT="${GOSSIP_ROOT:?Please set GOSSIP_ROOT to your Gossip dataset root.}"
WEIBO_ROOT="${WEIBO_ROOT:?Please set WEIBO_ROOT to your Weibo dataset root.}"
GOSSIP_IMAGE_ROOT="${GOSSIP_IMAGE_ROOT:-${GOSSIP_ROOT}}"

python "${SCRIPT_DIR}/run_all_models.py" \
  --dataset all \
  --model all \
  --prepare-only \
  --device "${DEVICE}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --runs-dir "${RUNS_DIR}" \
  --cfnd-root "${CFND_ROOT}" \
  --gossip-root "${GOSSIP_ROOT}" \
  --weibo-root "${WEIBO_ROOT}" \
  --gossip-image-root "${GOSSIP_IMAGE_ROOT}"

python "${SCRIPT_DIR}/run_all_models.py" \
  --dataset cfnd \
  --model all \
  --train-only \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --runs-dir "${RUNS_DIR}" \
  --cfnd-root "${CFND_ROOT}" \
  --gossip-root "${GOSSIP_ROOT}" \
  --weibo-root "${WEIBO_ROOT}" \
  --gossip-image-root "${GOSSIP_IMAGE_ROOT}"

python "${SCRIPT_DIR}/run_all_models.py" \
  --dataset gossip \
  --model all \
  --train-only \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --runs-dir "${RUNS_DIR}" \
  --cfnd-root "${CFND_ROOT}" \
  --gossip-root "${GOSSIP_ROOT}" \
  --weibo-root "${WEIBO_ROOT}" \
  --gossip-image-root "${GOSSIP_IMAGE_ROOT}"

python "${SCRIPT_DIR}/run_all_models.py" \
  --dataset weibo \
  --model all \
  --train-only \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --runs-dir "${RUNS_DIR}" \
  --cfnd-root "${CFND_ROOT}" \
  --gossip-root "${GOSSIP_ROOT}" \
  --weibo-root "${WEIBO_ROOT}" \
  --gossip-image-root "${GOSSIP_IMAGE_ROOT}"

echo "Done. Check report: ${RUNS_DIR}/run_report.json"
