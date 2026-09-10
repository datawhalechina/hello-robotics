#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
GPUS="${1:-2}"
CONFIG="${2:-configs/stage3_g2_lora.yaml}"
PORT="${MASTER_PORT:-29516}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

mkdir -p logs checkpoints
if [[ ! -d weights/Motus ]]; then
  echo "Missing weights/Motus. Run: bash download_weights.sh" >&2
  exit 2
fi

echo "Stage-3 config=$CONFIG GPUs=$GPUS attention=${MOTUS_ATTN_BACKEND:-auto}"
"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$PORT" \
  train_stage3.py \
  --deepspeed configs/zero1.json \
  --config "$CONFIG" \
  --report_to tensorboard \
  2>&1 | tee "logs/$(basename "${CONFIG%.yaml}")_$(date +%Y%m%d_%H%M%S).log"
