#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p weights

# Reuse Chapter-16-old files as hard links when available. This is fast and does not
# duplicate tens of gigabytes on the same filesystem.
reuse_old() {
  local name="$1"
  local old="../code_chapter16_old/weights/$name"
  local target="weights/$name"
  if [[ ! -e "$target" && -d "$old" ]]; then
    echo "[reuse] $old -> $target (hard links)"
    cp -al "$old" "$target"
  fi
}

reuse_old Wan2.2-TI2V-5B
reuse_old Qwen3-VL-2B-Instruct
reuse_old Motus_robotwin2

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install: pip install -U huggingface_hub" >&2
  exit 2
fi

download() {
  local repo="$1" target="$2" marker="$3"
  if [[ -e "$target/$marker" ]]; then
    echo "[skip] $target already has $marker"
  else
    echo "[download] $repo -> $target"
    huggingface-cli download "$repo" --local-dir "$target"
  fi
}

# Formal Stage 3 starts from the Stage-2 Motus checkpoint, not Motus_robotwin2.
download motus-robotics/Motus weights/Motus mp_rank_00_model_states.pt
# Backbone configs/VAE/UMT5 and frozen Qwen processor/config.
download Wan-AI/Wan2.2-TI2V-5B weights/Wan2.2-TI2V-5B models_t5_umt5-xxl-enc-bf16.pth
download Qwen/Qwen3-VL-2B-Instruct weights/Qwen3-VL-2B-Instruct model.safetensors

if [[ "${DOWNLOAD_ROBOTWIN2:-0}" == "1" ]]; then
  download motus-robotics/Motus_robotwin2 weights/Motus_robotwin2 mp_rank_00_model_states.pt
fi

echo "Weights ready. Motus_robotwin2 is optional and is not the 16D G2 initializer."
