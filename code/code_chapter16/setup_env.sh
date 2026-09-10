#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 - <<'PY'
import torch
print(f"Detected torch={torch.__version__}, CUDA runtime={torch.version.cuda}, CUDA available={torch.cuda.is_available()}")
major, minor = map(int, torch.__version__.split('+', 1)[0].split('.')[:2])
if (major, minor) < (2, 7):
    raise SystemExit("Motus expects PyTorch 2.7+; install the CUDA wheel before running setup_env.sh")
PY
then
  echo "Official CUDA 12.8 example:" >&2
  echo "  pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128" >&2
  exit 2
fi

python3 -m pip install -r "$ROOT/requirements.txt"
echo "Optional official fast path: pip install flash-attn --no-build-isolation"
echo "Without flash-attn, this chapter uses PyTorch SDPA (higher VRAM use)."
