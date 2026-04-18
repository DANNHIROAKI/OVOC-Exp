#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# Use the NVIDIA GPU with the least used memory (OpenViking local mode).
if command -v nvidia-smi >/dev/null 2>&1; then
  _ov_gpu="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | sort -t, -k2 -n \
      | head -1 \
      | cut -d, -f1 \
      | tr -d ' '
  )"
  if [[ -n "${_ov_gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${_ov_gpu}"
  fi
  unset _ov_gpu
fi

for cmd in python3 git curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

VENV_DIR="$ROOT/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel >/dev/null
python -m pip install -r "$ROOT/requirements.txt"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec python -m ovoc_bench.cli --config "$ROOT/configs/experiment.yaml" "$@"
