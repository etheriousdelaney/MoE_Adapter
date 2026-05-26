#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.ubuntu1804-cu121.lock.txt}"
export VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

"$ROOT_DIR/scripts/setup_uv_env.sh"

"$VENV_DIR/bin/python" - <<'PY'
import sys

import torch

print("[v100-check] torch:", torch.__version__)
print("[v100-check] torch cuda:", torch.version.cuda)
print("[v100-check] arch list:", torch.cuda.get_arch_list())

if not torch.cuda.is_available():
    raise SystemExit("[v100-check][error] CUDA is not available to PyTorch.")

if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit(
        "[v100-check][error] this PyTorch build does not include sm_70, "
        "so it cannot run on Tesla V100."
    )

device_count = torch.cuda.device_count()
print("[v100-check] cuda device count:", device_count)
for idx in range(device_count):
    name = torch.cuda.get_device_name(idx)
    capability = torch.cuda.get_device_capability(idx)
    print(f"[v100-check] cuda device {idx}: {name}, capability={capability[0]}.{capability[1]}")

has_v100 = any(torch.cuda.get_device_capability(idx) == (7, 0) for idx in range(device_count))
if not has_v100:
    print("[v100-check][warn] no compute capability 7.0 GPU was detected.")

x = torch.randn(2, 2, device="cuda")
_ = x @ x
torch.cuda.synchronize()
print("[v100-check] CUDA matmul smoke test passed.")
PY
