#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock.txt}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
VENV_PY="$VENV_DIR/bin/python"

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

die() {
  printf '[setup][error] %s\n' "$*" >&2
  exit 1
}

command -v uv >/dev/null 2>&1 || die "uv is not installed. Install uv first: https://docs.astral.sh/uv/"
test -f "$LOCK_FILE" || die "missing lock file: $LOCK_FILE"

log "project root: $ROOT_DIR"
log "uv: $(uv --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  warn "nvidia-smi not found; this server may not expose NVIDIA GPUs."
fi

log "creating virtualenv at $VENV_DIR with Python $PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

log "syncing packages from $LOCK_FILE"
uv pip sync --python "$VENV_PY" "$LOCK_FILE"

log "checking flash_attn"
if "$VENV_PY" -c "import flash_attn" >/dev/null 2>&1; then
  log "flash_attn already importable"
else
  shopt -s nullglob
  wheels=("$ROOT_DIR"/flash_attn-*.whl)
  shopt -u nullglob
  if [ "${#wheels[@]}" -gt 0 ]; then
    log "trying local wheel: ${wheels[0]}"
    if uv pip install --python "$VENV_PY" "${wheels[0]}"; then
      log "flash_attn installed from local wheel"
    else
      warn "flash_attn wheel install failed; check Python, torch, CUDA, and GPU compatibility."
    fi
  else
    warn "flash_attn is not installed and no flash_attn-*.whl exists in the project root."
  fi
fi

log "environment check"
"$VENV_PY" "$ROOT_DIR/scripts/check_env.py"

log "done"
