#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.ubuntu1804-cu121.lock.txt}"

exec "$ROOT_DIR/scripts/setup_uv_env.sh"
