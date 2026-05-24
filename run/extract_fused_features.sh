#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON:-${repo_root}/.venv/bin/python}"
encoder="moonshotai/Kimi-Audio-7B-Instruct"
tokenizer="THUDM/glm-4-voice-tokenizer"
sample_rate=16000
output_root="dump/fused"
force=false
resume=false
data_dirs=()

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--data_dir PATH ...] [--encoder MODEL_OR_PATH] [--force] [--resume]

Options:
  --data_dir PATH         Dataset directory containing wav.scp. May be repeated.
  --encoder PATH          Encoder model repo/path. Default: ${encoder}
  --output-root PATH      Root directory for extracted fused feature .pt files.
  --resume                Continue partial extraction by reusing existing .pt files and rebuilding scp/shape.
  --force                 Regenerate fused features even if outputs already exist.
  -h, --help              Show this help message.
EOF
)

while [ $# -gt 0 ]; do
    case "$1" in
        --data_dir)
            data_dirs+=("$2")
            shift 2
            ;;
        --encoder)
            encoder="$2"
            shift 2
            ;;
        --output-root)
            output_root="$2"
            shift 2
            ;;
        --resume)
            resume=true
            shift
            ;;
        --force)
            force=true
            shift
            ;;
        -h|--help)
            echo "${help_message}"
            exit 0
            ;;
        *)
            echo "${help_message}"
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [ "${#data_dirs[@]}" -eq 0 ]; then
    echo "${help_message}"
    echo "--data_dir is required" >&2
    exit 2
fi

force_suffix=""
if ${force}; then
    force_suffix=" --force"
fi

log "Encoder: ${encoder}"
log "Output root: ${output_root}"
log "Resume: ${resume}"
for data_dir in "${data_dirs[@]}"; do
    log "Data dir: ${data_dir}"
done

cmd=(
    "${python_bin}" "${repo_root}/train/extract_fused_features.py"
    --encoder "${encoder}"
    --tokenizer "${tokenizer}"
    --sample_rate "${sample_rate}"
    --output_root "${output_root}"
)

for data_dir in "${data_dirs[@]}"; do
    cmd+=(--data_dir "${data_dir}")
done

if ${force}; then
    cmd+=(--force)
fi
if ${resume}; then
    cmd+=(--resume)
fi

"${cmd[@]}"

log "Finished extracting fused features"
