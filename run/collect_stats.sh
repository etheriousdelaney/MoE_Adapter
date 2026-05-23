#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

nj="${NJ:-32}"
python_bin="${PYTHON:-./.venv/bin/python}"
force=false
progress_every="${PROGRESS_EVERY:-500}"
cleanup_logs=true
config=""
log_dir=""

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--config conf/MoEClassfier.yaml] [--nj 8] [--log-dir exp/collect_stats/logdir] [--force] [--keep-logs]

Options:
  --config PATH           YAML config to read dataset_conf.train_data / valid_data from.
  --nj INT                Number of parallel jobs used to generate speech_shape.
  --log-dir PATH          Directory for collect_stats logs. Defaults to a temporary directory under exp/collect_stats.
  --progress-every INT    Write a worker progress update every N utterances.
  --python PATH           Python interpreter to use.
  --force                 Regenerate speech_shape / instruction_fused_shape even if it already exists.
  --keep-logs             Keep collect_stats logs after a successful run. Logs are always kept on failure.
  -h, --help              Show this help message.
EOF
)

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            config="$2"
            shift 2
            ;;
        --nj)
            nj="$2"
            shift 2
            ;;
        --log-dir)
            log_dir="$2"
            shift 2
            ;;
        --progress-every)
            progress_every="$2"
            shift 2
            ;;
        --python)
            python_bin="$2"
            shift 2
            ;;
        --force)
            force=true
            shift
            ;;
        --keep-logs)
            cleanup_logs=false
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

if [ -z "${config}" ]; then
    echo "${help_message}"
    echo "--config is required" >&2
    exit 2
fi

if [ -z "${log_dir}" ]; then
    config_stem="$(basename "${config}")"
    config_stem="${config_stem%.*}"
    mkdir -p "exp/collect_stats"
    log_dir="$(mktemp -d "exp/collect_stats/${config_stem}.XXXXXX")"
else
    mkdir -p "${log_dir}"
fi

success=false
cleanup() {
    if ${success}; then
        if ${cleanup_logs}; then
            log "collect_stats finished successfully; removing logs under ${log_dir}"
            rm -rf "${log_dir}"
        else
            log "collect_stats finished successfully; logs kept under ${log_dir}"
            log "Main log: ${log_dir}/collect_stats.log"
        fi
    else
        log "collect_stats did not finish successfully; logs kept under ${log_dir}"
    fi
}
trap cleanup EXIT

force_suffix=""
if ${force}; then
    force_suffix=" --force"
fi

log "$0 --config ${config} --nj ${nj} --log-dir ${log_dir}${force_suffix}"
log "Logs will be written under ${log_dir}"

cmd=(
    "${python_bin}" -m train.shape_files
    --config "${config}"
    --nj "${nj}"
    --log_dir "${log_dir}"
    --progress_every "${progress_every}"
)

if ${force}; then
    cmd+=(--force)
fi

"${cmd[@]}" 2>&1 | tee "${log_dir}/collect_stats.log"

success=true
log "Finished generating shape files"
