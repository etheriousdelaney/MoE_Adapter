#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

config="${TRAIN_CONFIG:-conf/MoEClassfier.yaml}"
nj="${NJ:-8}"
python_bin="${PYTHON:-./.venv/bin/python}"
force=false
progress_every="${PROGRESS_EVERY:-500}"
log_dir=""

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--config conf/MoEClassfier.yaml] [--nj 8] [--log-dir exp/collect_stats/logdir] [--force]

Options:
  --config PATH           YAML config to read dataset_conf.train_data / valid_data from.
  --nj INT                Number of parallel jobs used to generate speech_shape.
  --log-dir PATH          Directory for per-job progress logs.
  --progress-every INT    Write a worker progress update every N utterances.
  --python PATH           Python interpreter to use.
  --force                 Regenerate speech_shape even if it already exists.
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

if [ -z "${log_dir}" ]; then
    config_stem="$(basename "${config}")"
    config_stem="${config_stem%.*}"
    log_dir="exp/collect_stats/${config_stem}"
fi

mkdir -p "${log_dir}"

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

log "Finished generating speech_shape files"
log "Main log: ${log_dir}/collect_stats.log"
