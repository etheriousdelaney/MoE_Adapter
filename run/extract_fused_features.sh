#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

config="${TRAIN_CONFIG:-conf/asr.yaml}"
nj="${NJ:-1}"
python_bin="${PYTHON:-./.venv/bin/python}"
force=false
progress_every="${PROGRESS_EVERY:-100}"
log_dir=""
output_root="${OUTPUT_ROOT:-dump/fused}"
splits=(train valid)

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--config conf/asr.yaml] [--nj 1] [--log-dir exp/extract_fused/logdir] [--output-root dump/fused] [--splits train valid] [--force]

Options:
  --config PATH           YAML config to read dataset_conf.train_data / valid_data from.
  --nj INT                Number of extraction worker jobs.
  --log-dir PATH          Directory for main log and per-job worker logs.
  --output-root PATH      Root directory for extracted fused feature .pt files.
  --splits LIST           Space-separated split names from: train valid.
  --progress-every INT    Worker progress update interval. The terminal also shows total progress and ETA.
  --python PATH           Python interpreter to use.
  --force                 Regenerate fused features even if outputs already exist.
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
        --output-root)
            output_root="$2"
            shift 2
            ;;
        --splits)
            shift
            splits=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                splits+=("$1")
                shift
            done
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
    log_dir="exp/extract_fused/${config_stem}"
fi

mkdir -p "${log_dir}"

force_suffix=""
if ${force}; then
    force_suffix=" --force"
fi

log "$0 --config ${config} --nj ${nj} --log-dir ${log_dir} --output-root ${output_root}${force_suffix}"
log "Logs will be written under ${log_dir}"

cmd=(
    "${python_bin}" -m train.extract_fused_features
    --config "${config}"
    --nj "${nj}"
    --log_dir "${log_dir}"
    --output_root "${output_root}"
    --progress_every "${progress_every}"
    --splits "${splits[@]}"
)

if ${force}; then
    cmd+=(--force)
fi

"${cmd[@]}" 2>&1 | tee "${log_dir}/extract_fused_features.log"

log "Finished extracting fused features"
log "Main log: ${log_dir}/extract_fused_features.log"
