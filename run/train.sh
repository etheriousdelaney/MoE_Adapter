#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export WANDB_MODE=offline

tag="${TAG:-asr_test}"
output_dir="${OUTPUT_DIR:-exp/${tag}}"
train_config="${TRAIN_CONFIG:-conf/asr.yaml}"
python_bin="${PYTHON:-./.venv/bin/python}"
init_model="${INIT_MODEL:-}"
job_runner="${JOB_RUNNER:-bin/run.py}"
write_log="${WRITE_LOG:-true}"
trainer_extra_args=()
if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--tag TAG] [--output_dir PATH] [--config PATH] [--init_model PATH] [--python PATH] [-- ...extra trainer args]

Options:
  --tag TAG              Experiment tag. Default: \${TAG:-asr_test}
  --output_dir PATH      Experiment output directory. Default: \${OUTPUT_DIR:-exp/<tag>}
  --config PATH          Training config YAML. Default: \${TRAIN_CONFIG:-conf/asr.yaml}
  --init_model PATH      Initialize model weights from a .ckpt or .pth without resuming optimizer/callback state.
  --python PATH          Python interpreter to use. Default: \${PYTHON:-./.venv/bin/python}
  --job_runner PATH      Job runner script. Default: \${JOB_RUNNER:-bin/run.py}
  -h, --help             Show this help message.

Any unknown arguments after -- are forwarded to train.trainer.
EOF
)

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)
            tag="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --config)
            train_config="$2"
            shift 2
            ;;
        --log)
            write_log="$2"
            shift 2
            ;;
        --init_model)
            init_model="$2"
            shift 2
            ;;
        --python)
            python_bin="$2"
            shift 2
            ;;
        --job_runner)
            job_runner="$2"
            shift 2
            ;;
        --)
            shift
            trainer_extra_args+=("$@")
            break
            ;;
        -h|--help)
            echo "${help_message}"
            exit 0
            ;;
        *)
            trainer_extra_args+=("$1")
            shift
            ;;
    esac
done

if [ "${output_dir}" = "exp/${TAG:-asr_test}" ] || [ "${output_dir}" = "exp/asr_test" ]; then
    output_dir="exp/${tag}"
fi

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log "$0 $*"


log "Open the curtains"
sleep 1
log "Lights on"
sleep 0.8
log "Don't miss a moment of this experiment"
sleep 2
log "Oh, the book is strange"
sleep 1
log "Like clockwork orange"
sleep 1
log "Keep your eyes buttered till the end ♪ ~~ ♫ ~~"

log "Start training log in ${output_dir}/train.log"
log "Experiment tag: ${tag}"
log "Training config: ${train_config}"

if [ -n "${init_model}" ]; then
    log "Init model: ${init_model}"
    trainer_extra_args+=(--init_model "${init_model}")
fi

mkdir -p "${output_dir}"

if [ "$write_log" = "true" ]; then
    "${python_bin}" utils/rotate_logfiles.py "${output_dir}/train.log"
    "${python_bin}" "${job_runner}" "${output_dir}"/train.log "${python_bin}" -m train.trainer \
                                    --exp_tag "${tag}" \
                                    --output_dir "${output_dir}" \
                                    --config "${train_config}" \
                                    "${trainer_extra_args[@]}"
else
    "${python_bin}" -m train.trainer \
        --exp_tag "${tag}" \
        --output_dir "${output_dir}" \
        --config "${train_config}"
fi



