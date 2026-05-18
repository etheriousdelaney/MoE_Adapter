#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=0


data_dir="/mnt/disk2/m11315045/MoE_Adapter/data/chime4/et05_real_isolated_1ch_track"
prompt="/mnt/disk2/m11315045/MoE_Adapter/data/desta/prompt"
system_prompt=""
decoder="/mnt/disk2/m11315045/MoE_Adapter/model/decoder/qwen_ntp.py"
model="Qwen/Qwen3-1.7B"
message_output=""
response_output=""
full_response_output=""
build_message=true
run_response=true
resume=false
limit=0
python_bin="${PYTHON:-./.venv/bin/python}"
batch_size=10000000000
max_new_tokens=256
device="auto"
torch_dtype="auto"
response_extra_args=()

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

normalize_bool() {
    local value
    value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "${value}" in
        true|1|yes|y) echo "true" ;;
        false|0|no|n) echo "false" ;;
        *)
            echo "Invalid boolean value: $1" >&2
            exit 2
            ;;
    esac
}

help_message=$(cat << EOF
Usage: $0 --data_dir PATH [options]

Options:
  --data_dir PATH          Data directory containing metadata.json.
  --prompt PATH            Prompt JSON list. Default: ${prompt}
  --system_prompt TEXT     Override build_messages.py default system prompt.
  --decoder PATH           Decoder Python file. Default: ${decoder}
  --model NAME_OR_PATH     Model repo/path. Default: ${model}
  --message_output PATH    Message JSONL path. Default: <data_dir>/message.jsonl
  --response_output PATH   Response JSONL path. Default: <data_dir>/response.jsonl
  --full_response_output PATH
                           Full response JSONL path. Default: <data_dir>/response_full.jsonl
  --build_message BOOL     Build message JSONL. Default: ${build_message}
  --run_response BOOL      Run LLM responses. Default: ${run_response}
  --resume BOOL            Resume existing response JSONL. Default: ${resume}
  --limit N                Limit response generation to first N messages. Default: ${limit}
  --python PATH            Python interpreter. Default: \${PYTHON:-./.venv/bin/python}
  --batch_size N           Reserved compatibility option. Default: ${batch_size}
  --max_new_tokens N       Max generated tokens. Default: ${max_new_tokens}
  --device DEVICE          auto/cpu/cuda/cuda:0. Default: ${device}
  --torch_dtype DTYPE      auto/float32/float16/bfloat16. Default: ${torch_dtype}
  --                       Forward remaining args to run_responses.py.
  -h, --help               Show this help message.
EOF
)

while [ $# -gt 0 ]; do
    case "$1" in
        --data_dir) data_dir="$2"; shift 2 ;;
        --prompt) prompt="$2"; shift 2 ;;
        --system_prompt) system_prompt="$2"; shift 2 ;;
        --decoder) decoder="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --message_output) message_output="$2"; shift 2 ;;
        --response_output) response_output="$2"; shift 2 ;;
        --full_response_output) full_response_output="$2"; shift 2 ;;
        --build_message) build_message="$(normalize_bool "$2")"; shift 2 ;;
        --run_response) run_response="$(normalize_bool "$2")"; shift 2 ;;
        --resume) resume="$(normalize_bool "$2")"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --torch_dtype) torch_dtype="$2"; shift 2 ;;
        --)
            shift
            response_extra_args+=("$@")
            break
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

if [ -z "${data_dir}" ]; then
    echo "--data_dir is required" >&2
    exit 2
fi
if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi
if [ -z "${message_output}" ]; then
    message_output="${data_dir}/message.jsonl"
fi
if [ -z "${response_output}" ]; then
    response_output="${data_dir}/response.jsonl"
fi
if [ -z "${full_response_output}" ]; then
    full_response_output="${data_dir}/response_full.jsonl"
fi

log "Data dir: ${data_dir}"
log "Message output: ${message_output}"
log "Response output: ${response_output}"
log "Full response output: ${full_response_output}"

# if ${build_message}; then
#     log "Building message JSONL"
#     build_args=(
#         data/desta/build_messages.py
#         --data-dir "${data_dir}"
#         --prompt "${prompt}"
#         --output "${message_output}"
#     )
#     if [ -n "${system_prompt}" ]; then
#         build_args+=(--system-prompt "${system_prompt}")
#     fi
#     "${python_bin}" "${build_args[@]}"
# fi

if ${run_response}; then
    log "Generating responses"
    response_args=(
        data/desta/run_responses.py
        --messages "${message_output}"
        --decoder "${decoder}"
        --model "${model}"
        --output "${response_output}"
        --full-output "${full_response_output}"
        --metadata "${data_dir}/metadata.json"
        --batch-size "${batch_size}"
        --max-new-tokens "${max_new_tokens}"
        --device "${device}"
        --torch-dtype "${torch_dtype}"
    )
    if ${resume}; then
        response_args+=(--resume)
    fi
    if [ "${limit}" -gt 0 ]; then
        response_args+=(--limit "${limit}")
    fi
    response_args+=("${response_extra_args[@]}")
    "${python_bin}" "${response_args[@]}"
fi
