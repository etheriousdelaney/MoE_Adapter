#!/usr/bin/env bash
set -e
set -u
set -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

data_dir="${repo_root}/data/chime4/et05_real_isolated_1ch_track"
prompt="${repo_root}/dataset/desta/prompt"
system_prompt=""
decoder="${repo_root}/model/decoder/qwen.py"
model="Qwen/Qwen3-1.7B"
python_bin="${PYTHON:-${repo_root}/.venv/bin/python}"

build_metadata=false
build_message=true
run_response=true

resume=false
limit=0
max_new_tokens=256
device="auto"
torch_dtype="auto"
annotation_root="/mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations"
speaker_gender="${repo_root}/data/chime4/all_speaker/speaker"
metadata_nj=32
metadata_progress_every=100
annotations=()

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

infer_chime4_annotations() {
    local split_name
    split_name="$(basename "${data_dir}")"
    annotations=()
    case "${split_name}" in
        tr05_all_noisy)
            annotations+=("${annotation_root}/tr05_real.json" "${annotation_root}/tr05_simu.json")
            ;;
        dt05_multi_isolated_1ch_track)
            annotations+=("${annotation_root}/dt05_real.json" "${annotation_root}/dt05_simu.json")
            ;;
        dt05_real_isolated_1ch_track)
            annotations+=("${annotation_root}/dt05_real.json")
            ;;
        dt05_simu_isolated_1ch_track)
            annotations+=("${annotation_root}/dt05_simu.json")
            ;;
        et05_real_isolated_1ch_track)
            annotations+=("${annotation_root}/et05_real.json")
            ;;
        et05_simu_isolated_1ch_track)
            annotations+=("${annotation_root}/et05_simu.json")
            ;;
        *)
            echo "Cannot infer CHiME4 annotation files for data_dir: ${data_dir}" >&2
            echo "Use --annotation PATH one or more times, or set --build_metadata false." >&2
            exit 2
            ;;
    esac
}

help_message=$(cat << EOF
Usage: $0 --data_dir PATH [options]

Options:
  --data_dir PATH          Data directory containing wav.scp/text/metadata.json.
  --build_metadata BOOL    Build <data_dir>/metadata.json before message JSONL. Default: ${build_metadata}
  --annotation PATH        Annotation JSON path. May be repeated. If omitted, CHiME4 split names are inferred.
  --annotation_root PATH   CHiME3 annotation root used for inference. Default: ${annotation_root}
  --speaker_gender PATH    Speaker gender file for metadata. Default: ${speaker_gender}
  --metadata_nj N          Number of workers for metadata duration reading. Default: ${metadata_nj}
  --metadata_progress_every N
                           Metadata duration progress frequency. Default: ${metadata_progress_every}
  --prompt PATH            Prompt JSON list. Default: ${prompt}
  --system_prompt TEXT     Override build_messages.py default system prompt.
  --decoder PATH           Decoder Python file. Default: ${decoder}
  --model NAME_OR_PATH     Model repo/path. Default: ${model}
  --build_message BOOL     Build message JSONL. Default: ${build_message}
  --run_response BOOL      Run LLM responses. Default: ${run_response}
  --resume BOOL            Resume existing response JSONL. Default: ${resume}
  --limit N                Limit response generation to first N messages. Default: ${limit}
  --max_new_tokens N       Max generated tokens. Default: ${max_new_tokens}
  --device DEVICE          auto/cpu/cuda/cuda:0. Default: ${device}
  --torch_dtype DTYPE      auto/float32/float16/bfloat16. Default: ${torch_dtype}
  -h, --help               Show this help message.
EOF
)

while [ $# -gt 0 ]; do
    case "$1" in
        --data_dir) data_dir="$2"; shift 2 ;;
        --build_metadata) build_metadata="$(normalize_bool "$2")"; shift 2 ;;
        --annotation) annotations+=("$2"); shift 2 ;;
        --annotation_root) annotation_root="$2"; shift 2 ;;
        --speaker_gender) speaker_gender="$2"; shift 2 ;;
        --metadata_nj) metadata_nj="$2"; shift 2 ;;
        --metadata_progress_every) metadata_progress_every="$2"; shift 2 ;;
        --prompt) prompt="$2"; shift 2 ;;
        --system_prompt) system_prompt="$2"; shift 2 ;;
        --decoder) decoder="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --build_message) build_message="$(normalize_bool "$2")"; shift 2 ;;
        --run_response) run_response="$(normalize_bool "$2")"; shift 2 ;;
        --resume) resume="$(normalize_bool "$2")"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --torch_dtype) torch_dtype="$2"; shift 2 ;;
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

metadata_path="${data_dir}/metadata.json"
message_output="${data_dir}/message.jsonl"
response_output="${data_dir}/response.jsonl"
full_response_output="${data_dir}/response_full.jsonl"

log "Data dir: ${data_dir}"
log "Metadata output: ${metadata_path}"
log "Message output: ${message_output}"
log "Response output: ${response_output}"
log "Full response output: ${full_response_output}"

if ${build_metadata}; then
    if [ "${#annotations[@]}" -eq 0 ]; then
        infer_chime4_annotations
    fi
    log "Building metadata JSON"
    metadata_args=(
        "${repo_root}/dataset/desta/build_metadata_json.py"
        --data-dir "${data_dir}"
        --output "${metadata_path}"
        --speaker-gender "${speaker_gender}"
        --nj "${metadata_nj}"
        --progress-every "${metadata_progress_every}"
        --annotations
    )
    metadata_args+=("${annotations[@]}")
    "${python_bin}" "${metadata_args[@]}"
fi

if ${build_message}; then
    log "Building message JSONL"
    build_args=(
        "${repo_root}/dataset/desta/build_messages.py"
        --data-dir "${data_dir}"
        --metadata "${metadata_path}"
        --prompt "${prompt}"
        --output "${message_output}"
    )
    if [ -n "${system_prompt}" ]; then
        build_args+=(--system-prompt "${system_prompt}")
    fi
    "${python_bin}" "${build_args[@]}"
fi

if ${run_response}; then
    log "Generating responses"
    response_args=(
        "${repo_root}/dataset/desta/run_responses.py"
        --messages "${message_output}"
        --decoder "${decoder}"
        --model "${model}"
        --output "${response_output}"
        --full-output "${full_response_output}"
        --metadata "${metadata_path}"
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
    "${python_bin}" "${response_args[@]}"
fi
