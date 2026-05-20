#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export HF_HUB_OFFLINE=1

tag=""
model=""
decode_config="conf/decode_asr_qwen_greedy.yaml"
datasets=()
decode=true
scoring=true
ngpu=1
nj=8
inference_nj=8
score_opts=""
python_bin="${PYTHON:-./.venv/bin/python}"
expert_heatmap="auto"

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 --tag TAG --model MODEL --decode_config conf/decode_asr_qwen_greedy.yaml --datasets dataset1 [dataset2 ...]

Options:
  --tag TAG               Experiment tag under exp/<tag>.
  --model MODEL           Model filename under exp/<tag>/checkpoint or exp/<tag>.
  --decode_config PATH    Decode yaml.
  --datasets LIST         One or more dataset names under data/.
  --decode BOOL           Whether to run decoding. Default: ${decode}
  --scoring BOOL          Whether to run scoring. Default: ${scoring}
  --ngpu INT              Number of gpus passed to inference workers.
  --nj INT                Reserved compatibility option.
  --inference_nj INT      Number of split decode jobs per dataset.
  --python PATH           Python interpreter.
  --score_opts STRING     Extra options forwarded to sclite.
  --expert_heatmap MODE   Expert heatmap policy: auto, true, or false. Default: ${expert_heatmap}
  -h, --help              Show this help message.
EOF
)

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

normalize_heatmap_policy() {
    local value
    value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "${value}" in
        auto|true|false) echo "${value}" ;;
        1|yes|y) echo "true" ;;
        0|no|n) echo "false" ;;
        *)
            echo "Invalid expert_heatmap value: $1" >&2
            exit 2
            ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) tag="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --decode_config) decode_config="$2"; shift 2 ;;
        --datasets)
            shift
            datasets=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --decode) decode="$(normalize_bool "$2")"; shift 2 ;;
        --scoring) scoring="$(normalize_bool "$2")"; shift 2 ;;
        --ngpu) ngpu="$2"; shift 2 ;;
        --nj) nj="$2"; shift 2 ;;
        --inference_nj) inference_nj="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --score_opts) score_opts="$2"; shift 2 ;;
        --expert_heatmap) expert_heatmap="$(normalize_heatmap_policy "$2")"; shift 2 ;;
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

if [ -z "${tag}" ] || [ -z "${model}" ] || [ ${#datasets[@]} -eq 0 ]; then
    echo "${help_message}" >&2
    exit 2
fi
if [ ! -f "${decode_config}" ]; then
    echo "decode config not found: ${decode_config}" >&2
    exit 1
fi

exp_dir="exp/${tag}"
train_config="${exp_dir}/config.yaml"
if [ ! -f "${train_config}" ]; then
    echo "train config not found: ${train_config}" >&2
    exit 1
fi

model_path="${exp_dir}/checkpoint/${model}"
if [ ! -f "${model_path}" ]; then
    model_path="${exp_dir}/${model}"
fi
if ${decode} && [ ! -f "${model_path}" ]; then
    echo "model file not found under ${exp_dir}: ${model}" >&2
    exit 1
fi

decode_tag="$(basename "${decode_config}")"
decode_tag="${decode_tag%.*}"
model_tag="$(basename "${model}")"
model_tag="${model_tag%.ckpt}"
model_tag="${model_tag%.pth}"
inference_tag="instruction_decode_${decode_tag}_${model_tag}"
decode_root="${exp_dir}/${inference_tag}"

if ${scoring}; then
    if ! command -v sclite >/dev/null 2>&1; then
        echo "sclite is required for scoring but was not found in PATH" >&2
        exit 1
    fi
fi

mkdir -p "${decode_root}"
log "Generate '${decode_root}/run.sh'. You can reuse this script for decode/score"
echo "$0 --tag ${tag} --model ${model} --decode_config ${decode_config} --datasets ${datasets[*]} --decode ${decode} --scoring ${scoring} --expert_heatmap ${expert_heatmap} \"\$@\"; exit \$?" > "${decode_root}/run.sh"
chmod +x "${decode_root}/run.sh"

if ${decode}; then
    log "Decoding"
    for dataset in "${datasets[@]}"; do
        data_dir="data/${dataset}"
        if [ ! -f "${data_dir}/response.jsonl" ]; then
            echo "response.jsonl not found for dataset: ${dataset}" >&2
            exit 1
        fi

        dataset_dir="${decode_root}/${dataset}"
        logdir="${dataset_dir}/logdir"
        mkdir -p "${logdir}"
        keys_file="${logdir}/keys.scp"
        "${python_bin}" -m inference.write_instruction_keys \
            --response_jsonl "${data_dir}/response.jsonl" \
            --output "${keys_file}"
        rm -f "${logdir}"/keys.*.scp "${logdir}"/instruction_inference.*.log
        split -n "l/${inference_nj}" -d -a 3 "${keys_file}" "${logdir}/keys."
        job_count=0
        for split_file in "${logdir}"/keys.*; do
            [ -f "${split_file}" ] || continue
            if [[ "${split_file}" == *.scp ]]; then
                continue
            fi
            job_count=$((job_count + 1))
            mv "${split_file}" "${split_file}.scp"
        done

        if [ "${job_count}" -eq 0 ]; then
            echo "No key splits were created for dataset ${dataset}" >&2
            exit 1
        fi

        log "Instruction decoding started... log: '${logdir}/instruction_inference.*.log'"
        pids=()
        for split_file in "${logdir}"/keys.*.scp; do
            job_id="$(basename "${split_file}" | sed 's/keys\.//; s/\.scp//')"
            output_dir="${logdir}/output.${job_id}"
            log_path="${logdir}/instruction_inference.${job_id}.log"
            "${python_bin}" -m inference.instruction_inference \
                --train_config "${train_config}" \
                --decode_config "${decode_config}" \
                --dataset "${dataset}" \
                --model_file "${model_path}" \
                --output_dir "${output_dir}" \
                --key_file "${split_file}" \
                --ngpu "${ngpu}" \
                --expert_heatmap "${expert_heatmap}" \
                > "${log_path}" 2>&1 &
            pids+=("$!")
        done

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                failed=1
            fi
        done
        if [ "${failed}" -ne 0 ]; then
            grep -l -i error "${logdir}"/instruction_inference.*.log 2>/dev/null | while read -r err_log; do
                cat "${err_log}"
            done
            exit 1
        fi

        for name in text prompt token token_int score; do
            : > "${dataset_dir}/${name}"
            for output_dir in "${logdir}"/output.*; do
                cat "${output_dir}/1best_recog/${name}" >> "${dataset_dir}/${name}"
            done
            sort -k1 "${dataset_dir}/${name}" -o "${dataset_dir}/${name}"
        done

        stats_files=()
        for output_dir in "${logdir}"/output.*; do
            if [ -f "${output_dir}/expert_heatmap_stats.pt" ]; then
                stats_files+=("${output_dir}/expert_heatmap_stats.pt")
            fi
        done
        if [ "${expert_heatmap}" != "false" ] && [ ${#stats_files[@]} -gt 0 ]; then
            "${python_bin}" -m inference.merge_expert_heatmap \
                --stats "${stats_files[@]}" \
                --output "${dataset_dir}/expert_heatmap.png" \
                --title "Instruction Inference Expert Usage: ${dataset}"
            log "Write expert heatmap in ${dataset_dir}/expert_heatmap.png"
        fi
    done
fi

if ${scoring}; then
    log "Scoring"
    for dataset in "${datasets[@]}"; do
        dataset_dir="${decode_root}/${dataset}"
        "${python_bin}" -m inference.instruction_score \
            --dataset "${dataset}" \
            --decode_dir "${dataset_dir}" \
            --score_opts "${score_opts}"
        log "Write CER result in ${dataset_dir}/score_cer/result.txt"
        grep -e Avg -m 1 "${dataset_dir}/score_cer/result.txt" || true
        log "Write WER result in ${dataset_dir}/score_wer/result.txt"
        grep -e Avg -m 1 "${dataset_dir}/score_wer/result.txt" || true
    done

    "${python_bin}" -m inference.show_asr_result "${decode_root}" > "${decode_root}/RESULTS.md"
    log "Write summary result in ${decode_root}/RESULTS.md"
    cat "${decode_root}/RESULTS.md"
fi
