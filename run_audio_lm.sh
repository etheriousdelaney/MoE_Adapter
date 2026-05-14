#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=1

tag="asr_qwen_8_expert"
model="valid.lm_loss.ave_5best.pth"
wav="/mnt/disk2/m11315045/MoE_Adapter/IC0001W0001.wav"
prompt="Transcribe the following speech"
max_new_tokens=256
device="cuda"
python_bin="${PYTHON:-./.venv/bin/python}"

if [ ! -x "${python_bin}" ]; then
    python_bin=python3
fi

"${python_bin}" -m inference.audio_lm_demo \
  --tag "${tag}" \
  --model "${model}" \
  --wav "${wav}" \
  --prompt "${prompt}" \
  --max_new_tokens "${max_new_tokens}" \
  --device "${device}"
