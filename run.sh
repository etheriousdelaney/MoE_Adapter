#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export PATH=/mnt/disk2/m11315045/espnet/tools/sctk/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

tag="asr_qwen_moe_multitask_0525"
train_config="conf/MoE_all_task.yaml"

decode_config="conf/decode_asr_qwen_greedy.yaml"
decode_model="valid.lm_loss.ave_3best.pth"

datasets=(
  chime4/dt05_real_isolated_1ch_track
  chime4/dt05_simu_isolated_1ch_track
  chime4/et05_real_isolated_1ch_track
  chime4/et05_simu_isolated_1ch_track
)

train_args=(
  --tag "${tag}"
  --config "${train_config}"
  --log true
  # --init_model "${init_model}"
)

# init_model="/mnt/disk2/m11315045/MoE_Adapter/exp/asr_qwen_fused_frozen_adapter_instruction/checkpoint/last.ckpt"

# ./run/collect_stats.sh --config "${train_config}" --force --nj 32

./run/train.sh "${train_args[@]}"

inference_runner="./run/instruction_inference.sh"

"${inference_runner}" \
  --tag "${tag}" \
  --model "${decode_model}" \
  --decode_config "${decode_config}" \
  --datasets "${datasets[@]}" \
  --inference_nj 1 \
  --decode true \
  --scoring true
