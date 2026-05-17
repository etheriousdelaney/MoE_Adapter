python build_messages.py --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/tr05_all_noisy
python build_metadata_json.py \
  --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/tr05_all_noisy \
  --annotations /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/tr05_real.json \
                /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/tr05_simu.json \
  --nj 32

python build_messages.py --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/dt05_multi_isolated_1ch_track
python build_metadata_json.py \
  --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/dt05_multi_isolated_1ch_track \
  --annotations /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/dt05_real.json \
                /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/dt05_simu.json \
  --nj 32

python build_messages.py --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/dt05_real_isolated_1ch_track
python build_metadata_json.py \
  --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/dt05_real_isolated_1ch_track \
  --annotations /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/dt05_real.json \
  --nj 32

python build_messages.py --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/dt05_simu_isolated_1ch_track
python build_metadata_json.py \
  --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/dt05_simu_isolated_1ch_track \
  --annotations /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/dt05_simu.json \
  --nj 32

python build_messages.py --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/et05_real_isolated_1ch_track
python build_metadata_json.py \
  --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/et05_real_isolated_1ch_track \
  --annotations /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/et05_real.json \
  --nj 32

python build_messages.py --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/et05_simu_isolated_1ch_track
python build_metadata_json.py \
  --data-dir /mnt/disk2/m11315045/MoE_Adapter/data/chime4/et05_simu_isolated_1ch_track \
  --annotations /mnt/disk2/ASR_corpus/CHiME3/CHiME3/data/annotations/et05_simu.json \
  --nj 32