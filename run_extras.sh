#!/bin/bash
# run_extras.sh — nohup bash run_extras.sh > extras.log 2>&1 &
BASE="--store_dir ensemble_ckpt_p09 --E 20 --trials_per_soa 50 \
      --soa_start 1 --soa_end 20 --soa_step 2 --ITI 1.8 --workers 6 --plot"

for p in 0.50 0.80; do                                   # strategic extremes (~35 min each)
  python -m scripts.run_prp_sweep $BASE --persistence $p --optimize_onset
done
for p in 0.00 0.50 0.65 0.80; do                         # greedy line padding (~4 min each)
  python -m scripts.run_prp_sweep $BASE --persistence $p
done