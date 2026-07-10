#!/bin/bash
# run_finals.sh — launch: nohup bash run_finals.sh > finals.log 2>&1 &
BASE="--store_dir ensemble_ckpt_p09 --E 20 --trials_per_soa 50 \
      --soa_start 1 --soa_end 20 --soa_step 2 --ITI 1.8 --workers 6 --plot"

for p in 0.60 0.65 0.70; do                     # M1–M4: main (policy on)
  python -m scripts.run_prp_sweep $BASE --persistence $p --optimize_onset
done
for p in 0.60 0.70; do                               # S1–S2: greedy secondary
  python -m scripts.run_prp_sweep $BASE --persistence $p
done
for iti in 0.5 4.0; do                               # R1–R2: ITI robustness
  python -m scripts.run_prp_sweep --store_dir ensemble_ckpt_p09 --E 20 \
    --trials_per_soa 50 --soa_start 1 --soa_end 20 --soa_step 2 \
    --ITI $iti --workers 6 --plot --persistence 0.65 --optimize_onset
done