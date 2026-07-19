#!/bin/bash
# run_overnight.sh — launch: nohup bash run_overnight.sh > overnight.log 2>&1 &
#
# Clock-corrected pipeline (dt=0.05, tau=0.5), unconstrained RR argmax (af00).
# Estimated total: ~4 hours with 6 workers on Ryzen 5 7530U.
#
# Conditions:
#   M1–M5 : strategic (onset policy ON), p = {0.50, 0.60, 0.65, 0.70, 0.80}
#   G0–G5 : greedy (onset policy OFF),   p = {0.00, 0.50, 0.60, 0.65, 0.70, 0.80}
#   R1–R2 : ITI robustness at p = 0.65,  ITI = {0.5, 4.0}

set -e  # stop on first error

BASE="--store_dir ensemble_ckpt_p09 --E 20 --trials_per_soa 50 \
      --soa_start 1 --soa_end 20 --soa_step 2 --workers 6 --plot \
      --acc_floor_task1 0.98 --acc_floor_dual 0.0"

echo "============================================"
echo " Overnight run — started $(date)"
echo "============================================"

# ── M1–M5: strategic (main results, money figure) ────────────────────────
echo ""
echo ">>> STRATEGIC RUNS (onset policy ON)"
for p in 0.00 0.50 0.60 0.65 0.70 0.80; do
  echo "--- Strategic p=$p  $(date) ---"
  python -m scripts.run_prp_sweep $BASE --persistence $p --optimize_onset
done

# ── G0–G5: greedy (dynamics-only decomposition) ──────────────────────────
echo ""
echo ">>> GREEDY RUNS (onset policy OFF)"
for p in 0.00 0.50 0.60 0.65 0.70 0.80; do
  echo "--- Greedy p=$p  $(date) ---"
  python -m scripts.run_prp_sweep $BASE --persistence $p
done

# ── R1–R2: ITI robustness at p=0.65 ─────────────────────────────────────
echo ""
echo ">>> ITI ROBUSTNESS (p=0.65, strategic)"
for iti in 0.5 4.0; do
  echo "--- ITI=$iti  $(date) ---"
  python -m scripts.run_prp_sweep $BASE --persistence 0.65 --optimize_onset --ITI $iti
done

echo ""
echo "============================================"
echo " Overnight run — finished $(date)"
echo "============================================"