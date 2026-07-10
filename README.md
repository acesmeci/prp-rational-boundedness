# A Connectionist Model of the Psychological Refractory Period and Cognitive Control

Repository for the Master's thesis *"A Connectionist Model of the Psychological
Refractory Period and Cognitive Control"* (A. Cesmeci, Osnabrück University,
2026; supervisor: S. Musslick). The project replicates and extends the PRP
simulation (Simulation Study 3) from:

> Musslick, S., Saxe, A., Hoskin, A., Sagiv, Y., Reichman, D., Petri,
> G., & Cohen, J. (2020). On the rational boundedness of cognitive 
> control: Shared versus separated representations.

A feedforward task network (shared vs. separated task representations, task
cues as control signals, temporal persistence of task-set activity) is read
out by a Leaky Competing Accumulator (LCA). Dual-task (PRP) trials contrast a
functionally **dependent** pairing (B→A, shared representations) with an
**independent** pairing (C→A, separated representations) across SOAs, under
two strategic regimes: **greedy** Task-2 engagement (cue on at stimulus onset)
and **strategic** engagement (reward-rate-optimal onset; Eq. 7 of the
preprint).

## Headline result

Task-2 head slopes are lawfully task-dependent (dependent ≈ −0.5 to −0.9,
independent ≈ −0.4 to −0.5 under the strategic regime) and vary with
persistence and strategy, jointly spanning the empirical head-slope range
documented in the thesis' literature evaluation (−0.27 to −1.60) — while only
the strategic regime reproduces the empirically observed flat Task-2 error
profile. See `output/plots/ensemble/thesis/` and
`model_results_table_100726.md`.

## Repository structure

```
prp_model/            Core package
  task_network.py       Feedforward task network (paper Fig. 13 architecture)
  nn_wrapper.py         Training / integration wrapper (persistence EMA)
  training_set.py       MATLAB-style single-task training patterns
  lca.py                LCA dynamics (Eq. 4): run_lca / run_lca_avg / run_lca_dist
  threshold_utils.py    Session-level threshold selection (dual-task SOA-mixture,
                        accuracy-constrained expected reward rate) + onset policy
  prp_simulator.py      Two-pass PRP trial and SOA sweep
  utils.py              Trial generation, checkpoint I/O, aggregation, units
scripts/
  run_prp_sweep.py      Ensemble PRP sweep CLI (training, thresholds, sweeps)
  plot_prp_sweep.py     Thesis figures: RT1+RT2 panels, error rates, onset delay
  plot_money_figure.py  Head slope × persistence × condition × strategy
  make_results_table.py Summary table (markdown + LaTeX) across runs
run_finals.sh           Provenance: exact thesis simulation runs (M/S/R)
run_extras.sh           Provenance: extended persistence range runs
prp_old/                Archived pre-refactor code (do not use)
output/                 Results JSONs, plots, tables (gitignored except tables)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline for one configuration (trains 20 networks if checkpoints
# are missing, computes session-level thresholds, runs both conditions):
python -m scripts.run_prp_sweep --store_dir ensemble_ckpt_p09 --E 20 \
    --train_if_missing --persistence 0.65 --trials_per_soa 50 \
    --soa_start 1 --soa_end 20 --soa_step 2 --ITI 1.8 \
    --optimize_onset --workers 6 --plot

# All thesis runs (main / greedy / robustness):
bash run_finals.sh && bash run_extras.sh

# Figures and summary table from saved JSONs:
python -m scripts.plot_prp_sweep --json "output/results/E20_*_ITI18_*zcD*.json" --context talk
python -m scripts.plot_money_figure --json "output/results/E20_p0[5-8]*_ITI18_*zcD*.json" \
    --empirical_band -1.60 -0.27
python -m scripts.make_results_table --json "output/results/E20_*_ITI18_*zcD*.json"
```

## Key configuration (thesis-final; full rationale in simulation_spec_090726.md)

| Parameter | Value |
|---|---|
| LCA (Eq. 4) | dt/τ = 0.1, λ = 0.4, α = 0.2, β = 0.2, σ = 0.2, t0 = 0.15 |
| Time calibration | 1 step = 50 ms |
| ITI | 1.8 s (Pashler & Johnston, 1989); robust across 0.5–4.0 s |
| Thresholds | per task role, session-level, dual-task SOA-mixture context; accuracy floors 0.99 (T1) / 0.95 (T2); z2 shared across conditions |
| Onset policy | Eq. 7 reward-rate-optimal onset (main); greedy engagement (secondary) |
| Ensemble | 20 networks; deterministic seeding of training, thresholds, stimuli (yoked across conditions), and LCA noise |

## Reproducibility

All reported simulations regenerate from the committed shell scripts and
seeds; result filenames encode the full configuration
(`E{n}_p{persistence}_..._ITI{iti}_s{noise}_zc{context}_af{floor}_fx{z1}_oo{onset}_od{window}`).
Trained checkpoints are not tracked; they regenerate deterministically via
`--train_if_missing` (seed = network index). Development history and the
rationale for every modeling decision are documented in `troubleshoot.md`.