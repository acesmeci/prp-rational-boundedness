# PRP-Rational-Boundedness

Connectionist model of the Psychological Refractory Period (PRP) effect,
based on the Rational Boundedness Account (Musslick et al., 2020; Musslick
& Cohen, 2021). A feedforward neural network learns shared task
representations, and dual-task interference emerges from representational
overlap — without any structural bottleneck.

Developed for the Master's thesis of Ahmet Cesmeci (Osnabrück University,
2026; supervisor: Prof. Dr. Sebastian Musslick).

## Setup

```bash
git clone https://github.com/acesmeci/prp-rational-boundedness.git
cd prp-rational-boundedness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and a working PyTorch installation (CPU is fine;
training 20 networks takes ~5 minutes on a modern laptop).

## Usage

The pipeline has one entry point. Training, threshold
optimization, PRP simulation, and plotting, everything runs through a single script:

```bash
# Train 20 networks, compute thresholds, run PRP sweeps, generate figures
python -m scripts.run_prp_sweep \
    --store_dir ensemble_ckpt \
    --E 20 \
    --persistence 0.65 \
    --optimize_onset \
    --workers 6 \
    --plot
```

Networks are trained once and cached in `--store_dir`; subsequent runs
reuse the trained weights automatically.

### Key options

| Flag | Default | What it does |
|---|---|---|
| `--persistence` | — | Temporal persistence parameter (0.0–1.0) |
| `--optimize_onset` | off | Enable strategic Task 2 deferment (omit for greedy) |
| `--E` | 20 | Number of networks in the ensemble |
| `--ITI` | 1.8 | Inter-trial interval in seconds |
| `--soa_start/end/step` | 1/20/2 | SOA range in simulation steps (1 step = 50 ms) |
| `--acc_floor_task1` | 0.0 | Task 1 accuracy floor for threshold selection |
| `--workers` | 0 | Parallel workers (0 = serial) |
| `--plot` | off | Generate figures after sweep |

### Generating figures from saved results

```bash
# RT and error figures for all completed runs
python -m scripts.plot_prp_sweep \
    --json "output/results/E20_*.json" --context paper

# Head slope vs persistence (money figure)
python -m scripts.plot_money_figure \
    --json "output/results/E20_*_ITI18_*.json" --context paper

# Summary table (markdown + LaTeX)
python -m scripts.make_results_table --json "output/results/E20_*.json"
```

### Running all thesis configurations

```bash
bash run_simulations.sh
```

## Repository structure

```
prp_model/              Core library
  task_network.py         Three-layer feedforward network (PyTorch)
  nn_wrapper.py           Training wrapper + temporal integration (persistence)
  training_set.py         Single-task training patterns
  lca.py                  Leaky Competing Accumulator (decision process)
  threshold_utils.py      Reward-rate threshold optimization + onset policy
  prp_simulator.py        Dual-task PRP trial simulation
  utils.py                Trial generation, I/O, time-unit conversions

scripts/                Pipeline entry points
  run_prp_sweep.py        Full ensemble pipeline (train → threshold → sweep)
  plot_prp_sweep.py       RT, error, and onset-delay figures
  plot_money_figure.py    Head slope × persistence × condition figure
  make_results_table.py   Summary table across runs

ensemble_ckpt_p09/      Trained network weights (not tracked; regeneratable)
output/                 Results and figures (not tracked; regeneratable)
```

## How it works

1. **Training:** 20 networks learn five tasks (A–E) over shared pathways.
   Tasks B and A become functionally dependent through intermediary tasks
   D and E; Tasks C and A remain independent.

2. **Threshold optimization:** For each condition (B→A, C→A), LCA decision
   thresholds are selected to maximize expected reward rate in dual-task
   context, pooled over representative SOAs.

3. **PRP simulation:** Dual-task trials across SOAs (50–950 ms). Task 1's
   residual activity (governed by persistence) interferes with Task 2
   processing at short SOAs, producing the PRP curve.

4. **Strategic vs. greedy:** Under the strategic regime, Task 2 engagement
   is deferred to maximize joint reward rate. Under greedy engagement,
   Task 2 starts immediately at stimulus onset.

## References

- Musslick, S., Saxe, A., Hoskin, A., Sagiv, Y., Reichman, D., Petri, G.,
  & Cohen, J. D. (2020). On the rational boundedness of cognitive control:
  Shared versus separated representations. *Manuscript submitted for
  publication.*
- Musslick, S., & Cohen, J. D. (2021). Rationalizing constraints on the
  capacity for cognitive control. *Trends in Cognitive Sciences, 25*(9),
  757–775.

## Documentation

See `CODEBASE_SUMMARY.md` for detailed module documentation, parameter
conventions, and development history.