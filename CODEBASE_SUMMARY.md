# PRP-Rational-Boundedness: Codebase Summary

Detailed module documentation for collaborators and future development.
For setup and quick start, see `README.md`.

---

## 1. What This Project Does

This codebase implements a **connectionist model of the Psychological Refractory Period (PRP)** effect, replicating and extending **Simulation Study 3** from Musslick et al. (2023, preprint). A shared-representation neural network + persistence mechanism + LCA decision process + reward-rate optimization reproduces PRP curves *without* any structural bottleneck — consistent with the Rational Boundedness Account (RBA).

**Pipeline stages:**
1. **Train** a feedforward task network on single-task mappings (Tasks A–E)
2. **Optimize LCA thresholds** per condition via reward-rate maximization in dual-task context
3. **Simulate PRP trials** across SOA values (with optional onset-policy optimization)
4. **Plot** RT2 vs SOA curves, error rates, onset delays, and the money figure (head slope × persistence)

---

## 2. Directory Layout

```
prp-rational-boundedness/
├── prp_model/                  # Core library
│   ├── task_network.py         # TaskNetwork (PyTorch nn.Module)
│   ├── nn_wrapper.py           # TaskNetworkWrapper (training + temporal integration)
│   ├── lca.py                  # Leaky Competing Accumulator (dt=0.05, tau=0.5)
│   ├── prp_simulator.py        # PRP trial simulation + SOA sweep
│   ├── threshold_utils.py      # LCA threshold optimization (dual-context, session-level)
│   ├── training_set.py         # Stimulus/target generation (MATLAB-faithful)
│   └── utils.py                # Shared helpers, constants, I/O
├── scripts/                    # CLI entry points
│   ├── run_prp_sweep.py        # Full ensemble pipeline: train/load → threshold → sweep → save
│   ├── plot_prp_sweep.py       # RT1/RT2/error-rate/onset figures
│   ├── plot_money_figure.py    # Head slope vs persistence (central thesis figure)
│   └── make_results_table.py   # Markdown + LaTeX summary table
├── ensemble_ckpt_p09/          # Trained weights + threshold caches (gitignored)
├── output/                     # Results JSONs + plots (gitignored)
├── run_simulations.sh          # Batch script for all thesis configurations
├── CODEBASE_SUMMARY.md         # This file
├── README.md                   # Setup and usage
└── requirements.txt
```

---

## 3. Core Modules

### 3.1 `task_network.py` — TaskNetwork

A 3-layer feedforward network with task-control inputs (PyTorch `nn.Module`).

**Architecture** (Musslick et al., 2023, Fig. 9):
```
stim_input (9-dim) ──► hidden (100 units) ──► output (9-dim)
                    ▲                       ▲
       task_input ──┘                       └── task_input
```

- **4 weight matrices** (all `nn.Linear`, no learnable biases):
  - `fc_input_hidden`: stim → hidden (9 × 100)
  - `fc_task_hidden`: task → hidden (9 × 100)
  - `fc_hidden_output`: hidden → output (100 × 9)
  - `fc_task_output`: task → output (9 × 9)
- **Fixed bias offset**: −2.0 added to pre-activations at both layers
- **Activation**: sigmoid
- **Init scale**: 0.1 (uniform random)

### 3.2 `nn_wrapper.py` — TaskNetworkWrapper

Wraps TaskNetwork for training and temporal integration.

- `train_online(...)` — SGD + MSE, single-sample online learning, stops at loss ≤ 1e-3
- `predict(stim_input, task_input)` — single forward pass
- `integrate(stim_sequence, task_sequence, persistence=0.0)` — runs the network over T timesteps with temporal persistence (EMA):

```python
net_h[t] = (1 - p) * net_h_fresh[t] + p * net_h[t-1]
net_o[t] = (1 - p) * net_o_fresh[t] + p * net_o[t-1]
```

### 3.3 `lca.py` — Leaky Competing Accumulator

Implements the LCA decision process (Usher & McClelland, 2001; Musslick et al., 2023, Eq. 4):

```
dr_i = [y_o - λ·r_i + α·f(r_i) - β·Σ_{j≠i} f(r_j)] · (dt/τ) + ξ_i · √(dt/τ)
```

**Parameters:** λ=0.4, α=0.2, β=0.2, σ=0.2, t0=0.15 s, dt=0.05 s, τ=0.5 s (dt/τ=0.1; each step = 50 ms). All returned RTs are in **physical seconds**: `rt = (t+1) * dt + t0`.

**Functions:**
- `run_lca(...)` → `(rt, choice, trajectory)` — single stochastic trajectory
- `run_lca_avg(...)` → dict with `rt`, `rt_correct`, `p_correct`, `frac_decided`
- `run_lca_dist(...)` → per-threshold Acc/RT/Reward-Rate arrays (for z-optimization)

### 3.4 `prp_simulator.py` — PRP Trial Simulation

**`run_prp_trial(...)`** — two-pass dual-task trial:

1. **Pass 1** (both tasks active): stim1 from t=0, stim2 added at SOA, cue2 at onset. Measures Task 1 RT via LCA.
2. **Pass 2** (Task 1 gated off): cue1 zeroed after Task 1's decision time. Measures Task 2 RT on the tail from cue2 onset.

When `optimize_onset=True`, Task 2 cue may be deferred beyond SOA to maximize joint reward rate (strategic deferment).

**`sweep_soa(...)`** — runs trials across SOAs; `rt_task2_from_stim_correct` is the reported dependent variable.

### 3.5 `threshold_utils.py` — Threshold Optimization

`compute_condition_thresholds(...)` — dual-context session-level selection:
- Simulates dual-task trials at three reference SOAs (150, 400, 800 ms)
- Pools reward rate across SOAs and stimuli
- Selects z via constrained argmax (Task 1 floor configurable; Task 2 unconstrained)
- z2 shared across conditions (max over B→A and C→A)

Cache filenames encode all configuration parameters to prevent stale-cache collisions.

### 3.6 `training_set.py` — Training Data Generation

27 stimuli per task (3³ exhaustive combinations). Tasks A–E defined by input→output pathway mappings:

```python
TASK_MAP = {"A": (0,0), "B": (1,1), "C": (2,2), "D": (0,1), "E": (1,0)}
```

Tasks D and E create the representational overlap between A and B that produces the PRP effect.

### 3.7 `utils.py` — Shared Utilities

- `generate_trial_pair(prp_pair, seed)` — PRP trial stimuli and cues
- `steepest_adjacent_slope(soa_steps, rt_values)` — steepest pairwise slope (s/s)
- `steps_to_ms(steps)` — steps × dt × 1000
- `sim_seconds_to_ms(seconds)` — seconds × 1000
- Checkpoint I/O: `save_state`, `load_state`, `save_threshold`, `load_threshold`

---

## 4. Key Concepts

### Time Units
- Each simulation step = dt = 0.05 s = 50 ms
- All RTs from the LCA are in **seconds**
- SOA values in CLI are in **steps** (e.g., SOA=3 = 150 ms)
- Non-decision time t0 = 0.15 s (150 ms)

### Threshold z
- Per-condition (z1, z2) from dual-context session-level selection
- z2 shared across B→A and C→A (prevents criterion confounding)
- Selection: reward-rate argmax over threshold grid [0.1, 1.5]
- Typical z ≈ 0.30–0.80, tracks persistence

### Evaluation Criteria (from thesis Ch. 2)
- **Head slope**: slope between the two shortest SOAs
- **Tail slope**: slope between the two longest SOAs
- **SOA\***: 0.80 × RT1 at the longest SOA (head-tail boundary)
