"""
ts_simulator.py
---------------
Task-switching simulation for the Rational Boundedness model.

Implements cued task switching following Musslick et al. (2020/2023),
Simulation Study 3 (pp. 70-74). The network performs a current task
after having just performed a previous task, and the switch cost is
the RT difference between switch trials (prev ≠ curr) and repeat
trials (prev == curr).

Trial structure (matching MATLAB switchTasks):
  Phase 1 (1 step):  Previous task cue + stimulus (blank by default).
                      Produces a task-specific hidden-layer state.
  Phase 2 (N steps): Current task cue + bivalent stimulus.
                      Persistence carries the phase-1 state forward.
                      LCA reads the output series from this phase.

Key differences from PRP trials:
  - Bivalent stimuli (all input dimensions active)
  - No SOA manipulation; sequential, not overlapping
  - Single LCA readout (current task only)
  - Congruency as a within-trial factor
  - No onset policy or cue gating

Convention note (MATLAB ↔ Python persistence):
  MATLAB's tau = 0.15 means new = 0.15*fresh + 0.85*prev
  Python's  p  = 0.85 means new = 0.15*fresh + 0.85*prev
  So tau_matlab = 1 - p_python.
"""

import numpy as np
import torch

from prp_model.lca import run_lca_avg, run_lca_dist, _DEFAULTS
from prp_model.threshold_utils import optimize_lca_threshold_dist
from prp_model.utils import (
    TASK_MAP, N_PATHWAYS, N_FEATURES,
    generate_switch_trial,
)

DEFAULT_N_REPEATS = 100


def run_switch_trial(
    wrapper,
    stim_prev: np.ndarray,
    stim_curr: np.ndarray,
    cue_prev: np.ndarray,
    cue_curr: np.ndarray,
    correct_idx: int,
    resp_indices: list[int],
    persistence: float = 0.85,
    n_phase2_steps: int = 50,
    n_repeats: int = DEFAULT_N_REPEATS,
    thresholds: np.ndarray = np.arange(0.1, 1.1, 0.1),
    ITI: float = 4.0,
    dt: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    z_fixed: float | None = None,
) -> dict:
    """
    Simulate a single task-switching trial.

    The trial consists of two phases fed as a contiguous sequence to
    wrapper.integrate():
      Phase 1 (step 0):           prev task cue + prev stimulus
      Phase 2 (steps 1..N):       curr task cue + curr stimulus

    The LCA reads only phase 2 outputs.

    Parameters
    ----------
    wrapper : TaskNetworkWrapper
        Trained network wrapper.
    stim_prev, stim_curr : np.ndarray
        Stimulus vectors for previous and current trials.
    cue_prev, cue_curr : np.ndarray
        One-hot task cue vectors.
    correct_idx : int
        Correct feature index within the current task's response dimension.
    resp_indices : list[int]
        Output unit indices for the current task's response dimension.
    persistence : float
        Persistence parameter p (Python convention: p=0.85 means 85%
        carry-over from previous timestep).
    n_phase2_steps : int
        Number of integration steps for phase 2 (current task).
    n_repeats : int
        Stochastic LCA repeats per threshold.
    thresholds : np.ndarray
        Threshold grid for reward-rate optimization.
    ITI : float
        Inter-trial interval for reward rate computation.
    dt, tau, t0, noise_std : float
        LCA parameters.
    z_fixed : float or None
        If given, skip threshold optimization and use this value.

    Returns
    -------
    dict with keys:
        rt          : float | None   mean RT (all decided trials)
        rt_correct  : float | None   mean RT (correct trials only)
        acc         : float | None   fraction correct
        decided     : float          fraction of LCA runs reaching threshold
        z           : float          threshold used
    """
    I = stim_prev.shape[0]
    T_dim = cue_prev.shape[0]
    total_steps = 1 + n_phase2_steps

    # Build contiguous input sequences
    stim_seq = np.zeros((total_steps, I), dtype=np.float32)
    cue_seq = np.zeros((total_steps, T_dim), dtype=np.float32)

    # Phase 1: one step with previous task
    stim_seq[0] = stim_prev
    cue_seq[0] = cue_prev

    # Phase 2: current task
    stim_seq[1:] = stim_curr[None, :]
    cue_seq[1:] = cue_curr[None, :]

    # Integrate
    outputs = wrapper.integrate(
        torch.from_numpy(stim_seq),
        torch.from_numpy(cue_seq),
        persistence=persistence,
    )
    out_np = np.stack([o.numpy() for o in outputs], axis=0)

    # LCA reads phase 2 only
    phase2_out = out_np[1:]

    # Threshold selection
    if z_fixed is not None:
        z = z_fixed
    else:
        z, _ = optimize_lca_threshold_dist(
            phase2_out, resp_indices,
            correct_response_idx=correct_idx,
            thresholds=thresholds,
            ITI=ITI, n_repeats=n_repeats,
            dt=dt, tau=tau, noise_std=noise_std,
        )

    # LCA measurement
    res = run_lca_avg(
        phase2_out, resp_indices,
        threshold=z, n_repeats=n_repeats,
        dt=dt, tau=tau, noise_std=noise_std,
        correct_response_idx=correct_idx,
    )

    return {
        "rt": res["rt"],
        "rt_correct": res["rt_correct"],
        "acc": res["p_correct"],
        "decided": res["frac_decided"],
        "z": float(z),
    }


def sweep_conditions(
    wrapper,
    persistence: float = 0.85,
    n_stim: int = 100,
    n_phase2_steps: int = 50,
    n_repeats: int = DEFAULT_N_REPEATS,
    thresholds: np.ndarray = np.arange(0.1, 1.1, 0.1),
    ITI: float = 4.0,
    dt: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    seed: int = 0,
    verbose: bool = True,
    blank_prev_stimulus: bool = True,
    conditions: list[tuple[str, str, str]] | None = None,
) -> dict:
    """
    Run task-switching simulation across conditions and congruency.

    Default conditions match Musslick et al. (2020) Simulation Study 3:
      E→A (structural dependence), B→A (functional dependence),
      C→A (independence), plus A→A (repeat baseline).

    Parameters
    ----------
    wrapper : TaskNetworkWrapper
        Trained network.
    persistence : float
        Python-convention persistence (p=0.85 = MATLAB tau=0.15).
    n_stim : int
        Number of stimulus samples per cell.
    conditions : list of (prev_task, curr_task, label) or None
        If None, uses the default four conditions.

    Returns
    -------
    dict mapping condition labels to sub-dicts, each containing:
        congruent / incongruent, each with:
            rt_mean, rt_se, rt_correct_mean, rt_correct_se,
            acc_mean, acc_se, n_decided, n_total
    """
    if conditions is None:
        conditions = [
            ("E", "A", "structural"),
            ("B", "A", "functional"),
            ("C", "A", "independent"),
            ("A", "A", "repeat"),
        ]

    results = {}

    for prev_task, curr_task, label in conditions:
        cond_results = {"congruent": [], "incongruent": []}

        for i in range(n_stim):
            trial = generate_switch_trial(
                prev_task, curr_task,
                seed=seed + i,
                blank_prev_stimulus=blank_prev_stimulus,
            )

            res = run_switch_trial(
                wrapper,
                stim_prev=trial["stim_prev"],
                stim_curr=trial["stim_curr"],
                cue_prev=trial["cue_prev"],
                cue_curr=trial["cue_curr"],
                correct_idx=trial["correct_idx"],
                resp_indices=trial["resp_indices"],
                persistence=persistence,
                n_phase2_steps=n_phase2_steps,
                n_repeats=n_repeats,
                thresholds=thresholds,
                ITI=ITI, dt=dt, tau=tau, t0=t0,
                noise_std=noise_std,
            )

            key = "congruent" if trial["congruent"] else "incongruent"
            cond_results[key].append(res)

        # Aggregate
        agg = {}
        for cong_key in ("congruent", "incongruent"):
            trials = cond_results[cong_key]
            if not trials:
                agg[cong_key] = {
                    "rt_mean": np.nan, "rt_se": np.nan,
                    "rt_correct_mean": np.nan, "rt_correct_se": np.nan,
                    "acc_mean": np.nan, "acc_se": np.nan,
                    "n_decided": 0, "n_total": 0,
                }
                continue

            rts = [t["rt"] for t in trials if t["rt"] is not None]
            rts_c = [t["rt_correct"] for t in trials if t["rt_correct"] is not None]
            accs = [t["acc"] for t in trials if t["acc"] is not None]

            agg[cong_key] = {
                "rt_mean": float(np.mean(rts)) if rts else np.nan,
                "rt_se": float(np.std(rts, ddof=1) / np.sqrt(len(rts))) if len(rts) > 1 else np.nan,
                "rt_correct_mean": float(np.mean(rts_c)) if rts_c else np.nan,
                "rt_correct_se": float(np.std(rts_c, ddof=1) / np.sqrt(len(rts_c))) if len(rts_c) > 1 else np.nan,
                "acc_mean": float(np.mean(accs)) if accs else np.nan,
                "acc_se": float(np.std(accs, ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else np.nan,
                "n_decided": sum(1 for t in trials if t["rt"] is not None),
                "n_total": len(trials),
            }

        results[label] = agg

        if verbose:
            for cong_key in ("congruent", "incongruent"):
                a = agg[cong_key]
                print(f"{label:12s} {cong_key:12s} | "
                      f"RT(correct) = {a['rt_correct_mean']:.3f} +- {a['rt_correct_se']:.3f} | "
                      f"Acc = {a['acc_mean']:.3f} | "
                      f"n = {a['n_decided']}/{a['n_total']}")

    # Compute switch costs
    if verbose and "repeat" in results:
        print("\n--- Switch costs (RT_correct, switch - repeat) ---")
        for label in [l for _, _, l in conditions if l != "repeat"]:
            if label not in results:
                continue
            for cong_key in ("congruent", "incongruent"):
                sw = results[label][cong_key]["rt_correct_mean"]
                rep = results["repeat"][cong_key]["rt_correct_mean"]
                if np.isfinite(sw) and np.isfinite(rep):
                    print(f"  {label:12s} {cong_key:12s}: {sw - rep:+.3f}s")

    return results


def sweep_persistence(
    wrapper,
    persistence_values: np.ndarray | list[float] = np.arange(0.0, 1.0, 0.05),
    n_stim: int = 50,
    n_phase2_steps: int = 50,
    n_repeats: int = 100,
    thresholds: np.ndarray = np.arange(0.1, 1.1, 0.1),
    ITI: float = 4.0,
    seed: int = 0,
    verbose: bool = True,
    blank_prev_stimulus: bool = True,
    conditions: list[tuple[str, str, str]] | None = None,
) -> dict:
    """
    Sweep persistence values and collect switch costs per condition.

    This produces the data for Fig. 21-style plots: switch cost as a
    function of persistence, separately for each dependence condition.

    Returns
    -------
    dict with keys:
        persistence   : list[float]
        Then per condition label, per congruency:
            "{label}_{cong}_rt"     : list[float]  mean RT(correct)
            "{label}_{cong}_cost"   : list[float]  switch cost (switch - repeat)
            "{label}_{cong}_acc"    : list[float]  mean accuracy
    """
    persistence_values = list(persistence_values)
    all_results = {
        "persistence": persistence_values,
    }

    for pi, p in enumerate(persistence_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Persistence p = {p:.2f}  (MATLAB tau = {1-p:.2f})")
            print(f"{'='*60}")

        res = sweep_conditions(
            wrapper, persistence=p,
            n_stim=n_stim, n_phase2_steps=n_phase2_steps,
            n_repeats=n_repeats, thresholds=thresholds,
            ITI=ITI, seed=seed, verbose=verbose,
            blank_prev_stimulus=blank_prev_stimulus,
            conditions=conditions,
        )

        for label, cond_data in res.items():
            for cong_key in ("congruent", "incongruent"):
                rt_key = f"{label}_{cong_key}_rt"
                acc_key = f"{label}_{cong_key}_acc"
                if rt_key not in all_results:
                    all_results[rt_key] = []
                    all_results[acc_key] = []
                all_results[rt_key].append(cond_data[cong_key]["rt_correct_mean"])
                all_results[acc_key].append(cond_data[cong_key]["acc_mean"])

        # Compute costs against repeat
        if "repeat" in res:
            for label in [l for l, d in res.items() if l != "repeat"]:
                for cong_key in ("congruent", "incongruent"):
                    cost_key = f"{label}_{cong_key}_cost"
                    if cost_key not in all_results:
                        all_results[cost_key] = []
                    sw = res[label][cong_key]["rt_correct_mean"]
                    rep = res["repeat"][cong_key]["rt_correct_mean"]
                    cost = sw - rep if np.isfinite(sw) and np.isfinite(rep) else np.nan
                    all_results[cost_key].append(cost)

    return all_results
