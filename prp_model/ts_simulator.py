"""
ts_simulator.py
---------------
Task-switching simulation for the Rational Boundedness model.

Implements cued task switching following Musslick et al. (2020/2023),
Simulation Study 3 (pp. 70-74). The network performs a current task
after having just performed a previous task, and the switch cost is
the RT difference between switch trials (prev != curr) and repeat
trials (prev == curr).

Trial structure (matching MATLAB switchTasks / Part1_Sim3_Transition_Analysis):
  Phase 1 (1 step):  Previous task cue + stimulus.
                      Switch trials: independently sampled bivalent stimulus.
                      Repeat trials: blank (zeros) stimulus.
  Phase 2 (N steps): Current task cue + bivalent stimulus.
                      Persistence carries the phase-1 state forward.
                      LCA reads the output series from this phase.

Design (matching MATLAB Part1_Sim3_TaskSwitching_Analysis):
  - Switch and repeat both measure RT for the SAME current task (always A).
  - Repeat baseline is A->A with blank previous stimulus.
  - Congruency for repeat stimuli is classified using the switch pair's
    input dimensions (e.g. E and A for structural), not A and A.

Convention note (MATLAB <-> Python persistence):
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


def _integrate_switch_trial(
    wrapper,
    stim_prev: np.ndarray,
    stim_curr: np.ndarray,
    cue_prev: np.ndarray,
    cue_curr: np.ndarray,
    persistence: float,
    n_phase2_steps: int,
) -> np.ndarray:
    """Run integration and return phase-2 output series (n_phase2_steps, D_out)."""
    I = stim_prev.shape[0]
    T_dim = cue_prev.shape[0]
    total_steps = 1 + n_phase2_steps

    stim_seq = np.zeros((total_steps, I), dtype=np.float32)
    cue_seq = np.zeros((total_steps, T_dim), dtype=np.float32)
    stim_seq[0] = stim_prev
    cue_seq[0] = cue_prev
    stim_seq[1:] = stim_curr[None, :]
    cue_seq[1:] = cue_curr[None, :]

    outputs = wrapper.integrate(
        torch.from_numpy(stim_seq),
        torch.from_numpy(cue_seq),
        persistence=persistence,
    )
    out_np = np.stack([o.numpy() for o in outputs], axis=0)
    return out_np[1:]  # phase 2 only


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
    thresholds: np.ndarray = np.arange(0.1, 1.6, 0.1),
    ITI: float = 4.0,
    dt: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    z_fixed: float | None = None,
) -> dict:
    """
    Simulate a single task-switching trial.

    Returns dict with keys: rt, rt_correct, acc, decided, z
    """
    phase2_out = _integrate_switch_trial(
        wrapper, stim_prev, stim_curr, cue_prev, cue_curr,
        persistence=persistence, n_phase2_steps=n_phase2_steps,
    )

    if z_fixed is not None:
        z = z_fixed
    else:
        z, _ = optimize_lca_threshold_dist(
            phase2_out, resp_indices,
            correct_response_idx=correct_idx,
            thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
            dt=dt, tau=tau, noise_std=noise_std,
        )

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


def _aggregate_cell(trials: list[dict]) -> dict:
    """Aggregate a list of per-trial result dicts into cell-level summary."""
    if not trials:
        return {
            "rt_mean": np.nan, "rt_se": np.nan,
            "rt_correct_mean": np.nan, "rt_correct_se": np.nan,
            "acc_mean": np.nan, "acc_se": np.nan,
            "z": np.nan, "n_decided": 0, "n_total": 0,
        }

    def _ms(vals):
        if not vals:
            return np.nan, np.nan
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        return m, se

    rts = [t["rt"] for t in trials if t["rt"] is not None]
    rts_c = [t["rt_correct"] for t in trials if t["rt_correct"] is not None]
    accs = [t["acc"] for t in trials if t["acc"] is not None]
    zs = [t["z"] for t in trials if np.isfinite(t["z"])]

    rt_m, rt_se = _ms(rts)
    rtc_m, rtc_se = _ms(rts_c)
    acc_m, acc_se = _ms(accs)

    return {
        "rt_mean": rt_m, "rt_se": rt_se,
        "rt_correct_mean": rtc_m, "rt_correct_se": rtc_se,
        "acc_mean": acc_m, "acc_se": acc_se,
        "z": float(np.mean(zs)) if zs else np.nan,
        "n_decided": len(rts), "n_total": len(trials),
    }


def sweep_task_switching(
    wrapper,
    persistence: float = 0.85,
    n_stim: int = 100,
    n_phase2_steps: int = 50,
    n_repeats: int = DEFAULT_N_REPEATS,
    thresholds: np.ndarray = np.arange(0.1, 1.6, 0.1),
    ITI: float = 4.0,
    dt: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    seed: int = 0,
    verbose: bool = True,
    switch_pairs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """
    Run the full task-switching design: switch conditions with real previous
    stimuli, plus a matched A->A repeat baseline with blank previous stimuli.

    For each switch pair, congruency of repeat stimuli is classified using
    that pair's input dimensions, providing matched congruent and incongruent
    repeat baselines per condition.

    Parameters
    ----------
    wrapper : TaskNetworkWrapper
        Trained network.
    persistence : float
        Python-convention persistence (p=0.85 = MATLAB tau=0.15).
    n_stim : int
        Number of stimulus samples per condition.
    switch_pairs : list of (prev_task, curr_task, label) or None
        If None, uses [(E,A,structural), (B,A,functional), (C,A,independent)].

    Returns
    -------
    dict with structure:
        results[label]["switch"]["congruent"]   -> cell summary dict
        results[label]["switch"]["incongruent"] -> cell summary dict
        results[label]["repeat"]["congruent"]   -> cell summary dict
        results[label]["repeat"]["incongruent"] -> cell summary dict
        results[label]["cost_congruent"]        -> float (switch - repeat)
        results[label]["cost_incongruent"]      -> float (switch - repeat)
    """
    if switch_pairs is None:
        switch_pairs = [
            ("E", "A", "structural"),
            ("B", "A", "functional"),
            ("C", "A", "independent"),
        ]

    # --- Step 1: Generate and integrate ALL repeat trials once ---
    # We run n_stim A->A trials with blank prev stimulus.
    # Congruency will be classified per switch pair later.
    repeat_trials_raw = []
    for i in range(n_stim):
        # Generate with dummy congruency (will reclassify per pair)
        trial = generate_switch_trial(
            "A", "A", seed=seed + 10000 + i,
            blank_prev_stimulus=True,
            congruency_tasks=None,  # will override below
        )
        phase2_out = _integrate_switch_trial(
            wrapper, trial["stim_prev"], trial["stim_curr"],
            trial["cue_prev"], trial["cue_curr"],
            persistence=persistence, n_phase2_steps=n_phase2_steps,
        )
        repeat_trials_raw.append({
            "phase2_out": phase2_out,
            "correct_idx": trial["correct_idx"],
            "resp_indices": trial["resp_indices"],
            "stim_curr": trial["stim_curr"],
        })

    results = {}

    for prev_task, curr_task, label in switch_pairs:
        congruency_ref = (prev_task, curr_task)

        # --- Step 2: Switch trials (real previous stimuli) ---
        switch_data = {"congruent": [], "incongruent": []}

        for i in range(n_stim):
            trial = generate_switch_trial(
                prev_task, curr_task, seed=seed + i,
                blank_prev_stimulus=False,
                congruency_tasks=congruency_ref,
            )
            phase2_out = _integrate_switch_trial(
                wrapper, trial["stim_prev"], trial["stim_curr"],
                trial["cue_prev"], trial["cue_curr"],
                persistence=persistence, n_phase2_steps=n_phase2_steps,
            )

            # Pass 1: get RR curve for this stimulus
            _, rr_res = optimize_lca_threshold_dist(
                phase2_out, trial["resp_indices"],
                correct_response_idx=trial["correct_idx"],
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                dt=dt, tau=tau, noise_std=noise_std,
            )
            key = "congruent" if trial["congruent"] else "incongruent"
            switch_data[key].append({
                "phase2_out": phase2_out,
                "correct_idx": trial["correct_idx"],
                "resp_indices": trial["resp_indices"],
                "rr_curve": rr_res["reward_rates"],
                "acc_curve": rr_res["accuracies"],
            })

        # --- Step 3: Classify repeat trials using this pair's congruency ---
        repeat_data = {"congruent": [], "incongruent": []}
        in_prev_ref = TASK_MAP[prev_task][0]
        in_curr_ref = TASK_MAP[curr_task][0]

        for rt in repeat_trials_raw:
            # Extract features from stimulus to check congruency
            stim = rt["stim_curr"]
            feat_prev = int(np.argmax(stim[in_prev_ref * N_FEATURES:(in_prev_ref + 1) * N_FEATURES]))
            feat_curr = int(np.argmax(stim[in_curr_ref * N_FEATURES:(in_curr_ref + 1) * N_FEATURES]))
            is_cong = (feat_prev == feat_curr)

            _, rr_res = optimize_lca_threshold_dist(
                rt["phase2_out"], rt["resp_indices"],
                correct_response_idx=rt["correct_idx"],
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                dt=dt, tau=tau, noise_std=noise_std,
            )
            key = "congruent" if is_cong else "incongruent"
            repeat_data[key].append({
                "phase2_out": rt["phase2_out"],
                "correct_idx": rt["correct_idx"],
                "resp_indices": rt["resp_indices"],
                "rr_curve": rr_res["reward_rates"],
                "acc_curve": rr_res["accuracies"],
            })

        # --- Step 4: Optimize one z per cell, measure RT ---
        cond_results = {"switch": {}, "repeat": {}}

        for trial_type, data in [("switch", switch_data), ("repeat", repeat_data)]:
            for cong_key in ("congruent", "incongruent"):
                trials = data[cong_key]
                if not trials:
                    cond_results[trial_type][cong_key] = _aggregate_cell([])
                    continue

                # Pool RR curves, find single z
                mean_rr = np.stack([t["rr_curve"] for t in trials]).mean(axis=0)
                z_cell = float(thresholds[int(np.argmax(mean_rr))])

                if verbose:
                    mean_acc = np.stack([t["acc_curve"] for t in trials]).mean(axis=0)
                    best_idx = int(np.argmax(mean_rr))
                    print(f"  {label:12s} {trial_type:6s} {cong_key:12s} | "
                          f"z={z_cell:.2f} (RR={mean_rr[best_idx]:.3f}, "
                          f"Acc={mean_acc[best_idx]:.3f}) [{len(trials)} stim]")

                # Measure RT with fixed z
                measured = []
                resp_indices = trials[0]["resp_indices"]
                for t in trials:
                    res = run_lca_avg(
                        t["phase2_out"], resp_indices,
                        threshold=z_cell, n_repeats=n_repeats,
                        dt=dt, tau=tau, noise_std=noise_std,
                        correct_response_idx=t["correct_idx"],
                    )
                    measured.append({
                        "rt": res["rt"],
                        "rt_correct": res["rt_correct"],
                        "acc": res["p_correct"],
                        "decided": res["frac_decided"],
                        "z": z_cell,
                    })
                cond_results[trial_type][cong_key] = _aggregate_cell(measured)

        # --- Step 5: Compute switch costs ---
        costs = {}
        for cong_key in ("congruent", "incongruent"):
            sw = cond_results["switch"][cong_key]["rt_correct_mean"]
            rep = cond_results["repeat"][cong_key]["rt_correct_mean"]
            costs[cong_key] = sw - rep if np.isfinite(sw) and np.isfinite(rep) else np.nan

        cond_results["cost_congruent"] = costs["congruent"]
        cond_results["cost_incongruent"] = costs["incongruent"]
        results[label] = cond_results

        if verbose:
            for trial_type in ("switch", "repeat"):
                for cong_key in ("congruent", "incongruent"):
                    a = cond_results[trial_type][cong_key]
                    print(f"{label:12s} {trial_type:6s} {cong_key:12s} | "
                          f"RT(c)={a['rt_correct_mean']:.3f}+-{a['rt_correct_se']:.3f} | "
                          f"Acc={a['acc_mean']:.3f} | z={a['z']:.2f} | "
                          f"n={a['n_decided']}/{a['n_total']}")
            print(f"  -> cost con={costs['congruent']:+.3f}  "
                  f"inc={costs['incongruent']:+.3f}")
            print()

    # Summary
    if verbose:
        print("=" * 60)
        print("SWITCH COSTS SUMMARY")
        print("=" * 60)
        for _, _, label in switch_pairs:
            c = results[label]
            print(f"  {label:12s} | congruent: {c['cost_congruent']:+.3f} | "
                  f"incongruent: {c['cost_incongruent']:+.3f}")

    return results


def sweep_persistence(
    wrapper,
    persistence_values: np.ndarray | list[float] = np.arange(0.0, 1.0, 0.05),
    n_stim: int = 50,
    n_phase2_steps: int = 50,
    n_repeats: int = 100,
    thresholds: np.ndarray = np.arange(0.1, 1.6, 0.1),
    ITI: float = 4.0,
    seed: int = 0,
    verbose: bool = True,
    switch_pairs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """
    Sweep persistence values and collect switch costs per condition.

    Produces data for Fig. 21-style plots: switch cost as a function of
    persistence, separately for each dependence condition and congruency.

    Returns
    -------
    dict with keys:
        persistence   : list[float]
        "{label}_cost_congruent"   : list[float]
        "{label}_cost_incongruent" : list[float]
        "{label}_switch_{cong}_rt" : list[float]
        "{label}_repeat_{cong}_rt" : list[float]
    """
    persistence_values = list(persistence_values)
    all_results = {"persistence": persistence_values}

    if switch_pairs is None:
        switch_pairs = [
            ("E", "A", "structural"),
            ("B", "A", "functional"),
            ("C", "A", "independent"),
        ]

    for pi, p in enumerate(persistence_values):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Persistence p = {p:.2f}  (MATLAB tau = {1 - p:.2f})")
            print(f"{'=' * 60}")

        res = sweep_task_switching(
            wrapper, persistence=p,
            n_stim=n_stim, n_phase2_steps=n_phase2_steps,
            n_repeats=n_repeats, thresholds=thresholds,
            ITI=ITI, seed=seed, verbose=verbose,
            switch_pairs=switch_pairs,
        )

        for _, _, label in switch_pairs:
            for cost_key in ("cost_congruent", "cost_incongruent"):
                k = f"{label}_{cost_key}"
                if k not in all_results:
                    all_results[k] = []
                all_results[k].append(res[label][cost_key])

            for trial_type in ("switch", "repeat"):
                for cong_key in ("congruent", "incongruent"):
                    k = f"{label}_{trial_type}_{cong_key}_rt"
                    if k not in all_results:
                        all_results[k] = []
                    all_results[k].append(
                        res[label][trial_type][cong_key]["rt_correct_mean"]
                    )

    return all_results