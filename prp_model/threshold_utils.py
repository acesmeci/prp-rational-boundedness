"""
Threshold utilities for the LCA readout layer.

This module provides:
- A reward-rate threshold optimizer that wraps `run_lca_dist` and forwards
  all LCA parameters (`optimize_lca_threshold_dist`).
- A per-task fixed-threshold precomputation over single-task stimuli with
  ACCURACY-CONSTRAINED reward-rate selection
  (`compute_fixed_threshold_for_task_meanargmax`).
- A PRP onset policy helper (`choose_onset_policy`).

Threshold selection rule (08 Jul 2026):
    Among thresholds whose mean single-task accuracy across stimuli is
    >= acc_floor (default 0.99), select the one maximizing mean reward rate.
    Rationale: standard PRP instructions require responding as fast as
    possible WHILE MAINTAINING ACCURACY — accuracy is an instructed
    constraint that participants satisfice near ceiling, not a quantity
    traded linearly against speed. Unconstrained RR = acc/(ITI+RT) is flat
    across a wide z range (single-task accuracy saturates early), so its
    argmax resolves sampling noise rather than a real optimum and can land
    on hair-trigger thresholds that behave pathologically under dual-task
    interference.

Conventions:
- Task cues are **row-major** one-hots (index = in_dim * N_pathways + out_dim).
- Reward-rate = accuracy / (ITI + RT). In `run_lca_dist`, trials with no
  decision yield RT=NaN and are treated as RR=0 so extreme thresholds
  cannot win spuriously.
- ALL LCA parameter defaults are sourced from `prp_model.lca._DEFAULTS`
  (single source of truth). Never re-declare dt/tau literals here (a stale
  local default previously ran threshold selection under collapsed
  dt/tau=1.0 dynamics; fixed 08 Jul 2026).
"""

import numpy as np
import torch

from prp_model.lca import run_lca_avg, run_lca_dist, _DEFAULTS
from prp_model.training_set import generate_training_set_matlab_style

DEFAULT_N_REPEATS = 100


# ─────────────────────────────────────────────────────────────────────────────
# Reward-rate threshold optimization on a given output series
# ─────────────────────────────────────────────────────────────────────────────

def optimize_lca_threshold_dist(
    input_series,
    relevant_output_indices,
    correct_response_idx=None,
    thresholds=np.arange(0.1, 1.6, 0.1),
    ITI=0.5,
    n_repeats=DEFAULT_N_REPEATS,
    dt=_DEFAULTS["dt"],
    tau=_DEFAULTS["tau"],
    lambda_=_DEFAULTS["lambda_"],
    alpha=_DEFAULTS["alpha"],
    beta=_DEFAULTS["beta"],
    noise_std=_DEFAULTS["noise_std"],
    t0=_DEFAULTS["t0"],
    verbose=False,
):
    """
    Sweep thresholds with full LCA dynamics and choose the z with max reward-rate.

    Returns
    -------
    (best_threshold, results) : (float, dict)
        `results` is the dictionary returned by `run_lca_dist` and includes:
        thresholds, reward_rates, accuracies, rts, all_rts, all_accs.

    Notes
    -----
    - Unconstrained argmax; used for per-trial fits. The fixed per-task
      precompute applies the accuracy-constrained rule instead (see
      compute_fixed_threshold_for_task_meanargmax).
    - `run_lca_dist` treats "no decision" trials as RR=0.
    - Tiebreaks: if several units cross z in the same discretized step,
      `run_lca_dist` picks uniformly among them.
    """
    results = run_lca_dist(
        input_series=input_series,
        relevant_output_indices=relevant_output_indices,
        thresholds=thresholds,
        n_repeats=n_repeats,
        dt=dt, tau=tau, lambda_=lambda_, alpha=alpha, beta=beta,
        noise_std=noise_std, t0=t0, ITI=ITI,
        correct_response_idx=correct_response_idx,
    )

    rr = results["reward_rates"]
    best_idx = int(np.argmax(rr))  # safe: RR invalids were set to 0
    best_threshold = float(results["thresholds"][best_idx])

    if verbose:
        accs, rts, zs = results["accuracies"], results["rts"], results["thresholds"]
        for i in range(len(zs)):
            print(f"z={zs[i]:.2f} | Acc={accs[i]:.2f} | RT={rts[i]:.3f} | RR={rr[i]:.3f}")
        print(f"Best threshold z: {best_threshold:.2f}")

    return best_threshold, results


# ─────────────────────────────────────────────────────────────────────────────
# Fixed per-task threshold from single-task performance
# ─────────────────────────────────────────────────────────────────────────────

def _decode_task(task_vec, input_vec, N_pathways=3, N_features=3):
    """Row-major: decode task cue to relevant output indices and correct feature index."""
    M = task_vec.reshape(N_pathways, N_pathways)
    i_in, i_out = np.argwhere(M == 1)[0]
    correct = int(np.argmax(input_vec[i_in * N_features : (i_in + 1) * N_features]))
    idxs = list(range(i_out * N_features, (i_out + 1) * N_features))
    return idxs, correct


def compute_fixed_threshold_for_task_meanargmax(
    wrapper,
    task_name="A",
    K=27,
    thresholds=None,
    ITI=0.5,
    n_repeats=DEFAULT_N_REPEATS,
    persistence=0.0,
    seed=42,
    verbose=False,
    N_pathways=3,
    N_features=3,
    dt=_DEFAULTS["dt"],
    tau=_DEFAULTS["tau"],
    n_timesteps: int = 100,
    acc_floor: float = 0.99,
):
    """
    Select a fixed LCA threshold z for one task by ACCURACY-CONSTRAINED
    argmax of the mean reward-rate curve over K single-task stimuli.

    Procedure:
      1. Sample K single-task patterns for the chosen task.
      2. Integrate each as a sustained trial (constant stimulus + cue for
         n_timesteps steps; a 1-step series cannot support accumulation
         under dt/tau = 0.1 and degenerates z-selection — fixed 08 Jul 2026).
      3. Sweep thresholds per stimulus (run_lca_dist), stack per-stimulus
         accuracy and RR curves, and average both across stimuli.
      4. Among thresholds with mean accuracy >= acc_floor, return the one
         with maximal mean reward rate. If no threshold satisfies the
         constraint, fall back to the most accurate threshold (and warn).

    (Note for methods text: this is a constrained argmax over the mean-RR
    curve across stimuli, NOT a median of per-stimulus optima.)

    Returns
    -------
    float
        Selected threshold z_star.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.5, 0.1)
    thresholds = np.asarray(thresholds)

    X, T, _, meta = generate_training_set_matlab_style()
    mask = meta["task_indices"] == task_name
    X, T = X[mask], T[mask]

    rng = np.random.RandomState(seed)
    n_avail = len(X)
    pick = rng.choice(n_avail, size=min(K, n_avail), replace=False)

    rr_curves, acc_curves = [], []
    for k in pick:
        # Sustained single-task trial (matches PRP trial presentation).
        x = torch.from_numpy(
            np.tile(X[k][None, :], (n_timesteps, 1)).astype(np.float32)
        )
        t = torch.from_numpy(
            np.tile(T[k][None, :], (n_timesteps, 1)).astype(np.float32)
        )
        out_th = wrapper.integrate(x, t, persistence=persistence)
        out_np = np.stack([o.numpy() for o in out_th], axis=0)

        rel_idxs, correct_idx = _decode_task(
            T[k], X[k], N_pathways=N_pathways, N_features=N_features
        )

        _, res = optimize_lca_threshold_dist(
            out_np,
            rel_idxs,
            correct_response_idx=correct_idx,
            thresholds=thresholds,
            ITI=ITI,
            n_repeats=n_repeats,
            dt=dt, tau=tau,
        )
        rr_curves.append(res["reward_rates"])
        acc_curves.append(res["accuracies"])

    rr_mean = np.stack(rr_curves, axis=0).mean(axis=0)
    acc_mean = np.stack(acc_curves, axis=0).mean(axis=0)

    eligible = acc_mean >= acc_floor
    if eligible.any():
        idx = np.where(eligible)[0]
        z_star = float(thresholds[idx[int(np.argmax(rr_mean[idx]))]])
    else:
        z_star = float(thresholds[int(np.argmax(acc_mean))])
        print(f"[compute_fixed_threshold] WARNING: no threshold met "
              f"acc_floor={acc_floor} for task {task_name}; falling back to "
              f"most accurate z={z_star:.2f} (max mean acc={acc_mean.max():.3f})")

    if verbose:
        for i, z in enumerate(thresholds):
            flag = "*" if eligible[i] else " "
            print(f"{flag} z={z:.2f} | mean Acc={acc_mean[i]:.3f} | mean RR={rr_mean[i]:.3f}")
        print(f"Selected fixed z_{task_name} (constrained argmax, "
              f"acc_floor={acc_floor}): {z_star:.3f}")
    return z_star


# ─────────────────────────────────────────────────────────────────────────────
# Task-2 onset policy (reward-rate optimal onset, paper Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

def choose_onset_policy(
    task_net,
    input_a, input_b,
    task_a, task_b,
    soa: int = 0,
    max_onset_delay: int = 5,      # small, fast search window
    max_timesteps: int = 100,
    persistence: float = 0.5,
    ITI: float = 0.5,
    dt_lca: float = _DEFAULTS["dt"],
    t0: float = _DEFAULTS["t0"],
    tau: float = _DEFAULTS["tau"],
    # speedups for policy search:
    z_a_fixed: float | None = None,
    z_b_fixed: float | None = None,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,  # coarse grid for policy search
):
    """
    Reward-rate onset policy for Task-2.

    Evaluates candidate onsets in [SOA, SOA + max_onset_delay] and returns the
    one that maximizes:
        RR = (P_corr_1 * P_corr_2) / (ITI + max(RT_1, RT_2_abs)).

    Accuracies are graded P(correct) across LCA repeats (paper-faithful,
    Eq. 7). RTs are means over all decided repeats (the policy models the
    agent's expected trial duration, errors included).

    Returns
    -------
    int
        Best onset (in steps) according to the policy.
    """
    if thresholds_policy is None:
        # Coarse grid is enough for relative comparisons (faster).
        thresholds_policy = np.linspace(0.1, 0.6, 6)

    def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
        mat = task_vec.reshape(N_pathways, N_pathways)
        in_dim, out_dim = np.argwhere(mat == 1)[0]
        correct = int(np.argmax(input_vec[in_dim*N_features:(in_dim+1)*N_features]))
        idxs = list(range(out_dim*N_features, (out_dim+1)*N_features))
        return idxs, correct

    def _integrate_once(onset_b: int, gate_after_a_steps: int | None):
        inp_dim, task_dim = input_a.shape[0], task_a.shape[0]
        input_series, task_series = [], []
        for t in range(max_timesteps):
            stim_t = np.zeros(inp_dim, dtype=np.float32)
            task_t = np.zeros(task_dim, dtype=np.float32)
            stim_t += input_a
            if t >= soa:
                stim_t += input_b
            if gate_after_a_steps is None or t < gate_after_a_steps:
                task_t += task_a
            if t >= onset_b:
                task_t += task_b
            input_series.append(stim_t)
            task_series.append(task_t)
        input_np = np.stack(input_series, axis=0).astype(np.float32)
        task_np = np.stack(task_series, axis=0).astype(np.float32)
        out_th = task_net.integrate(torch.from_numpy(input_np),
                                    torch.from_numpy(task_np),
                                    persistence=persistence)
        return np.stack([o.numpy() for o in out_th], axis=0)  # [T, D_out]

    idxs_a, corr_a = _decode(task_a, input_a)
    idxs_b, corr_b = _decode(task_b, input_b)

    best_rr = -np.inf
    best_onset = int(soa)
    onset_upper = min(int(soa) + int(max_onset_delay), max_timesteps - 1)

    for onset in range(int(soa), onset_upper + 1):
        # ---- Pass 1: Task-1 timing ----
        out1 = _integrate_once(onset_b=onset, gate_after_a_steps=None)
        if z_a_fixed is None:
            z_a, _ = optimize_lca_threshold_dist(
                out1, idxs_a,
                correct_response_idx=corr_a,
                thresholds=thresholds_policy,
                ITI=ITI,
                n_repeats=policy_n_repeats,
                dt=dt_lca, tau=tau,
            )
        else:
            z_a = z_a_fixed
        res_a = run_lca_avg(
            out1, idxs_a, threshold=z_a,
            n_repeats=policy_n_repeats, dt=dt_lca, tau=tau,
            correct_response_idx=corr_a,
        )
        if res_a["rt"] is None:
            continue
        # Convert RT (sec) to step index for gating Task-1 off
        t_off_a = int(np.ceil(max(0.0, (res_a["rt"] - t0) / dt_lca)))

        # ---- Pass 2: gate Task-1, then evaluate Task-2 ----
        out2 = _integrate_once(onset_b=onset, gate_after_a_steps=t_off_a)
        tail = out2[onset:]
        if tail.shape[0] == 0:
            continue

        if z_b_fixed is None:
            z_b, _ = optimize_lca_threshold_dist(
                tail, idxs_b,
                correct_response_idx=corr_b,
                thresholds=thresholds_policy,
                ITI=ITI,
                n_repeats=policy_n_repeats,
                dt=dt_lca, tau=tau,
            )
        else:
            z_b = z_b_fixed
        res_b = run_lca_avg(
            tail, idxs_b, threshold=z_b,
            n_repeats=policy_n_repeats, dt=dt_lca, tau=tau,
            correct_response_idx=corr_b,
        )
        if res_b["rt"] is None:
            continue
        rt_b_abs = res_b["rt"] + onset * dt_lca  # convert to absolute time

        rr = (res_a["p_correct"] * res_b["p_correct"]) / (ITI + max(res_a["rt"], rt_b_abs))
        if rr > best_rr:
            best_rr = rr
            best_onset = onset

    return int(best_onset)