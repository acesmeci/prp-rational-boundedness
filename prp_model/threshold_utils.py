"""
Threshold utilities for the LCA readout layer.

This module provides:
- A reward-rate threshold optimizer that wraps `run_lca_dist` and forwards
  all LCA parameters (`optimize_lca_threshold_dist`).
- A per-task fixed-threshold precomputation over single-task stimuli
  (`compute_fixed_threshold_for_task_meanargmax`).
- A PRP onset policy helper (`choose_onset_policy`) that evaluates a small
  window of candidate Task-2 onsets using two-pass integration.

Conventions:
- Task cues are **row-major** one-hots (index = in_dim * N_pathways + out_dim).
- Reward-rate = accuracy / (ITI + RT). In `run_lca_dist`, trials with no
  decision yield RT=NaN and are treated as RR=0 so extreme thresholds
  cannot win spuriously.
- ALL LCA parameter defaults are sourced from `prp_model.lca._DEFAULTS`
  (single source of truth). Never re-declare dt/tau literals here: a stale
  local default (dt=0.1, tau=0.1 -> dt/tau=1.0) previously caused threshold
  selection to run under collapsed dynamics while RT measurement ran under
  the correct dt/tau=0.1 regime. Fixed 08 Jul 2026.
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

    Parameters
    ----------
    input_series : np.ndarray
        Output time series, shape [T, D_out].
    relevant_output_indices : Sequence[int]
        Indices of the response units for the current task (within D_out).
    correct_response_idx : int | None
        Correct feature index **within** the relevant outputs. If None,
        `run_lca_dist` will infer a label (fallback; not recommended).
    thresholds : np.ndarray
        Threshold grid to test.
    ITI : float
        Inter-trial interval used in reward-rate.
    n_repeats : int
        Number of LCA simulations per threshold.
    dt, tau, lambda_, alpha, beta, noise_std, t0 : float
        LCA parameters, defaulting to lca._DEFAULTS (dt/tau = 0.1).
    verbose : bool
        If True, print per-threshold Acc/RT/RR and the selected z.

    Returns
    -------
    (best_threshold, results) : (float, dict)
        `results` is the dictionary returned by `run_lca_dist` and includes:
        thresholds, reward_rates, accuracies, rts, all_rts, all_accs.

    Notes
    -----
    - `run_lca_dist` treats "no decision" trials as RR=0, preventing large z
      values from being selected due to NaNs.
    - Tiebreaks: if several units cross z in the same discretized step,
      `run_lca_dist` picks uniformly among them (not literally first-crosser).
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
):
    """
    Select a fixed LCA threshold z for one task by argmax of MEAN reward-rate
    curve over K single-task stimuli.

    Samples K single-task patterns for the chosen task, integrates once per
    stimulus, runs optimize_lca_threshold_dist for each, stacks RR curves,
    takes the mean over K, then returns the threshold that maximizes this
    mean RR curve.

    (Note for methods text: this is argmax-of-mean-RR across stimuli, NOT the
    median of per-stimulus optimal thresholds.)

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

    rr_curves = []
    for k in pick:
        x = torch.from_numpy(X[k][None, :]).float()
        t = torch.from_numpy(T[k][None, :]).float()
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

    rr_curves = np.stack(rr_curves, axis=0)
    rr_mean = rr_curves.mean(axis=0)
    z_star = float(thresholds[int(np.argmax(rr_mean))])

    if verbose:
        print(f"Selected fixed z_{task_name} (argmax of mean RR): {z_star:.3f}")
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
    Eq. 7), not modal-choice indicators.

    Two-pass evaluation per candidate onset:
      Pass 1: Task-1 cue ON (Task-2 from onset) -> RT_1 and gate time.
      Pass 2: Turn OFF Task-1 after its decision -> evaluate Task-2 on tail.

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
        rt_a, _, pcorr_a, _ = run_lca_avg(
            out1, idxs_a, threshold=z_a,
            n_repeats=policy_n_repeats, dt=dt_lca, tau=tau,
            correct_response_idx=corr_a,
        )
        if rt_a is None:
            continue
        # Convert RT (sec) to step index for gating Task-1 off
        t_off_a = int(np.ceil(max(0.0, (rt_a - t0) / dt_lca)))

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
        rt_b, _, pcorr_b, _ = run_lca_avg(
            tail, idxs_b, threshold=z_b,
            n_repeats=policy_n_repeats, dt=dt_lca, tau=tau,
            correct_response_idx=corr_b,
        )
        if rt_b is None:
            continue
        rt_b_abs = rt_b + onset * dt_lca  # convert to absolute time

        rr = (pcorr_a * pcorr_b) / (ITI + max(rt_a, rt_b_abs))
        if rr > best_rr:
            best_rr = rr
            best_onset = onset

    return int(best_onset)