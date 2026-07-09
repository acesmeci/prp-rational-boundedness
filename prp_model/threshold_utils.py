"""
Threshold utilities for the LCA readout layer.

Provides:
- optimize_lca_threshold_dist: RR threshold sweep on a given output series.
- compute_fixed_threshold_for_task_meanargmax: per-task fixed z from
  SINGLE-TASK performance. RETAINED AS A DIAGNOSTIC ONLY — not used in the
  PRP pipeline (single-task calibration is systematically under-set for
  dual-task demands; see troubleshoot.md 08 Jul 2026, cells a/b).
- compute_condition_thresholds: per-condition fixed (z1, z2) from DUAL-TASK
  context, selected against the SESSION'S SOA MIXTURE (see below).
- choose_onset_policy: Task-2 onset policy (paper Eq. 7 / EVC).

Session-level criterion rationale (09 Jul 2026):
    In standard PRP designs SOA varies randomly within blocks, so an
    SOA-specific criterion is impossible for a participant to set; and
    criterion adaptation is empirically block-level, so per-trial fitting
    grants oracle foresight. What a participant CAN set is one criterion per
    task role (Task 1 / Task 2 are instructed and known), calibrated to the
    session's environment as a whole. We therefore select each threshold by
    accuracy-constrained argmax of the EXPECTED reward-rate curve, pooled
    over reference SOAs spanning the experimental range (soa_refs) and over
    stimuli. The accuracy floor is context-appropriate: dual-task selection
    uses acc_floor ~0.95, matching empirically maintained dual-task accuracy
    (90-95%), vs ~0.99 for single-task contexts.

Conventions:
- Task cues are row-major one-hots (index = in_dim * N_pathways + out_dim).
- ALL LCA parameter defaults sourced from prp_model.lca._DEFAULTS.
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
    Sweep thresholds with full LCA dynamics; return (best z by plain RR
    argmax, full results dict incl. per-threshold accuracies/RTs/RRs).
    Constrained selection is applied by the callers that need it.
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


def _constrained_argmax(thresholds, acc_mean, rr_mean, acc_floor, label="",
                        verbose=False):
    """Accuracy-constrained RR argmax with most-accurate fallback."""
    eligible = acc_mean >= acc_floor
    if eligible.any():
        idx = np.where(eligible)[0]
        z_star = float(thresholds[idx[int(np.argmax(rr_mean[idx]))]])
    else:
        z_star = float(thresholds[int(np.argmax(acc_mean))])
        print(f"[threshold:{label}] WARNING: no threshold met acc_floor="
              f"{acc_floor}; falling back to most accurate z={z_star:.2f} "
              f"(max mean acc={acc_mean.max():.3f})")
    if verbose:
        for i, z in enumerate(thresholds):
            flag = "*" if eligible[i] else " "
            print(f"{flag} z={z:.2f} | mean Acc={acc_mean[i]:.3f} "
                  f"| mean RR={rr_mean[i]:.3f}")
        print(f"Selected z_{label} (constrained argmax, "
              f"acc_floor={acc_floor}): {z_star:.3f}")
    return z_star


# ─────────────────────────────────────────────────────────────────────────────
# Fixed per-task threshold from SINGLE-TASK performance (DIAGNOSTIC ONLY)
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
    noise_std=_DEFAULTS["noise_std"],
    n_timesteps: int = 100,
    acc_floor: float = 0.99,
):
    """
    Fixed z for one task from sustained single-task trials
    (accuracy-constrained RR argmax). DIAGNOSTIC ONLY — see module docstring.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.5, 0.1)
    thresholds = np.asarray(thresholds)

    X, T, _, meta = generate_training_set_matlab_style()
    mask = meta["task_indices"] == task_name
    X, T = X[mask], T[mask]

    rng = np.random.RandomState(seed)
    pick = rng.choice(len(X), size=min(K, len(X)), replace=False)

    rr_curves, acc_curves = [], []
    for k in pick:
        x = torch.from_numpy(np.tile(X[k][None, :], (n_timesteps, 1)).astype(np.float32))
        t = torch.from_numpy(np.tile(T[k][None, :], (n_timesteps, 1)).astype(np.float32))
        out_th = wrapper.integrate(x, t, persistence=persistence)
        out_np = np.stack([o.numpy() for o in out_th], axis=0)

        rel_idxs, correct_idx = _decode_task(T[k], X[k],
                                             N_pathways=N_pathways,
                                             N_features=N_features)
        _, res = optimize_lca_threshold_dist(
            out_np, rel_idxs, correct_response_idx=correct_idx,
            thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
            dt=dt, tau=tau, noise_std=noise_std,
        )
        rr_curves.append(res["reward_rates"])
        acc_curves.append(res["accuracies"])

    rr_mean = np.stack(rr_curves, axis=0).mean(axis=0)
    acc_mean = np.stack(acc_curves, axis=0).mean(axis=0)
    return _constrained_argmax(thresholds, acc_mean, rr_mean, acc_floor,
                               label=task_name, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed per-condition thresholds from DUAL-TASK context (SOA mixture)
# ─────────────────────────────────────────────────────────────────────────────

def compute_condition_thresholds(
    wrapper,
    task1_name: str,
    task2_name: str,
    soa_refs=(3, 8, 16),
    n_stim: int = 20,
    thresholds=None,
    ITI=0.5,
    n_repeats=DEFAULT_N_REPEATS,
    persistence: float = 0.75,
    seed: int = 42,
    verbose=False,
    dt=_DEFAULTS["dt"],
    tau=_DEFAULTS["tau"],
    noise_std=_DEFAULTS["noise_std"],
    t0=_DEFAULTS["t0"],
    max_timesteps: int = 100,
    acc_floor: float = 0.95,
    acc_floor_task1: float = 0.99,   # <-- NEW: Task 1 is the protected task
):
    """
    Session-level fixed (z1, z2) for one PRP condition, selected from
    dual-task context against the session's SOA MIXTURE.

    Accuracy and RR curves are pooled over len(soa_refs) reference SOAs x
    n_stim stimuli; each threshold is then a single accuracy-constrained
    argmax of the pooled (expected) curves. Models a participant setting one
    criterion per task role for a mixed-SOA session (see module docstring).

    Procedure per (soa_ref, stimulus): mirrors run_prp_trial's two passes
    with greedy Task-2 onset. Pass 1 pools Task-1 curves -> z1; pass 2
    (gated by Task-1 RT under z1) pools Task-2 tail curves -> z2.

    Returns
    -------
    (z1, z2) : (float, float)
    """
    from prp_model.utils import generate_trial_pair  # local import (no cycle)

    if thresholds is None:
        thresholds = np.arange(0.1, 1.5, 0.1)
    thresholds = np.asarray(thresholds)
    soa_refs = tuple(int(s) for s in soa_refs)

    def _integrate(inp_series, cue_series):
        x = np.stack(inp_series, axis=0).astype(np.float32)
        t = np.stack(cue_series, axis=0).astype(np.float32)
        out_th = wrapper.integrate(torch.from_numpy(x), torch.from_numpy(t),
                                   persistence=persistence)
        return np.stack([o.numpy() for o in out_th], axis=0)

    trials = []          # (soa_ref, s1, s2, c1, c2, out1, idxs1, corr1)
    rr1, acc1 = [], []

    # ---- Pass 1, pooled over soa_refs x stimuli: Task-1 threshold curves ----
    for r_idx, soa_ref in enumerate(soa_refs):
        for i in range(n_stim):
            s1, s2, c1, c2 = generate_trial_pair(
                (task1_name, task2_name), seed=seed + i + 10000 * r_idx
            )
            I, T = s1.shape[0], c1.shape[0]
            inp_series, cue_series = [], []
            for t in range(max_timesteps):
                s = np.zeros(I, dtype=np.float32); s += s1
                if t >= soa_ref: s += s2
                c = np.zeros(T, dtype=np.float32); c += c1
                if t >= soa_ref: c += c2      # greedy onset (policy off)
                inp_series.append(s); cue_series.append(c)
            out1 = _integrate(inp_series, cue_series)

            idxs1, corr1 = _decode_task(c1, s1)
            _, res = optimize_lca_threshold_dist(
                out1, idxs1, correct_response_idx=corr1,
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                dt=dt, tau=tau, noise_std=noise_std,
            )
            rr1.append(res["reward_rates"]); acc1.append(res["accuracies"])
            trials.append((soa_ref, s1, s2, c1, c2, out1, idxs1, corr1))

    z1 = _constrained_argmax(thresholds,
                             np.stack(acc1).mean(axis=0),
                             np.stack(rr1).mean(axis=0),
                             acc_floor_task1,
                             label=f"{task1_name}|{task1_name}->{task2_name}",
                             verbose=verbose)

    # ---- Pass 2, pooled: Task-2 tail curves under z1 gating ----
    rr2, acc2 = [], []
    for (soa_ref, s1, s2, c1, c2, out1, idxs1, corr1) in trials:
        res1 = run_lca_avg(out1, idxs1, threshold=z1, n_repeats=n_repeats,
                           dt=dt, tau=tau, noise_std=noise_std,
                           correct_response_idx=corr1)
        rt1 = res1["rt"]
        t_off1 = (int(np.ceil(max(0.0, (rt1 - t0) / dt)))
                  if rt1 is not None else max_timesteps)

        I, T = s1.shape[0], c1.shape[0]
        inp_series, cue_series = [], []
        for t in range(max_timesteps):
            s = np.zeros(I, dtype=np.float32); s += s1
            if t >= soa_ref: s += s2
            c = np.zeros(T, dtype=np.float32)
            if t < t_off1: c += c1
            if t >= soa_ref: c += c2
            inp_series.append(s); cue_series.append(c)
        out2 = _integrate(inp_series, cue_series)

        tail = out2[soa_ref:]
        if tail.shape[0] == 0:
            continue
        idxs2, corr2 = _decode_task(c2, s2)
        _, res = optimize_lca_threshold_dist(
            tail, idxs2, correct_response_idx=corr2,
            thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
            dt=dt, tau=tau, noise_std=noise_std,
        )
        rr2.append(res["reward_rates"]); acc2.append(res["accuracies"])

    z2 = _constrained_argmax(thresholds,
                             np.stack(acc2).mean(axis=0),
                             np.stack(rr2).mean(axis=0),
                             acc_floor,
                             label=f"{task2_name}|{task1_name}->{task2_name}",
                             verbose=verbose)

    return z1, z2


# ─────────────────────────────────────────────────────────────────────────────
# Task-2 onset policy (reward-rate optimal onset, paper Eq. 7 / EVC)
# ─────────────────────────────────────────────────────────────────────────────

def choose_onset_policy(
    task_net,
    input_a, input_b,
    task_a, task_b,
    soa: int = 0,
    max_onset_delay: int = 15,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    ITI: float = 0.5,
    dt_lca: float = _DEFAULTS["dt"],
    t0: float = _DEFAULTS["t0"],
    tau: float = _DEFAULTS["tau"],
    noise_std: float = _DEFAULTS["noise_std"],
    z_a_fixed: float | None = None,
    z_b_fixed: float | None = None,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,
):
    """
    Reward-rate onset policy for Task-2 (paper Eq. 7; EVC: control-signal
    onset as a decision variable — Musslick, Shenhav, Botvinick & Cohen 2015).
    RR = (P_corr_1 * P_corr_2) / (ITI + max(RT_1, RT_2_abs)).
    Warns if the optimum sits at the search-window edge.
    """
    if thresholds_policy is None:
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
        return np.stack([o.numpy() for o in out_th], axis=0)

    idxs_a, corr_a = _decode(task_a, input_a)
    idxs_b, corr_b = _decode(task_b, input_b)

    best_rr = -np.inf
    best_onset = int(soa)
    onset_upper = min(int(soa) + int(max_onset_delay), max_timesteps - 1)

    for onset in range(int(soa), onset_upper + 1):
        out1 = _integrate_once(onset_b=onset, gate_after_a_steps=None)
        if z_a_fixed is None:
            z_a, _ = optimize_lca_threshold_dist(
                out1, idxs_a, correct_response_idx=corr_a,
                thresholds=thresholds_policy, ITI=ITI,
                n_repeats=policy_n_repeats,
                dt=dt_lca, tau=tau, noise_std=noise_std,
            )
        else:
            z_a = z_a_fixed
        res_a = run_lca_avg(out1, idxs_a, threshold=z_a,
                            n_repeats=policy_n_repeats, dt=dt_lca, tau=tau,
                            noise_std=noise_std, correct_response_idx=corr_a)
        if res_a["rt"] is None:
            continue
        t_off_a = int(np.ceil(max(0.0, (res_a["rt"] - t0) / dt_lca)))

        out2 = _integrate_once(onset_b=onset, gate_after_a_steps=t_off_a)
        tail = out2[onset:]
        if tail.shape[0] == 0:
            continue

        if z_b_fixed is None:
            z_b, _ = optimize_lca_threshold_dist(
                tail, idxs_b, correct_response_idx=corr_b,
                thresholds=thresholds_policy, ITI=ITI,
                n_repeats=policy_n_repeats,
                dt=dt_lca, tau=tau, noise_std=noise_std,
            )
        else:
            z_b = z_b_fixed
        res_b = run_lca_avg(tail, idxs_b, threshold=z_b,
                            n_repeats=policy_n_repeats, dt=dt_lca, tau=tau,
                            noise_std=noise_std, correct_response_idx=corr_b)
        if res_b["rt"] is None:
            continue
        rt_b_abs = res_b["rt"] + onset * dt_lca

        rr = (res_a["p_correct"] * res_b["p_correct"]) / (ITI + max(res_a["rt"], rt_b_abs))
        if rr > best_rr:
            best_rr = rr
            best_onset = onset

    if best_onset == onset_upper:
        print(f"[choose_onset_policy] WARNING: optimum at window edge "
              f"(onset={best_onset}, soa={soa}, max_onset_delay={max_onset_delay}) "
              f"— consider a larger window.")

    return int(best_onset)