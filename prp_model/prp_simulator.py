# prp_model/prp_simulator.py
"""
PRP (Psychological Refractory Period) simulation helpers.

Runs dual-task (PRP) trials on a trained TaskNetworkWrapper and aggregates
results across SOAs. See run_prp_trial for trial structure; conventions
unchanged from 08 Jul 2026 refactor (dict returns; graded P(correct);
correct-trials-only RT fields; Task-2 readout from cue onset with
stimulus-locked rt2_from_stim as the reported measure).

noise_std is threaded through ALL threshold fits and RT measurements so the
same LCA noise regime governs selection and measurement.
"""

import numpy as np
import torch

from prp_model.lca import run_lca_avg, _DEFAULTS
from prp_model.threshold_utils import optimize_lca_threshold_dist, choose_onset_policy

DEFAULT_N_REPEATS = 100


def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
    """Map a ROW-MAJOR task cue to (relevant output indices, correct feature)."""
    M = task_vec.reshape(N_pathways, N_pathways)   # row-major (no transpose)
    in_dim, out_dim = np.argwhere(M == 1)[0]
    correct = int(np.argmax(input_vec[in_dim*N_features:(in_dim+1)*N_features]))
    idxs = list(range(out_dim*N_features, (out_dim+1)*N_features))
    return idxs, correct


def run_prp_trial(
    task_net,
    stim1, stim2,              # partial one-hot stimuli (same length)
    cue1,  cue2,               # one-hot task cues (row-major: in*N + out)
    soa: int,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    thresholds=np.arange(0.1, 1.6, 0.1),
    ITI: float = 0.5,
    n_repeats: int = DEFAULT_N_REPEATS,
    z_task1_fixed: float | None = None,
    z_task2_fixed: float | None = None,
    dt_lca: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    optimize_onset: bool = False,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,
    max_onset_delay: int = 5,
    return_outputs: bool = False,
):
    """
    Simulate a single PRP trial (two-pass; see module docstring).

    Returns
    -------
    dict with keys:
        rt1, rt1_correct, acc1, decided1,
        rt2_abs, rt2_from_stim, rt2_tail, rt2_from_stim_correct,
        acc2, decided2, onset2, z1, z2, outputs
    (semantics as documented in the 08 Jul 2026 refactor; rt*_correct are
    the empirical-convention correct-trials-only measures.)
    """

    def _integrate(input_series, task_series):
        x = np.stack(input_series, axis=0).astype(np.float32)
        t = np.stack(task_series,  axis=0).astype(np.float32)
        out_th = task_net.integrate(
            torch.from_numpy(x), torch.from_numpy(t), persistence=persistence
        )
        return np.stack([o.numpy() for o in out_th], axis=0)

    # --- 0) Decide Task-2 onset (fixed SOA or reward-rate policy) ---
    if optimize_onset:
        onset2 = choose_onset_policy(
            task_net, stim1, stim2, cue1, cue2,
            soa=soa, max_onset_delay=max_onset_delay, max_timesteps=max_timesteps,
            persistence=persistence, ITI=ITI, dt_lca=dt_lca, t0=t0, tau=tau,
            noise_std=noise_std,
            z_a_fixed=z_task1_fixed, z_b_fixed=z_task2_fixed,
            policy_n_repeats=policy_n_repeats,
            thresholds_policy=thresholds_policy,
        )
    else:
        onset2 = soa

    # --- 1) Pass 1: both cues from their onsets -> measure Task-1 RT ---
    inp_series, cue_series = [], []
    I, T = stim1.shape[0], cue1.shape[0]
    for t in range(max_timesteps):
        s = np.zeros(I, dtype=np.float32); s += stim1
        if t >= soa:
            s += stim2                  # stim2 appears at SOA
        c = np.zeros(T, dtype=np.float32); c += cue1
        if t >= onset2:
            c += cue2                   # task-2 cue at (possibly delayed) onset
        inp_series.append(s); cue_series.append(c)
    out1 = _integrate(inp_series, cue_series)

    idxs1, corr1 = _decode(cue1, stim1)
    if z_task1_fixed is not None:
        z1 = z_task1_fixed
    else:
        z1, _ = optimize_lca_threshold_dist(
            out1, idxs1, correct_response_idx=corr1,
            thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
            dt=dt_lca, tau=tau, noise_std=noise_std,
        )
    res1 = run_lca_avg(
        out1, idxs1, threshold=z1, n_repeats=n_repeats,
        dt=dt_lca, tau=tau, noise_std=noise_std, correct_response_idx=corr1,
    )
    rt1 = res1["rt"]

    # Cue-gating time from the all-decided mean RT (response occurs when it
    # occurs, right or wrong).
    t_off1 = int(np.ceil(max(0.0, (rt1 - t0) / dt_lca))) if rt1 is not None else max_timesteps

    # --- 2) Pass 2: turn OFF Task-1 after its decision -> evaluate Task-2 tail ---
    inp_series, cue_series = [], []
    for t in range(max_timesteps):
        s = np.zeros(I, dtype=np.float32); s += stim1
        if t >= soa: s += stim2
        c = np.zeros(T, dtype=np.float32)
        if t < t_off1: c += cue1        # Task-1 only until its decision
        if t >= onset2: c += cue2       # Task-2 from onset
        inp_series.append(s); cue_series.append(c)
    out2 = _integrate(inp_series, cue_series)

    idxs2, corr2 = _decode(cue2, stim2)
    tail = out2[onset2:]                # readout starts at task engagement

    result = {
        "rt1": rt1, "rt1_correct": res1["rt_correct"],
        "acc1": res1["p_correct"], "decided1": res1["frac_decided"],
        "rt2_abs": None, "rt2_from_stim": None, "rt2_tail": None,
        "rt2_from_stim_correct": None,
        "acc2": None, "decided2": 0.0,
        "onset2": int(onset2),
        "z1": float(z1), "z2": np.nan,
        "outputs": out2 if return_outputs else None,
    }

    if tail.shape[0] == 0:
        return result

    if z_task2_fixed is None:
        z2, _ = optimize_lca_threshold_dist(
            tail, idxs2, correct_response_idx=corr2,
            thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
            dt=dt_lca, tau=tau, noise_std=noise_std,
        )
    else:
        z2 = z_task2_fixed
    result["z2"] = float(z2)

    res2 = run_lca_avg(
        tail, idxs2, threshold=z2, n_repeats=n_repeats,
        dt=dt_lca, tau=tau, noise_std=noise_std, correct_response_idx=corr2,
    )

    if res2["rt"] is not None:
        result["rt2_tail"] = res2["rt"]
        result["rt2_abs"] = res2["rt"] + onset2 * dt_lca
        result["rt2_from_stim"] = res2["rt"] + (onset2 - soa) * dt_lca
    if res2["rt_correct"] is not None:
        result["rt2_from_stim_correct"] = res2["rt_correct"] + (onset2 - soa) * dt_lca
    result["acc2"] = res2["p_correct"]
    result["decided2"] = res2["frac_decided"]

    return result


def sweep_soa(
    task_net,
    trial_generator,                 # returns (stim1, stim2, cue1, cue2)
    soa_values,
    n_trials_per_soa: int = 10,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    n_repeats: int = DEFAULT_N_REPEATS,
    verbose: bool = False,
    z_task1_fixed: float | None = None,
    z_task2_fixed: float | None = None,
    dt_lca: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    ITI: float = 0.5,
    optimize_onset: bool = False,
    thresholds=np.arange(0.1, 1.6, 0.1),
):
    """
    Run PRP simulations across a list of SOAs and aggregate RT/ACC.
    Returns per-SOA means (see 08 Jul 2026 refactor docs);
    rt_task2_from_stim_correct is the reported dependent measure.
    """
    keys = (
        "soa",
        "rt_task1", "rt_task1_correct", "acc_task1", "decided_task1",
        "rt_task2", "acc_task2", "decided_task2",
        "onset2", "rt_task2_tail", "rt_task2_from_stim",
        "rt_task2_from_stim_correct",
    )
    results = {k: [] for k in keys}

    for soa in soa_values:
        r1, r1c, a1, d1 = [], [], [], []
        r2, a2, d2 = [], [], []
        onsets, r2_tail, r2_from_stim, r2_from_stim_c = [], [], [], []

        for _ in range(n_trials_per_soa):
            s1, s2, c1, c2 = trial_generator()
            tr = run_prp_trial(
                task_net, s1, s2, c1, c2, soa,
                max_timesteps=max_timesteps, persistence=persistence,
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                z_task1_fixed=z_task1_fixed, z_task2_fixed=z_task2_fixed,
                dt_lca=dt_lca, tau=tau, t0=t0, noise_std=noise_std,
                optimize_onset=optimize_onset,
            )

            d1.append(tr["decided1"])
            d2.append(tr["decided2"])

            if tr["rt1"] is not None:
                r1.append(tr["rt1"]); a1.append(tr["acc1"])
            if tr["rt1_correct"] is not None:
                r1c.append(tr["rt1_correct"])

            if tr["rt2_tail"] is not None:
                r2.append(tr["rt2_abs"]); a2.append(tr["acc2"])
                r2_tail.append(tr["rt2_tail"])
                r2_from_stim.append(tr["rt2_from_stim"])
                onsets.append(tr["onset2"])
            if tr["rt2_from_stim_correct"] is not None:
                r2_from_stim_c.append(tr["rt2_from_stim_correct"])

        results["soa"].append(soa)
        results["rt_task1"].append(np.mean(r1) if r1 else np.nan)
        results["rt_task1_correct"].append(np.mean(r1c) if r1c else np.nan)
        results["acc_task1"].append(np.mean(a1) if a1 else np.nan)
        results["decided_task1"].append(np.mean(d1) if d1 else np.nan)
        results["rt_task2"].append(np.mean(r2) if r2 else np.nan)
        results["acc_task2"].append(np.mean(a2) if a2 else np.nan)
        results["decided_task2"].append(np.mean(d2) if d2 else np.nan)
        results["onset2"].append(np.mean(onsets) if onsets else np.nan)
        results["rt_task2_tail"].append(np.mean(r2_tail) if r2_tail else np.nan)
        results["rt_task2_from_stim"].append(np.mean(r2_from_stim) if r2_from_stim else np.nan)
        results["rt_task2_from_stim_correct"].append(
            np.mean(r2_from_stim_c) if r2_from_stim_c else np.nan
        )

        if verbose:
            print(f"SOA={soa} | T1 RT={results['rt_task1'][-1]:.2f} "
                  f"| T2 RT(correct)={results['rt_task2_from_stim_correct'][-1]:.2f} "
                  f"| T2 acc={results['acc_task2'][-1]:.2f}")

    return results