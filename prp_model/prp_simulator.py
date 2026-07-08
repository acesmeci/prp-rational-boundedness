# prp_model/prp_simulator.py
"""
PRP (Psychological Refractory Period) simulation helpers.

Runs dual-task (PRP) trials on a trained TaskNetworkWrapper and aggregates
results across SOAs.

Conventions (matches paper/MATLAB):
- Task cues use ROW-MAJOR indexing: index = (input_dim * N_pathways) + output_dim.
- SOA is measured in LCA STEPS. One step = dt_lca sim-seconds = MS_PER_STEP
  display-milliseconds.
- Stimuli follow the partial-stimulus design (Musslick et al., 2023, p. 68):
  each task's stimulus activates only its task-relevant dimension; stim2 is
  superimposed at the SOA.
- Task-1 cue is ON from t=0 until its decision time; then it is turned OFF.
- Task-2 cue is OFF until its onset (SOA or policy-chosen); then it stays ON.
- Task-2 LCA readout starts at the Task-2 cue onset (onset2), NOT at the SOA:
  evidence accumulation for Task 2 begins when control engages the task. The
  strategic delay (onset2 - soa) is instead counted in the stimulus-locked
  reaction time rt2_from_stim, which is the paper-faithful dependent measure.

Accuracy semantics:
- acc_task1 / acc_task2 are graded P(correct): the fraction of stochastic LCA
  repeats (within the trial) on which the correct accumulator crossed first.
- decided_task1 / decided_task2 are the fractions of LCA repeats on which ANY
  accumulator crossed threshold (non-decisions are excluded from RT and
  accuracy, so these fields make that exclusion visible).
"""

import numpy as np
import torch

from prp_model.lca import run_lca_avg, _DEFAULTS
from prp_model.threshold_utils import optimize_lca_threshold_dist, choose_onset_policy

# Number of stochastic LCA runs used when averaging within a trial
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
    optimize_onset: bool = False,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,
    max_onset_delay: int = 5,
    return_outputs: bool = False,
):
    """
    Simulate a single PRP trial with explicit Task-1 then Task-2.

    Two-pass procedure:
      (1) Integrate with Task-1 from t=0 and Task-2 from onset to obtain
          Task-1 RT.
      (2) Rebuild the series with Task-1 turned OFF after its decision time
          and evaluate Task-2 on the tail (time >= onset2).

    Parameters (selected)
    ---------------------
    z_task1_fixed : float | None
        If provided, fixes Task-1's threshold (precomputed from single-task
        performance, recommended). If None, a reward-rate maximizing threshold
        is fit per-trial on the dual-task output series (legacy behavior;
        expensive and criterion adapts to interference level).
        TODO(design): decide with Sebastian / MATLAB check whether per-trial
        fitting should be removed entirely.
    z_task2_fixed : float | None
        If provided, fixes Task-2's threshold (recommended: precomputed z_A).
    dt_lca, tau, t0 : float
        LCA timing parameters, defaulting to lca._DEFAULTS (dt/tau = 0.1).
        These are threaded through ALL threshold fits and RT measurements so
        selection and measurement always run under identical dynamics.
    return_outputs : bool
        If True, include the pass-2 output time series under key "outputs"
        (memory-heavy; keep False in sweeps).

    Returns
    -------
    dict with keys:
        rt1            : float | None   Task-1 RT (sim-seconds), mean over decided repeats
        acc1           : float | None   Task-1 P(correct) over decided repeats
        decided1       : float          fraction of repeats with a Task-1 decision
        rt2_abs        : float | None   Task-2 RT from trial onset (sim-seconds)
        rt2_from_stim  : float | None   Task-2 RT from S2 onset (paper-faithful)
        rt2_tail       : float | None   Task-2 RT from cue-2 onset (LCA-internal)
        acc2           : float | None   Task-2 P(correct) over decided repeats
        decided2       : float          fraction of repeats with a Task-2 decision
        onset2         : int            Task-2 cue onset actually used (steps)
        z1, z2         : float          thresholds used for Task-1 / Task-2
        outputs        : np.ndarray | None  pass-2 output series (if requested)
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
            dt=dt_lca, tau=tau,
        )
    rt1, _, acc1, decided1 = run_lca_avg(
        out1, idxs1, threshold=z1, n_repeats=n_repeats,
        dt=dt_lca, tau=tau, correct_response_idx=corr1,
    )

    # Convert Task-1 RT (sim-seconds) to the step index for gating the cue off
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
    tail = out2[onset2:]                # readout starts at task engagement (see module docstring)

    result = {
        "rt1": rt1, "acc1": acc1, "decided1": decided1,
        "rt2_abs": None, "rt2_from_stim": None, "rt2_tail": None,
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
            dt=dt_lca, tau=tau,
        )
    else:
        z2 = z_task2_fixed
    result["z2"] = float(z2)

    rt2_tail, _, acc2, decided2 = run_lca_avg(
        tail, idxs2, threshold=z2, n_repeats=n_repeats,
        dt=dt_lca, tau=tau, correct_response_idx=corr2,
    )

    if rt2_tail is not None:
        result["rt2_tail"] = rt2_tail
        result["rt2_abs"] = rt2_tail + onset2 * dt_lca
        result["rt2_from_stim"] = rt2_tail + (onset2 - soa) * dt_lca  # paper-faithful RT2
    result["acc2"] = acc2
    result["decided2"] = decided2

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
    ITI: float = 0.5,
    optimize_onset: bool = False,
    thresholds=np.arange(0.1, 1.6, 0.1),
):
    """
    Run PRP simulations across a list of SOAs and aggregate RT/ACC.

    Returns
    -------
    dict
        Keys (each a per-SOA list of means across valid trials, NaN if none):
          "soa", "rt_task1", "acc_task1", "decided_task1",
          "rt_task2", "acc_task2", "decided_task2",
          "onset2", "rt_task2_tail", "rt_task2_from_stim"
        RT/accuracy means include only trials where the corresponding task
        produced at least one decided LCA repeat; "decided_task*" reports the
        mean decided fraction over ALL trials, making exclusions visible.
    """
    keys = (
        "soa",
        "rt_task1", "acc_task1", "decided_task1",
        "rt_task2", "acc_task2", "decided_task2",
        "onset2", "rt_task2_tail", "rt_task2_from_stim",
    )
    results = {k: [] for k in keys}

    for soa in soa_values:
        r1, a1, d1 = [], [], []
        r2, a2, d2 = [], [], []
        onsets, r2_tail, r2_from_stim = [], [], []

        for _ in range(n_trials_per_soa):
            s1, s2, c1, c2 = trial_generator()
            tr = run_prp_trial(
                task_net, s1, s2, c1, c2, soa,
                max_timesteps=max_timesteps, persistence=persistence,
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                z_task1_fixed=z_task1_fixed, z_task2_fixed=z_task2_fixed,
                dt_lca=dt_lca, tau=tau, t0=t0,
                optimize_onset=optimize_onset,
            )

            d1.append(tr["decided1"])
            d2.append(tr["decided2"])

            if tr["rt1"] is not None:
                r1.append(tr["rt1"]); a1.append(tr["acc1"])

            if tr["rt2_tail"] is not None:
                r2.append(tr["rt2_abs"]); a2.append(tr["acc2"])
                r2_tail.append(tr["rt2_tail"])
                r2_from_stim.append(tr["rt2_from_stim"])
                onsets.append(tr["onset2"])

        results["soa"].append(soa)
        results["rt_task1"].append(np.mean(r1) if r1 else np.nan)
        results["acc_task1"].append(np.mean(a1) if a1 else np.nan)
        results["decided_task1"].append(np.mean(d1) if d1 else np.nan)
        results["rt_task2"].append(np.mean(r2) if r2 else np.nan)
        results["acc_task2"].append(np.mean(a2) if a2 else np.nan)
        results["decided_task2"].append(np.mean(d2) if d2 else np.nan)
        results["onset2"].append(np.mean(onsets) if onsets else np.nan)
        results["rt_task2_tail"].append(np.mean(r2_tail) if r2_tail else np.nan)
        results["rt_task2_from_stim"].append(np.mean(r2_from_stim) if r2_from_stim else np.nan)

        if verbose:
            print(f"SOA={soa} | T1 RT={results['rt_task1'][-1]:.2f} "
                  f"| T2 RT={results['rt_task2'][-1]:.2f} "
                  f"| T2 acc={results['acc_task2'][-1]:.2f}")

    return results