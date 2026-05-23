# prp/prp_simulator.py
"""
PRP (Psychological Refractory Period) simulation helpers.

This module runs dual-task (PRP) trials on a trained TaskNetworkWrapper and
aggregates results across SOAs.

Conventions (matches paper/MATLAB):
- Task cues use ROW-MAJOR indexing: index = (input_dim * N_pathways) + output_dim.
  No transposes anywhere in this file.
- SOA is measured in LCA STEPS (not seconds). With dt_lca=0.1, SOA=1 → 0.1 s.
- Task-1 cue is ON from t=0 until its decision time; then it is turned OFF.
- Task-2 cue is OFF until its onset (SOA or policy-chosen); then it stays ON.
- Task-2 RT is read from the “tail” (time ≥ onset), then shifted to absolute time.

Typical usage:
    results = sweep_soa(
        task_net=wrapper,
        trial_generator=lambda: generate_trial_pair(("B","A")),  # returns (stim1, stim2, cue1, cue2)
        soa_values=range(1, 9),
        n_trials_per_soa=30,
        persistence=0.95,
        z_task2_fixed=z_A,        # fixed LCA threshold for Task-2 (recommended)
        optimize_onset=False
    )
    # plot results["rt_task2"] vs results["soa"]
"""

import numpy as np
import torch
from prp_model.lca import run_lca_avg
from prp_model.threshold_utils import optimize_lca_threshold_dist, choose_onset_policy

# Number of stochastic LCA runs used when averaging within a trial
DEFAULT_N_REPEATS = 100


def run_prp_trial(
    task_net,
    stim1, stim2,              # one-hot stimuli (same length)
    cue1,  cue2,               # one-hot task cues (row-major: in*N + out)
    soa: int,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    thresholds=np.arange(0.1, 1.6, 0.1),
    ITI: float = 0.5, # 0.5
    n_repeats: int = DEFAULT_N_REPEATS,
    z_task2_fixed: float | None = None,
    dt_lca: float = 0.1,
    t0: float = 0.15,
    optimize_onset: bool = False,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,
    max_onset_delay: int = 5,
):
    """
    Simulate a single PRP trial with explicit Task-1 then Task-2.

    Two-pass procedure:
      (1) Integrate with Task-1 from t=0 and Task-2 from onset to obtain Task-1 RT.
      (2) Rebuild the series with Task-1 turned OFF after its decision time and
          evaluate Task-2 on the tail (time ≥ onset).

    Parameters
    ----------
    task_net : TaskNetworkWrapper
        Must expose .integrate(input_series, task_series, persistence).
    stim1, stim2 : np.ndarray
        One-hot stimuli (length = N_pathways * N_features). Should be identical
        if you want a shared stimulus per trial (recommended).
    cue1, cue2 : np.ndarray
        One-hot task cues (length = N_pathways**2) using ROW-MAJOR indexing.
    soa : int
        Task-2 onset in LCA steps (seconds = soa * dt_lca).
    max_timesteps : int
        Length of the synthetic trial in LCA steps.
    persistence : float
        Network carry-over parameter p.
    thresholds : np.ndarray
        Grid of LCA thresholds used when optimizing (if needed).
    ITI : float
        Inter-trial interval used in reward-rate computations.
    n_repeats : int
        Number of LCA repeats for averaging stochastic dynamics.
    z_task2_fixed : float | None
        If provided, fixes Task-2’s threshold to this value. Otherwise a
        reward-rate maximizing threshold is fit on the Task-2 tail.
    dt_lca : float
        LCA time step in seconds (unit conversion for RT).
    t0 : float
        Non-decision time added to LCA hitting times (seconds).
    optimize_onset : bool
        If True, choose Task-2 onset via reward-rate policy (bounded by
        max_onset_delay). If False, onset = SOA.
    policy_n_repeats : int
        LCA repeats used inside the onset policy.
    thresholds_policy : np.ndarray | None
        Threshold grid used by the onset policy (use a coarse grid for speed).
    max_onset_delay : int
        Maximum extra delay (in steps) the policy may add to the given SOA.

    Returns
    -------
    rt_task1 : float | None
        Absolute RT (seconds) for Task-1, or None if no decision occurred.
    acc_task1 : bool
        Whether Task-1’s choice matched the correct feature (False if no decision).
    rt_task2 : float | None
        Absolute RT (seconds) for Task-2 (tail RT + onset*dt_lca), or None.
    acc_task2 : bool
        Whether Task-2’s choice matched the correct feature (False if no decision).
    outputs_np : np.ndarray
        Final pass output time series (after Task-1 gating).

    Notes
    -----
    - Reward-rate optimization (optimize_onset / threshold fitting) relies on
      optimize_lca_threshold_dist / run_lca_dist, which set RR=0 when no decision
      occurs (prevents extreme thresholds from “winning” due to NaNs).
    """

    def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
        """Map a ROW-MAJOR task cue to (relevant output indices, correct feature)."""
        M = task_vec.reshape(N_pathways, N_pathways)   # row-major (no transpose)
        in_dim, out_dim = np.argwhere(M == 1)[0]
        # Correct feature is taken from the relevant input dimension
        correct = np.argmax(input_vec[in_dim*N_features:(in_dim+1)*N_features])
        # Absolute indices for the relevant output dimension
        idxs = list(range(out_dim*N_features, (out_dim+1)*N_features))
        return idxs, correct

    def _integrate(input_series, task_series):
        """Run network integration over a full trial and return output time series."""
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
            persistence=persistence, ITI=ITI, dt_lca=dt_lca, t0=t0,
            z_b_fixed=z_task2_fixed, policy_n_repeats=policy_n_repeats,
            thresholds_policy=thresholds_policy
        )
    else:
        onset2 = soa

    # --- 1) Pass 1: both cues from their onsets → measure Task-1 RT ---
    inp_series, cue_series = [], []
    I, T = stim1.shape[0], cue1.shape[0]
    # Pass 1: Disentange task_stim onset and task_cue onset, by making stim2 appear at SOA
    for t in range(max_timesteps):
        # Stimuli
        s = np.zeros(I, dtype=np.float32); s += stim1
        if t >= soa:     # <-- CHANGED (stim2 appears at SOA)
            s += stim2
        # Task cues
        c = np.zeros(T, dtype=np.float32); c += cue1
        if t >= onset2:  # <-- unchanged (task-2 cue at optimized onset)
            c += cue2

        inp_series.append(s); cue_series.append(c)
    out1 = _integrate(inp_series, cue_series)

    idxs1, corr1 = _decode(cue1, stim1)
    z1, _ = optimize_lca_threshold_dist(out1, idxs1, corr1, thresholds, ITI, n_repeats)
    rt1, choice1 = run_lca_avg(out1, idxs1, threshold=z1, n_repeats=n_repeats, dt=dt_lca)
    acc1 = (choice1 == corr1) if rt1 is not None else False

    # Convert Task-1 RT (sec) to the step index when the cue should be gated off
    t_off1 = int(np.ceil(max(0.0, (rt1 - t0) / dt_lca))) if rt1 is not None else max_timesteps

    # --- 2) Pass 2: turn OFF Task-1 after its decision → evaluate Task-2 tail ---
    inp_series, cue_series = [], []
    for t in range(max_timesteps):
        s = np.zeros(I, dtype=np.float32); s += stim1
        if t >= soa: s += stim2         # stim2 appears at SOA, instead of onset2
        c = np.zeros(T, dtype=np.float32)
        if t < t_off1: c += cue1        # Task-1 only until its decision
        if t >= onset2: c += cue2       # Task-2 from onset
        inp_series.append(s); cue_series.append(c)
    out2 = _integrate(inp_series, cue_series)

    idxs2, corr2 = _decode(cue2, stim2)
    tail = out2[onset2:]                 # readout for Task-2 starts at onset !!! I think this should start from SOA
    
    # returns onset2 as well
    if tail.shape[0] == 0:
        return rt1, acc1, None, False, out2, onset2, None


    if z_task2_fixed is None:
        z2, _ = optimize_lca_threshold_dist(tail, idxs2, corr2, thresholds, ITI, n_repeats)
    else:
        z2 = z_task2_fixed

    # **Returns rt2_tail + onset2, so that I dont have to compute tail_rt in notebook using raw SOA
    rt2_tail, choice2 = run_lca_avg(tail, idxs2, threshold=z2,
                                n_repeats=n_repeats, dt=dt_lca)

    rt2_abs = None
    rt2_from_stim = None
    if rt2_tail is not None:
        rt2_abs = rt2_tail + onset2 * dt_lca
        rt2_from_stim = rt2_tail + (onset2 - soa) * dt_lca  # <-- NEW, paper-faithful RT2

    acc2 = (choice2 == corr2) if rt2_tail is not None else False

    return rt1, acc1, rt2_abs, acc2, out2, onset2, rt2_tail, rt2_from_stim



def sweep_soa(
    task_net,
    trial_generator,                 # returns (stim1, stim2, cue1, cue2)
    soa_values,
    n_trials_per_soa: int = 10,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    n_repeats: int = DEFAULT_N_REPEATS,
    verbose: bool = False,
    z_task2_fixed: float | None = None,
    dt_lca: float = 0.1,
    t0: float = 0.15,
    ITI: float = 0.5, #0.5
    optimize_onset: bool = False,
    thresholds=np.arange(0.1, 1.6, 0.1),
):
    """
    Run PRP simulations across a list of SOAs and aggregate RT/ACC.

    Parameters
    ----------
    task_net : TaskNetworkWrapper
    trial_generator : callable
        Must return a tuple (stim1, stim2, cue1, cue2). For cleaner PRP curves,
        prefer a generator that uses the same stimulus features for both tasks.
    soa_values : iterable[int]
        SOAs in LCA steps to evaluate (seconds = soa * dt_lca).
    n_trials_per_soa : int
        Number of trials to run per SOA.
    max_timesteps, persistence, n_repeats, z_task2_fixed, dt_lca, t0, ITI,
    optimize_onset, thresholds :
        Passed through to run_prp_trial (see its docstring).

    Returns
    -------
    dict
        Keys:
          "soa"        : list[int]
          "rt_task1"   : list[float]
          "acc_task1"  : list[float]
          "rt_task2"   : list[float]
          "acc_task2"  : list[float]
        Each list contains the per-SOA mean across valid trials (NaNs dropped).
    """
    
    results = {k: [] for k in (
        "soa", "rt_task1", "acc_task1",
        "rt_task2", "acc_task2",
        "onset2", "rt_task2_tail", "rt_task2_from_stim"
    )}

    for soa in soa_values:
        r1, a1, r2, a2 = [], [], [], []
        onsets, r2_tail, r2_from_stim = [], [], []   # <-- moved here

        for _ in range(n_trials_per_soa):
            s1, s2, c1, c2 = trial_generator()
            rt1, acc1, rt2, acc2, _, onset2, rt2_tail_i, rt2_from_stim_i = run_prp_trial(  # <-- rename
                task_net, s1, s2, c1, c2, soa,
                max_timesteps=max_timesteps, persistence=persistence,
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                z_task2_fixed=z_task2_fixed, dt_lca=dt_lca, t0=t0,
                optimize_onset=optimize_onset
            )

            if rt1 is not None:
                r1.append(rt1); a1.append(acc1)

            if rt2 is not None:
                r2.append(rt2); a2.append(acc2)

            if rt2_tail_i is not None:
                r2_tail.append(rt2_tail_i)
                onsets.append(onset2)
            if rt2_from_stim_i is not None:
                r2_from_stim.append(rt2_from_stim_i)


        results["soa"].append(soa)
        results["rt_task1"].append(np.mean(r1) if r1 else np.nan)
        results["acc_task1"].append(np.mean(a1) if a1 else np.nan)
        results["rt_task2"].append(np.mean(r2) if r2 else np.nan)
        results["acc_task2"].append(np.mean(a2) if a2 else np.nan)
        results["onset2"].append(np.mean(onsets) if onsets else np.nan)
        results["rt_task2_tail"].append(np.mean(r2_tail) if r2_tail else np.nan)
        results["rt_task2_from_stim"].append(np.mean(r2_from_stim) if r2_from_stim else np.nan)


        if verbose:
            print(f"SOA={soa} | T1 RT={results['rt_task1'][-1]:.2f} "
                f"| T2 RT={results['rt_task2'][-1]:.2f}")

    return results

