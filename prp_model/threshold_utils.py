"""
Threshold utilities for the LCA readout layer.

This module provides:
- A modern reward-rate threshold optimizer that wraps `run_lca_dist` and
  passes through all LCA parameters (`optimize_lca_threshold_dist`).
- A PRP onset policy helper (`choose_onset_policy`) that evaluates a small
  window of candidate Task-2 onsets using two-pass integration.
- Two legacy helpers retained for reference/debugging.

Conventions:
- Task cues are **row-major** one-hots (index = in_dim * N_pathways + out_dim).
- Reward-rate = accuracy / (ITI + RT). In our `run_lca_dist`, trials with
  no decision yield RT=NaN and are treated as RR=0 so extreme thresholds
  cannot win spuriously.
"""

# Legacy threshold optimizers (currently unused, kept for fallback/debug):
# - optimize_reward_rate_threshold
# - optimize_lca_threshold

import numpy as np
import torch
from prp_model.lca import run_lca_avg
from prp_model.lca import run_lca_dist
from prp_model.training_set import generate_training_set_matlab_style

DEFAULT_N_REPEATS = 100  # Default number of repeats for LCA simulations. Use 100 if you have access to a GPU.


# Not used in the current implementation. Replaced by optimize_lca_threshold_dist.
def optimize_lca_threshold(input_series, relevant_output_indices, correct_response_idx,
                           thresholds=np.arange(0.0, 1.6, 0.1),
                           ITI=0.5,
                           n_repeats=DEFAULT_N_REPEATS):
    """
    [LEGACY] Maximize reward-rate over a grid of thresholds using `run_lca_dist`.

    Parameters
    ----------
    input_series : np.ndarray
        Output time series, shape [T, D_out].
    relevant_output_indices : Sequence[int]
        Indices of the response units for this task (within D_out).
    correct_response_idx : int
        Index of the correct feature **within** the relevant outputs.
        (Ignored by the legacy implementation of `run_lca_dist`.)
    thresholds : np.ndarray
        Candidate thresholds (z) to evaluate.
    ITI : float
        Inter-trial interval (not propagated in this legacy helper).
    n_repeats : int
        Number of LCA simulations per threshold.

    Returns
    -------
    float
        Threshold z that maximizes reward-rate on this series.

    Notes
    -----
    Superseded by `optimize_lca_threshold_dist`, which forwards all LCA
    parameters and ITI properly.
    """
    results = run_lca_dist(
        input_series=input_series,
        relevant_output_indices=relevant_output_indices,
        thresholds=thresholds,
        n_repeats=n_repeats
    )

    reward_rates = results['reward_rates']
    best_idx = np.argmax(reward_rates)
    best_threshold = results['thresholds'][best_idx]

    return best_threshold


# Optimize LCA threshold with run_lca_dist. Faithful to MATLAB implementation
def optimize_lca_threshold_dist(
    input_series,
    relevant_output_indices,
    correct_response_idx=None,
    thresholds=np.arange(0.1, 1.6, 0.1),
    ITI=0.5,
    n_repeats=100,
    dt=0.1,
    tau=0.1,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2,
    noise_std=0.2, #0.2 in paper, check if it is 0.1 in MATLAB
    t0=0.15,
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
        `run_lca_dist` will infer a label (fallback).
    thresholds : np.ndarray
        Threshold grid to test.
    ITI : float
        Inter-trial interval used in reward-rate.
    n_repeats : int
        Number of LCA simulations per threshold.
    dt, tau, lambda_, alpha, beta, noise_std, t0 : float
        LCA parameters (forwarded to `run_lca_dist`).
    verbose : bool
        If True, print per-threshold Acc/RT/RR and the selected z.

    Returns
    -------
    (best_threshold, results) : (float, dict)
        `results` is the dictionary returned by `run_lca_dist` and includes:
        thresholds, reward_rates, accuracies, rts, all_rts, all_accs.

    Notes
    -----
    Our `run_lca_dist` treats “no decision” trials as RR=0, preventing large z
    values from being selected due to NaNs.
    !!!For tiebreak situations the function checks all units above threshold in the same timestep 
    and then randomly picks one. Its not literally the first unit to cross the threshold.!!!
    # This shouldn't be a huge deal but tiebreaks in discretized timesteps *is* possible.
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
        print(f"✅ Best threshold z: {best_threshold:.2f}")

    return best_threshold, results


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
    K=20,
    thresholds=None,
    ITI=0.5,
    n_repeats=100,
    persistence=0.9,
    seed=42,
    verbose=False,
    N_pathways=3,
    N_features=3,
):
    """
    Select a fixed LCA threshold z for one task by argmax of mean reward-rate over K stimuli.

    Samples K single-task patterns for the chosen task, integrates once per stimulus,
    runs optimize_lca_threshold_dist for each, stacks RR curves, takes mean over K,
    then returns the threshold that maximizes this mean RR.

    Parameters
    ----------
    wrapper : TaskNetworkWrapper
        Network wrapper with .integrate(x, t, persistence=...) returning list of tensors.
    task_name : str
        Task label, e.g. "A", "B".
    K : int
        Number of stimuli to sample for the task.
    thresholds : np.ndarray or None
        Threshold grid. If None, uses np.arange(0.1, 1.5, 0.1).
    ITI : float
        Inter-trial interval for reward-rate.
    n_repeats : int
        LCA repeats per threshold per stimulus.
    persistence : float
        Integration persistence.
    seed : int
        Random seed for sampling stimuli.
    verbose : bool
        If True, print selected z.
    N_pathways, N_features : int
        Used for row-major decoding.

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

        rel_idxs, correct_idx = _decode_task(T[k], X[k], N_pathways=N_pathways, N_features=N_features)

        _, res = optimize_lca_threshold_dist(
            out_np,
            rel_idxs,
            correct_response_idx=correct_idx,
            thresholds=thresholds,
            ITI=ITI,
            n_repeats=n_repeats,
        )
        rr_curves.append(res["reward_rates"])

    rr_curves = np.stack(rr_curves, axis=0)
    rr_mean = rr_curves.mean(axis=0)
    z_star = float(thresholds[int(np.argmax(rr_mean))])

    if verbose:
        print(f"Selected fixed z_{task_name} (argmax of mean RR): {z_star:.3f}")
    return z_star


def choose_onset_policy(
    task_net,
    input_a, input_b,
    task_a, task_b,
    soa: int = 0,
    max_onset_delay: int = 5,      # small, fast search window
    max_timesteps: int = 100,
    persistence: float = 0.5,
    ITI: float = 0.5,
    dt_lca: float = 0.1,
    t0: float = 0.15,
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
        RR = (Acc_A * Acc_B) / (ITI + max(RT_A, RT_B_abs)).

    Two-pass evaluation per candidate onset:
      Pass 1: Task-1 cue ON (Task-2 from onset) → get RT_A and gate time for Task-1.
      Pass 2: Turn OFF Task-1 after its decision → evaluate Task-2 on the tail.

    Parameters
    ----------
    task_net : TaskNetworkWrapper
        Must expose .integrate(input_series, task_series, persistence).
    input_a, input_b : np.ndarray
        One-hot stimuli for Task-1/Task-2 (length = N_pathways * N_features).
    task_a, task_b : np.ndarray
        One-hot task cues in **row-major** indexing (length = N_pathways**2).
    soa : int
        Base onset (in LCA steps). Candidate onsets start here.
    max_onset_delay : int
        Search window size in steps (inclusive).
    max_timesteps : int
        Total steps per simulated trial.
    persistence : float
        Carry-over parameter p for the network integration.
    ITI, dt_lca, t0 : float
        Reward-rate / LCA timing parameters.
    z_a_fixed, z_b_fixed : float | None
        If provided, reuse these thresholds to speed up policy evaluation.
    policy_n_repeats : int
        LCA repeats to average per candidate onset.
    thresholds_policy : np.ndarray | None
        Coarse threshold grid for policy evaluation (default 0.1–0.6).

    Returns
    -------
    int
        Best onset (in steps) according to the policy.

    Notes
    -----
    - This function is optional and is usually kept OFF in Sim Study 3 sweeps.
    - Task decoding is **row-major** (no transpose) to stay consistent across modules.
    """
    if thresholds_policy is None:
        # Coarse grid is enough for relative comparisons (faster).
        thresholds_policy = np.linspace(0.1, 0.6, 6)  # full grid would be 0.1..1.6

    def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
        # ROW-MAJOR task cue → (relevant output indices, correct feature)
        mat = task_vec.reshape(N_pathways, N_pathways)
        in_dim, out_dim = np.argwhere(mat == 1)[0]
        correct = np.argmax(input_vec[in_dim*N_features:(in_dim+1)*N_features])
        idxs = list(range(out_dim*N_features, (out_dim+1)*N_features))
        return idxs, correct

    def _integrate_once(onset_b: int, gate_after_a_steps: int | None):
        # Build one trial with optional gating of Task-1 after its decision.
        inp_dim, task_dim = input_a.shape[0], task_a.shape[0]
        input_series, task_series = [], []
        for t in range(max_timesteps):
            stim_t = np.zeros(inp_dim, dtype=np.float32)
            task_t = np.zeros(task_dim, dtype=np.float32)
            # stimuli
            stim_t += input_a
            if t >= soa:
                stim_t += input_b
            # control (task cues)
            if gate_after_a_steps is None or t < gate_after_a_steps:
                task_t += task_a
            if t >= onset_b:
                task_t += task_b
            input_series.append(stim_t)
            task_series.append(task_t)
        input_np = np.stack(input_series, axis=0).astype(np.float32)
        task_np  = np.stack(task_series,  axis=0).astype(np.float32)
        out_th = task_net.integrate(torch.from_numpy(input_np),
                                    torch.from_numpy(task_np),
                                    persistence=persistence)
        return np.stack([o.numpy() for o in out_th], axis=0)  # [T, D_out]

    # Decode once (row-major)
    idxs_a, corr_a = _decode(task_a, input_a)
    idxs_b, corr_b = _decode(task_b, input_b)

    best_rr = -np.inf
    best_onset = int(soa)
    onset_upper = min(int(soa) + int(max_onset_delay), max_timesteps - 1)

    for onset in range(int(soa), onset_upper + 1):
        # ---- Pass 1: A timing ----
        out1 = _integrate_once(onset_b=onset, gate_after_a_steps=None)
        if z_a_fixed is None:
            z_a, _ = optimize_lca_threshold_dist(
                out1, idxs_a,
                correct_response_idx=corr_a,
                thresholds=thresholds_policy,
                ITI=ITI,
                n_repeats=policy_n_repeats
            )
        else:
            z_a = z_a_fixed
        rt_a, choice_a = run_lca_avg(out1, idxs_a, threshold=z_a,
                                     n_repeats=policy_n_repeats, dt=dt_lca)
        if rt_a is None:
            continue
        acc_a = 1.0 if choice_a == corr_a else 0.0
        # Convert RT (sec) to step index for gating Task-1 off
        t_off_a = int(np.ceil(max(0.0, (rt_a - t0) / dt_lca)))

        # ---- Pass 2: gate A then evaluate B ----
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
                n_repeats=policy_n_repeats
            )
        else:
            z_b = z_b_fixed
        rt_b, choice_b = run_lca_avg(tail, idxs_b, threshold=z_b,
                                     n_repeats=policy_n_repeats, dt=dt_lca)
        if rt_b is None:
            continue
        acc_b = 1.0 if choice_b == corr_b else 0.0
        rt_b_abs = rt_b + onset * dt_lca  # convert to absolute time

        rr = (acc_a * acc_b) / (ITI + max(rt_a, rt_b_abs))
        if rr > best_rr:
            best_rr = rr
            best_onset = onset

    return int(best_onset)



def optimize_reward_rate_threshold(net, input_a, input_b, task_a, task_b,
                                   soa, max_timesteps=100,
                                   thresholds=np.arange(0.0, 1.6, 0.1),
                                   tau_net=0.2, tau_task=0.2, persistence=0.5,
                                   ITI=0.5,
                                   n_repeats=DEFAULT_N_REPEATS):
    """
    [LEGACY] Find the best Task-1 threshold (z) by simulating a PRP trial once.

    Parameters mirror older MATLAB code and an older wrapper API:
    - Expects `net.integrate` to accept `tau_net`/`tau_task`.
    - Uses a transpose when decoding the task cue (column-major assumption).
    - Calls `optimize_lca_threshold_dist` with a keyword `output_series`.
      (Kept for reference; not used in the current pipeline.)

    Returns
    -------
    float
        Best z for Task-1 according to reward-rate on that single synthetic trial.

    Notes
    -----
    This function is **not** wired into the current PRP sweeps and may be
    incompatible with the modern `TaskNetworkWrapper`. Retained only to aid
    debugging/porting.
    """
    input_dim = input_a.shape[0]
    task_dim = task_a.shape[0]

    onset_b = choose_onset_policy(
        task_net=net,
        input_a=input_a,
        input_b=input_b,
        task_a=task_a,
        task_b=task_b,
        soa=soa,
        tau_net=tau_net,
        tau_task=tau_task,
        persistence=persistence,
        ITI=ITI
    )

    # Create full input + task series
    input_series, task_series = [], []
    for t in range(max_timesteps):
        stim_t = np.zeros(input_dim)
        task_t = np.zeros(task_dim)

        if t >= 0:
            stim_t += input_a
            task_t += task_a
        if t >= onset_b:
            stim_t += input_b
            task_t += task_b

        input_series.append(stim_t)
        task_series.append(task_t)

    # Run integration
    output_series = net.integrate(
        input_series, task_series,
        tau_net=tau_net,
        tau_task=tau_task,
        persistence=persistence
    )

    # Target output for Task A (first) — legacy column-major decoding (uses .T)
    N_pathways = 3
    N_features = 3
    task_matrix = task_a.reshape(N_pathways, N_pathways).T
    in_dim, out_dim = np.argwhere(task_matrix == 1)[0]
    output_idxs = list(range(out_dim * N_features, (out_dim + 1) * N_features))
    correct_idx = np.argmax(input_a[in_dim * N_features:(in_dim + 1) * N_features])

    # Use full LCA reward-rate scan
    best_z, _ = optimize_lca_threshold_dist(
        output_series=output_series,   # legacy keyword; kept as-is
        relevant_output_indices=output_idxs,
        correct_response_idx=correct_idx,
        thresholds=thresholds,
        ITI=ITI,
        n_repeats=n_repeats,
    )

    return best_z
