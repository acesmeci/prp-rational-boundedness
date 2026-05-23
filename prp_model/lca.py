"""
Leaky Competing Accumulator (LCA) utilities.

This module implements three readout helpers used by the PRP pipeline:

- run_lca:       One stochastic LCA run for a given threshold (single pass).
- run_lca_avg:   Average RT / most-common choice across repeated LCA runs.
- run_lca_dist:  Full threshold sweep (distribution over repeats) returning
                 Acc/RT/Reward-Rate per threshold, used for z-optimization.

Conventions
-----------
- Time is simulated in discrete steps of size `dt` (seconds).
- Reported RTs are in seconds and include non-decision time `t0`.
- `relevant_output_indices` selects the task’s output dimension (e.g., the 3
  units belonging to one response pathway).
- Unless otherwise noted, parameters match the paper defaults:
  lambda_=0.4, alpha=0.2, beta=0.2, noise_std=0.2, t0=0.15, dt=0.1.
"""

import numpy as np

# Don't change lambda = 0.4, alpha = 0.2, beta = 0.2, noise_std = 0.2, t0 = 0.15
# These are the parameters used in the paper
def run_lca(input_series,
            relevant_output_indices,
            dt=0.1,  # 0.1 in MATLAB # UPDATE: It is actually 0.01, and the SOA range is 5:5:60 (range 5-60, steps of 5; corresponds to 50 to 600 ms)
            max_timesteps=100,  # 1000 in MATLAB?, I used 100 for all my simulations, p.45 says 100 sims of LCA
            lambda_=0.4,  # 0.4
            alpha=0.2,  # 0.2
            beta=0.2,  # 0.2 # lateral inhibition: has the highest potential for steep PRP
            noise_std=0.2,  # 0.2 in paper, 0.1 in MATLAB
            threshold=1.0,  # decided by reward maximization (optimize_lca_threshold)
            t0=0.15  # 0.15
           ):
    """
    Run a single LCA trajectory for one task/response dimension.

    Parameters
    ----------
    input_series : array-like, shape [T, D_out]
        Time series of model outputs for all units.
    relevant_output_indices : sequence[int]
        Indices of the output units belonging to the current task’s response
        dimension (e.g., the 3 units for one pathway).
    dt : float
        Simulation time step in seconds.
    max_timesteps : int
        Hard cap on simulation length (steps).
    lambda_, alpha, beta : float
        Leak, self-excitation, and lateral inhibition coefficients.
    noise_std : float
        Gaussian noise scale for the state update.
    threshold : float
        Decision threshold (unitless activation).
    t0 : float
        Non-decision time added to hitting times (seconds).

    Returns
    -------
    rt : float | None
        Decision time in seconds (includes t0). None if no threshold crossing.
    choice : int | None
        Argmax index (within the relevant outputs) at crossing time.
    trajectory : list[np.ndarray]
        State history for the relevant units (one array per step).

    Notes
    -----
    - This is a simple “textbook” LCA: accumulator state `x` is updated with
      leak/self/lateral terms and noise; threshold crossing ends the trial.
    - For threshold selection you will likely prefer `run_lca_dist` which
      averages over many repeats and thresholds.
    """
    n_units = len(relevant_output_indices)
    x = np.zeros(n_units)  # accumulator activations
    trajectory = []
    input_series = np.array(input_series)  # convert to numpy array if not already

    for t in range(min(len(input_series), max_timesteps)):
        inp = input_series[t][relevant_output_indices]
        noise = np.random.randn(n_units) * noise_std

        dx = -lambda_ * x + alpha * x + inp - beta * (np.sum(x) - x) + noise
        x += dt * dx
        trajectory.append(x.copy())

        if np.any(x >= threshold):
            rt = t * dt + t0
            choice = np.argmax(x)
            return rt, choice, trajectory

    return None, None, trajectory


def run_lca_avg(input_series, relevant_output_indices,
                n_repeats=100, dt=0.1, max_timesteps=100,
                lambda_=0.4, alpha=0.2, beta=0.2, #0.2
                noise_std=0.2,  # 0.2 in paper, 0.1 in MATLAB
                threshold=1.0, t0=0.15):
    """
    Repeat `run_lca` and return mean RT and most-common choice.

    Parameters
    ----------
    input_series, relevant_output_indices, dt, max_timesteps, lambda_, alpha,
    beta, threshold, t0 : see `run_lca`.
    n_repeats : int
        Number of independent stochastic runs to average.
    noise_std : float
        Noise level per run. Default (0.1) mirrors the MATLAB variant;
        use 0.2 to match the paper’s main text.

    Returns
    -------
    avg_rt : float | None
        Mean RT across successful runs, or None if no crossings occurred.
    most_common_choice : int | None
        The mode of choices across successful runs, or None if none crossed.

    Notes
    -----
    Only successful runs (threshold crossed) contribute to the mean RT and
    the choice histogram.
    """
    rts = []
    corrects = []
    for _ in range(n_repeats):
        rt, choice, _ = run_lca(
            input_series,
            relevant_output_indices,
            dt=dt,
            max_timesteps=max_timesteps,
            lambda_=lambda_,
            alpha=alpha,
            beta=beta,
            noise_std=noise_std,
            threshold=threshold,
            t0=t0
        )
        if rt is not None:
            rts.append(rt)
            corrects.append(choice)

    if not rts:
        return None, None  # No threshold crossings

    avg_rt = np.mean(rts)
    most_common_choice = max(set(corrects), key=corrects.count)
    return avg_rt, most_common_choice


def run_lca_dist(
    input_series,
    relevant_output_indices,
    thresholds=np.arange(0.1, 1.6, 0.1),
    n_repeats=100,
    dt=0.1,
    tau=0.1,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2, # 0.2
    noise_std=0.2,
    t0=0.15,
    ITI=0.5, #0.5
    correct_response_idx=None,   # true label (relative to relevant indices)
):
    """
    Full LCA sweep across thresholds with repeat sampling (used for RR-optimal z).

    Dynamics
    --------
    Uses a rectified-state variant:
        x_{t+1} = x_t + (I_ext - λ x_t + α f_t + β f_t W_inhib) * (dt/tau) + ξ
        f_t     = max(x_t, 0)

    For each threshold z and each repeat, the process stops on first crossing
    (if any), returning an RT (seconds) and a choice. Accuracy is 1 when
    choice == correct_response_idx, else 0. When no crossing occurs the RT is
    NaN and accuracy is 0. Reward-rate is set to 0 for such cases.

    Parameters
    ----------
    input_series : np.ndarray, shape [T, D_out]
        Output time series for all units.
    relevant_output_indices : sequence[int]
        Indices of the response units for this task (within D_out).
    thresholds : np.ndarray
        Threshold grid to evaluate.
    n_repeats : int
        Number of stochastic LCA runs per threshold.
    dt : float
        LCA step size in seconds (used in RT computation).
    tau : float
        Time constant used to scale updates (dt/tau).
    lambda_, alpha, beta : float
        Leak, self-excitation, and lateral inhibition coefficients.
    noise_std : float
        Gaussian noise scale.
    t0 : float
        Non-decision time added to hitting times (seconds).
    ITI : float
        Inter-trial interval for reward-rate.
    correct_response_idx : int | None
        Correct feature index within the relevant block. If None, falls back
        to argmax at t=0 within that block.

    Returns
    -------
    dict
        {
          "thresholds": np.ndarray [Z],
          "reward_rates": np.ndarray [Z],
          "accuracies":   np.ndarray [Z],
          "rts":          np.ndarray [Z],
          "all_choices":  None,
          "all_rts":      np.ndarray [Z, n_repeats],
          "all_accs":     np.ndarray [Z, n_repeats],
        }

    Notes
    -----
    - Reward-rate is 0 when RT is NaN (no decision), so extreme thresholds
      cannot be selected spuriously.
    - `correct_response_idx` should be passed by the caller (preferred) to
      avoid relying on the argmax fallback at t=0.
    """
    import numpy as np

    input_series = np.array(input_series)
    p = input_series[:, relevant_output_indices]          # [T, n_units]
    n_steps, n_units = p.shape
    n_thresholds = len(thresholds)

    # Lateral inhibition matrix
    W_inhib = -np.ones((n_units, n_units)) + np.eye(n_units)

    # Storage
    all_rts = np.full((n_thresholds, n_repeats), np.nan, dtype=float)
    all_accs = np.zeros((n_thresholds, n_repeats), dtype=float)

    dt_tau = dt / tau
    sqrt_dt_tau = np.sqrt(dt_tau)

    # Label to score against
    if correct_response_idx is None:
        # fallback: most-active at t=0 within relevant block
        correct_response_idx = int(np.argmax(p[0]))

    for ti, z in enumerate(thresholds):
        for rep in range(n_repeats):
            x = np.zeros(n_units)  # state
            f = np.zeros(n_units)  # rectified activation
            noise = noise_std * np.random.randn(n_steps, n_units)

            rt = np.nan
            choice = -1

            for t in range(n_steps):
                I_ext = p[t]
                lateral = beta * f @ W_inhib
                dx = (I_ext - lambda_ * x + alpha * f + lateral) * dt_tau + noise[t] * sqrt_dt_tau
                x += dx
                f = np.maximum(x, 0.0)

                above = np.where(f > z)[0]
                if len(above) > 0:
                    choice = int(np.random.choice(above))
                    rt = t * dt + t0
                    break

            all_rts[ti, rep] = rt
            if not np.isnan(rt):
                all_accs[ti, rep] = 1.0 if (choice == correct_response_idx) else 0.0
            # else accuracy stays 0

    # Aggregate
    accs = np.nanmean(all_accs, axis=1)                   # [n_thresholds]
    rts  = np.nanmean(all_rts,  axis=1)                   # [n_thresholds]

    # Reward-rate: 0 when no decision (rts is NaN)
    reward_rates = np.zeros_like(rts, dtype=float)
    valid = ~np.isnan(rts)
    reward_rates[valid] = accs[valid] / (ITI + rts[valid])

    return {
        "thresholds": np.array(thresholds, dtype=float),
        "reward_rates": reward_rates,
        "accuracies": accs,
        "rts": rts,
        "all_choices": None,
        "all_rts": all_rts,
        "all_accs": all_accs,
    }
