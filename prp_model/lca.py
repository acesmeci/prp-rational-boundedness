"""
Leaky Competing Accumulator (LCA) — unified, clock-faithful implementation.

Implements the LCA decision process from Musslick et al. (2023), Eq. 4:

    dr_i = [y_o - λ r_i + α f(r_i) - β Σ_{j≠i} f(r_j)] · (dt/τ)
           + ξ_i · √(dt/τ)

where f(r) = max(r, 0)  (rectified activation).

This module provides three public functions, all sharing the same core
dynamics (_lca_step):

    run_lca       – Single stochastic LCA trajectory → (rt, choice, trajectory).
    run_lca_avg   – Average RT / mode choice over N repeats → (mean_rt, mode_choice).
    run_lca_dist  – Full threshold sweep with repeat sampling → dict of
                    Acc / RT / Reward-Rate per threshold (used for z-optimization).

Conventions
-----------
- Each simulation step corresponds to dt = 0.05 physical seconds (50 ms).
- Returned RTs are in seconds: (completed_steps) * dt + t0.
- `relevant_output_indices` selects the output units belonging to one
  response dimension (e.g. the 3 units for one pathway).

Default parameters (paper, p. 45):
    λ = 0.4,  α = 0.2,  β = 0.2,  σ = 0.2,  t0 = 0.15,
    dt = 0.05,  τ = 0.5  (effective dt/τ = 0.1),  ITI = 1.8

Note on dt/τ: The MATLAB repository uses dt = 0.01, τ = 0.1 (dt/τ = 0.1),
where each step is 10 ms. Our implementation uses dt = 0.05, τ = 0.5
(dt/τ = 0.1) so that each step is 50 ms. The per-iteration accumulation
dynamics are identical (same dt/τ ratio); only the physical duration
assigned to each step differs. All returned times are in seconds.
"""

import numpy as np


# ── Default LCA parameters ───────────────────────────────────────────────────
# λ, α, β, σ from paper p. 45.  dt = 0.05, τ = 0.5 gives dt/τ = 0.1,
# matching the MATLAB's per-iteration accumulation rate (dt=0.01, τ=0.1)
# while assigning each step a physical duration of 50 ms.
_DEFAULTS = dict(
    dt=0.05,
    tau=0.5,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2,
    noise_std=0.2,
    t0=0.15,
    max_timesteps=100,
)


# ── Core single-trial LCA ───────────────────────────────────────────────────

def run_lca(
    input_series,
    relevant_output_indices,
    threshold=1.0,
    dt=_DEFAULTS["dt"],
    tau=_DEFAULTS["tau"],
    max_timesteps=_DEFAULTS["max_timesteps"],
    lambda_=_DEFAULTS["lambda_"],
    alpha=_DEFAULTS["alpha"],
    beta=_DEFAULTS["beta"],
    noise_std=_DEFAULTS["noise_std"],
    t0=_DEFAULTS["t0"],
):
    """
    Run a single stochastic LCA trajectory for one response dimension.

    Parameters
    ----------
    input_series : array-like, shape [T, D_out]
        Time series of network output activations for all units.
    relevant_output_indices : sequence of int
        Indices of the output units belonging to the current task's response
        dimension (e.g. the 3 units for one pathway).
    threshold : float
        Decision threshold z (unitless activation level).
    dt : float
        Simulation time-step size.
    tau : float
        Time constant for scaling updates.
    max_timesteps : int
        Hard cap on simulation length (in LCA steps).
    lambda_, alpha, beta : float
        Leak, self-excitation, and lateral-inhibition coefficients.
    noise_std : float
        Standard deviation of Gaussian noise (σ in the paper).
    t0 : float
        Non-decision time added to threshold-crossing times.

    Returns
    -------
    rt : float or None
        Decision time in seconds: (completed_steps) * dt + t0.
        None if threshold not reached.
    choice : int or None
        Index (within relevant outputs) of the winning accumulator.
    trajectory : list of np.ndarray
        Rectified state f(x) at each step, for the relevant units.
    """
    input_series = np.asarray(input_series)
    p = input_series[:, relevant_output_indices]
    n_steps = min(len(p), max_timesteps)
    n_units = p.shape[1]

    dt_tau = dt / tau
    sqrt_dt_tau = np.sqrt(dt_tau)

    # Lateral inhibition matrix: W_inhib[i,j] = -1 for i≠j, 0 for i==j
    W_inhib = -np.ones((n_units, n_units)) + np.eye(n_units)

    x = np.zeros(n_units)           # accumulator state
    f = np.zeros(n_units)           # rectified activation
    trajectory = []

    for t in range(n_steps):
        noise = noise_std * np.random.randn(n_units) * sqrt_dt_tau
        lateral = beta * f @ W_inhib
        dx = (p[t] - lambda_ * x + alpha * f + lateral) * dt_tau + noise
        x += dx
        f = np.maximum(x, 0.0)
        trajectory.append(f.copy())

        above = np.where(f >= threshold)[0]
        if len(above) > 0:
            choice = int(np.random.choice(above)) if len(above) > 1 else int(above[0])
            rt = (t + 1) * dt + t0
            return rt, choice, trajectory

    return None, None, trajectory


# ── Averaged single-trial LCA ───────────────────────────────────────────────

def run_lca_avg(
    input_series,
    relevant_output_indices,
    threshold=1.0,
    n_repeats=100,
    dt=_DEFAULTS["dt"],
    tau=_DEFAULTS["tau"],
    max_timesteps=_DEFAULTS["max_timesteps"],
    lambda_=_DEFAULTS["lambda_"],
    alpha=_DEFAULTS["alpha"],
    beta=_DEFAULTS["beta"],
    noise_std=_DEFAULTS["noise_std"],
    t0=_DEFAULTS["t0"],
    correct_response_idx=None,
):
    """
    Repeat ``run_lca`` over the same output series and aggregate.

    Returns
    -------
    dict with keys:
        rt           : float | None  mean RT over ALL decided repeats
                       (errors included; used for cue-gating timing)
        rt_correct   : float | None  mean RT over CORRECT decided repeats only
                       (the empirical-convention dependent measure; None if
                       correct_response_idx not given or no correct repeat)
        choice       : int | None    modal choice across decided repeats
        p_correct    : float | None  fraction of decided repeats won by the
                       correct accumulator (needs correct_response_idx)
        frac_decided : float         fraction of repeats with any crossing
    """
    rts, choices = [], []

    for _ in range(n_repeats):
        rt, choice, _ = run_lca(
            input_series, relevant_output_indices,
            threshold=threshold,
            dt=dt, tau=tau, max_timesteps=max_timesteps,
            lambda_=lambda_, alpha=alpha, beta=beta,
            noise_std=noise_std, t0=t0,
        )
        if rt is not None:
            rts.append(rt)
            choices.append(choice)

    out = {"rt": None, "rt_correct": None, "choice": None,
           "p_correct": None, "frac_decided": len(rts) / float(n_repeats)}
    if not rts:
        return out

    rts = np.asarray(rts, float)
    choices_arr = np.asarray(choices, int)

    out["rt"] = float(np.mean(rts))
    out["choice"] = int(max(set(choices), key=choices.count))

    if correct_response_idx is not None:
        correct_mask = choices_arr == int(correct_response_idx)
        out["p_correct"] = float(np.mean(correct_mask))
        if correct_mask.any():
            out["rt_correct"] = float(np.mean(rts[correct_mask]))

    return out

# ── Threshold sweep (for reward-rate optimization) ──────────────────────────

def run_lca_dist(
    input_series,
    relevant_output_indices,
    correct_response_idx=None,
    thresholds=np.arange(0.1, 1.6, 0.1),
    n_repeats=100,
    dt=_DEFAULTS["dt"],
    tau=_DEFAULTS["tau"],
    lambda_=_DEFAULTS["lambda_"],
    alpha=_DEFAULTS["alpha"],
    beta=_DEFAULTS["beta"],
    noise_std=_DEFAULTS["noise_std"],
    t0=_DEFAULTS["t0"],
    ITI=0.5,
):
    """
    Sweep thresholds with repeated LCA runs, returning per-threshold
    accuracy, RT, and reward-rate.

    This is the function used by ``optimize_lca_threshold_dist`` to find
    the reward-rate-maximising threshold z.

    Parameters
    ----------
    input_series : np.ndarray, shape [T, D_out]
        Output time series for all units.
    relevant_output_indices : sequence of int
        Indices of the response units for this task.
    correct_response_idx : int or None
        Correct feature index within the relevant outputs. If None,
        falls back to argmax of the first timestep (not recommended).
    thresholds : np.ndarray
        Threshold grid to evaluate.
    n_repeats : int
        Stochastic LCA runs per threshold.
    dt, tau, lambda_, alpha, beta, noise_std, t0 :
        LCA parameters (see ``run_lca``).
    ITI : float
        Inter-trial interval for reward-rate = acc / (ITI + RT).

    Returns
    -------
    dict with keys:
        thresholds    : np.ndarray [Z]
        reward_rates  : np.ndarray [Z]
        accuracies    : np.ndarray [Z]
        rts           : np.ndarray [Z]       (mean RT per threshold)
        all_rts       : np.ndarray [Z, R]    (per-repeat RTs)
        all_accs      : np.ndarray [Z, R]    (per-repeat accuracies)

    Notes
    -----
    Trials where no threshold is reached yield RT = NaN and acc = 0.
    Reward-rate is set to 0 for such cases, preventing extreme thresholds
    from winning spuriously.
    """
    input_series = np.asarray(input_series)
    p = input_series[:, relevant_output_indices]
    n_steps, n_units = p.shape
    n_thresholds = len(thresholds)

    dt_tau = dt / tau
    sqrt_dt_tau = np.sqrt(dt_tau)
    W_inhib = -np.ones((n_units, n_units)) + np.eye(n_units)

    if correct_response_idx is None:
        correct_response_idx = int(np.argmax(p[0]))

    all_rts = np.full((n_thresholds, n_repeats), np.nan)
    all_accs = np.zeros((n_thresholds, n_repeats))

    for ti, z in enumerate(thresholds):
        for rep in range(n_repeats):
            x = np.zeros(n_units)
            f = np.zeros(n_units)
            noise = noise_std * np.random.randn(n_steps, n_units)

            for t in range(n_steps):
                lateral = beta * f @ W_inhib
                dx = (p[t] - lambda_ * x + alpha * f + lateral) * dt_tau \
                     + noise[t] * sqrt_dt_tau
                x += dx
                f = np.maximum(x, 0.0)

                above = np.where(f >= z)[0]
                if len(above) > 0:
                    choice = int(np.random.choice(above))
                    all_rts[ti, rep] = (t + 1) * dt + t0
                    all_accs[ti, rep] = 1.0 if choice == correct_response_idx else 0.0
                    break

    # Aggregate
    accs = np.nanmean(all_accs, axis=1)
    rts = np.nanmean(all_rts, axis=1)

    reward_rates = np.zeros_like(rts)
    valid = ~np.isnan(rts)
    reward_rates[valid] = accs[valid] / (ITI + rts[valid])

    return {
        "thresholds": np.array(thresholds, dtype=float),
        "reward_rates": reward_rates,
        "accuracies": accs,
        "rts": rts,
        "all_rts": all_rts,
        "all_accs": all_accs,
    }