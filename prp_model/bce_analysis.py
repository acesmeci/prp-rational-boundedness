"""
bce_analysis.py
---------------
Backward Crosstalk Effect (BCE) analysis using the existing PRP pipeline.

BCE = RT1_incongruent - RT1_congruent at a given SOA. If Task 2's stimulus
properties influence Task 1's RT before Task 1 has finished responding,
information is flowing backward through shared representations.

The RBA predicts:
  - BCE should be present at short SOAs (where Task 2's cue is active
    before Task 1 responds) and disappear at long SOAs.
  - BCE should be larger for functionally dependent pairs (B->A) than
    for independent pairs (C->A), because the hidden route carrying
    backward crosstalk is stronger for dependent tasks.

This module wraps the existing run_prp_trial and sweep_soa machinery,
adding congruency classification to split RT1 and RT2 by congruency.

No changes to the simulation code; BCE falls out of the same model
that produces PRP curves.
"""

import numpy as np
from prp_model.prp_simulator import run_prp_trial, DEFAULT_N_REPEATS
from prp_model.lca import _DEFAULTS
from prp_model.utils import (
    TASK_MAP, N_PATHWAYS, N_FEATURES,
    generate_trial_pair, prp_trial_congruency,
)

# One z1 per task pair, optimized across SOAs and congruencies jointly
def calibrate_z1(
    task_net,
    prp_pair: tuple[str, str] = ("B", "A"),
    soa_values: list[int] = [1, 5, 11],
    n_trials: int = 30,
    persistence: float = 0.65,
    thresholds: np.ndarray = np.arange(0.1, 1.6, 0.1),
    ITI: float = 1.8,
    n_repeats: int = 100,
    dt_lca: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    base_seed: int = 0,
    max_timesteps: int = 100,
    acc_floor: float = 0.98,
    verbose: bool = True,
) -> float:
    """
    Find a single z1 for a task pair by pooling reward-rate curves across
    SOAs and congruencies, subject to an accuracy floor.

    The accuracy floor prevents the optimizer from picking very low
    thresholds where the LCA commits before backward crosstalk has
    time to manifest. This matches the PRP pipeline's approach in
    compute_condition_thresholds.
    """
    from prp_model.prp_simulator import _decode
    from prp_model.threshold_utils import optimize_lca_threshold_dist

    rr_curves = []
    acc_curves = []

    for soa in soa_values:
        for j in range(n_trials):
            seed = base_seed + soa * 1000 + j
            s1, s2, c1, c2 = generate_trial_pair(prp_pair, seed=seed)

            I, T = s1.shape[0], c1.shape[0]
            inp, cue = [], []
            for t in range(max_timesteps):
                s = np.zeros(I, np.float32) + s1
                if t >= soa:
                    s += s2
                c = np.zeros(T, np.float32) + c1
                if t >= soa:
                    c += c2
                inp.append(s); cue.append(c)

            import torch
            x = np.stack(inp).astype(np.float32)
            tt = np.stack(cue).astype(np.float32)
            outs = task_net.integrate(
                torch.from_numpy(x), torch.from_numpy(tt),
                persistence=persistence,
            )
            out_np = np.stack([o.numpy() for o in outs], axis=0)

            idxs1, corr1 = _decode(c1, s1)
            _, res = optimize_lca_threshold_dist(
                out_np, idxs1, correct_response_idx=corr1,
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                dt=dt_lca, tau=tau, noise_std=noise_std,
            )
            rr_curves.append(res["reward_rates"])
            acc_curves.append(res["accuracies"])

    mean_rr = np.stack(rr_curves).mean(axis=0)
    mean_acc = np.stack(acc_curves).mean(axis=0)

    # Constrained argmax: best RR where accuracy >= floor
    valid = mean_acc >= acc_floor
    if valid.any():
        # Among thresholds meeting the floor, pick highest RR
        masked_rr = np.where(valid, mean_rr, -np.inf)
        best_idx = int(np.argmax(masked_rr))
    else:
        # No threshold meets floor — pick highest accuracy
        best_idx = int(np.argmax(mean_acc))
        if verbose:
            print(f"  WARNING: no threshold meets acc >= {acc_floor:.2f}, "
                  f"best acc = {mean_acc[best_idx]:.3f}")

    z_opt = float(thresholds[best_idx])

    if verbose:
        print(f"Calibrated z1 for {prp_pair[0]}->{prp_pair[1]}: "
              f"z={z_opt:.2f} (RR={mean_rr[best_idx]:.3f}, "
              f"Acc={mean_acc[best_idx]:.3f}, floor={acc_floor}) "
              f"[{len(rr_curves)} trials across {len(soa_values)} SOAs]")

    return z_opt

def sweep_soa_bce(
    task_net,
    prp_pair: tuple[str, str] = ("B", "A"),
    soa_values: list[int] | np.ndarray = None,
    n_trials_per_soa: int = 30,
    max_timesteps: int = 100,
    persistence: float = 0.65,
    n_repeats: int = DEFAULT_N_REPEATS,
    z_task1_fixed: float | None = None,
    z_task2_fixed: float | None = None,
    dt_lca: float = _DEFAULTS["dt"],
    tau: float = _DEFAULTS["tau"],
    t0: float = _DEFAULTS["t0"],
    noise_std: float = _DEFAULTS["noise_std"],
    ITI: float = 1.8,
    thresholds: np.ndarray = np.arange(0.1, 1.6, 0.1),
    optimize_onset: bool = False,
    max_onset_delay: int = 15,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Run PRP sweep with congruency-split RT1 and RT2 for BCE analysis.

    Parameters
    ----------
    task_net : TaskNetworkWrapper
        Trained network.
    prp_pair : (str, str)
        (Task1, Task2), e.g. ("B", "A").
    soa_values : list[int]
        SOA values in simulation steps. If None, uses a default grid.
    n_trials_per_soa : int
        Total trials per SOA (split ~1/3 congruent, ~2/3 incongruent).

    Returns
    -------
    dict with keys:
        soa                     : list[int]
        rt1_congruent           : list[float]  mean RT1 on congruent trials
        rt1_incongruent         : list[float]  mean RT1 on incongruent trials
        rt1_congruent_se        : list[float]
        rt1_incongruent_se      : list[float]
        bce                     : list[float]  RT1_inc - RT1_con per SOA
        bce_se                  : list[float]
        rt2_congruent           : list[float]  mean RT2 (from stim, correct)
        rt2_incongruent         : list[float]
        rt2_congruent_se        : list[float]
        rt2_incongruent_se      : list[float]
        acc1_congruent          : list[float]
        acc1_incongruent        : list[float]
        acc2_congruent          : list[float]
        acc2_incongruent        : list[float]
        n_congruent             : list[int]   trials per SOA
        n_incongruent           : list[int]
    """
    if soa_values is None:
        soa_values = [1, 3, 5, 8, 11, 16]

    results = {
        "soa": [],
        "rt1_congruent": [], "rt1_incongruent": [],
        "rt1_congruent_se": [], "rt1_incongruent_se": [],
        "bce": [], "bce_se": [],
        "rt2_congruent": [], "rt2_incongruent": [],
        "rt2_congruent_se": [], "rt2_incongruent_se": [],
        "acc1_congruent": [], "acc1_incongruent": [],
        "acc2_congruent": [], "acc2_incongruent": [],
        "n_congruent": [], "n_incongruent": [],
    }

    for soa in soa_values:
        rt1_con, rt1_inc = [], []
        rt2_con, rt2_inc = [], []
        acc1_con, acc1_inc = [], []
        acc2_con, acc2_inc = [], []

        for j in range(n_trials_per_soa):
            trial_seed = base_seed + soa * 1000 + j
            s1, s2, c1, c2 = generate_trial_pair(prp_pair, seed=trial_seed)
            is_cong = prp_trial_congruency(prp_pair, seed=trial_seed)

            tr = run_prp_trial(
                task_net, s1, s2, c1, c2, soa,
                max_timesteps=max_timesteps, persistence=persistence,
                thresholds=thresholds, ITI=ITI, n_repeats=n_repeats,
                z_task1_fixed=z_task1_fixed, z_task2_fixed=z_task2_fixed,
                dt_lca=dt_lca, tau=tau, t0=t0, noise_std=noise_std,
                optimize_onset=optimize_onset,
                max_onset_delay=max_onset_delay,
            )

            # Collect RT1
            if tr["rt1_correct"] is not None:
                if is_cong:
                    rt1_con.append(tr["rt1_correct"])
                    acc1_con.append(tr["acc1"])
                else:
                    rt1_inc.append(tr["rt1_correct"])
                    acc1_inc.append(tr["acc1"])

            # Collect RT2
            if tr["rt2_from_stim_correct"] is not None:
                if is_cong:
                    rt2_con.append(tr["rt2_from_stim_correct"])
                    acc2_con.append(tr["acc2"])
                else:
                    rt2_inc.append(tr["rt2_from_stim_correct"])
                    acc2_inc.append(tr["acc2"])

        def _ms(vals):
            if not vals:
                return np.nan, np.nan
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            return m, se

        r1c_m, r1c_se = _ms(rt1_con)
        r1i_m, r1i_se = _ms(rt1_inc)
        r2c_m, r2c_se = _ms(rt2_con)
        r2i_m, r2i_se = _ms(rt2_inc)

        # BCE = RT1_inc - RT1_con
        if np.isfinite(r1i_m) and np.isfinite(r1c_m):
            bce_val = r1i_m - r1c_m
            # SE of difference (assuming independence)
            bce_se = np.sqrt(r1c_se**2 + r1i_se**2) if np.isfinite(r1c_se) and np.isfinite(r1i_se) else np.nan
        else:
            bce_val, bce_se = np.nan, np.nan

        results["soa"].append(soa)
        results["rt1_congruent"].append(r1c_m)
        results["rt1_incongruent"].append(r1i_m)
        results["rt1_congruent_se"].append(r1c_se)
        results["rt1_incongruent_se"].append(r1i_se)
        results["bce"].append(bce_val)
        results["bce_se"].append(bce_se)
        results["rt2_congruent"].append(r2c_m)
        results["rt2_incongruent"].append(r2i_m)
        results["rt2_congruent_se"].append(r2c_se)
        results["rt2_incongruent_se"].append(r2i_se)
        results["acc1_congruent"].append(float(np.mean(acc1_con)) if acc1_con else np.nan)
        results["acc1_incongruent"].append(float(np.mean(acc1_inc)) if acc1_inc else np.nan)
        results["acc2_congruent"].append(float(np.mean(acc2_con)) if acc2_con else np.nan)
        results["acc2_incongruent"].append(float(np.mean(acc2_inc)) if acc2_inc else np.nan)
        results["n_congruent"].append(len(rt1_con))
        results["n_incongruent"].append(len(rt1_inc))

        if verbose:
            soa_ms = int(soa * dt_lca * 1000)
            print(f"SOA={soa_ms:>4d}ms | "
                  f"RT1 con={r1c_m:.3f} inc={r1i_m:.3f} BCE={bce_val:+.3f} | "
                  f"RT2 con={r2c_m:.3f} inc={r2i_m:.3f} | "
                  f"n={len(rt1_con)}/{len(rt1_inc)}")

    return results


def run_bce_comparison(
    task_net,
    pairs: list[tuple[str, str, str]] | None = None,
    soa_values: list[int] | np.ndarray = None,
    n_trials_per_soa: int = 30,
    persistence: float = 0.65,
    z_task1_fixed: float | None = None,
    z_task2_fixed: float | None = None,
    noise_std: float = _DEFAULTS["noise_std"],
    ITI: float = 1.8,
    thresholds: np.ndarray = np.arange(0.1, 1.6, 0.1),
    base_seed: int = 0,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Run BCE analysis across multiple task pairs for comparison.

    Default pairs: B->A (functional) and C->A (independent).
    The RBA predicts larger BCE for B->A than C->A.

    Returns
    -------
    dict mapping labels to sweep_soa_bce results.
    """
    if pairs is None:
        pairs = [
            ("B", "A", "functional"),
            ("C", "A", "independent"),
        ]

    if soa_values is None:
        soa_values = [1, 3, 5, 8, 11, 16]

    all_results = {}

    for t1, t2, label in pairs:
        if verbose:
            print(f"\n{'='*50}")
            print(f"{label}: {t1} -> {t2}")
            print(f"{'='*50}")

        # Calibrate session-level z1 for this pair
        z1 = calibrate_z1(
            task_net, prp_pair=(t1, t2),
            soa_values=[soa_values[0], soa_values[len(soa_values)//2], soa_values[-1]],
            n_trials=20, persistence=persistence,
            noise_std=noise_std, ITI=ITI, thresholds=thresholds,
            base_seed=base_seed + 99000,  # different seeds from measurement
            verbose=verbose,
        )

        res = sweep_soa_bce(
            task_net, prp_pair=(t1, t2),
            soa_values=soa_values,
            n_trials_per_soa=n_trials_per_soa,
            persistence=persistence,
            z_task1_fixed=z1,
            z_task2_fixed=z_task2_fixed,
            noise_std=noise_std,
            ITI=ITI,
            thresholds=thresholds,
            base_seed=base_seed,
            verbose=verbose,
            **kwargs,
        )
        all_results[label] = res

    # Summary
    if verbose:
        print(f"\n{'='*50}")
        print("BCE SUMMARY (mean across SOAs)")
        print(f"{'='*50}")
        for label, res in all_results.items():
            bce_vals = [b for b in res["bce"] if np.isfinite(b)]
            mean_bce = float(np.mean(bce_vals)) if bce_vals else np.nan
            print(f"  {label:12s}: mean BCE = {mean_bce:+.4f}s "
                  f"({mean_bce*1000:+.1f}ms)")

    return all_results