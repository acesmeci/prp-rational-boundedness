#!/usr/bin/env python3
import json, argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from prp.nn_wrapper import TaskNetworkWrapper
from prp.prp_simulator import sweep_soa


# ---------------------------
# Trial generator (copy from notebook)
# ---------------------------
def generate_trial_pair(prp_pair=("B","A"), N_pathways=3, N_features=3, seed=None):
    task_map = {'A': (0,0), 'B': (1,1), 'C': (2,2), 'D': (0,1), 'E': (1,0)}
    rng = np.random.RandomState(seed)

    def sample_single_task(task_name, shared_features=None):
        in_dim, out_dim = task_map[task_name]
        feats = shared_features if shared_features is not None \
                else rng.randint(0, N_features, size=N_pathways)

        stim = np.zeros(N_pathways*N_features, dtype=np.float32)
        for i in range(N_pathways):
            stim[i*N_features + feats[i]] = 1

        cue = np.zeros(N_pathways**2, dtype=np.float32)
        cue[in_dim*N_pathways + out_dim] = 1
        return stim, cue

    feats = rng.randint(0, N_features, size=N_pathways)
    stim1, cue1 = sample_single_task(prp_pair[0], shared_features=feats)
    stim2, cue2 = sample_single_task(prp_pair[1], shared_features=feats)
    return stim1, stim2, cue1, cue2


# ---------------------------
# Stats helpers
# ---------------------------
def nanmean(a):
    a = np.asarray(a, float)
    return np.nanmean(a, axis=0)

def nanse(a):
    a = np.asarray(a, float)
    # standard error ignoring NaNs
    n = np.sum(np.isfinite(a), axis=0)
    sd = np.nanstd(a, axis=0, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))

def steepest_adjacent_slope(soa_steps, y, dt_lca):
    """Return the most negative adjacent-pair slope in s/s (and its segment)."""
    soa_steps = np.asarray(soa_steps, float)
    y = np.asarray(y, float)

    m = np.isfinite(soa_steps) & np.isfinite(y)
    soa_steps, y = soa_steps[m], y[m]
    order = np.argsort(soa_steps)
    soa_steps, y = soa_steps[order], y[order]

    dsoa = np.diff(soa_steps)          # steps
    dy   = np.diff(y)                  # seconds
    slope_s_per_step = dy / dsoa
    slope_s_per_s    = slope_s_per_step / dt_lca

    i = int(np.nanargmin(slope_s_per_s))  # most negative
    return {
        "seg": (float(soa_steps[i]), float(soa_steps[i+1])),
        "slope_s_per_s": float(slope_s_per_s[i]),
    }


# ---------------------------
# I/O helpers
# ---------------------------
def load_threshold_json(path):
    with open(path, "r") as f:
        return float(json.load(f)["z"])

def load_model(make_wrapper_fn, model_path, device="cpu"):
    wrapper = make_wrapper_fn()
    state = torch.load(model_path, map_location=device)
    wrapper.model.load_state_dict(state)
    wrapper.model.eval()
    return wrapper


def _make_wrapper():
    """Default wrapper constructor (must be at module level for pickling in multiprocessing)."""
    return TaskNetworkWrapper(
        stim_input_dim=9, task_input_dim=9, hidden_dim=100, output_dim=9,
        learning_rate=0.3,
        init_scale=0.1,
        init_task_scale=None,
        bias_offset=-2.0,
        default_weight_decay=0.0,
        device="cpu",
    )


def _sweep_one_network(args_tuple):
    """
    Run PRP sweeps for one network. Top-level for multiprocessing (picklable).
    args_tuple: (idx, ckpt_dir_str, z_task, soa, trials_per_soa, persistence, dt_lca, t0, ITI, optimize_onset)
    Returns: (idx, z_A, dep_rt2, ind_rt2) or (idx, None, None, None) if model/z missing.
    """
    (idx, ckpt_dir_str, z_task, soa, trials_per_soa, persistence,
     dt_lca, t0, ITI, optimize_onset) = args_tuple
    ckpt_dir = Path(ckpt_dir_str)
    model_path = ckpt_dir / f"net_{idx:02d}.pt"
    z_path = ckpt_dir / f"net_{idx:02d}_z_{z_task}.json"
    if not model_path.exists() or not z_path.exists():
        return (idx, None, None, None)
    wrapper = load_model(_make_wrapper, str(model_path))
    z_A = load_threshold_json(str(z_path))
    gen_dep = lambda: generate_trial_pair(("B", "A"))
    gen_ind = lambda: generate_trial_pair(("C", "A"))
    results_ba = sweep_soa(
        wrapper, gen_dep, soa,
        n_trials_per_soa=trials_per_soa,
        persistence=persistence,
        dt_lca=dt_lca, t0=t0, ITI=ITI,
        z_task2_fixed=z_A,
        optimize_onset=optimize_onset,
    )
    results_ca = sweep_soa(
        wrapper, gen_ind, soa,
        n_trials_per_soa=trials_per_soa,
        persistence=persistence,
        dt_lca=dt_lca, t0=t0, ITI=ITI,
        z_task2_fixed=z_A,
        optimize_onset=optimize_onset,
    )
    return (idx, z_A, results_ba["rt_task2_from_stim"], results_ca["rt_task2_from_stim"])


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Directory containing net_XX.pt and net_XX_z_A.json")
    ap.add_argument("--E", type=int, default=None,
                    help="How many nets to use. Default: infer from files.")
    ap.add_argument("--z_task", type=str, default="A")
    ap.add_argument("--dt_lca", type=float, default=0.1)
    ap.add_argument("--t0", type=float, default=0.15)
    ap.add_argument("--ITI", type=float, default=0.5)
    ap.add_argument("--persistence", type=float, default=0.90)
    ap.add_argument("--trials_per_soa", type=int, default=30)
    ap.add_argument("--soa_start", type=int, default=5)
    ap.add_argument("--soa_end", type=int, default=60)
    ap.add_argument("--soa_step", type=int, default=5)
    ap.add_argument("--optimize_onset", action="store_true")
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel workers for sweep (0=serial). Default 6.")
    ap.add_argument("--out_png", type=str, default="prp_ensemble_rt2_from_stim.png")
    
    # --- New Pashler-related Arguments ---
    ap.add_argument("--add_pashler", action="store_true", 
                    help="Overlay the Pashler (1994) Figure 1 empirical curve.")
    ap.add_argument("--pashler_mode", type=str, choices=["absolute", "relative"], default="absolute",
                    help="absolute: starts at 50ms; relative: starts at the simulation's first SOA.")
    ap.add_argument("--align_pashler_rt", action="store_true",
                    help="Vertically shift Pashler curve to match simulation RT2 at the first point.")
    
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)

    # Infer net indices from files if E not provided
    pt_files = sorted(ckpt_dir.glob("net_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No net_*.pt found in {ckpt_dir}")

    indices = sorted({int(p.stem.split("_")[1]) for p in pt_files})
    if args.E is not None:
        indices = indices[:args.E]

    soa = list(range(args.soa_start, args.soa_end + 1, args.soa_step))

    job_tuples = [
        (idx, str(ckpt_dir), args.z_task, soa, args.trials_per_soa,
         args.persistence, args.dt_lca, args.t0, args.ITI, args.optimize_onset)
        for idx in indices
    ]

    if args.workers > 0:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(_sweep_one_network, job_tuples)
    else:
        results = [_sweep_one_network(t) for t in job_tuples]

    dep_curves = []
    ind_curves = []
    for idx, z_A, dep_rt2, ind_rt2 in sorted(results, key=lambda r: r[0]):
        if z_A is None:
            continue
        dep_curves.append(dep_rt2)
        ind_curves.append(ind_rt2)
        print(f"Done net {idx:02d} (z_A={z_A:.3f})")

    dep_curves = np.asarray(dep_curves, float)
    ind_curves = np.asarray(ind_curves, float)

    dep_mean = nanmean(dep_curves)
    ind_mean = nanmean(ind_curves)
    dep_se   = nanse(dep_curves)
    ind_se   = nanse(ind_curves)

    # Slope indicators (steepest adjacent segment)
    dep_slope = steepest_adjacent_slope(soa, dep_mean, args.dt_lca)
    ind_slope = steepest_adjacent_slope(soa, ind_mean, args.dt_lca)

    # Data transformation for plotting
    soa_arr = np.asarray(soa, float)
    soa_ms = soa_arr * args.dt_lca * 500.0
    dep_mean_ms = dep_mean * 500.0
    ind_mean_ms = ind_mean * 500.0
    dep_se_ms   = dep_se   * 500.0
    ind_se_ms   = ind_se   * 500.0

    dep_seg_ms = (dep_slope["seg"][0] * args.dt_lca * 500.0,
                  dep_slope["seg"][1] * args.dt_lca * 500.0)

    plt.figure(figsize=(8, 5))

    # Plot Simulation Results
    plt.plot(soa_ms, dep_mean_ms, "x--", color="#1f77b4", linewidth=1.5,
             label=(f"Simulation B→A | steepest: {dep_slope['slope_s_per_s']:.2f}"))
    plt.fill_between(soa_ms, dep_mean_ms - dep_se_ms, dep_mean_ms + dep_se_ms, color="#1f77b4", alpha=0.15)

    plt.plot(soa_ms, ind_mean_ms, "x--", color="#2ca02c", linewidth=1.5,
             label=f"Simulation C→A")
    plt.fill_between(soa_ms, ind_mean_ms - ind_se_ms, ind_mean_ms + ind_se_ms, color="#2ca02c", alpha=0.15)

    # Add Pashler Curve
    if args.add_pashler:
        # Empirical points based on Pashler (1994) Fig 1
        p_soa = np.array([50, 150, 300, 900], dtype=float)
        p_rt2 = np.array([700, 600, 525, 500], dtype=float)
        
        if args.pashler_mode == "relative":
            # Shift x so the Pashler curve starts at the same SOA as your simulation
            p_soa = p_soa - p_soa[0] + soa_ms[0]
            
        if args.align_pashler_rt:
            # Shift y so the Pashler curve starts at the same RT2 as your simulation
            p_rt2 = p_rt2 - p_rt2[0] + dep_mean_ms[0]
            
        plt.plot(p_soa, p_rt2, "ko-", linewidth=2.5, markersize=8, alpha=0.7,
                 label=f"Pashler (1994) Curve ({args.pashler_mode})")
        
        # Add visual slope marker near the first segment of Pashler curve
        #plt.annotate("slope ≈ -1.0", xy=(p_soa[1], p_rt2[1]), xytext=(p_soa[1]+50, p_rt2[1]+40),
        #             arrowprops=dict(arrowstyle="->", color='black'), fontsize=9)

    plt.xlabel("SOA (milliseconds)")
    plt.ylabel("RT2 (milliseconds)")
    plt.title(f"Task 2 RT Comparison | Persistence p={args.persistence:.2f}")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print("Saved plot:", out_path)
    plt.show()


if __name__ == "__main__":
    main()

