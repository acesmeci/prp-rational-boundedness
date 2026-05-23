#!/usr/bin/env python3
"""
Run PRP ensemble: train E networks (only if .pt missing), run PRP sweeps in parallel, optionally plot.

Equivalent to the last 4–5 cells of lca_experiments.ipynb:
- Train 20 networks with different seeds, skipping any that already have net_XX.pt.
- Use --workers N to run N networks in parallel (recommended; without it, 20 nets run serially and take ~1h+).
- Saves ensemble_results.json; use --plot to also save a PNG.
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import torch

# ---- your project imports (adjust paths) ----
from prp.nn_wrapper import TaskNetworkWrapper
from prp.training_set import generate_training_set_matlab_style
from prp.prp_simulator import sweep_soa
from prp.threshold_utils import compute_fixed_threshold_for_task_meanargmax

# ---------------------------
# Utilities: mean + SE
# ---------------------------
def _nanmean(x): 
    return float(np.nanmean(np.asarray(x, float)))

def _nanse(x):
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

def average_with_se(results_list, keys):
    """Assumes identical SOA grid across results_list."""
    soa = results_list[0]["soa"]
    out = {"soa": soa}
    for k in keys:
        out[k] = []
        out[k + "_se"] = []
        for i in range(len(soa)):
            vals = [r[k][i] for r in results_list]
            out[k].append(_nanmean(vals))
            out[k + "_se"].append(_nanse(vals))
    return out

# ---------------------------
# Disk cache
# ---------------------------
def save_state(wrapper, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.model.state_dict(), path)

def load_state(make_wrapper_fn, path, device="cpu"):
    wrapper = make_wrapper_fn()
    wrapper.model.load_state_dict(torch.load(path, map_location=device))
    wrapper.model.eval()
    return wrapper

def save_threshold(z, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"z": float(z)}, f)

def load_threshold(path):
    with open(path, "r") as f:
        return float(json.load(f)["z"])

# ---------------------------
# Trial generator (as in notebook)
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
# Train network (optional)
# ---------------------------
def train_single_network(make_wrapper_fn, train_epochs=5000, stop_loss=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    wrapper = make_wrapper_fn()
    X, T, Y, _ = generate_training_set_matlab_style()
    wrapper.train_online(torch.tensor(X), torch.tensor(T), torch.tensor(Y),
                         max_epochs=train_epochs, stop_loss=stop_loss, print_every=200)
    return wrapper

# ---------------------------
# Per-network job
# ---------------------------
def per_network_job(
    net_idx, make_wrapper_fn, store_dir,
    train_if_missing, train_epochs, stop_loss,
    z_task, z_K, z_repeats, thresholds, ITI,
    prp_persistence, prp_trials_per_soa, prp_soa,
    dt_lca, t0, optimize_onset
):
    model_path = os.path.join(store_dir, f"net_{net_idx:02d}.pt")
    z_path     = os.path.join(store_dir, f"net_{net_idx:02d}_z_{z_task}.json")

    # 1) Load or train
    if os.path.exists(model_path):
        wrapper = load_state(make_wrapper_fn, model_path)
    else:
        if not train_if_missing:
            raise FileNotFoundError(model_path)
        wrapper = train_single_network(make_wrapper_fn, train_epochs, stop_loss, seed=net_idx)
        save_state(wrapper, model_path)

    # 2) Load or compute z
    if os.path.exists(z_path):
        z_A = load_threshold(z_path)
    else:
        z_A = compute_fixed_threshold_for_task_meanargmax(
            wrapper, task_name=z_task, K=z_K,
            thresholds=thresholds, ITI=ITI, n_repeats=z_repeats,
            persistence=0.0, seed=1000 + net_idx, verbose=False
        )
        save_threshold(z_A, z_path)

    # 3) PRP sweeps
    gen_dep = lambda: generate_trial_pair(("B","A"))
    gen_ind = lambda: generate_trial_pair(("C","A"))

    dep = sweep_soa(wrapper, gen_dep, prp_soa,
                    n_trials_per_soa=prp_trials_per_soa,
                    persistence=prp_persistence,
                    dt_lca=dt_lca, t0=t0, ITI=ITI,
                    z_task2_fixed=z_A,
                    optimize_onset=optimize_onset)

    ind = sweep_soa(wrapper, gen_ind, prp_soa,
                    n_trials_per_soa=prp_trials_per_soa,
                    persistence=prp_persistence,
                    dt_lca=dt_lca, t0=t0, ITI=ITI,
                    z_task2_fixed=z_A,
                    optimize_onset=optimize_onset)

    return {"net_idx": net_idx, "z": z_A, "dep": dep, "ind": ind}

# ---------------------------
# Orchestrator
# ---------------------------
def run_ensemble(args, make_wrapper_fn):
    store_dir = args.store_dir
    os.makedirs(store_dir, exist_ok=True)

    # Run jobs (serial or multiprocessing)
    if args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            jobs = [pool.apply_async(
                        per_network_job,
                        (i, make_wrapper_fn, store_dir,
                         args.train_if_missing, args.train_epochs, args.stop_loss,
                         args.z_task, args.z_K, args.z_repeats, np.arange(*args.thresholds), args.ITI,
                         args.persistence, args.trials_per_soa, list(range(args.soa_start, args.soa_end+1, args.soa_step)),
                         args.dt_lca, args.t0, args.optimize_onset)
                    )
                    for i in range(args.E)]
            per_net = [j.get() for j in jobs]
    else:
        per_net = [per_network_job(
                    i, make_wrapper_fn, store_dir,
                    args.train_if_missing, args.train_epochs, args.stop_loss,
                    args.z_task, args.z_K, args.z_repeats, np.arange(*args.thresholds), args.ITI,
                    args.persistence, args.trials_per_soa, list(range(args.soa_start, args.soa_end+1, args.soa_step)),
                    args.dt_lca, args.t0, args.optimize_onset
                )
                for i in range(args.E)]

    # Average + SE for key metrics
    keys_to_avg = [
        "rt_task1", "acc_task1",
        "rt_task2_from_stim", "rt_task2_tail",
        "acc_task2", "onset2"
    ]
    dep_avg = average_with_se([d["dep"] for d in per_net], keys_to_avg)
    ind_avg = average_with_se([d["ind"] for d in per_net], keys_to_avg)

    out = {"per_net": per_net,
           "avg": {"dep": dep_avg, "ind": ind_avg},
           "z_list": [d["z"] for d in per_net]}

    # Save raw results for re-plotting without rerunning (serialize without full per_net for size)
    out_path = os.path.join(store_dir, "ensemble_results.json")
    # Save summary + per-net sweep arrays for plotting
    to_save = {
        "avg": out["avg"],
        "z_list": out["z_list"],
        "soa": list(out["avg"]["dep"]["soa"]),
    }
    with open(out_path, "w") as f:
        json.dump(to_save, f)
    print("Saved:", out_path)
    print("z_A:", np.round(out["z_list"], 3))

    return out


# ---------------------------
# Plot (same style as plot_prp_ensemble)
# ---------------------------
def steepest_adjacent_slope(soa_steps, y, dt_lca):
    soa_steps = np.asarray(soa_steps, float)
    y = np.asarray(y, float)
    m = np.isfinite(soa_steps) & np.isfinite(y)
    soa_steps, y = soa_steps[m], y[m]
    order = np.argsort(soa_steps)
    soa_steps, y = soa_steps[order], y[order]
    dsoa = np.diff(soa_steps)
    dy = np.diff(y)
    slope_s_per_step = dy / dsoa
    slope_s_per_s = slope_s_per_step / dt_lca
    i = int(np.nanargmin(slope_s_per_s))
    return {
        "seg": (float(soa_steps[i]), float(soa_steps[i + 1])),
        "slope_s_per_s": float(slope_s_per_s[i]),
    }


def plot_ensemble_results(avg, dt_lca, persistence, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    soa = np.asarray(avg["dep"]["soa"], float)
    dep_mean = np.asarray(avg["dep"]["rt_task2_from_stim"], float)
    ind_mean = np.asarray(avg["ind"]["rt_task2_from_stim"], float)
    dep_se = np.asarray(avg["dep"].get("rt_task2_from_stim_se", [0] * len(soa)), float)
    ind_se = np.asarray(avg["ind"].get("rt_task2_from_stim_se", [0] * len(soa)), float)

    dep_slope = steepest_adjacent_slope(soa, dep_mean, dt_lca)
    ind_slope = steepest_adjacent_slope(soa, ind_mean, dt_lca)

    soa_ms = soa * dt_lca * 500.0
    dep_mean_ms = dep_mean * 500.0
    ind_mean_ms = ind_mean * 500.0
    dep_se_ms = dep_se * 500.0
    ind_se_ms = ind_se * 500.0
    dep_seg_ms = (dep_slope["seg"][0] * dt_lca * 500.0, dep_slope["seg"][1] * dt_lca * 500.0)
    ind_seg_ms = (ind_slope["seg"][0] * dt_lca * 500.0, ind_slope["seg"][1] * dt_lca * 500.0)

    plt.figure(figsize=(7, 4))
    plt.plot(soa_ms, dep_mean_ms, "x--",
             label=(f"Dependent B→A | steepest {dep_seg_ms[0]:.0f}-{dep_seg_ms[1]:.0f} ms: "
                    f"{dep_slope['slope_s_per_s']:.2f}"))
    plt.fill_between(soa_ms, dep_mean_ms - dep_se_ms, dep_mean_ms + dep_se_ms, alpha=0.2)
    plt.plot(soa_ms, ind_mean_ms, "x--",
             label=(f"Independent C→A | steepest {ind_seg_ms[0]:.0f}-{ind_seg_ms[1]:.0f} ms: "
                    f"{ind_slope['slope_s_per_s']:.2f}"))
    plt.fill_between(soa_ms, ind_mean_ms - ind_se_ms, ind_mean_ms + ind_se_ms, alpha=0.2)
    plt.xlabel("SOA (milliseconds)")
    plt.ylabel("RT2 (milliseconds)")
    plt.title(f"Task 2 | p={persistence:.2f}")
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Saved plot:", out_png)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train (if missing) and run PRP ensemble; use --workers for parallel speed.")
    p.add_argument("--E", type=int, default=20, help="Number of ensemble networks (default 20)")
    p.add_argument("--store_dir", type=str, default="ensemble_ckpt")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel workers (0=serial). Default 6 for faster runs; set to CPU count for max speed.")

    p.add_argument("--train_if_missing", action="store_true",
                   help="Train networks that don't have net_XX.pt yet; otherwise require existing checkpoints")
    p.add_argument("--train_epochs", type=int, default=5000)
    p.add_argument("--stop_loss", type=float, default=1e-3)

    p.add_argument("--z_task", type=str, default="A")
    p.add_argument("--z_K", type=int, default=27)
    p.add_argument("--z_repeats", type=int, default=100)
    # np.arange(start, stop, step) tuple
    p.add_argument("--thresholds", type=float, nargs=3, default=[0.1, 1.5, 0.1])

    p.add_argument("--persistence", type=float, default=0.90)
    p.add_argument("--trials_per_soa", type=int, default=30)
    p.add_argument("--soa_start", type=int, default=5)
    p.add_argument("--soa_end", type=int, default=60)
    p.add_argument("--soa_step", type=int, default=5)

    p.add_argument("--dt_lca", type=float, default=0.1)
    p.add_argument("--t0", type=float, default=0.15)
    p.add_argument("--ITI", type=float, default=0.5)

    p.add_argument("--optimize_onset", action="store_true")
    p.add_argument("--plot", action="store_true", help="Save PNG plot after running (same style as plot_prp_ensemble)")
    p.add_argument("--out_png", type=str, default="prp_ensemble_rt2_from_stim.png",
                   help="Output path when --plot. If left default and --plot is set, auto-generated name under output/plots/ensemble/ is used.")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- wrapper factory (fill with your real constructor args) ----
    def make_wrapper_fn():
        return TaskNetworkWrapper(
            stim_input_dim=9, task_input_dim=9, hidden_dim=100, output_dim=9,
            learning_rate=0.3,
            init_scale=0.1,
            init_task_scale=None,
            bias_offset=-2.0,
            default_weight_decay=0.0,
            device="cpu",
        )

    if args.workers == 0 and args.E > 1:
        print("Hint: running serially (--workers 0). Use --workers 6 or higher for parallel speed.")
    out = run_ensemble(args, make_wrapper_fn)
    if args.plot:
        out_png = args.out_png
        if out_png == "prp_ensemble_rt2_from_stim.png":
            # Auto-generate path: output/plots/ensemble/E20_p08_ITI05_nt50_dt005_SOA3-21-3.png
            p_str = f"p{int(round(args.persistence * 100)):02d}"
            iti_str = f"ITI{args.ITI:.2f}".replace(".", "")
            dt_str = f"dt{int(round(args.dt_lca * 1000)):03d}"
            out_png = os.path.join(
                "output", "plots", "ensemble",
                f"E{args.E}_{p_str}_{iti_str}_nt{args.trials_per_soa}_{dt_str}_SOA{args.soa_start}-{args.soa_end}-{args.soa_step}.png",
            )
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plot_ensemble_results(out["avg"], args.dt_lca, args.persistence, out_png)

if __name__ == "__main__":
    main()
