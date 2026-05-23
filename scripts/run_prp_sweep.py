#!/usr/bin/env python3
"""
Run PRP ensemble sweep: train E networks (if missing), run PRP SOA sweeps, save results as JSON.

Usage examples:
    # Single run (sanity check):
    python -m scripts.run_prp_sweep \
        --store_dir notebooks/ensemble_ckpt_p09 \
        --E 10 \
        --persistence 0.80 \
        --trials_per_soa 30 \
        --soa_start 1 --soa_end 20 --soa_step 2 \
        --dt_lca 0.1 --t0 0.15 --ITI 0.5 \
        --optimize_onset \
        --workers 6 \
        --plot

    # If networks haven't been trained yet, add --train_if_missing

    # Persistence sweep (reuses same trained networks):
    for p in 0.50 0.70 0.75 0.90; do
        python -m scripts.run_prp_sweep \
            --store_dir notebooks/ensemble_ckpt_p09 \
            --E 10 --persistence $p \
            --trials_per_soa 30 \
            --soa_start 1 --soa_end 20 --soa_step 2 \
            --dt_lca 0.1 --t0 0.15 --ITI 0.5 \
            --optimize_onset --workers 6 --plot
    done

Outputs:
    JSON results -> output/results/<auto_tag>.json
    RT2 plots    -> output/plots/ensemble/pashler/<auto_tag>.png  (via --plot)
    ER plots     -> output/plots/ensemble/pashler/ER/<auto_tag>.png  (via --plot)

Tunable hyperparameters:
    --persistence          Carry-over p (0.0 to 0.99)
    --dt_lca               LCA timestep; 0.1 -> 50ms/step, 0.2 -> 100ms/step
    --soa_start/end/step   SOA range in simulation steps
    --trials_per_soa       Trials per SOA (30 ok, 50 cleaner)
    --E                    Ensemble size (10 or 20)
    --ITI                  Inter-trial interval in seconds (0.5 standard)
    --optimize_onset       Enable reward-rate Task-2 onset optimization
    --workers              Parallel workers (set to CPU count for speed)
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import torch

from prp_model.nn_wrapper import TaskNetworkWrapper
from prp_model.training_set import generate_training_set_matlab_style
from prp_model.prp_simulator import sweep_soa
from prp_model.threshold_utils import compute_fixed_threshold_for_task_meanargmax


# ===================================================================
# Naming
# ===================================================================
def make_tag(E, persistence, trials_per_soa, soa_start, soa_end, soa_step, dt_lca, ITI):
    """Auto-generate a filename tag from simulation parameters.

    Convention: E{E}_p{p}_nt{nt}_soa{start}-{end}-{step}_dt{dt}_ITI{ITI}
    - p:   persistence * 100, zero-padded to 2 digits  (0.80 -> "080", 0.5 -> "050")
    - dt:  timestep in seconds * 100, zero-padded to 3 digits  (50ms=0.05s -> "005", 100ms=0.1s -> "010")
    - ITI: ITI * 10, as integer (0.5 -> "05", 4.0 -> "40")
    """
    p_tag = f"{int(round(persistence * 100)):03d}"
    step_s = dt_lca * 0.5  # one simulation step in seconds
    dt_tag = f"{int(round(step_s * 100)):03d}"
    iti_tag = f"{int(round(ITI * 10)):02d}"
    return f"E{E}_p{p_tag}_nt{trials_per_soa}_soa{soa_start}-{soa_end}-{soa_step}_dt{dt_tag}_ITI{iti_tag}"


# ===================================================================
# Utilities
# ===================================================================
def _nanmean(x):
    return float(np.nanmean(np.asarray(x, float)))

def _nanse(x):
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

def average_with_se(results_list, keys):
    """Compute mean and SE across networks for each SOA point."""
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

def steepest_adjacent_slope(soa_steps, y, dt_lca):
    """Find the steepest adjacent-pair slope in the RT2-vs-SOA curve."""
    soa_steps = np.asarray(soa_steps, float)
    y = np.asarray(y, float)
    m = np.isfinite(soa_steps) & np.isfinite(y)
    soa_steps, y = soa_steps[m], y[m]
    order = np.argsort(soa_steps)
    soa_steps, y = soa_steps[order], y[order]
    if len(soa_steps) < 2:
        return {"seg": (np.nan, np.nan), "slope_s_per_s": np.nan}
    dsoa = np.diff(soa_steps)
    dy = np.diff(y)
    slope_s_per_step = dy / dsoa
    slope_s_per_s = slope_s_per_step / dt_lca
    i = int(np.nanargmin(slope_s_per_s))
    return {
        "seg": (float(soa_steps[i]), float(soa_steps[i + 1])),
        "slope_s_per_s": float(slope_s_per_s[i]),
    }


# ===================================================================
# Disk cache helpers
# ===================================================================
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


# ===================================================================
# Trial generator
# ===================================================================
def generate_trial_pair(prp_pair=("B", "A"), N_pathways=3, N_features=3, seed=None):
    task_map = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2), 'D': (0, 1), 'E': (1, 0)}
    rng = np.random.RandomState(seed)

    def sample_single_task(task_name, shared_features=None):
        in_dim, out_dim = task_map[task_name]
        feats = shared_features if shared_features is not None \
                else rng.randint(0, N_features, size=N_pathways)
        stim = np.zeros(N_pathways * N_features, dtype=np.float32)
        for i in range(N_pathways):
            stim[i * N_features + feats[i]] = 1
        cue = np.zeros(N_pathways ** 2, dtype=np.float32)
        cue[in_dim * N_pathways + out_dim] = 1
        return stim, cue

    feats = rng.randint(0, N_features, size=N_pathways)
    stim1, cue1 = sample_single_task(prp_pair[0], shared_features=feats)
    stim2, cue2 = sample_single_task(prp_pair[1], shared_features=feats)
    return stim1, stim2, cue1, cue2


# ===================================================================
# Training
# ===================================================================
def train_single_network(make_wrapper_fn, train_epochs=5000, stop_loss=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    wrapper = make_wrapper_fn()
    X, T, Y, _ = generate_training_set_matlab_style()
    wrapper.train_online(torch.tensor(X), torch.tensor(T), torch.tensor(Y),
                         max_epochs=train_epochs, stop_loss=stop_loss, print_every=200)
    return wrapper


# ===================================================================
# Per-network job
# ===================================================================
def per_network_job(
    net_idx, make_wrapper_fn, store_dir,
    train_if_missing, train_epochs, stop_loss,
    z_task, z_K, z_repeats, thresholds, ITI,
    prp_persistence, prp_trials_per_soa, prp_soa,
    dt_lca, t0, optimize_onset
):
    import time
    t_start = time.time()
    model_path = os.path.join(store_dir, f"net_{net_idx:02d}.pt")
    z_path = os.path.join(store_dir, f"net_{net_idx:02d}_z_{z_task}.json")

    # 1) Load or train
    if os.path.exists(model_path):
        wrapper = load_state(make_wrapper_fn, model_path)
        print(f"  [net {net_idx:02d}] Loaded checkpoint")
    else:
        if not train_if_missing:
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        print(f"  [net {net_idx:02d}] Training from scratch...")
        wrapper = train_single_network(make_wrapper_fn, train_epochs, stop_loss, seed=net_idx)
        save_state(wrapper, model_path)

    # 2) Load or compute threshold
    if os.path.exists(z_path):
        z_A = load_threshold(z_path)
    else:
        z_A = compute_fixed_threshold_for_task_meanargmax(
            wrapper, task_name=z_task, K=z_K,
            thresholds=thresholds, ITI=ITI, n_repeats=z_repeats,
            persistence=0.0, seed=1000 + net_idx, verbose=False
        )
        save_threshold(z_A, z_path)
    print(f"  [net {net_idx:02d}] z_A={z_A:.3f}, starting PRP sweep...")

    # 3) PRP sweeps
    gen_dep = lambda: generate_trial_pair(("B", "A"))
    gen_ind = lambda: generate_trial_pair(("C", "A"))

    dep = sweep_soa(wrapper, gen_dep, prp_soa,
                    n_trials_per_soa=prp_trials_per_soa,
                    persistence=prp_persistence,
                    dt_lca=dt_lca, t0=t0, ITI=ITI,
                    z_task2_fixed=z_A, optimize_onset=optimize_onset)

    ind = sweep_soa(wrapper, gen_ind, prp_soa,
                    n_trials_per_soa=prp_trials_per_soa,
                    persistence=prp_persistence,
                    dt_lca=dt_lca, t0=t0, ITI=ITI,
                    z_task2_fixed=z_A, optimize_onset=optimize_onset)

    elapsed = time.time() - t_start
    print(f"  [net {net_idx:02d}] Done in {elapsed:.1f}s")
    return {"net_idx": net_idx, "z": z_A, "dep": dep, "ind": ind}


# ===================================================================
# Orchestrator
# ===================================================================
def run_ensemble(args, make_wrapper_fn):
    import time
    store_dir = args.store_dir
    os.makedirs(store_dir, exist_ok=True)

    soa_list = list(range(args.soa_start, args.soa_end + 1, args.soa_step))
    n_soa = len(soa_list)
    soa_ms_range = (soa_list[0] * args.dt_lca * 500, soa_list[-1] * args.dt_lca * 500)

    tag = make_tag(args.E, args.persistence, args.trials_per_soa,
                   args.soa_start, args.soa_end, args.soa_step,
                   args.dt_lca, args.ITI)

    print(f"\n{'='*60}")
    print(f"PRP Ensemble Sweep: {tag}")
    print(f"{'='*60}")
    print(f"  Networks:     {args.E}")
    print(f"  Persistence:  {args.persistence}")
    print(f"  SOA range:    {n_soa} points, {soa_ms_range[0]:.0f}-{soa_ms_range[1]:.0f} ms")
    print(f"  Trials/SOA:   {args.trials_per_soa}")
    print(f"  dt_lca:       {args.dt_lca} ({args.dt_lca*500:.0f} ms/step)")
    print(f"  ITI:          {args.ITI}s")
    print(f"  Onset optim:  {args.optimize_onset}")
    print(f"  Workers:      {args.workers}")
    print(f"  Checkpoints:  {store_dir}")
    print(f"{'='*60}\n")

    t_total_start = time.time()

    job_args = (
        make_wrapper_fn, store_dir,
        args.train_if_missing, args.train_epochs, args.stop_loss,
        args.z_task, args.z_K, args.z_repeats, np.arange(*args.thresholds), args.ITI,
        args.persistence, args.trials_per_soa, soa_list,
        args.dt_lca, args.t0, args.optimize_onset
    )

    if args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            jobs = [pool.apply_async(per_network_job, (i,) + job_args)
                    for i in range(args.E)]
            per_net = [j.get() for j in jobs]
    else:
        per_net = [per_network_job(i, *job_args) for i in range(args.E)]

    t_total = time.time() - t_total_start
    print(f"\nAll {args.E} networks completed in {t_total:.1f}s ({t_total/60:.1f} min)")

    # Compute averages and SE
    keys_to_avg = [
        "rt_task1", "acc_task1",
        "rt_task2", "rt_task2_from_stim", "rt_task2_tail",
        "acc_task2", "onset2"
    ]
    dep_avg = average_with_se([d["dep"] for d in per_net], keys_to_avg)
    ind_avg = average_with_se([d["ind"] for d in per_net], keys_to_avg)

    # Also store per-network raw curves for flexible re-analysis
    per_net_serializable = []
    for d in per_net:
        entry = {"net_idx": d["net_idx"], "z": d["z"]}
        for cond_key in ("dep", "ind"):
            entry[cond_key] = {}
            for k in keys_to_avg:
                vals = d[cond_key].get(k)
                if vals is not None:
                    entry[cond_key][k] = [float(v) if np.isfinite(v) else None
                                          for v in np.asarray(vals, float)]
        per_net_serializable.append(entry)

    # Build output dict with parameters for the plotter
    tag = make_tag(args.E, args.persistence, args.trials_per_soa,
                   args.soa_start, args.soa_end, args.soa_step,
                   args.dt_lca, args.ITI)

    out_dict = {
        "tag": tag,
        "params": {
            "E": args.E,
            "persistence": args.persistence,
            "trials_per_soa": args.trials_per_soa,
            "soa_start": args.soa_start,
            "soa_end": args.soa_end,
            "soa_step": args.soa_step,
            "dt_lca": args.dt_lca,
            "t0": args.t0,
            "ITI": args.ITI,
            "optimize_onset": args.optimize_onset,
        },
        "soa": soa_list,
        "avg": {"dep": dep_avg, "ind": ind_avg},
        "z_list": [float(d["z"]) for d in per_net],
        "per_net": per_net_serializable,
    }

    # Save JSON
    results_dir = Path("output/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved results: {json_path}")
    print(f"Tag: {tag}")
    print(f"z_A values: {np.round(out_dict['z_list'], 3)}")

    # Compute and print slope summary
    dep_slope = steepest_adjacent_slope(
        np.array(soa_list, float),
        np.array(dep_avg["rt_task2_from_stim"], float),
        args.dt_lca
    )
    print(f"Steepest B→A slope: {dep_slope['slope_s_per_s']:.2f} "
          f"(segment {dep_slope['seg'][0]:.0f}-{dep_slope['seg'][1]:.0f} steps)")

    return out_dict, tag


# ===================================================================
# Wrapper factory
# ===================================================================
def _make_wrapper():
    return TaskNetworkWrapper(
        stim_input_dim=9, task_input_dim=9, hidden_dim=100, output_dim=9,
        learning_rate=0.3, init_scale=0.1, init_task_scale=None,
        bias_offset=-2.0, default_weight_decay=0.0, device="cpu",
    )


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Run PRP ensemble sweep. Results saved to output/results/<tag>.json"
    )
    p.add_argument("--E", type=int, default=20, help="Number of networks in ensemble")
    p.add_argument("--store_dir", type=str, default="ensemble_ckpt",
                   help="Directory for network checkpoints (.pt) and thresholds")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel workers (0=serial)")

    # Training
    p.add_argument("--train_if_missing", action="store_true",
                   help="Train networks whose .pt files are missing")
    p.add_argument("--train_epochs", type=int, default=5000)
    p.add_argument("--stop_loss", type=float, default=1e-3)

    # Threshold
    p.add_argument("--z_task", type=str, default="A")
    p.add_argument("--z_K", type=int, default=27)
    p.add_argument("--z_repeats", type=int, default=100)
    p.add_argument("--thresholds", type=float, nargs=3, default=[0.1, 1.5, 0.1],
                   help="np.arange(start, stop, step) for threshold search")

    # PRP sweep
    p.add_argument("--persistence", type=float, default=0.80)
    p.add_argument("--trials_per_soa", type=int, default=50)
    p.add_argument("--soa_start", type=int, default=1)
    p.add_argument("--soa_end", type=int, default=20)
    p.add_argument("--soa_step", type=int, default=2)

    # Timing
    p.add_argument("--dt_lca", type=float, default=0.1,
                   help="LCA dt parameter. 1 step = dt_lca * 500 ms (0.1 -> 50ms)")
    p.add_argument("--t0", type=float, default=0.15)
    p.add_argument("--ITI", type=float, default=0.5)

    p.add_argument("--optimize_onset", action="store_true")

    # Auto-plot after running
    p.add_argument("--plot", action="store_true",
                   help="Generate plots immediately after simulation")

    return p.parse_args()


def main():
    args = parse_args()

    if args.workers == 0 and args.E > 1:
        print("Hint: running serially. Use --workers 6 for parallel speed.")

    out_dict, tag = run_ensemble(args, _make_wrapper)

    if args.plot:
        json_path = f"output/results/{tag}.json"
        print(f"\nGenerating plots from {json_path} ...")
        os.system(f"python -m scripts.plot_prp_sweep --json {json_path}")


if __name__ == "__main__":
    main()