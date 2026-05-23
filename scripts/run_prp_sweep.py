#!/usr/bin/env python3
"""
Run PRP ensemble sweep: train E networks (if missing), run PRP SOA sweeps, save results as JSON.

Usage examples:
    # Single run:
    python -m scripts.run_prp_sweep \
        --store_dir ensemble_ckpt \
        --E 10 --persistence 0.80 \
        --trials_per_soa 30 \
        --soa_start 1 --soa_end 20 --soa_step 2 \
        --optimize_onset --workers 6 --plot

    # Persistence sweep (reuses same trained networks):
    for p in 0.50 0.70 0.75 0.90; do
        python -m scripts.run_prp_sweep \
            --store_dir ensemble_ckpt \
            --E 10 --persistence $p \
            --trials_per_soa 30 \
            --soa_start 1 --soa_end 20 --soa_step 2 \
            --optimize_onset --workers 6 --plot
    done

Outputs:
    JSON results -> output/results/<auto_tag>.json
    Plots        -> via --plot (calls scripts.plot_prp_sweep)
"""
import os, json, argparse, time
from pathlib import Path

import numpy as np
import torch

from prp_model.nn_wrapper import TaskNetworkWrapper
from prp_model.training_set import generate_training_set_matlab_style
from prp_model.prp_simulator import sweep_soa
from prp_model.threshold_utils import compute_fixed_threshold_for_task_meanargmax
from prp_model.lca import MS_PER_STEP
from prp_model.utils import (
    make_wrapper,
    generate_trial_pair,
    save_state, load_state,
    save_threshold, load_threshold,
    average_with_se,
    steepest_adjacent_slope,
)


# ===================================================================
# Naming
# ===================================================================
def make_tag(E, persistence, trials_per_soa, soa_start, soa_end, soa_step, dt_lca, ITI):
    """Auto-generate a filename tag from simulation parameters.

    Convention: E{E}_p{p}_nt{nt}_soa{start}-{end}-{step}_dt{dt}_ITI{ITI}
    """
    p_tag = f"{int(round(persistence * 100)):03d}"
    step_ms = MS_PER_STEP  # ms per simulation step
    dt_tag = f"{int(round(step_ms / 10)):03d}"
    iti_tag = f"{int(round(ITI * 10)):02d}"
    return f"E{E}_p{p_tag}_nt{trials_per_soa}_soa{soa_start}-{soa_end}-{soa_step}_dt{dt_tag}_ITI{iti_tag}"


# ===================================================================
# Training
# ===================================================================
def train_single_network(train_epochs=5000, stop_loss=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    wrapper = make_wrapper()
    X, T, Y, _ = generate_training_set_matlab_style()
    wrapper.train_online(
        torch.tensor(X), torch.tensor(T), torch.tensor(Y),
        max_epochs=train_epochs, stop_loss=stop_loss, print_every=200,
    )
    return wrapper


# ===================================================================
# Per-network job
# ===================================================================
def per_network_job(
    net_idx, store_dir,
    train_if_missing, train_epochs, stop_loss,
    z_task, z_K, z_repeats, thresholds, ITI,
    prp_persistence, prp_trials_per_soa, prp_soa,
    dt_lca, t0, optimize_onset,
):
    t_start = time.time()
    model_path = os.path.join(store_dir, f"net_{net_idx:02d}.pt")
    z_path = os.path.join(store_dir, f"net_{net_idx:02d}_z_{z_task}.json")

    # 1) Load or train
    if os.path.exists(model_path):
        wrapper = load_state(model_path)
        print(f"  [net {net_idx:02d}] Loaded checkpoint")
    else:
        if not train_if_missing:
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        print(f"  [net {net_idx:02d}] Training from scratch...")
        wrapper = train_single_network(train_epochs, stop_loss, seed=net_idx)
        save_state(wrapper, model_path)

    # 2) Load or compute threshold
    if os.path.exists(z_path):
        z_A = load_threshold(z_path)
    else:
        z_A = compute_fixed_threshold_for_task_meanargmax(
            wrapper, task_name=z_task, K=z_K,
            thresholds=thresholds, ITI=ITI, n_repeats=z_repeats,
            persistence=0.0, seed=1000 + net_idx, verbose=False,
        )
        save_threshold(z_A, z_path)
    print(f"  [net {net_idx:02d}] z_A={z_A:.3f}, starting PRP sweep...")

    # 3) PRP sweeps
    gen_dep = lambda: generate_trial_pair(("B", "A"))
    gen_ind = lambda: generate_trial_pair(("C", "A"))

    dep = sweep_soa(
        wrapper, gen_dep, prp_soa,
        n_trials_per_soa=prp_trials_per_soa,
        persistence=prp_persistence,
        dt_lca=dt_lca, t0=t0, ITI=ITI,
        z_task2_fixed=z_A, optimize_onset=optimize_onset,
    )
    ind = sweep_soa(
        wrapper, gen_ind, prp_soa,
        n_trials_per_soa=prp_trials_per_soa,
        persistence=prp_persistence,
        dt_lca=dt_lca, t0=t0, ITI=ITI,
        z_task2_fixed=z_A, optimize_onset=optimize_onset,
    )

    elapsed = time.time() - t_start
    print(f"  [net {net_idx:02d}] Done in {elapsed:.1f}s")
    return {"net_idx": net_idx, "z": z_A, "dep": dep, "ind": ind}


# ===================================================================
# Orchestrator
# ===================================================================
def run_ensemble(args):
    store_dir = args.store_dir
    os.makedirs(store_dir, exist_ok=True)

    soa_list = list(range(args.soa_start, args.soa_end + 1, args.soa_step))
    n_soa = len(soa_list)
    soa_ms_range = (soa_list[0] * MS_PER_STEP, soa_list[-1] * MS_PER_STEP)

    tag = make_tag(
        args.E, args.persistence, args.trials_per_soa,
        args.soa_start, args.soa_end, args.soa_step,
        args.dt_lca, args.ITI,
    )

    print(f"\n{'='*60}")
    print(f"PRP Ensemble Sweep: {tag}")
    print(f"{'='*60}")
    print(f"  Networks:     {args.E}")
    print(f"  Persistence:  {args.persistence}")
    print(f"  SOA range:    {n_soa} points, {soa_ms_range[0]:.0f}–{soa_ms_range[1]:.0f} ms")
    print(f"  Trials/SOA:   {args.trials_per_soa}")
    print(f"  Step size:    {MS_PER_STEP} ms/step")
    print(f"  ITI:          {args.ITI}s")
    print(f"  Onset optim:  {args.optimize_onset}")
    print(f"  Workers:      {args.workers}")
    print(f"  Checkpoints:  {store_dir}")
    print(f"{'='*60}\n")

    t_total_start = time.time()

    job_args = (
        store_dir,
        args.train_if_missing, args.train_epochs, args.stop_loss,
        args.z_task, args.z_K, args.z_repeats,
        np.arange(*args.thresholds), args.ITI,
        args.persistence, args.trials_per_soa, soa_list,
        args.dt_lca, args.t0, args.optimize_onset,
    )

    if args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            jobs = [
                pool.apply_async(per_network_job, (i,) + job_args)
                for i in range(args.E)
            ]
            per_net = [j.get() for j in jobs]
    else:
        per_net = [per_network_job(i, *job_args) for i in range(args.E)]

    t_total = time.time() - t_total_start
    print(f"\nAll {args.E} networks completed in {t_total:.1f}s ({t_total/60:.1f} min)")

    # Compute averages and SE
    keys_to_avg = [
        "rt_task1", "acc_task1",
        "rt_task2", "rt_task2_from_stim", "rt_task2_tail",
        "acc_task2", "onset2",
    ]
    dep_avg = average_with_se([d["dep"] for d in per_net], keys_to_avg)
    ind_avg = average_with_se([d["ind"] for d in per_net], keys_to_avg)

    # Store per-network raw curves for flexible re-analysis
    per_net_serializable = []
    for d in per_net:
        entry = {"net_idx": d["net_idx"], "z": d["z"]}
        for cond_key in ("dep", "ind"):
            entry[cond_key] = {}
            for k in keys_to_avg:
                vals = d[cond_key].get(k)
                if vals is not None:
                    entry[cond_key][k] = [
                        float(v) if np.isfinite(v) else None
                        for v in np.asarray(vals, float)
                    ]
        per_net_serializable.append(entry)

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
            "ms_per_step": MS_PER_STEP,
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
    print(f"z_A values: {np.round(out_dict['z_list'], 3)}")

    # Print slope summary
    dep_slope = steepest_adjacent_slope(
        np.array(soa_list, float),
        np.array(dep_avg["rt_task2_from_stim"], float),
    )
    print(
        f"Steepest B→A slope: {dep_slope['slope_s_per_s']:.2f} "
        f"(segment {dep_slope['seg'][0]:.0f}–{dep_slope['seg'][1]:.0f} steps)"
    )

    return out_dict, tag


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Run PRP ensemble sweep. Results saved to output/results/<tag>.json"
    )
    p.add_argument("--E", type=int, default=20)
    p.add_argument("--store_dir", type=str, default="ensemble_ckpt")
    p.add_argument("--workers", type=int, default=6)

    # Training
    p.add_argument("--train_if_missing", action="store_true")
    p.add_argument("--train_epochs", type=int, default=5000)
    p.add_argument("--stop_loss", type=float, default=1e-3)

    # Threshold
    p.add_argument("--z_task", type=str, default="A")
    p.add_argument("--z_K", type=int, default=27)
    p.add_argument("--z_repeats", type=int, default=100)
    p.add_argument("--thresholds", type=float, nargs=3, default=[0.1, 1.5, 0.1])

    # PRP sweep
    p.add_argument("--persistence", type=float, default=0.80)
    p.add_argument("--trials_per_soa", type=int, default=50)
    p.add_argument("--soa_start", type=int, default=1)
    p.add_argument("--soa_end", type=int, default=20)
    p.add_argument("--soa_step", type=int, default=2)

    # Timing
    p.add_argument("--dt_lca", type=float, default=0.1)
    p.add_argument("--t0", type=float, default=0.15)
    p.add_argument("--ITI", type=float, default=0.5)

    p.add_argument("--optimize_onset", action="store_true")
    p.add_argument("--plot", action="store_true",
                   help="Generate plots after simulation")

    return p.parse_args()


def main():
    args = parse_args()

    if args.workers == 0 and args.E > 1:
        print("Hint: running serially. Use --workers 6 for parallel speed.")

    out_dict, tag = run_ensemble(args)

    if args.plot:
        json_path = f"output/results/{tag}.json"
        print(f"\nGenerating plots from {json_path} ...")
        os.system(f"python -m scripts.plot_prp_sweep --json {json_path}")


if __name__ == "__main__":
    main()