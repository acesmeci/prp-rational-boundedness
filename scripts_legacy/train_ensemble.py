#!/usr/bin/env python3
"""
Train an ensemble of networks and compute LCA thresholds.

This is the first step in the pipeline. Run once, then use
run_prp_sweep.py to run PRP sweeps with different persistence values.

Usage:
    python -m scripts.train_ensemble --E 20 --store_dir ensemble_ckpt --workers 6

Outputs per network (saved to --store_dir):
    net_XX.pt          Trained model weights
    net_XX_z_A.json    Optimal LCA threshold for Task A
"""
import os, json, argparse, time
from pathlib import Path

import numpy as np
import torch

from prp_model.training_set import generate_training_set_matlab_style
from prp_model.threshold_utils import compute_fixed_threshold_for_task_meanargmax
from prp_model.utils import make_wrapper, save_state, load_state, save_threshold, load_threshold


# ===================================================================
# Train + threshold for one network
# ===================================================================
def train_and_threshold(
    net_idx, store_dir,
    train_epochs, stop_loss,
    z_task, z_K, z_repeats, thresholds, ITI,
):
    model_path = os.path.join(store_dir, f"net_{net_idx:02d}.pt")
    z_path = os.path.join(store_dir, f"net_{net_idx:02d}_z_{z_task}.json")

    t0 = time.time()

    # --- Train (skip if checkpoint exists) ---
    if os.path.exists(model_path):
        wrapper = load_state(model_path)
        print(f"  [net {net_idx:02d}] Loaded existing checkpoint")
    else:
        torch.manual_seed(net_idx)
        np.random.seed(net_idx)

        wrapper = make_wrapper()
        X, T, Y, _ = generate_training_set_matlab_style()
        wrapper.train_online(
            torch.tensor(X), torch.tensor(T), torch.tensor(Y),
            max_epochs=train_epochs, stop_loss=stop_loss, print_every=500,
        )
        save_state(wrapper, model_path)
        print(f"  [net {net_idx:02d}] Trained and saved")

    # --- Compute threshold (skip if exists) ---
    if os.path.exists(z_path):
        z = load_threshold(z_path)
        print(f"  [net {net_idx:02d}] Loaded threshold z={z:.3f}")
    else:
        z = compute_fixed_threshold_for_task_meanargmax(
            wrapper, task_name=z_task, K=z_K,
            thresholds=thresholds, ITI=ITI, n_repeats=z_repeats,
            persistence=0.0,
            seed=1000 + net_idx, verbose=False,
        )
        save_threshold(z, z_path)
        print(f"  [net {net_idx:02d}] Computed threshold z={z:.3f}")

    elapsed = time.time() - t0
    print(f"  [net {net_idx:02d}] Done in {elapsed:.1f}s")
    return net_idx, z


# ===================================================================
# Main
# ===================================================================
def main():
    p = argparse.ArgumentParser(description="Train ensemble + compute thresholds.")
    p.add_argument("--E", type=int, default=20)
    p.add_argument("--store_dir", type=str, default="ensemble_ckpt")
    p.add_argument("--workers", type=int, default=6)

    p.add_argument("--train_epochs", type=int, default=5000)
    p.add_argument("--stop_loss", type=float, default=1e-3)

    p.add_argument("--z_task", type=str, default="A")
    p.add_argument("--z_K", type=int, default=27)
    p.add_argument("--z_repeats", type=int, default=100)
    p.add_argument("--thresholds", type=float, nargs=3, default=[0.1, 1.5, 0.1])
    p.add_argument("--ITI", type=float, default=0.5)

    args = p.parse_args()
    os.makedirs(args.store_dir, exist_ok=True)
    thresh_grid = np.arange(*args.thresholds)

    print(f"\nTraining ensemble: {args.E} networks -> {args.store_dir}")
    print(f"  Threshold task: {args.z_task}, ITI: {args.ITI}")
    t_start = time.time()

    job_args = (
        args.store_dir, args.train_epochs, args.stop_loss,
        args.z_task, args.z_K, args.z_repeats, thresh_grid, args.ITI,
    )

    if args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            jobs = [
                pool.apply_async(train_and_threshold, (i,) + job_args)
                for i in range(args.E)
            ]
            results = [j.get() for j in jobs]
    else:
        results = [train_and_threshold(i, *job_args) for i in range(args.E)]

    t_total = time.time() - t_start
    z_vals = [z for _, z in sorted(results)]
    print(f"\nDone in {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Thresholds: {np.round(z_vals, 3)}")
    print(f"\nNext: python -m scripts.run_prp_sweep --store_dir {args.store_dir} --persistence 0.80 --plot")


if __name__ == "__main__":
    main()