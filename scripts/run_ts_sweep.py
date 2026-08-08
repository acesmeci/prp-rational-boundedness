#!/usr/bin/env python
"""
run_ts_sweep.py
---------------
Run the task-switching simulation across the trained ensemble and plot
results in the style of Musslick et al. (2020) Fig. 20.

Usage
-----
    python scripts/run_ts_sweep.py
    python scripts/run_ts_sweep.py --n-nets 5 --n-stim 50 --persistence 0.85
    python scripts/run_ts_sweep.py --ckpt-dir ensemble_ckpt --out-dir output
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from prp_model.utils import load_state  # noqa: E402
from prp_model.ts_simulator import sweep_task_switching  # noqa: E402
from scripts.plot_ts_results import plot_fig20, plot_fig20_ensemble  # noqa: E402

def _run_one_network(ckpt_path, persistence, n_stim, n_repeats, noise_std, seed):
    wrapper = load_state(ckpt_path)
    return sweep_task_switching(
        wrapper,
        persistence=persistence,
        n_stim=n_stim,
        n_repeats=n_repeats,
        noise_std=noise_std,
        seed=seed,
        verbose=False,
    )

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt-dir", default="ensemble_ckpt",
                    help="directory containing net_XX.pt files (default: ensemble_ckpt)")
    ap.add_argument("--n-nets", type=int, default=None,
                    help="number of networks to use (default: all in ckpt-dir)")
    ap.add_argument("--n-stim", type=int, default=100,
                    help="stimuli per condition (default: 100)")
    ap.add_argument("--persistence", type=float, default=0.85,
                    help="persistence parameter p (default: 0.85 = MATLAB tau 0.15)")
    ap.add_argument("--n-repeats", type=int, default=100,
                    help="LCA repeats per threshold (default: 100)")
    ap.add_argument("--noise-std", type=float, default=0.2,
                    help="LCA noise sigma (default: 0.2; MATLAB uses 0.1)")
    ap.add_argument("--out-dir", default="output",
                    help="output directory (default: output)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5,
                    help="parallel workers (default: all CPU cores)")
    args = ap.parse_args()

    ckpt_dir = os.path.join(PROJECT_ROOT, args.ckpt_dir)
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "net_*.pt")))
    if not ckpt_files:
        raise SystemExit(f"No net_*.pt files found in {ckpt_dir}")

    if args.n_nets is not None:
        ckpt_files = ckpt_files[:args.n_nets]

    n_nets = len(ckpt_files)
    print(f"Running task-switching sweep: {n_nets} networks, "
          f"p={args.persistence}, n_stim={args.n_stim}, "
          f"noise_std={args.noise_std}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print()

    all_costs = {
        "structural": {"congruent": [], "incongruent": []},
        "functional": {"congruent": [], "incongruent": []},
        "independent": {"congruent": [], "incongruent": []},
    }

    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = args.workers or min(n_nets, os.cpu_count() or 4)
    print(f"Running {n_nets} networks across {n_workers} workers\n")

    all_results = [None] * n_nets
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_one_network, ckpt,
                        args.persistence, args.n_stim, args.n_repeats,
                        args.noise_std, args.seed): i
            for i, ckpt in enumerate(ckpt_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            all_results[idx] = future.result()
            net_name = os.path.basename(ckpt_files[idx])
            elapsed = time.time() - t0
            done = sum(1 for r in all_results if r is not None)
            print(f"  [{done}/{n_nets}] {net_name} done ({elapsed:.0f}s total)", flush=True)

    print(f"\nAll networks finished in {time.time() - t0:.0f}s")
    for i, results in enumerate(all_results):
        for label in ("structural", "functional", "independent"):
            for cong in ("congruent", "incongruent"):
                all_costs[label][cong].append(results[label][f"cost_{cong}"])

    # Ensemble summary
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE SUMMARY ({n_nets} networks)")
    print(f"{'=' * 60}")
    for label in ("structural", "functional", "independent"):
        parts = []
        for cong in ("congruent", "incongruent"):
            vals = [v for v in all_costs[label][cong] if np.isfinite(v)]
            if vals:
                m = np.mean(vals)
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                parts.append(f"{cong}: {m:+.3f}+-{se:.3f}")
            else:
                parts.append(f"{cong}: nan")
        print(f"  {label:12s} | {' | '.join(parts)}")

    # Save results
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    # Save per-network costs for later analysis
    summary = {
        "n_nets": n_nets,
        "persistence": args.persistence,
        "n_stim": args.n_stim,
        "noise_std": args.noise_std,
        "costs": {},
    }
    for label in ("structural", "functional", "independent"):
        summary["costs"][label] = {}
        for cong in ("congruent", "incongruent"):
            vals = all_costs[label][cong]
            summary["costs"][label][cong] = {
                "per_network": vals,
                "mean": float(np.nanmean(vals)),
                "se": float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
            }

    json_path = os.path.join(out_dir, f"ts_sweep_summary_p{args.persistence:.2f}".replace(".", "") + ".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {json_path}")

    # Plot
    fig_path = os.path.join(out_dir, "figures", f"task_switching_fig20_p{args.persistence:.2f}".replace(".", "") + ".png")
    plot_fig20_ensemble(
        all_results,
        out_path=fig_path,
        title=f"Task Switching (n={n_nets}, p={args.persistence}, σ={args.noise_std})",
    )


if __name__ == "__main__":
    main()