#!/usr/bin/env python
"""
run_bce_sweep.py
----------------
Run backward crosstalk effect (BCE) analysis across the trained ensemble.

Compares RT1 congruent vs incongruent at each SOA for functionally
dependent (B->A) and independent (C->A) pairs. The RBA predicts larger
BCE for B->A than C->A.

Usage
-----
    python scripts/run_bce_sweep.py --n-nets 20 --persistence 0.65
    python scripts/run_bce_sweep.py --n-nets 5 --persistence 0.85 --n-trials 120
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
from prp_model.bce_analysis import run_bce_comparison  # noqa: E402


def _run_one_network(
    ckpt_path, persistence, n_trials, soa_values, noise_std, ITI,
    thresholds, seed, pairs, optimize_onset, max_onset_delay,
    z_task1_fixed, z_task2_fixed,
):
    wrapper = load_state(ckpt_path)
    return run_bce_comparison(
        wrapper,
        pairs=pairs,
        soa_values=soa_values,
        n_trials_per_soa=n_trials,
        persistence=persistence,
        noise_std=noise_std,
        ITI=ITI,
        thresholds=thresholds,
        base_seed=seed,
        verbose=False,
        optimize_onset=optimize_onset,
        max_onset_delay=max_onset_delay,
        z_task1_fixed=z_task1_fixed,
        z_task2_fixed=z_task2_fixed,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt-dir", default="ensemble_ckpt")
    ap.add_argument("--n-nets", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=60,
                    help="trials per SOA per condition (default: 60)")
    ap.add_argument("--persistence", type=float, default=0.65)
    ap.add_argument("--soa-steps", type=int, nargs="+",
                    default=[1, 3, 5, 8, 11, 16],
                    help="SOA values in simulation steps")
    ap.add_argument("--noise-std", type=float, default=0.2)
    ap.add_argument("--ITI", type=float, default=1.8)
    ap.add_argument("--optimize-onset", action="store_true",
                    help="enable strategic onset optimization")
    ap.add_argument("--max-onset-delay", type=int, default=15)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--pairs", nargs="+", default=["BA", "CA"],
                    help="task pairs, e.g. BA CA EA (default: BA CA)")
    args = ap.parse_args()

    # Parse pairs
    pair_map = {
        "BA": ("B", "A", "functional"),
        "CA": ("C", "A", "independent"),
        "EA": ("E", "A", "structural"),
    }
    pairs = [pair_map[p] for p in args.pairs if p in pair_map]

    ckpt_dir = os.path.join(PROJECT_ROOT, args.ckpt_dir)
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "net_*.pt")))
    if not ckpt_files:
        raise SystemExit(f"No net_*.pt files found in {ckpt_dir}")
    if args.n_nets is not None:
        ckpt_files = ckpt_files[:args.n_nets]

    n_nets = len(ckpt_files)
    thresholds = np.arange(0.1, 1.6, 0.1)

    print(f"BCE sweep: {n_nets} networks, p={args.persistence}, "
          f"n_trials={args.n_trials}, noise={args.noise_std}")
    print(f"SOAs (steps): {args.soa_steps}")
    print(f"Pairs: {[p[2] for p in pairs]}")
    print(f"Onset optimization: {args.optimize_onset}")
    print(flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = args.workers or min(n_nets, os.cpu_count() or 4)
    print(f"Running {n_nets} networks across {n_workers} workers\n", flush=True)

    all_results = [None] * n_nets
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_one_network, ckpt,
                args.persistence, args.n_trials, args.soa_steps,
                args.noise_std, args.ITI, thresholds, args.seed,
                pairs, args.optimize_onset, args.max_onset_delay,
                None, None,
            ): i
            for i, ckpt in enumerate(ckpt_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            all_results[idx] = future.result()
            net_name = os.path.basename(ckpt_files[idx])
            done = sum(1 for r in all_results if r is not None)
            elapsed = time.time() - t0
            print(f"  [{done}/{n_nets}] {net_name} done ({elapsed:.0f}s total)",
                  flush=True)

    print(f"\nAll networks finished in {time.time() - t0:.0f}s\n")

    # Ensemble averaging
    labels = [p[2] for p in pairs]
    soa_list = args.soa_steps
    n_soas = len(soa_list)
    dt = 0.05

    print(f"{'='*70}")
    print(f"ENSEMBLE BCE RESULTS ({n_nets} networks)")
    print(f"{'='*70}")

    summary = {
        "n_nets": n_nets,
        "persistence": args.persistence,
        "n_trials": args.n_trials,
        "noise_std": args.noise_std,
        "soa_steps": soa_list,
        "optimize_onset": args.optimize_onset,
        "conditions": {},
    }

    for label in labels:
        print(f"\n{label}:")
        print(f"  {'SOA(ms)':>8s}  {'RT1_con':>8s}  {'RT1_inc':>8s}  "
              f"{'BCE':>8s}  {'BCE_se':>8s}  {'RT2_con':>8s}  {'RT2_inc':>8s}")

        cond_summary = {"soa_ms": [], "bce_mean": [], "bce_se": [],
                        "rt1_con": [], "rt1_inc": [],
                        "rt2_con": [], "rt2_inc": []}

        for si, soa in enumerate(soa_list):
            soa_ms = int(soa * dt * 1000)

            bce_per_net = []
            rt1c_per_net, rt1i_per_net = [], []
            rt2c_per_net, rt2i_per_net = [], []

            for net_res in all_results:
                r = net_res[label]
                bce_per_net.append(r["bce"][si])
                rt1c_per_net.append(r["rt1_congruent"][si])
                rt1i_per_net.append(r["rt1_incongruent"][si])
                rt2c_per_net.append(r["rt2_congruent"][si])
                rt2i_per_net.append(r["rt2_incongruent"][si])

            def _stat(vals):
                v = [x for x in vals if np.isfinite(x)]
                if not v:
                    return np.nan, np.nan
                return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))

            bce_m, bce_se = _stat(bce_per_net)
            r1c_m, _ = _stat(rt1c_per_net)
            r1i_m, _ = _stat(rt1i_per_net)
            r2c_m, _ = _stat(rt2c_per_net)
            r2i_m, _ = _stat(rt2i_per_net)

            print(f"  {soa_ms:>8d}  {r1c_m:>8.3f}  {r1i_m:>8.3f}  "
                  f"{bce_m:>+8.3f}  {bce_se:>8.3f}  {r2c_m:>8.3f}  {r2i_m:>8.3f}")

            cond_summary["soa_ms"].append(soa_ms)
            cond_summary["bce_mean"].append(bce_m)
            cond_summary["bce_se"].append(bce_se)
            cond_summary["rt1_con"].append(r1c_m)
            cond_summary["rt1_inc"].append(r1i_m)
            cond_summary["rt2_con"].append(r2c_m)
            cond_summary["rt2_inc"].append(r2i_m)

        # Mean BCE across SOAs
        all_bce = [b for b in cond_summary["bce_mean"] if np.isfinite(b)]
        short_bce = [cond_summary["bce_mean"][i] for i, s in enumerate(soa_list)
                     if s <= 5 and np.isfinite(cond_summary["bce_mean"][i])]
        print(f"  mean BCE (all SOAs): {np.mean(all_bce)*1000:+.1f}ms")
        if short_bce:
            print(f"  mean BCE (short SOAs <=250ms): {np.mean(short_bce)*1000:+.1f}ms")

        summary["conditions"][label] = cond_summary

    # Save
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    p_tag = f"p{args.persistence:.2f}".replace(".", "")
    onset_tag = "_onset" if args.optimize_onset else ""
    json_path = os.path.join(out_dir, f"bce_sweep_{p_tag}{onset_tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()