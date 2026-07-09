#!/usr/bin/env python3
"""
Run PRP ensemble sweep: train E networks (if missing), run PRP SOA sweeps,
save results as JSON.

v4 (09 Jul 2026):
- Dual-task, session-level threshold selection is now the DEFAULT
  (--z_context dual): one (z1, z2) per condition, selected by
  accuracy-constrained expected-RR over the session's SOA mixture
  (--z_soa_refs) with a context-appropriate floor (--acc_floor_dual, 0.95).
  Single-task selection retained as a diagnostic (--z_context single).
- z2 is SHARED across conditions (max over conditions): criterion
  differences must not confound the representational comparison.
- --max_onset_delay (default 15) threaded to the onset policy.

Example (smoke cells d'/f', E=2, p=0.75):
    python -m scripts.run_prp_sweep --store_dir ensemble_ckpt_p09 --E 2 \
        --persistence 0.75 --trials_per_soa 40 --soa_start 1 --soa_end 20 \
        --soa_step 2 --ITI 4.0 --workers 0 --plot                    # (d')
    ... same + --optimize_onset                                       # (f')
"""
import os, json, argparse, time
from pathlib import Path

import numpy as np
import torch

from prp_model.nn_wrapper import TaskNetworkWrapper
from prp_model.training_set import generate_training_set_matlab_style
from prp_model.prp_simulator import sweep_soa
from prp_model.threshold_utils import (
    compute_fixed_threshold_for_task_meanargmax,
    compute_condition_thresholds,
)
from prp_model.lca import MS_PER_STEP, _DEFAULTS
from prp_model.utils import (
    make_wrapper,
    generate_trial_pair,
    save_state, load_state,
    save_threshold, load_threshold,
    average_with_se,
    steepest_adjacent_slope,
)

TASK1_BY_COND = {"dep": "B", "ind": "C"}  # Task 2 is always A


# ===================================================================
# Naming
# ===================================================================
def make_tag(args):
    p_tag = f"{int(round(args.persistence * 100)):03d}"
    iti_tag = f"{int(round(args.ITI * 10)):02d}"
    s_tag = f"{int(round(args.noise_std * 100)):03d}"
    zc_tag = "D" if args.z_context == "dual" else "S"
    af_tag = f"{int(round(args.acc_floor_dual * 100)):02d}"
    return (f"E{args.E}_p{p_tag}_nt{args.trials_per_soa}"
            f"_soa{args.soa_start}-{args.soa_end}-{args.soa_step}"
            f"_step{MS_PER_STEP:03d}ms_ITI{iti_tag}"
            f"_s{s_tag}_zc{zc_tag}_af{af_tag}"
            f"_fx{int(args.fix_z_task1)}_oo{int(args.optimize_onset)}"
            f"_od{args.max_onset_delay}")


def _noise_tag(noise_std):
    return f"{int(round(noise_std * 100)):03d}"


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
# Threshold precompute helpers (cached per network)
# ===================================================================
def _get_single_task_z(wrapper, store_dir, net_idx, task_name, thresholds,
                       ITI, z_K, z_repeats, noise_std, seed):
    path = os.path.join(
        store_dir, f"net_{net_idx:02d}_z_{task_name}_s{_noise_tag(noise_std)}.json"
    )
    if os.path.exists(path):
        return load_threshold(path)
    z = compute_fixed_threshold_for_task_meanargmax(
        wrapper, task_name=task_name, K=z_K,
        thresholds=thresholds, ITI=ITI, n_repeats=z_repeats,
        persistence=0.0, seed=seed, verbose=False,
        noise_std=noise_std,
    )
    save_threshold(z, path)
    return z


def _get_condition_zs(wrapper, store_dir, net_idx, task1_name, task2_name,
                      thresholds, ITI, z_repeats, noise_std, persistence,
                      soa_refs, acc_floor, acc_floor_task1, seed):
    refs_tag = "-".join(str(int(s)) for s in soa_refs)
    path = os.path.join(
        store_dir,
        f"net_{net_idx:02d}_zpair_{task1_name}{task2_name}_dual"
        f"_p{int(round(persistence*100)):03d}_s{_noise_tag(noise_std)}"
        f"_af{int(round(acc_floor_task1*100)):02d}-{int(round(acc_floor*100)):02d}"
        f"_iti{int(round(ITI*10)):02d}"
        f"_ref{refs_tag}.json"
    )
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return float(d["z1"]), float(d["z2"])
    z1, z2 = compute_condition_thresholds(
        wrapper, task1_name, task2_name,
        soa_refs=soa_refs, n_stim=20,
        thresholds=thresholds, ITI=ITI, n_repeats=z_repeats,
        persistence=persistence, seed=seed, verbose=False,
        noise_std=noise_std, acc_floor=acc_floor,
        acc_floor_task1=acc_floor_task1,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"z1": float(z1), "z2": float(z2)}, f)
    return z1, z2


# ===================================================================
# Per-network job
# ===================================================================
def per_network_job(
    net_idx, store_dir,
    train_if_missing, train_epochs, stop_loss,
    z_task, z_K, z_repeats, thresholds, ITI,
    prp_persistence, prp_trials_per_soa, prp_soa,
    dt_lca, t0, optimize_onset,
    fix_z_task1, z_context, noise_std,
    z_soa_refs, acc_floor_dual, acc_floor_task1,
    max_onset_delay,
):
    t_start = time.time()
    np.random.seed(777000 + net_idx)   # pins LCA noise per network (serial within job)
    model_path = os.path.join(store_dir, f"net_{net_idx:02d}.pt")

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

    # 2) Thresholds per condition
    cond_zs = {}  # cond -> (z1 or None, z2)
    if z_context == "dual":
        for cond, t1 in TASK1_BY_COND.items():
            z1, z2 = _get_condition_zs(
                wrapper, store_dir, net_idx, t1, z_task,
                thresholds, ITI, z_repeats, noise_std,
                prp_persistence, z_soa_refs, acc_floor_dual,
                acc_floor_task1, seed=1000 + net_idx,
            )
            cond_zs[cond] = (z1, z2)
            print(f"  [net {net_idx:02d}] {t1}->A dual-context "
                  f"z1={z1:.2f}, z2={z2:.2f}")
        # Shared Task-2 criterion across conditions (experimental control:
        # criterion differences must not confound the representational
        # comparison). Worst-case: caution set for the hardest condition.
        z2_shared = max(cond_zs[c][1] for c in cond_zs)
        cond_zs = {c: (cond_zs[c][0], z2_shared) for c in cond_zs}
        print(f"  [net {net_idx:02d}] shared z2 = {z2_shared:.2f}")
    else:
        z_A = _get_single_task_z(wrapper, store_dir, net_idx, z_task,
                                 thresholds, ITI, z_K, z_repeats,
                                 noise_std, seed=1000 + net_idx)
        for cond, t1 in TASK1_BY_COND.items():
            if fix_z_task1:
                z1 = _get_single_task_z(wrapper, store_dir, net_idx, t1,
                                        thresholds, ITI, z_K, z_repeats,
                                        noise_std, seed=1000 + net_idx)
            else:
                z1 = None  # legacy per-trial fitting
            cond_zs[cond] = (z1, z_A)
            z1_str = f"{z1:.2f}" if z1 is not None else "per-trial"
            print(f"  [net {net_idx:02d}] {t1}->A single-context "
                  f"z1={z1_str}, z2={z_A:.2f}")

    # 3) PRP sweeps
    gens = {"dep": lambda seed: generate_trial_pair(("B", "A"), seed=seed),
            "ind": lambda seed: generate_trial_pair(("C", "A"), seed=seed)}
    out = {"net_idx": net_idx,
           "z": {c: {"z1": cond_zs[c][0], "z2": cond_zs[c][1]} for c in cond_zs}}
    for cond in ("dep", "ind"):
        z1, z2 = cond_zs[cond]
        out[cond] = sweep_soa(
            wrapper, gens[cond], prp_soa,
            n_trials_per_soa=prp_trials_per_soa,
            persistence=prp_persistence,
            dt_lca=dt_lca, t0=t0, ITI=ITI, noise_std=noise_std,
            z_task1_fixed=z1, z_task2_fixed=z2,
            optimize_onset=optimize_onset,
            max_onset_delay=max_onset_delay,
            base_seed= 500000 + net_idx * 100000
        )

    print(f"  [net {net_idx:02d}] Done in {time.time() - t_start:.1f}s")
    return out


# ===================================================================
# Orchestrator
# ===================================================================
def run_ensemble(args):
    store_dir = args.store_dir
    os.makedirs(store_dir, exist_ok=True)

    soa_list = list(range(args.soa_start, args.soa_end + 1, args.soa_step))
    tag = make_tag(args)

    print(f"\n{'='*60}\nPRP Ensemble Sweep: {tag}\n{'='*60}")
    print(f"  Networks: {args.E} | p={args.persistence} | ITI={args.ITI}s "
          f"| sigma={args.noise_std} | z_context={args.z_context} "
          f"| acc_floor_dual={args.acc_floor_dual}")
    print(f"  z_soa_refs: {args.z_soa_refs} | fix_z1={args.fix_z_task1} "
          f"| onset_optim={args.optimize_onset} (window {args.max_onset_delay})")
    print(f"  SOA: {soa_list[0]}-{soa_list[-1]} step {args.soa_step} "
          f"({soa_list[0]*MS_PER_STEP:.0f}-{soa_list[-1]*MS_PER_STEP:.0f} ms) "
          f"| trials/SOA: {args.trials_per_soa} | workers: {args.workers}")
    print(f"{'='*60}\n")

    t_total_start = time.time()
    job_args = (
        store_dir,
        args.train_if_missing, args.train_epochs, args.stop_loss,
        args.z_task, args.z_K, args.z_repeats,
        np.arange(*args.thresholds), args.ITI,
        args.persistence, args.trials_per_soa, soa_list,
        args.dt_lca, args.t0, args.optimize_onset,
        args.fix_z_task1, args.z_context, args.noise_std,
        tuple(args.z_soa_refs), args.acc_floor_dual, args.acc_floor_task1,
        args.max_onset_delay,
    )

    if args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            jobs = [pool.apply_async(per_network_job, (i,) + job_args)
                    for i in range(args.E)]
            per_net = [j.get() for j in jobs]
    else:
        per_net = [per_network_job(i, *job_args) for i in range(args.E)]

    print(f"\nAll {args.E} networks completed in {time.time()-t_total_start:.1f}s")

    keys_to_avg = [
        "rt_task1", "rt_task1_correct", "acc_task1", "decided_task1",
        "rt_task2", "rt_task2_from_stim", "rt_task2_from_stim_correct",
        "rt_task2_tail", "acc_task2", "decided_task2", "onset2",
    ]
    dep_avg = average_with_se([d["dep"] for d in per_net], keys_to_avg)
    ind_avg = average_with_se([d["ind"] for d in per_net], keys_to_avg)

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
            "E": args.E, "persistence": args.persistence,
            "trials_per_soa": args.trials_per_soa,
            "soa_start": args.soa_start, "soa_end": args.soa_end,
            "soa_step": args.soa_step,
            "dt_lca": args.dt_lca, "t0": args.t0, "ITI": args.ITI,
            "noise_std": args.noise_std, "z_context": args.z_context,
            "z_soa_refs": list(args.z_soa_refs),
            "acc_floor_dual": args.acc_floor_dual,
            "acc_floor_task1": args.acc_floor_task1,
            "fix_z_task1": args.fix_z_task1,
            "optimize_onset": args.optimize_onset,
            "max_onset_delay": args.max_onset_delay,
            "ms_per_step": MS_PER_STEP,
        },
        "soa": soa_list,
        "avg": {"dep": dep_avg, "ind": ind_avg},
        "z_per_net": [d["z"] for d in per_net],
        "per_net": per_net_serializable,
    }

    results_dir = Path("output/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved results: {json_path}")
    print("z per net:", json.dumps(out_dict["z_per_net"]))

    dep_slope = steepest_adjacent_slope(
        np.array(soa_list, float),
        np.array(dep_avg["rt_task2_from_stim_correct"], float),
    )
    print(f"Steepest B->A slope (correct trials): {dep_slope['slope_s_per_s']:.2f} "
          f"(segment {dep_slope['seg'][0]:.0f}-{dep_slope['seg'][1]:.0f} steps)")

    return out_dict, tag


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Run PRP ensemble sweep. Results -> output/results/<tag>.json"
    )
    p.add_argument("--E", type=int, default=20)
    p.add_argument("--store_dir", type=str, default="ensemble_ckpt")
    p.add_argument("--workers", type=int, default=6)

    # Training
    p.add_argument("--train_if_missing", action="store_true")
    p.add_argument("--train_epochs", type=int, default=5000)
    p.add_argument("--stop_loss", type=float, default=1e-3)

    # Threshold selection
    p.add_argument("--z_task", type=str, default="A")
    p.add_argument("--z_K", type=int, default=27)
    p.add_argument("--z_repeats", type=int, default=100)
    p.add_argument("--thresholds", type=float, nargs=3, default=[0.1, 1.5, 0.1])
    p.add_argument("--z_context", type=str, choices=["single", "dual"],
                   default="dual",
                   help="dual (default): session-level (z1,z2) per condition "
                        "from dual-task SOA-mixture context, z2 shared across "
                        "conditions. single: legacy single-task selection "
                        "(diagnostic only).")
    p.add_argument("--z_soa_refs", type=int, nargs="+", default=[3, 8, 16],
                   help="Reference SOAs (steps) pooled for session-level "
                        "dual-context selection.")
    p.add_argument("--acc_floor_dual", type=float, default=0.95,
                   help="Accuracy floor for dual-context selection "
                        "(empirical dual-task accuracy: 90-95%%).")
    p.add_argument("--acc_floor_task1", type=float, default=0.99,
                   help="Accuracy floor for Task 1 (the protected task; "
                        "empirical T1 accuracy 97-99%%).")
    p.add_argument("--fix_z_task1", action="store_true",
                   help="[single context only] fixed z_B/z_C for Task 1 "
                        "instead of per-trial fitting.")

    # PRP sweep
    p.add_argument("--persistence", type=float, default=0.80)
    p.add_argument("--trials_per_soa", type=int, default=50)
    p.add_argument("--soa_start", type=int, default=1)
    p.add_argument("--soa_end", type=int, default=20)
    p.add_argument("--soa_step", type=int, default=2)

    # Timing / LCA
    p.add_argument("--dt_lca", type=float, default=_DEFAULTS["dt"])
    p.add_argument("--t0", type=float, default=_DEFAULTS["t0"])
    p.add_argument("--ITI", type=float, default=0.5)
    p.add_argument("--noise_std", type=float, default=_DEFAULTS["noise_std"])

    # Onset policy
    p.add_argument("--optimize_onset", action="store_true")
    p.add_argument("--max_onset_delay", type=int, default=15,
                   help="Onset-policy search window (steps beyond SOA).")

    p.add_argument("--plot", action="store_true")
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