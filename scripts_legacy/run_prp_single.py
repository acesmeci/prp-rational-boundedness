#!/usr/bin/env python3
import os, sys, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Make sure repo root is on PYTHONPATH when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from prp_model.nn_wrapper import TaskNetworkWrapper
from prp_model.prp_simulator import sweep_soa
from prp_model.training_set import generate_training_set_matlab_style
from prp_model.threshold_utils import optimize_lca_threshold_dist, compute_fixed_threshold_for_task_meanargmax


# ---------- Trial generator (same as notebook) ----------
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


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to trained .pth/.pt model state_dict")
    p.add_argument("--seed", type=int, default=42)

    # PRP params
    p.add_argument("--persistence", type=float, default=0.90)
    p.add_argument("--ITI", type=float, default=0.5)
    p.add_argument("--dt_lca", type=float, default=0.1)
    p.add_argument("--t0", type=float, default=0.15)
    p.add_argument("--trials_per_soa", type=int, default=30)
    p.add_argument("--soa_start", type=int, default=5)
    p.add_argument("--soa_end", type=int, default=60)
    p.add_argument("--soa_step", type=int, default=5)
    p.add_argument("--optimize_onset", action="store_true")

    # z computation params
    p.add_argument("--z_task", type=str, default="A")
    p.add_argument("--z_K", type=int, default=27)
    p.add_argument("--z_repeats", type=int, default=100)
    p.add_argument("--z_thresholds", type=float, nargs=3, default=[0.1, 1.5, 0.1])  # start stop step
    p.add_argument("--z_cache", type=str, default=None,
                   help="Optional path to load/save z (json with {'z': ...}).")

    # output
    p.add_argument("--out_png", type=str, default="output/plots/single/prp_single_rt2.png")
    return p.parse_args()

def main():
    args = parse_args()

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # build wrapper (match notebook)
    wrapper = TaskNetworkWrapper(
        stim_input_dim=9,
        task_input_dim=9,
        hidden_dim=100,
        output_dim=9,
        learning_rate=0.3,
        device="cpu"
    )

    # load model
    state = torch.load(args.model_path, map_location="cpu")
    wrapper.model.load_state_dict(state)
    wrapper.model.eval()
    print("✅ Loaded model:", args.model_path)

    # load or compute z
    z_A = None
    if args.z_cache and os.path.exists(args.z_cache):
        with open(args.z_cache, "r") as f:
            z_A = float(json.load(f)["z"])
        print(f"✅ Loaded z from cache: {z_A:.3f} ({args.z_cache})")
    else:
        z_thresholds = np.arange(*args.z_thresholds)
        z_A = compute_fixed_threshold_for_task_meanargmax(
            wrapper,
            task_name=args.z_task,
            K=args.z_K,
            thresholds=z_thresholds,
            ITI=args.ITI,
            n_repeats=args.z_repeats,
            persistence=0.0,
            seed=args.seed,
        )
        if args.z_cache:
            Path(args.z_cache).parent.mkdir(parents=True, exist_ok=True)
            with open(args.z_cache, "w") as f:
                json.dump({"z": float(z_A)}, f)
            print(f"💾 Saved z cache to: {args.z_cache}")

    # PRP sweeps
    soa = list(range(args.soa_start, args.soa_end + 1, args.soa_step))
    gen_dep = lambda: generate_trial_pair(("B","A"))
    gen_ind = lambda: generate_trial_pair(("C","A"))

    results_ba = sweep_soa(
        wrapper, gen_dep, soa,
        n_trials_per_soa=args.trials_per_soa,
        persistence=args.persistence,
        dt_lca=args.dt_lca, t0=args.t0, ITI=args.ITI,
        z_task2_fixed=z_A,
        optimize_onset=args.optimize_onset
    )
    results_ca = sweep_soa(
        wrapper, gen_ind, soa,
        n_trials_per_soa=args.trials_per_soa,
        persistence=args.persistence,
        dt_lca=args.dt_lca, t0=args.t0, ITI=args.ITI,
        z_task2_fixed=z_A,
        optimize_onset=args.optimize_onset
    )

    # plot RT2 from S2 onset (paper-faithful PRP metric)
    plt.figure(figsize=(7,4))
    plt.plot(results_ba["soa"], results_ba["rt_task2_from_stim"], "x--", label="Dependent B→A")
    plt.plot(results_ca["soa"], results_ca["rt_task2_from_stim"], "x--", label="Independent C→A")

    plt.xlabel(f"SOA (steps; dt={args.dt_lca:.1f}s)")
    plt.ylabel("RT2 from S2 onset (s)")
    plt.title(f"Single-net PRP | p={args.persistence:.2f} ITI={args.ITI:.1f} OptimOnset={args.optimize_onset}")
    plt.legend()
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print("Saved plot:", out_png)

    plt.show()


if __name__ == "__main__":
    main()


"""
python -m scripts.run_prp_single \
  --model_path output/trained_models/trained_model_sim3_init01.pth \
  --z_cache notebooks/ensemble_ckpt_p09/net_00_z_A.json \
  --persistence 0.90 \
  --ITI 0.5 \
  --dt_lca 0.1 --t0 0.15 \
  --trials_per_soa 30 \
  --soa_start 5 --soa_end 60 --soa_step 5 \
  --optimize_onset \
  --out_png output/plots/single/prp_single_rt2.png
"""