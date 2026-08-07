#!/usr/bin/env python
"""
plot_wot_structure.py
---------------------
Inspect the task->output weight matrix (W_ot) across the trained ensemble.

W_ot is the direct task-layer projection to the output layer (Eq. 3.3). It has
no access to the stimulus, so it cannot select *which* feature is correct within
a response dimension; the only thing it can encode is which response dimensions
are currently irrelevant. Training targets zero every output unit outside the
task-relevant response dimension, so that suppression has to live somewhere,
and W_ot is the cheapest place for it.

This script measures what actually got learned, and in particular how each
trained task projects into response dimension 0, which is Task A's (Task 2's)
response dimension in every PRP condition.

Index conventions (from utils.py / training_set.py):
  task unit index   = in_dim * N_PATHWAYS + out_dim        (row-major)
  output unit index = out_dim * N_FEATURES + feature
  torch Linear weight shape = (out_features, in_features), so W[output, task]

Usage
-----
    python scripts/plot_wot_structure.py
    python scripts/plot_wot_structure.py --ckpt-glob "ensemble_ckpt_p09/*.pt"
    python scripts/plot_wot_structure.py --out output/figures/wot_structure.png
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from prp_model.utils import TASK_MAP, N_PATHWAYS, N_FEATURES  # noqa: E402

TASKS = ["A", "B", "C", "D", "E"]
WOT_KEY = "fc_task_output.weight"


def task_unit_index(name: str) -> int:
    in_dim, out_dim = TASK_MAP[name]
    return in_dim * N_PATHWAYS + out_dim


def load_wot(ckpt_glob: str) -> np.ndarray:
    """Return (E, n_tasks, n_output_units) array of W_ot rows for TASKS."""
    paths = sorted(glob.glob(ckpt_glob))
    if not paths:
        raise SystemExit(
            f"No checkpoints matched {ckpt_glob!r}.\n"
            "Pass the right pattern with --ckpt-glob (e.g. 'ensemble_ckpt_p09/*.pt')."
        )

    cols = [task_unit_index(t) for t in TASKS]
    mats = []
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        if WOT_KEY not in state:
            raise SystemExit(
                f"{p} has no key {WOT_KEY!r}. Keys present: {sorted(state.keys())}"
            )
        w = state[WOT_KEY].detach().cpu().numpy()   # (out_units, task_units)
        mats.append(w[:, cols].T)                   # (n_tasks, out_units)

    print(f"Loaded {len(mats)} checkpoints from {ckpt_glob}")
    return np.stack(mats)


def summarise(W: np.ndarray) -> dict:
    """W: (E, n_tasks, n_out). Prints a summary and returns it as a dict."""
    E = W.shape[0]

    # collapse output units to response dimensions -> (E, n_tasks, N_PATHWAYS)
    Wdim = W.reshape(E, len(TASKS), N_PATHWAYS, N_FEATURES).mean(axis=-1)
    m = Wdim.mean(axis=0)
    se = Wdim.std(axis=0, ddof=1) / np.sqrt(E)

    print(f"\nEnsemble n = {E}")
    print("\nMean W_ot weight into each response dimension (mean +/- SE across nets)")
    header = f"{'task':<6}{'own':<6}" + "".join(f"{'resp ' + str(d):>18}" for d in range(N_PATHWAYS))
    print(header)
    print("-" * len(header))
    for k, t in enumerate(TASKS):
        own = TASK_MAP[t][1]
        cells = "".join(f"{m[k, d]:>+11.3f} +-{se[k, d]:<5.3f}" for d in range(N_PATHWAYS))
        print(f"{t:<6}{own:<6}{cells}")

    own_vals, other_vals = [], []
    for k, t in enumerate(TASKS):
        own = TASK_MAP[t][1]
        for d in range(N_PATHWAYS):
            (own_vals if d == own else other_vals).append(Wdim[:, k, d])
    own_vals = np.concatenate(own_vals)
    other_vals = np.concatenate(other_vals)

    print("\nOwn-dimension vs other-dimension weights, pooled over tasks and nets")
    print(f"  own    : {own_vals.mean():+.3f}  (SD {own_vals.std(ddof=1):.3f}, n={own_vals.size})")
    print(f"  other  : {other_vals.mean():+.3f}  (SD {other_vals.std(ddof=1):.3f}, n={other_vals.size})")

    print("\nProjection into RESPONSE DIMENSION 0 (Task A's response dimension)")
    for k, t in enumerate(TASKS):
        own_flag = "   <- this task uses resp dim 0" if TASK_MAP[t][1] == 0 else ""
        frac_neg = float((Wdim[:, k, 0] < 0).mean())
        print(f"  task {t}: {m[k, 0]:+.3f} +- {se[k, 0]:.3f}   "
              f"negative in {frac_neg * 100:>3.0f}% of nets{own_flag}")

    iB, iC = TASKS.index("B"), TASKS.index("C")
    diff = Wdim[:, iB, 0] - Wdim[:, iC, 0]
    print(f"\n  B minus C into resp dim 0: {diff.mean():+.3f} +- "
          f"{diff.std(ddof=1) / np.sqrt(E):.3f}   "
          f"(B more inhibitory in {float((diff < 0).mean()) * 100:.0f}% of nets)")
    print("  If this is near zero, W_ot inhibition of Task A is condition-general")
    print("  and cannot by itself produce the dependent/independent separation.")

    return {
        "n_networks": E,
        "tasks": TASKS,
        "resp_dim_mean": m.tolist(),
        "resp_dim_se": se.tolist(),
        "own_mean": float(own_vals.mean()),
        "other_mean": float(other_vals.mean()),
        "B_minus_C_into_resp0_mean": float(diff.mean()),
        "B_minus_C_into_resp0_se": float(diff.std(ddof=1) / np.sqrt(E)),
    }


def make_figure(W: np.ndarray, out_path: str) -> None:
    E = W.shape[0]
    Wm = W.mean(axis=0)
    Wdim = W.reshape(E, len(TASKS), N_PATHWAYS, N_FEATURES).mean(axis=-1)
    dm = Wdim.mean(axis=0)
    dse = Wdim.std(axis=0, ddof=1) / np.sqrt(E)

    fig = plt.figure(figsize=(13.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.45)

    # Panel A: full W_ot
    ax = fig.add_subplot(gs[0])
    lim = float(np.abs(Wm).max())
    im = ax.imshow(Wm, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")    
    ax.set_xticks(range(N_PATHWAYS * N_FEATURES))
    ax.set_xticklabels([f"{d}.{f}" for d in range(N_PATHWAYS) for f in range(N_FEATURES)],
                       fontsize=8)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([f"{t}  ({TASK_MAP[t][0]}\u2192{TASK_MAP[t][1]})" for t in TASKS])
    for b in np.arange(1, N_PATHWAYS) * N_FEATURES - 0.5:
        ax.axvline(b, color="k", lw=1.6)
    for k, t in enumerate(TASKS):
        own = TASK_MAP[t][1]
        ax.add_patch(plt.Rectangle((own * N_FEATURES - 0.5, k - 0.5), N_FEATURES, 1,
                                   fill=False, ec="k", lw=2.2, ls="--"))
    ax.set_xlabel("output unit  (response dim . feature)")
    ax.set_ylabel("task unit")
    ax.set_title("A.  $W_{ot}$, ensemble mean\ndashed box = task's own response dimension",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    # Panel B: collapsed to response dimensions
    ax = fig.add_subplot(gs[1])
    lim2 = float(np.abs(dm).max())
    im2 = ax.imshow(dm, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-lim2, vcenter=0.0, vmax=lim2), aspect="auto")    
    for k in range(len(TASKS)):
        for d in range(N_PATHWAYS):
            ax.text(d, k, f"{dm[k, d]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(N_PATHWAYS))
    ax.set_xticklabels([f"resp {d}" for d in range(N_PATHWAYS)])
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS)
    ax.set_xlabel("response dimension")
    ax.set_title("B.  mean weight per\nresponse dimension", fontsize=10, loc="left")
    fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.02)

    # Panel C: projection into Task A's response dimension
    ax = fig.add_subplot(gs[2])
    x = np.arange(len(TASKS))
    colors = ["#c44e52" if TASK_MAP[t][1] == 0 else "#4c72b0" for t in TASKS]
    ax.bar(x, dm[:, 0], yerr=dse[:, 0], color=colors, capsize=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_xlabel("task unit")
    ax.set_ylabel("mean weight into response dim 0")
    ax.set_title("C.  projection into Task A's\nresponse dimension", fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c44e52"),
               plt.Rectangle((0, 0), 1, 1, color="#4c72b0")]
    ax.legend(handles, ["uses resp dim 0", "does not"], fontsize=8, loc="lower left")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nFigure written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-glob", default="ensemble_ckpt_p09/*.pt",
                    help="glob for ensemble checkpoints, relative to project root")
    ap.add_argument("--out", default="output/figures/wot_structure.png")
    ap.add_argument("--json-out", default="output/wot_structure_summary.json")
    args = ap.parse_args()

    ckpt_glob = args.ckpt_glob
    if not os.path.isabs(ckpt_glob):
        ckpt_glob = os.path.join(PROJECT_ROOT, ckpt_glob)

    W = load_wot(ckpt_glob)
    summary = summarise(W)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    make_figure(W, out_path)

    json_path = (args.json_out if os.path.isabs(args.json_out)
                 else os.path.join(PROJECT_ROOT, args.json_out))
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {json_path}")


if __name__ == "__main__":
    main()
