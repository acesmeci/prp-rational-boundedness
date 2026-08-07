#!/usr/bin/env python
"""
diagnose_neto_decomposition.py
-------------------------------
Decompose the output-layer net input (Eq. 3.3) into its two additive
components during a PRP trial and plot their contribution to response
dimension 0 (Task A's response dimension) across time.

    net_o = Woh @ y_h  +  Wot @ x_t  +  bias

After persistence smoothing, each component tells us:
  - Woh @ y_h : the hidden-layer route (carries learned representational
                 structure; persists after Task 1 cue withdrawal because y_h
                 depends on the smoothed hidden net input)
  - Wot @ x_t : the direct task route (disappears instantly when cue is
                 withdrawn, except through the p * prev_net_o decay term)

The script runs one trial per condition (B->A, C->A) at each of several
SOAs, plotting the two components into response dimension 0 across time.

Usage
-----
    python scripts/diagnose_neto_decomposition.py
    python scripts/diagnose_neto_decomposition.py --persistence 0.65 --soa-steps 1 5 11
    python scripts/diagnose_neto_decomposition.py --ckpt ensemble_ckpt/net_00.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from prp_model.task_network import TaskNetwork  # noqa: E402
from prp_model.nn_wrapper import TaskNetworkWrapper  # noqa: E402
from prp_model.utils import (  # noqa: E402
    TASK_MAP, N_PATHWAYS, N_FEATURES,
    make_wrapper, generate_trial_pair, load_state,
)
from prp_model.lca import _DEFAULTS  # noqa: E402


RESP_DIM_0 = list(range(0, N_FEATURES))  # output units 0, 1, 2


def decompose_trial(
    wrapper: TaskNetworkWrapper,
    stim1, stim2, cue1, cue2,
    soa: int,
    persistence: float,
    max_timesteps: int = 60,
    t_off1: int | None = None,
):
    """
    Run one PRP trial and return per-timestep decomposition of net_o.

    If t_off1 is None, Task 1 cue stays on for the full trial (pass-1 style).
    If t_off1 is given, cue1 is withdrawn at that step (pass-2 style).

    Returns dict with arrays of shape (T,):
        woh_component  : mean of (Woh @ y_h) across resp dim 0 units
        wot_component  : mean of (Wot @ x_t) across resp dim 0 units
        total_net      : mean of full net_o across resp dim 0 units
        output_act     : mean of sigmoid(net_o) across resp dim 0 units
    """
    model = wrapper.model
    model.eval()
    device = next(model.parameters()).device

    I = stim1.shape[0]
    T_dim = cue1.shape[0]

    prev_net_h = None
    prev_net_o = None

    woh_log, wot_log, total_log, act_log = [], [], [], []

    with torch.no_grad():
        for t in range(max_timesteps):
            # Build stimulus: stim1 always on, stim2 added at SOA
            s = np.zeros(I, dtype=np.float32)
            s += stim1
            if t >= soa:
                s += stim2

            # Build cue: cue1 until t_off1 (or forever), cue2 from soa
            c = np.zeros(T_dim, dtype=np.float32)
            if t_off1 is None or t < t_off1:
                c += cue1
            if t >= soa:
                c += cue2

            x_t = torch.from_numpy(s[None, :]).to(device)
            t_t = torch.from_numpy(c[None, :]).to(device)

            # Hidden layer
            net_h_fresh = (
                model.fc_input_hidden(x_t)
                + model.fc_task_hidden(t_t)
                + model.bias_offset
            )
            if prev_net_h is not None:
                net_h = (1 - persistence) * net_h_fresh + persistence * prev_net_h
            else:
                net_h = net_h_fresh
            y_h = torch.sigmoid(net_h)
            prev_net_h = net_h

            # Output layer — decomposed
            comp_woh = model.fc_hidden_output(y_h)      # (1, 9)
            comp_wot = model.fc_task_output(t_t)         # (1, 9)
            net_o_fresh = comp_woh + comp_wot + model.bias_offset

            if prev_net_o is not None:
                net_o = (1 - persistence) * net_o_fresh + persistence * prev_net_o
            else:
                net_o = net_o_fresh
            y_o = torch.sigmoid(net_o)
            prev_net_o = net_o

            # Log mean across resp dim 0 units
            # Note: we log the FRESH components, not the smoothed ones,
            # so we can see what each route is injecting at each step.
            # The smoothed total is logged separately.
            woh_log.append(float(comp_woh[0, RESP_DIM_0].mean()))
            wot_log.append(float(comp_wot[0, RESP_DIM_0].mean()))
            total_log.append(float(net_o[0, RESP_DIM_0].mean()))
            act_log.append(float(y_o[0, RESP_DIM_0].mean()))

    return {
        "woh_component": np.array(woh_log),
        "wot_component": np.array(wot_log),
        "total_net": np.array(total_log),
        "output_act": np.array(act_log),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", default=None,
                    help="path to a single network checkpoint (default: first in ensemble_ckpt/)")
    ap.add_argument("--persistence", type=float, default=0.65)
    ap.add_argument("--soa-steps", type=int, nargs="+", default=[1, 5, 11],
                    help="SOA values in simulation steps (default: 1 5 11 = 50 250 550 ms)")
    ap.add_argument("--t-off1", type=int, default=8,
                    help="step at which Task 1 cue is withdrawn (default 8 = ~400ms, "
                         "roughly matching RT1 at long SOA; set to 0 to disable gating)")
    ap.add_argument("--max-timesteps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output/figures/neto_decomposition.png")
    args = ap.parse_args()

    # Load one network
    if args.ckpt:
        ckpt = args.ckpt
    else:
        import glob
        candidates = sorted(glob.glob(os.path.join(PROJECT_ROOT, "ensemble_ckpt*/net_00*.pt")))
        if not candidates:
            raise SystemExit("No checkpoint found. Pass --ckpt explicitly.")
        ckpt = candidates[0]
    print(f"Using checkpoint: {ckpt}")
    wrapper = load_state(ckpt)

    dt = _DEFAULTS["dt"]
    conditions = [("B", "A", "dependent"), ("C", "A", "independent")]
    soa_list = args.soa_steps

    n_rows = len(soa_list)
    n_cols = len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.0 * n_rows),
                             sharex=True, sharey="row", squeeze=False)

    for col, (t1, t2, label) in enumerate(conditions):
        s1, s2, c1, c2 = generate_trial_pair((t1, t2), seed=args.seed)

        for row, soa in enumerate(soa_list):
            ax = axes[row, col]
            soa_ms = int(soa * dt * 1000)

            t_off1 = args.t_off1 if args.t_off1 > 0 else None
            res = decompose_trial(
                wrapper, s1, s2, c1, c2,
                soa=soa, persistence=args.persistence,
                max_timesteps=args.max_timesteps, t_off1=t_off1,
            )

            ts = np.arange(args.max_timesteps) * dt * 1000  # ms

            ax.plot(ts, res["woh_component"], label="$W_{oh} \\cdot y_h$",
                    color="#4c72b0", lw=1.8)
            ax.plot(ts, res["wot_component"], label="$W_{ot} \\cdot x_t$",
                    color="#c44e52", lw=1.8)
            ax.axhline(-2.0, color="gray", ls=":", lw=0.8, label="bias (−2)")
            ax.axvline(soa_ms, color="green", ls="--", lw=1, alpha=0.7, label="SOA")
            if t_off1 is not None:
                ax.axvline(t_off1 * dt * 1000, color="orange", ls="--", lw=1,
                           alpha=0.7, label="cue1 off")

            ax.set_ylabel("mean net input\n(resp dim 0)")
            if row == 0:
                ax.set_title(f"{t1}→{t2}  ({label})", fontsize=11, fontweight="bold")
            ax.text(0.98, 0.95, f"SOA = {soa_ms} ms",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right", ncol=2)
            if row == n_rows - 1:
                ax.set_xlabel("time (ms)")

    fig.suptitle(
        f"Decomposition of $net_o$ into response dimension 0\n"
        f"p = {args.persistence},  t_off1 = {args.t_off1} steps "
        f"({int(args.t_off1 * dt * 1000)} ms)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
