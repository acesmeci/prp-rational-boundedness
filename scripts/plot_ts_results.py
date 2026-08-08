#!/usr/bin/env python
"""
plot_ts_results.py
------------------
Plot task-switching results in the style of Musslick et al. (2020) Fig. 20.

Three panels (structural, functional, independent), each showing RT
for switch (dashed) and repeat (solid) across congruent/incongruent.

Usage
-----
    # From a notebook, after running sweep_task_switching:
    from scripts.plot_ts_results import plot_fig20
    plot_fig20(results, out_path="output/figures/task_switching_fig20.png")

    # Or as a standalone script with a saved results dict:
    python scripts/plot_ts_results.py --results output/ts_results.json
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Colors matching the paper's scheme
COLORS = {
    "structural": "#2ca02c",   # green
    "functional": "#ff7f0e",   # orange
    "independent": "#1f77b4",  # blue
}

LABELS = {
    "structural": "Switch from E to A\n(Structural Dependence)",
    "functional": "Switch from B to A\n(Functional Dependence)",
    "independent": "Switch from C to A\n(Independence)",
}


def plot_fig20(
    results: dict,
    out_path: str = "output/figures/task_switching_fig20.png",
    title: str | None = None,
    figsize: tuple = (12, 4),
    y_limits: tuple | None = None,
):
    """
    Plot task-switching results matching Fig. 20 layout.

    Parameters
    ----------
    results : dict
        Output of sweep_task_switching().
    out_path : str
        Path to save the figure.
    title : str or None
        Optional suptitle.
    figsize : tuple
        Figure size.
    y_limits : (ymin, ymax) or None
        Shared y-axis limits. If None, auto-determined from data.
    """
    conditions = [k for k in ("structural", "functional", "independent") if k in results]
    n_panels = len(conditions)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    x = np.array([0, 1])
    x_labels = ["Congruent", "Incongruent"]

    all_rts = []

    for ax, cond in zip(axes, conditions):
        color = COLORS[cond]
        data = results[cond]

        # Extract RTs and SEs
        sw_con = data["switch"]["congruent"]
        sw_inc = data["switch"]["incongruent"]
        rep_con = data["repeat"]["congruent"]
        rep_inc = data["repeat"]["incongruent"]

        sw_rts = [sw_con["rt_correct_mean"], sw_inc["rt_correct_mean"]]
        sw_ses = [sw_con["rt_correct_se"], sw_inc["rt_correct_se"]]
        rep_rts = [rep_con["rt_correct_mean"], rep_inc["rt_correct_mean"]]
        rep_ses = [rep_con["rt_correct_se"], rep_inc["rt_correct_se"]]

        all_rts.extend(sw_rts + rep_rts)

        # Plot switch (dashed, square markers)
        ax.errorbar(x, sw_rts, yerr=sw_ses,
                    fmt="s--", color=color, markersize=7, lw=2,
                    capsize=3, label="Task Switch")

        # Plot repeat (solid, square markers)
        ax.errorbar(x, rep_rts, yerr=rep_ses,
                    fmt="s-", color=color, markersize=7, lw=2,
                    capsize=3, label="Task Repetition")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_title(LABELS[cond], fontsize=11, color=color, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

        # Annotate switch costs
        for i, cong in enumerate(("congruent", "incongruent")):
            cost_key = f"cost_{cong}"
            if cost_key in data and np.isfinite(data[cost_key]):
                cost = data[cost_key]
                y_mid = (sw_rts[i] + rep_rts[i]) / 2
                ax.annotate(f"{cost:+.2f}s",
                            xy=(x[i] + 0.08, y_mid),
                            fontsize=7, color="gray", style="italic")

    axes[0].set_ylabel("Reaction Time (s)", fontsize=11)

    if y_limits:
        axes[0].set_ylim(y_limits)
    else:
        valid = [r for r in all_rts if np.isfinite(r)]
        if valid:
            margin = (max(valid) - min(valid)) * 0.15
            axes[0].set_ylim(min(valid) - margin, max(valid) + margin)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


def plot_fig20_ensemble(
    all_results: list[dict],
    out_path: str = "output/figures/task_switching_fig20_ensemble.png",
    title: str | None = None,
    figsize: tuple = (12, 4),
):
    """
    Plot ensemble-averaged task-switching results.

    Parameters
    ----------
    all_results : list[dict]
        List of sweep_task_switching() outputs, one per network.
    """
    conditions = [k for k in ("structural", "functional", "independent")
                  if k in all_results[0]]

    # Average across networks
    averaged = {}
    for cond in conditions:
        averaged[cond] = {"switch": {}, "repeat": {}}
        for trial_type in ("switch", "repeat"):
            for cong_key in ("congruent", "incongruent"):
                rts = [r[cond][trial_type][cong_key]["rt_correct_mean"]
                       for r in all_results
                       if np.isfinite(r[cond][trial_type][cong_key]["rt_correct_mean"])]
                averaged[cond][trial_type][cong_key] = {
                    "rt_correct_mean": float(np.mean(rts)) if rts else np.nan,
                    "rt_correct_se": float(np.std(rts, ddof=1) / np.sqrt(len(rts))) if len(rts) > 1 else np.nan,
                }

        # Average costs
        for cong_key in ("congruent", "incongruent"):
            costs = [r[cond][f"cost_{cong_key}"]
                     for r in all_results
                     if np.isfinite(r[cond][f"cost_{cong_key}"])]
            averaged[cond][f"cost_{cong_key}"] = float(np.mean(costs)) if costs else np.nan

    n_nets = len(all_results)
    plot_title = title or f"Task-Switching Results (n={n_nets} networks)"
    plot_fig20(averaged, out_path=out_path, title=plot_title, figsize=figsize)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="Path to JSON results file")
    ap.add_argument("--out", default="output/figures/task_switching_fig20.png")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    plot_fig20(results, out_path=args.out)