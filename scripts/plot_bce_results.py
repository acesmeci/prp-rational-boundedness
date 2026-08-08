#!/usr/bin/env python
"""
plot_bce_results.py
-------------------
Plot backward crosstalk effect (BCE) results.

Panel A: RT1 by congruency across SOA for the functional condition,
         showing the raw effect (incongruent RT1 elevated at short SOAs).
Panel B: BCE (RT1_inc - RT1_con) across SOA for both conditions,
         showing the condition × SOA interaction.

Usage
-----
    from scripts.plot_bce_results import plot_bce, plot_bce_ensemble
    plot_bce(results, out_path="output/figures/bce.png")
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

COLORS = {
    "functional": "#ff7f0e",
    "independent": "#1f77b4",
    "structural": "#2ca02c",
}

DT = 0.05  # seconds per step


def _soa_to_ms(soa_steps):
    return [int(s * DT * 1000) for s in soa_steps]


def plot_bce(
    results: dict,
    out_path: str = "output/figures/bce.png",
    title: str | None = None,
    figsize: tuple = (11, 4.5),
):
    """
    Plot BCE results from run_bce_comparison output.

    Parameters
    ----------
    results : dict
        Output of run_bce_comparison(). Keys are condition labels,
        values are sweep_soa_bce dicts.
    """
    conditions = [k for k in ("functional", "independent", "structural") if k in results]
    primary = "functional" if "functional" in results else conditions[0]
    secondary = "independent" if "independent" in results else (
        [c for c in conditions if c != primary][0] if len(conditions) > 1 else None)

    n_panels = 3 if secondary else 2
    figsize = (figsize[0] if figsize[0] > 12 else 15, figsize[1])
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    # ── Panel A: RT1 by congruency for functional (B→A) ──
    ax = axes[0]
    res = results[primary]
    soa_ms = _soa_to_ms(res["soa"])
    color = COLORS.get(primary, "#ff7f0e")

    ax.errorbar(soa_ms, res["rt1_congruent"], yerr=res["rt1_congruent_se"],
                fmt="o-", color=color, lw=2, markersize=6, capsize=3,
                label="Congruent")
    ax.errorbar(soa_ms, res["rt1_incongruent"], yerr=res["rt1_incongruent_se"],
                fmt="s--", color=color, lw=2, markersize=6, capsize=3,
                label="Incongruent")

    t1_name, t2_name = primary, "B→A"
    if primary == "functional":
        t2_name = "B→A"
    elif primary == "structural":
        t2_name = "E→A"
    ax.set_xlabel("SOA (ms)", fontsize=11)
    ax.set_ylabel("Task 1 RT (s)", fontsize=11)
    ax.set_title(f"A.  Task 1 RT — {primary}\n({t2_name})",
                 fontsize=11, loc="left", color=color, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(soa_ms)
    if len(soa_ms) >= 3:
        ax.axvspan(soa_ms[0] - 20, soa_ms[2] + 20, alpha=0.06, color="gray")

    # ── Panel B: RT1 by congruency for independent (C→A) ──
    if secondary:
        ax = axes[1]
        res_ind = results[secondary]
        soa_ms_ind = _soa_to_ms(res_ind["soa"])
        color_ind = COLORS.get(secondary, "#1f77b4")

        ax.errorbar(soa_ms_ind, res_ind["rt1_congruent"], yerr=res_ind["rt1_congruent_se"],
                    fmt="o-", color=color_ind, lw=2, markersize=6, capsize=3,
                    label="Congruent")
        ax.errorbar(soa_ms_ind, res_ind["rt1_incongruent"], yerr=res_ind["rt1_incongruent_se"],
                    fmt="s--", color=color_ind, lw=2, markersize=6, capsize=3,
                    label="Incongruent")

        ax.set_xlabel("SOA (ms)", fontsize=11)
        ax.set_title(f"B.  Task 1 RT — {secondary}\n(C→A)",
                     fontsize=11, loc="left", color=color_ind, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xticks(soa_ms_ind)
        if len(soa_ms_ind) >= 3:
            ax.axvspan(soa_ms_ind[0] - 20, soa_ms_ind[2] + 20, alpha=0.06, color="gray")

        # Share y-axis with Panel A
        ylim_a = axes[0].get_ylim()
        ylim_b = ax.get_ylim()
        shared_ylim = (min(ylim_a[0], ylim_b[0]), max(ylim_a[1], ylim_b[1]))
        axes[0].set_ylim(shared_ylim)
        ax.set_ylim(shared_ylim)
        ax.set_yticklabels([])

    # ── Panel C: BCE across SOA for all conditions ──
    ax = axes[-1]
    for cond in conditions:
        res = results[cond]
        soa_ms = _soa_to_ms(res["soa"])
        color = COLORS.get(cond, "gray")

        ax.errorbar(soa_ms, np.array(res["bce"]) * 1000,
                     yerr=np.array(res["bce_se"]) * 1000,
                     fmt="o-" if cond == primary else "s--",
                     color=color, lw=2, markersize=6, capsize=3,
                     label=cond.capitalize())

    ax.axhline(0, color="black", lw=1, ls="-", alpha=0.5)
    ax.set_xlabel("SOA (ms)", fontsize=11)
    ax.set_ylabel("BCE (ms)", fontsize=11)
    panel_letter = "C" if secondary else "B"
    ax.set_title(f"{panel_letter}.  Backward Crosstalk Effect\n(RT1 incongruent − congruent)",
                  fontsize=11, loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(soa_ms)
    if len(soa_ms) >= 3:
        ax.axvspan(soa_ms[0] - 20, soa_ms[2] + 20, alpha=0.06, color="gray")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


def plot_bce_ensemble(
    all_results: list[dict],
    out_path: str = "output/figures/bce_ensemble.png",
    title: str | None = None,
    figsize: tuple = (11, 4.5),
):
    """
    Plot ensemble-averaged BCE results.

    Parameters
    ----------
    all_results : list[dict]
        List of run_bce_comparison outputs, one per network.
    """
    conditions = [k for k in ("functional", "independent", "structural")
                  if k in all_results[0]]

    # Average across networks
    soa_list = all_results[0][conditions[0]]["soa"]
    n_soas = len(soa_list)
    n_nets = len(all_results)

    averaged = {}
    for cond in conditions:
        avg = {
            "soa": soa_list,
            "rt1_congruent": [], "rt1_incongruent": [],
            "rt1_congruent_se": [], "rt1_incongruent_se": [],
            "bce": [], "bce_se": [],
        }

        for si in range(n_soas):
            rt1c = [r[cond]["rt1_congruent"][si] for r in all_results
                    if np.isfinite(r[cond]["rt1_congruent"][si])]
            rt1i = [r[cond]["rt1_incongruent"][si] for r in all_results
                    if np.isfinite(r[cond]["rt1_incongruent"][si])]
            bce = [r[cond]["bce"][si] for r in all_results
                   if np.isfinite(r[cond]["bce"][si])]

            def _stat(v):
                if not v:
                    return np.nan, np.nan
                return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))

            r1c_m, r1c_se = _stat(rt1c)
            r1i_m, r1i_se = _stat(rt1i)
            bce_m, bce_se = _stat(bce)

            avg["rt1_congruent"].append(r1c_m)
            avg["rt1_incongruent"].append(r1i_m)
            avg["rt1_congruent_se"].append(r1c_se)
            avg["rt1_incongruent_se"].append(r1i_se)
            avg["bce"].append(bce_m)
            avg["bce_se"].append(bce_se)

        averaged[cond] = avg

    plot_title = title or f"Backward Crosstalk Effect (n={n_nets} networks)"
    plot_bce(averaged, out_path=out_path, title=plot_title, figsize=figsize)


def plot_bce_from_json(
    json_path: str,
    out_path: str = "output/figures/bce.png",
):
    """Plot BCE from a saved sweep summary JSON."""
    with open(json_path) as f:
        summary = json.load(f)

    conditions = summary["conditions"]
    soa_ms = conditions[list(conditions.keys())[0]]["soa_ms"]
    dt = DT

    # Reconstruct results dict in the format plot_bce expects
    results = {}
    for cond, data in conditions.items():
        soa_steps = [int(ms / (dt * 1000)) for ms in data["soa_ms"]]
        results[cond] = {
            "soa": soa_steps,
            "rt1_congruent": data["rt1_con"],
            "rt1_incongruent": data["rt1_inc"],
            "rt1_congruent_se": [0] * len(soa_steps),
            "rt1_incongruent_se": [0] * len(soa_steps),
            "bce": data["bce_mean"],
            "bce_se": data["bce_se"],
        }

    n_nets = summary.get("n_nets", "?")
    p = summary.get("persistence", "?")
    plot_bce(results, out_path=out_path,
             title=f"BCE (n={n_nets}, p={p})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="path to bce_sweep JSON")
    ap.add_argument("--out", default="output/figures/bce.png")
    args = ap.parse_args()
    plot_bce_from_json(args.json, args.out)