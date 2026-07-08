#!/usr/bin/env python3
"""
Plot PRP sweep results from saved JSON.

Usage:
    # Plot a single result
    python -m scripts.plot_prp_sweep --json output/results/E10_p080_*.json

    # Plot all results in a directory
    python -m scripts.plot_prp_sweep --json output/results/*.json

    # Skip Pashler overlay
    python -m scripts.plot_prp_sweep --json output/results/*.json --no_pashler

Outputs:
    RT2 vs SOA  -> output/plots/ensemble/<tag>.png
    Error rates -> output/plots/ensemble/ER/<tag>.png
"""
import os, json, argparse, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prp_model.lca import MS_PER_STEP
from prp_model.utils import steps_to_ms, sim_seconds_to_ms


# ===================================================================
# Helpers
# ===================================================================
def steepest_slope_ms(soa_ms, rt_ms):
    """Steepest adjacent-pair slope in ms/ms (dimensionless, = s/s)."""
    soa_ms = np.asarray(soa_ms, float)
    rt_ms = np.asarray(rt_ms, float)
    m = np.isfinite(soa_ms) & np.isfinite(rt_ms)
    soa_ms, rt_ms = soa_ms[m], rt_ms[m]
    order = np.argsort(soa_ms)
    soa_ms, rt_ms = soa_ms[order], rt_ms[order]
    if len(soa_ms) < 2:
        return {"seg_ms": (np.nan, np.nan), "slope": np.nan}
    slopes = np.diff(rt_ms) / np.diff(soa_ms)
    i = int(np.nanargmin(slopes))
    return {
        "seg_ms": (float(soa_ms[i]), float(soa_ms[i + 1])),
        "slope": float(slopes[i]),
    }


def get_pashler_curve():
    """Pashler (1994) Figure 1 reference points (hand-drawn schematic)."""
    return {
        "soa_ms": np.array([50, 150, 300, 900], dtype=float),
        "rt2_ms": np.array([700, 600, 525, 500], dtype=float),
    }


# ===================================================================
# RT2 vs SOA plot
# ===================================================================
def plot_rt2(data, out_path, add_pashler=True):
    """Plot RT2 vs SOA with B→A and C→A, plus optional Pashler overlay."""
    params = data["params"]
    persistence = params["persistence"]
    soa_steps = np.array(data["soa"], float)

    soa_ms = steps_to_ms(soa_steps)

    rt_key = ("rt_task2_from_stim_correct"
              if "rt_task2_from_stim_correct" in data["avg"]["dep"]
              else "rt_task2_from_stim")

    # B→A (dependent)
    dep_mean_ms = sim_seconds_to_ms(data["avg"]["dep"][rt_key])
    dep_se_ms = sim_seconds_to_ms(
        data["avg"]["dep"].get(rt_key + "_se", [0] * len(soa_ms))
    )

    # C→A (independent)
    ind_mean_ms = sim_seconds_to_ms(data["avg"]["ind"][rt_key])
    ind_se_ms = sim_seconds_to_ms(
        data["avg"]["ind"].get(rt_key + "_se", [0] * len(soa_ms))
    )

    dep_slope = steepest_slope_ms(soa_ms, dep_mean_ms)

    fig, ax = plt.subplots(figsize=(8, 5))

    # B→A
    ax.plot(
        soa_ms, dep_mean_ms, "x--", color="#1f77b4",
        label=f"Simulation B→A | steepest: {dep_slope['slope']:.2f}",
    )
    ax.fill_between(
        soa_ms, dep_mean_ms - dep_se_ms, dep_mean_ms + dep_se_ms,
        color="#1f77b4", alpha=0.15,
    )

    # C→A
    ax.plot(soa_ms, ind_mean_ms, "x--", color="#2ca02c", label="Simulation C→A")
    ax.fill_between(
        soa_ms, ind_mean_ms - ind_se_ms, ind_mean_ms + ind_se_ms,
        color="#2ca02c", alpha=0.15,
    )

    # Pashler overlay
    if add_pashler:
        pashler = get_pashler_curve()
        p_soa = pashler["soa_ms"].copy()
        p_rt2 = pashler["rt2_ms"].copy()
        # Shift horizontally if our SOA starts later than Pashler's 50 ms
        if soa_ms[0] > p_soa[0]:
            p_soa = p_soa - p_soa[0] + soa_ms[0]
        ax.plot(
            p_soa, p_rt2, "ko-", linewidth=2, alpha=0.6, markersize=6,
            label="Pashler (1994) Fig 1",
        )

    ax.set_xlabel("SOA (milliseconds)")
    ax.set_ylabel("RT2 (milliseconds)")
    ax.set_title(f"Task 2 RT | Persistence p={persistence:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  RT2 plot: {out_path}")


# ===================================================================
# Error rate plot
# ===================================================================
def plot_error_rates(data, out_path):
    """Plot error rates for Task 1 and Task 2, both conditions."""
    params = data["params"]
    persistence = params["persistence"]
    soa_steps = np.array(data["soa"], float)
    soa_ms = steps_to_ms(soa_steps)

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond_key, cond_label, color_t2, color_t1 in [
        ("dep", "B→A", "#1f77b4", "#aec7e8"),
        ("ind", "C→A", "#2ca02c", "#98df8a"),
    ]:
        acc_t2 = np.array(data["avg"][cond_key]["acc_task2"], float)
        err_t2 = 1.0 - acc_t2
        acc_t2_se = np.array(
            data["avg"][cond_key].get("acc_task2_se", [0] * len(soa_ms)), float
        )
        ax.plot(
            soa_ms, err_t2, "x--", color=color_t2,
            label=f"Task 2 (A) Error | {cond_label}",
        )
        ax.fill_between(
            soa_ms, err_t2 - acc_t2_se, err_t2 + acc_t2_se,
            color=color_t2, alpha=0.1,
        )

        acc_t1 = np.array(data["avg"][cond_key]["acc_task1"], float)
        err_t1 = 1.0 - acc_t1
        acc_t1_se = np.array(
            data["avg"][cond_key].get("acc_task1_se", [0] * len(soa_ms)), float
        )
        ax.plot(
            soa_ms, err_t1, "o-", color=color_t1, alpha=0.8,
            label=f"Task 1 Error | {cond_label}",
        )
        ax.fill_between(
            soa_ms, err_t1 - acc_t1_se, err_t1 + acc_t1_se,
            color=color_t1, alpha=0.1,
        )

    ax.set_xlabel("SOA (milliseconds)")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"Error Rates | Persistence p={persistence:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  ER plot:  {out_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Plot PRP sweep results from saved JSON files."
    )
    ap.add_argument(
        "--json", type=str, nargs="+", required=True,
        help="Path(s) to result JSON files (supports glob)",
    )
    ap.add_argument("--no_pashler", action="store_true",
                    help="Omit Pashler (1994) reference curve")
    ap.add_argument("--rt2_dir", type=str, default="output/plots/ensemble",
                    help="Output directory for RT2 plots")
    ap.add_argument("--er_dir", type=str, default="output/plots/ensemble/ER",
                    help="Output directory for error rate plots")
    args = ap.parse_args()

    # Expand globs
    json_paths = []
    for pattern in args.json:
        expanded = sorted(glob.glob(pattern))
        json_paths.extend(expanded if expanded else [pattern])

    for jp in json_paths:
        print(f"\nProcessing: {jp}")
        with open(jp, "r") as f:
            data = json.load(f)

        tag = data.get("tag", Path(jp).stem)

        plot_rt2(data, os.path.join(args.rt2_dir, f"{tag}.png"),
                 add_pashler=not args.no_pashler)
        plot_error_rates(data, os.path.join(args.er_dir, f"{tag}.png"))


if __name__ == "__main__":
    main()