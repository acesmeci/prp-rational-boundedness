#!/usr/bin/env python3
"""
Plot PRP sweep results from saved JSON (thesis-quality figures).

Usage:
    python -m scripts.plot_prp_sweep --json output/results/E20_*.json

Outputs per JSON (PNG + PDF):
    main   -> output/plots/ensemble/<tag>_main.{png,pdf}   (RT2 + RT1 panels)
    er     -> output/plots/ensemble/ER/<tag>_er.{png,pdf}
    onset  -> output/plots/ensemble/onset/<tag>_onset.{png,pdf}

Slope reporting follows the thesis Ch.2 evaluation criteria:
    SOA* = 0.80 x RT1 (RT1 = mean correct-trials Task-1 RT, flat across SOA)
    head slope = OLS over SOA points <= SOA*; tail slope = OLS over points >= SOA*
    two-shortest-SOA slope also reported (primary empirical criterion).
"""
import os, json, argparse, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prp_model.lca import MS_PER_STEP
from prp_model.utils import steps_to_ms, sim_seconds_to_ms

COLORS = {"dep": "#1f77b4", "ind": "#2ca02c"}
LABELS = {"dep": "B\u2192A (shared)", "ind": "C\u2192A (separated)"}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ===================================================================
# Helpers
# ===================================================================
def _get(data, cond, key):
    """Fetch avg series with correct-trials preference and old-JSON fallback."""
    avg = data["avg"][cond]
    if key in avg:
        return np.asarray(avg[key], float)
    return None


def _rt_key(data, base):
    return (base + "_correct"
            if base + "_correct" in data["avg"]["dep"] else base)


def ols_slope(x, y):
    """Dimensionless OLS slope over finite pairs; nan if <2 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def two_shortest_slope(soa_ms, rt_ms):
    order = np.argsort(soa_ms)
    s, r = np.asarray(soa_ms)[order], np.asarray(rt_ms)[order]
    if len(s) < 2 or not (np.isfinite(r[0]) and np.isfinite(r[1])):
        return np.nan
    return float((r[1] - r[0]) / (s[1] - s[0]))


def soa_star_ms(data, cond="dep"):
    """SOA* = 0.80 x mean correct-trials RT1 (flat across SOA), in ms."""
    key = _rt_key(data, "rt_task1")
    rt1 = _get(data, cond, key)
    if rt1 is None or not np.isfinite(rt1).any():
        return np.nan
    return 0.80 * float(np.nanmean(sim_seconds_to_ms(rt1)))


def head_tail_slopes(soa_ms, rt_ms, soa_star):
    head = ols_slope(soa_ms[soa_ms <= soa_star], rt_ms[soa_ms <= soa_star])
    tail = ols_slope(soa_ms[soa_ms >= soa_star], rt_ms[soa_ms >= soa_star])
    return head, tail


def get_pashler_curve():
    """Pashler (1994) Fig 1 (hand-drawn schematic; opt-in overlay only)."""
    return {"soa_ms": np.array([50, 150, 300, 900], float),
            "rt2_ms": np.array([700, 600, 525, 500], float)}


def _save(fig, out_base):
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    print(f"  saved: {out_base}.png/.pdf")


# ===================================================================
# F2: Main figure — RT2 (left) + RT1 (right)
# ===================================================================
def plot_main(data, out_base, add_pashler=False):
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))
    rt2_key = _rt_key(data, "rt_task2_from_stim")
    rt1_key = _rt_key(data, "rt_task1")
    star = soa_star_ms(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- RT2 panel ---
    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(_get(data, cond, rt2_key))
        se = sim_seconds_to_ms(
            data["avg"][cond].get(rt2_key + "_se", np.zeros(len(soa_ms))))
        ts = two_shortest_slope(soa_ms, mean)
        head, tail = (head_tail_slopes(soa_ms, mean, star)
                      if np.isfinite(star) else (np.nan, np.nan))
        ax1.plot(soa_ms, mean, "x--", color=COLORS[cond],
                 label=(f"{LABELS[cond]}  "
                        f"[{ts:.2f} | head {head:.2f} | tail {tail:.2f}]"))
        ax1.fill_between(soa_ms, mean - se, mean + se,
                         color=COLORS[cond], alpha=0.15)

    if np.isfinite(star):
        ax1.axvline(star, color="gray", linestyle=":", linewidth=1.2)
        ax1.text(star, ax1.get_ylim()[1], "  SOA*", color="gray",
                 va="top", ha="left", fontsize=10)

    if add_pashler:
        pa = get_pashler_curve()
        ax1.plot(pa["soa_ms"], pa["rt2_ms"], "ko-", alpha=0.5,
                 label="Pashler (1994) Fig 1 (schematic)")

    ax1.set_xlabel("SOA (ms)")
    ax1.set_ylabel("RT2 (ms)")
    ax1.set_title(f"Task 2 RT  (p = {p:.2f})")
    ax1.legend(title="[2-shortest | head | tail slopes]",
               title_fontsize=9, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.4)

    # --- RT1 panel ---
    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(_get(data, cond, rt1_key))
        se = sim_seconds_to_ms(
            data["avg"][cond].get(rt1_key + "_se", np.zeros(len(soa_ms))))
        ax2.plot(soa_ms, mean, "o--", color=COLORS[cond],
                 markersize=4, label=LABELS[cond])
        ax2.fill_between(soa_ms, mean - se, mean + se,
                         color=COLORS[cond], alpha=0.15)

    ax2.set_xlabel("SOA (ms)")
    ax2.set_ylabel("RT1 (ms)")
    ax2.set_title(f"Task 1 RT  (p = {p:.2f})")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.4)
    # y-range: pad around data so flatness is visible but honest
    lo, hi = ax2.get_ylim()
    pad = max(50.0, 0.15 * (hi - lo))
    ax2.set_ylim(lo - pad, hi + pad)

    _save(fig, out_base)


# ===================================================================
# F3: Error rates
# ===================================================================
def plot_error_rates(data, out_base):
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cond in ("dep", "ind"):
        for task, marker, alpha in (("acc_task2", "x--", 1.0),
                                    ("acc_task1", "o-", 0.45)):
            acc = _get(data, cond, task)
            se = np.asarray(
                data["avg"][cond].get(task + "_se", np.zeros(len(soa_ms))),
                float)
            err = 1.0 - acc
            tlab = "Task 2" if task == "acc_task2" else "Task 1"
            ax.plot(soa_ms, err, marker, color=COLORS[cond], alpha=alpha,
                    markersize=5, label=f"{tlab} | {LABELS[cond]}")
            ax.fill_between(soa_ms, err - se, err + se,
                            color=COLORS[cond], alpha=0.08)

    ax.set_xlabel("SOA (ms)")
    ax.set_ylabel("Error rate")
    ax.set_title(f"Error rates  (p = {p:.2f})")
    ax.set_ylim(bottom=-0.005)
    ax.legend(ncol=2)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_base)


# ===================================================================
# F5: Strategic deferment — onset delay vs SOA
# ===================================================================
def plot_onset_delay(data, out_base):
    if not data["params"].get("optimize_onset", False):
        return  # greedy runs: nothing to plot
    p = data["params"]["persistence"]
    soa_steps = np.asarray(data["soa"], float)
    soa_ms = steps_to_ms(soa_steps)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cond in ("dep", "ind"):
        onset = _get(data, cond, "onset2")
        se = np.asarray(
            data["avg"][cond].get("onset2_se", np.zeros(len(soa_ms))), float)
        delay_ms = (onset - soa_steps) * MS_PER_STEP
        se_ms = se * MS_PER_STEP
        ax.plot(soa_ms, delay_ms, "s--", color=COLORS[cond],
                markersize=5, label=LABELS[cond])
        ax.fill_between(soa_ms, delay_ms - se_ms, delay_ms + se_ms,
                        color=COLORS[cond], alpha=0.15)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SOA (ms)")
    ax.set_ylabel("Strategic onset delay (ms)")
    ax.set_title(f"Task 2 engagement delay  (p = {p:.2f})")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_base)


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="Plot PRP sweep results (thesis-quality).")
    ap.add_argument("--json", type=str, nargs="+", required=True)
    ap.add_argument("--pashler", action="store_true",
                    help="Overlay Pashler (1994) Fig 1 schematic (off by default)")
    ap.add_argument("--out_dir", type=str, default="output/plots/ensemble")
    args = ap.parse_args()

    json_paths = []
    for pattern in args.json:
        expanded = sorted(glob.glob(pattern))
        json_paths.extend(expanded if expanded else [pattern])

    for jp in json_paths:
        print(f"\nProcessing: {jp}")
        with open(jp, "r") as f:
            data = json.load(f)
        tag = data.get("tag", Path(jp).stem)
        plot_main(data, os.path.join(args.out_dir, f"{tag}_main"),
                  add_pashler=args.pashler)
        plot_error_rates(data, os.path.join(args.out_dir, "ER", f"{tag}_er"))
        plot_onset_delay(data, os.path.join(args.out_dir, "onset", f"{tag}_onset"))


if __name__ == "__main__":
    main()