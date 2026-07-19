#!/usr/bin/env python3
"""
Plot PRP sweep results from saved JSON (thesis-quality figures).

Usage:
    python -m scripts.plot_prp_sweep --json output/results/E20_*.json \
        [--context talk] [--rt2_ylim 440 700] [--rt1_ylim 240 340] [--pashler]

Outputs per JSON (PNG + PDF):
    main   -> output/plots/ensemble/<tag>_main.{png,pdf}   (RT1 left, RT2 right)
    er     -> output/plots/ensemble/ER/<tag>_er.{png,pdf}
    onset  -> output/plots/ensemble/onset/<tag>_onset.{png,pdf}

Slope reporting follows the thesis Ch.2 evaluation criteria:
    Head slope = two shortest SOAs (Ch.2 primary criterion).
    Full-head = OLS over SOA <= SOA*.
    SOA* = SOA_STAR_FACTOR x RT1 at longest SOA.
"""
import os, json, argparse, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prp_model.lca import _DEFAULTS
from prp_model.utils import steps_to_ms, sim_seconds_to_ms

SOA_STAR_FACTOR = 0.80

COLORS = {"dep": "#1f77b4", "ind": "#2ca02c"}
LABELS = {"dep": "B\u2192A (dependent)", "ind": "C\u2192A (independent)"}

CONTEXTS = {  # name -> (base font size, figure scale)
    "paper": (16, 1.0),   # larger fonts: figures are scaled ~50% in thesis
    "talk": (14, 1.15),
    "poster": (18, 1.35),
}

_SHOW_TITLES = True  # set False for paper context (titles go in captions)


def set_context(name):
    global _SHOW_TITLES
    base, scale = CONTEXTS[name]
    _SHOW_TITLES = (name != "paper")
    legend_size = base - 4 if name == "paper" else base - 1
    plt.rcParams.update({
        "font.size": base,
        "axes.labelsize": base + 2,
        "axes.titlesize": base + 2,
        "legend.fontsize": legend_size,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "lines.linewidth": 1.6 * scale,
        "lines.markersize": 6 * scale,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return scale


# ===================================================================
# Helpers
# ===================================================================
def _get(data, cond, key):
    avg = data["avg"][cond]
    return np.asarray(avg[key], float) if key in avg else None


def _rt_key(data, base):
    return (base + "_correct"
            if base + "_correct" in data["avg"]["dep"] else base)


def ols_slope(x, y):
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

def two_longest_slope(soa_ms, rt_ms):
    order = np.argsort(soa_ms)
    s, r = np.asarray(soa_ms)[order][-2:], np.asarray(rt_ms)[order][-2:]
    if len(s) < 2 or not np.isfinite(r).all():
        return np.nan
    return float((r[1] - r[0]) / (s[1] - s[0]))

def soa_star_ms(data, cond="dep"):
    """SOA* = SOA_STAR_FACTOR x RT1 at longest SOA, in ms."""
    rt1 = _get(data, cond, _rt_key(data, "rt_task1"))
    if rt1 is None or not np.isfinite(rt1).any():
        return np.nan
    return SOA_STAR_FACTOR * float(sim_seconds_to_ms(rt1)[-1])


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
# F2: Main figure — RT1 (left) + RT2 (right)
# ===================================================================
def plot_main(data, out_base, scale=1.15, add_pashler=False,
              rt1_ylim=None, rt2_ylim=None):
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))
    rt2_key = _rt_key(data, "rt_task2_from_stim")
    rt1_key = _rt_key(data, "rt_task1")
    star = soa_star_ms(data)
    tag = data.get("tag", "")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * scale, 4.8 * scale))

    # --- RT1 panel (LEFT: Task 1 comes first) ---
    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(_get(data, cond, rt1_key))
        se = sim_seconds_to_ms(
            data["avg"][cond].get(rt1_key + "_se", np.zeros(len(soa_ms))))
        ax1.errorbar(soa_ms, mean, yerr=se, fmt="o-", color=COLORS[cond],
                     capsize=3, label=LABELS[cond])
    ax1.set_xlabel("SOA (ms)")
    ax1.set_ylabel("RT1 (ms)")
    if _SHOW_TITLES: ax1.set_title(f"Task 1 RT  (p = {p:.2f})")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(True, linestyle=":", alpha=0.4)
    if rt1_ylim:
        ax1.set_ylim(*rt1_ylim)
    else:
        lo, hi = ax1.get_ylim()
        pad = max(40.0, 0.15 * (hi - lo))
        ax1.set_ylim(lo - pad, hi + pad)

    # --- RT2 panel (RIGHT) ---
    console = [f"[{tag}] SOA* = {star:.0f} ms (factor {SOA_STAR_FACTOR})"]
    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(_get(data, cond, rt2_key))
        se = sim_seconds_to_ms(
            data["avg"][cond].get(rt2_key + "_se", np.zeros(len(soa_ms))))
        ts = two_shortest_slope(soa_ms, mean)
        tl = two_longest_slope(soa_ms, mean)
        head, tail = (head_tail_slopes(soa_ms, mean, star)
                      if np.isfinite(star) else (np.nan, np.nan))
        ax2.errorbar(soa_ms, mean, yerr=se, fmt="o-", color=COLORS[cond],
                     capsize=3,
                     label=f"{LABELS[cond]}, head slope {ts:.2f}")
        console.append(f"  {cond}: head {ts:.2f} | full-head {head:.2f} "
                       f"| tail {tail:.2f}")
        console.append(f"  {cond}: 2-longest {tl:.2f}")

    if np.isfinite(star):
        ax2.axvline(star, color="gray", linestyle=":", linewidth=1.2)
        ax2.text(star, ax2.get_ylim()[0], " SOA*", color="gray",
                 va="bottom", ha="left")

    if add_pashler:
        pa = get_pashler_curve()
        ax2.plot(pa["soa_ms"], pa["rt2_ms"], "ko-", alpha=0.5,
                 label="Pashler (1994) Fig 1 (schematic)")

    ax2.set_xlabel("SOA (ms)")
    ax2.set_ylabel("RT2 (ms)")
    if _SHOW_TITLES: ax2.set_title(f"Task 2 RT  (p = {p:.2f})")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, linestyle=":", alpha=0.4)
    if rt2_ylim:
        ax2.set_ylim(*rt2_ylim)

    print("\n".join(console))  # tail + 2-shortest slopes: report in text
    _save(fig, out_base)


# ===================================================================
# F3: Error rates
# ===================================================================
def plot_error_rates(data, out_base, scale=1.15):
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))

    fig, ax = plt.subplots(figsize=(8 * scale, 4.8 * scale))
    for cond in ("dep", "ind"):
        for task, alpha in (("acc_task2", 1.0),
                            ("acc_task1", 0.45)):
            acc = _get(data, cond, task)
            se = np.asarray(
                data["avg"][cond].get(task + "_se", np.zeros(len(soa_ms))),
                float)
            err = 1.0 - acc
            tlab = "Task 2" if task == "acc_task2" else "Task 1"
            ax.errorbar(soa_ms, err, yerr=se, fmt="o-", color=COLORS[cond],
                        alpha=alpha, capsize=3,
                        label=f"{tlab} | {LABELS[cond]}")

    ax.set_xlabel("SOA (ms)")
    ax.set_ylabel("Error rate")
    if _SHOW_TITLES: ax.set_title(f"Error rates  (p = {p:.2f})")
    ax.set_ylim(bottom=-0.005)
    ax.legend(ncol=2, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_base)


# ===================================================================
# F5: Strategic deferment — onset delay vs SOA
# ===================================================================
def plot_onset_delay(data, out_base, scale=1.15):
    if not data["params"].get("optimize_onset", False):
        return  # greedy runs: nothing to plot
    p = data["params"]["persistence"]
    soa_steps = np.asarray(data["soa"], float)
    soa_ms = steps_to_ms(soa_steps)

    fig, ax = plt.subplots(figsize=(8 * scale, 4.8 * scale))
    for cond in ("dep", "ind"):
        onset = _get(data, cond, "onset2")
        se = np.asarray(
            data["avg"][cond].get("onset2_se", np.zeros(len(soa_ms))), float)
        delay_ms = (onset - soa_steps) * _DEFAULTS["dt"] * 1000
        se_ms = se * _DEFAULTS["dt"] * 1000
        ax.errorbar(soa_ms, delay_ms, yerr=se_ms, fmt="o-",
                    color=COLORS[cond], capsize=3, label=LABELS[cond])

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SOA (ms)")
    ax.set_ylabel("Strategic onset delay (ms)")
    if _SHOW_TITLES: ax.set_title(f"Task 2 engagement delay  (p = {p:.2f})")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_base)


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="Plot PRP sweep results (thesis-quality).")
    ap.add_argument("--json", type=str, nargs="+", required=True)
    ap.add_argument("--context", type=str, choices=list(CONTEXTS),
                    default="talk", help="Font/figure size preset")
    ap.add_argument("--rt2_ylim", type=float, nargs=2, default=None,
                    help="Shared RT2 y-limits across runs, e.g. 440 700")
    ap.add_argument("--rt1_ylim", type=float, nargs=2, default=None,
                    help="Shared RT1 y-limits across runs, e.g. 240 340")
    ap.add_argument("--pashler", action="store_true",
                    help="Overlay Pashler (1994) Fig 1 schematic (off by default)")
    ap.add_argument("--out_dir", type=str, default="output/plots/ensemble")
    args = ap.parse_args()

    scale = set_context(args.context)

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
                  scale=scale, add_pashler=args.pashler,
                  rt1_ylim=args.rt1_ylim, rt2_ylim=args.rt2_ylim)
        plot_error_rates(data, os.path.join(args.out_dir, "ER", f"{tag}_er"),
                         scale=scale)
        plot_onset_delay(data, os.path.join(args.out_dir, "onset",
                                            f"{tag}_onset"), scale=scale)


if __name__ == "__main__":
    main()