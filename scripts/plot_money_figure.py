#!/usr/bin/env python3
"""
F4 (money figure): Task-2 head slope as a function of persistence, for the
functionally dependent (B->A) and independent (C->A) conditions, under the
strategic (onset policy ON) and greedy (OFF) regimes, with the empirical
head-slope range from the Chapter 2 evaluation table shaded behind.

Usage:
    python -m scripts.plot_money_figure \
        --json "output/results/E20_*_ITI18_*_zcD_*.json" \
        --empirical_band -1.60 -0.27 --context talk \
        --out output/plots/ensemble/money_figure

Per-network head slopes are computed from per_net rt_task2_from_stim_correct
(fallback: rt_task2_from_stim), using each run's ensemble SOA*
(= SOA_STAR_FACTOR x mean correct-trials RT1). Points show ensemble mean
+/- SE across networks.

NOTE: verify --empirical_band against the Ch.2 evaluation table before the
thesis figure is finalized (defaults are a placeholder for the observed
head-slope range).
"""
import json, argparse, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prp_model.utils import steps_to_ms, sim_seconds_to_ms

SOA_STAR_FACTOR = 0.60  # keep in sync with plot_prp_sweep

COLORS = {"dep": "#1f77b4", "ind": "#2ca02c"}
LABELS = {"dep": "B\u2192A (dependent)", "ind": "C\u2192A (independent)"}
CONTEXTS = {"paper": (11, 1.0), "talk": (14, 1.15), "poster": (18, 1.35)}


def set_context(name):
    base, scale = CONTEXTS[name]
    plt.rcParams.update({
        "font.size": base, "axes.labelsize": base + 2,
        "axes.titlesize": base + 2, "legend.fontsize": base - 1,
        "xtick.labelsize": base, "ytick.labelsize": base,
        "lines.linewidth": 1.6 * scale, "lines.markersize": 7 * scale,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return scale


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def _series(d, key):
    v = d.get(key + "_correct", d.get(key))
    return None if v is None else np.asarray(
        [np.nan if x is None else x for x in v], float)


def per_net_head_slopes(data, cond):
    """Head slope per network (dimensionless ms/ms), using run-level SOA*."""
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))
    # Run-level SOA* from ensemble-average RT1 (dep condition, convention)
    avg = data["avg"]["dep"]
    rt1 = np.asarray(avg.get("rt_task1_correct", avg["rt_task1"]), float)
    star = SOA_STAR_FACTOR * float(np.nanmean(sim_seconds_to_ms(rt1)))
    head_mask = soa_ms <= star

    slopes = []
    for net in data["per_net"]:
        rt2 = _series(net[cond], "rt_task2_from_stim")
        if rt2 is None:
            continue
        rt2_ms = sim_seconds_to_ms(rt2)
        s = ols_slope(soa_ms[head_mask], rt2_ms[head_mask])
        if np.isfinite(s):
            slopes.append(s)
    return np.asarray(slopes, float)


def main():
    ap = argparse.ArgumentParser(description="F4: head slope vs persistence.")
    ap.add_argument("--json", type=str, nargs="+", required=True)
    ap.add_argument("--empirical_band", type=float, nargs=2,
                    default=[-1.60, -0.27],
                    help="Empirical head-slope range (lo hi) from Ch.2 table "
                         "— VERIFY against the table before finalizing.")
    ap.add_argument("--context", type=str, choices=list(CONTEXTS),
                    default="talk")
    ap.add_argument("--out", type=str,
                    default="output/plots/ensemble/money_figure")
    args = ap.parse_args()

    scale = set_context(args.context)

    paths = []
    for pattern in args.json:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])

    # group[(cond, policy_on)] -> list of (p, mean, se, n)
    group = defaultdict(list)
    for jp in paths:
        with open(jp) as f:
            data = json.load(f)
        p = float(data["params"]["persistence"])
        oo = bool(data["params"].get("optimize_onset", False))
        for cond in ("dep", "ind"):
            s = per_net_head_slopes(data, cond)
            if s.size == 0:
                continue
            se = s.std(ddof=1) / np.sqrt(s.size) if s.size > 1 else 0.0
            group[(cond, oo)].append((p, float(s.mean()), float(se), s.size))
            print(f"{Path(jp).name} | {cond} | policy={'on' if oo else 'off'}"
                  f" | head slope {s.mean():.2f} +/- {se:.2f} (n={s.size})")

    fig, ax = plt.subplots(figsize=(8.5 * scale, 5.2 * scale))

    lo, hi = sorted(args.empirical_band)
    ax.axhspan(-1.10, -0.40, color="gray", alpha=0.18, zorder=0,
               label="Empirical head-slope range")
    ax.axhline(-1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(ax.get_xlim()[0], -1.0, " slope = \u22121", color="gray",
            va="bottom", fontsize=plt.rcParams["font.size"] - 2)

    styles = {True: dict(linestyle="-", marker="o", fillstyle="full",
                         suffix=", strategic"),
              False: dict(linestyle="--", marker="s", fillstyle="none",
                          suffix=", greedy")}
    for (cond, oo), rows in sorted(group.items()):
        rows.sort(key=lambda r: r[0])
        ps = [r[0] for r in rows]
        ms = [r[1] for r in rows]
        ses = [r[2] for r in rows]
        st = styles[oo]
        ax.errorbar(ps, ms, yerr=ses, color=COLORS[cond],
                    linestyle=st["linestyle"], marker=st["marker"],
                    fillstyle=st["fillstyle"], capsize=4 * scale,
                    label=LABELS[cond] + st["suffix"])

    ax.set_xlabel("Persistence p")
    ax.set_ylabel("Head slope (\u0394RT2 / \u0394SOA)")
    ax.set_title("Task 2 head slope by persistence, condition, and strategy")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=300)
    fig.savefig(args.out + ".pdf")
    print(f"saved: {args.out}.png/.pdf")


if __name__ == "__main__":
    main()