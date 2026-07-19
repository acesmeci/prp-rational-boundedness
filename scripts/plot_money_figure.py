#!/usr/bin/env python3
"""
F4 (money figure): Task-2 head slope as a function of persistence, for the
functionally dependent (B->A) and independent (C->A) conditions, under the
strategic (onset policy ON) and greedy (OFF) regimes, with the empirical
head-slope range from the Chapter 2 evaluation table shaded behind.

Usage:
    python -m scripts.plot_money_figure \
        --json "output/results/E20_*_ITI18_*_zcD_*.json" \
        --empirical_band -1.10 -0.30 --context talk \
        --out output/plots/ensemble/money_figure

Per-network head slopes are computed from per_net rt_task2_from_stim_correct
(fallback: rt_task2_from_stim), using each run's ensemble SOA*
(= SOA_STAR_FACTOR x mean correct-trials RT1). Points show ensemble mean
+/- SE across networks.

Empirical band: -1.10 to -0.30 covers 51/54 (94%) of the primary
conditions in the Ch.2 evaluation table (appendix table, 17 Jul 2026).
"""
import json, argparse, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prp_model.utils import steps_to_ms, sim_seconds_to_ms

COLORS = {"dep": "#1f77b4", "ind": "#2ca02c"}
LABELS = {"dep": "B\u2192A (dependent)", "ind": "C\u2192A (independent)"}
CONTEXTS = {"paper": (16, 1.0), "talk": (14, 1.15), "poster": (18, 1.35)}


def set_context(name):
    base, scale = CONTEXTS[name]
    legend_size = base - 4 if name == "paper" else base - 1
    plt.rcParams.update({
        "font.size": base, "axes.labelsize": base + 2,
        "axes.titlesize": base + 2, "legend.fontsize": legend_size,
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
    """Head slope per network: slope between the two shortest SOAs
    (Ch.2 primary criterion), dimensionless ms/ms."""
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))
    order = np.argsort(soa_ms)
    i0, i1 = order[0], order[1]

    slopes = []
    for net in data["per_net"]:
        rt2 = _series(net[cond], "rt_task2_from_stim")
        if rt2 is None:
            continue
        rt2_ms = sim_seconds_to_ms(rt2)
        s = (rt2_ms[i1] - rt2_ms[i0]) / (soa_ms[i1] - soa_ms[i0])
        if np.isfinite(s):
            slopes.append(s)
    return np.asarray(slopes, float)


def main():
    ap = argparse.ArgumentParser(description="F4: head slope vs persistence.")
    ap.add_argument("--json", type=str, nargs="+", required=True)
    ap.add_argument("--empirical_band", type=float, nargs=2,
                    default=[-1.10, -0.30],
                    help="Empirical head-slope band (lo hi); default covers "
                         "94% (51/54) of primary Ch.2 conditions.")
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
    ax.axhspan(lo, hi, color="gray", alpha=0.18, zorder=0,
               label="Empirical head slope range")
    ax.axhline(-1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)

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

    ax.set_xlabel("Persistence")
    ax.set_ylabel("Head slope (\u0394RT2 / \u0394SOA)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97, bottom=0.13)
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"saved: {args.out}.png/.pdf")


if __name__ == "__main__":
    main()