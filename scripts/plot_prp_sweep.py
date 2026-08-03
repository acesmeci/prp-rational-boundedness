#!/usr/bin/env python3
"""
Plot PRP sweep results from saved JSON (thesis-quality figures).

Usage:
    python -m scripts.plot_prp_sweep --json output/results/E20_*.json \
        [--context talk] [--rt2_ylim 440 700] [--rt1_ylim 240 340] \
        [--pashler] [--only all|main|error|onset|greedy_summary]

Outputs per JSON (PNG + PDF):
    main
        -> output/plots/ensemble/<tag>_main.{png,pdf}
           RT1 left, RT2 right

    error
        -> output/plots/ensemble/ER/<tag>_er.{png,pdf}

    onset
        -> output/plots/ensemble/onset/<tag>_onset.{png,pdf}

    greedy_summary
        -> output/plots/ensemble/greedy_summary/
           <tag>_greedy_summary.{png,pdf}
           RT2 left, Task-1/Task-2 error rates right

By default, --only all preserves the original behavior and generates the
main, error-rate, and onset figures. The greedy-summary figure is generated
only when explicitly requested with:

    --only greedy_summary

Slope reporting follows the thesis Chapter 2 evaluation criteria:
    Head slope = slope between the two shortest SOAs.
    Full-head slope = OLS over SOA <= SOA*.
    SOA* = SOA_STAR_FACTOR x RT1 at the longest SOA.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prp_model.lca import _DEFAULTS
from prp_model.utils import sim_seconds_to_ms, steps_to_ms


SOA_STAR_FACTOR = 0.80

COLORS = {
    "dep": "#1f77b4",
    "ind": "#2ca02c",
}

LABELS = {
    "dep": "B\u2192A (dependent)",
    "ind": "C\u2192A (independent)",
}

CONTEXTS = {
    # name -> (base font size, figure scale)
    "paper": (16, 1.0),
    "talk": (14, 1.15),
    "poster": (18, 1.35),
}

_SHOW_TITLES = False  # True for talk/poster, False for paper


def set_context(name):
    global _SHOW_TITLES

    base, scale = CONTEXTS[name]
    _SHOW_TITLES = name != "paper"
    legend_size = base - 4 if name == "paper" else base - 1

    plt.rcParams.update(
        {
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
        }
    )
    return scale


# ===================================================================
# Helpers
# ===================================================================

def _get(data, cond, key):
    avg = data["avg"][cond]
    return np.asarray(avg[key], float) if key in avg else None


def _rt_key(data, base):
    return (
        base + "_correct"
        if base + "_correct" in data["avg"]["dep"]
        else base
    )


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    return float(np.polyfit(x[mask], y[mask], 1)[0])


def two_shortest_slope(soa_ms, rt_ms):
    order = np.argsort(soa_ms)
    soa_sorted = np.asarray(soa_ms)[order]
    rt_sorted = np.asarray(rt_ms)[order]

    if (
        len(soa_sorted) < 2
        or not np.isfinite(rt_sorted[0])
        or not np.isfinite(rt_sorted[1])
    ):
        return np.nan

    return float(
        (rt_sorted[1] - rt_sorted[0])
        / (soa_sorted[1] - soa_sorted[0])
    )


def two_longest_slope(soa_ms, rt_ms):
    order = np.argsort(soa_ms)
    soa_sorted = np.asarray(soa_ms)[order][-2:]
    rt_sorted = np.asarray(rt_ms)[order][-2:]

    if len(soa_sorted) < 2 or not np.isfinite(rt_sorted).all():
        return np.nan

    return float(
        (rt_sorted[1] - rt_sorted[0])
        / (soa_sorted[1] - soa_sorted[0])
    )


def soa_star_ms(data, cond="dep"):
    """SOA* = SOA_STAR_FACTOR x RT1 at the longest SOA, in ms."""
    rt1 = _get(data, cond, _rt_key(data, "rt_task1"))

    if rt1 is None or not np.isfinite(rt1).any():
        return np.nan

    return SOA_STAR_FACTOR * float(sim_seconds_to_ms(rt1)[-1])


def head_tail_slopes(soa_ms, rt_ms, soa_star):
    head_mask = soa_ms <= soa_star
    tail_mask = soa_ms >= soa_star

    head = ols_slope(soa_ms[head_mask], rt_ms[head_mask])
    tail = ols_slope(soa_ms[tail_mask], rt_ms[tail_mask])

    return head, tail


def get_pashler_curve():
    """Pashler (1994) Fig. 1 hand-drawn schematic; opt-in overlay only."""
    return {
        "soa_ms": np.array([50, 150, 300, 900], float),
        "rt2_ms": np.array([700, 600, 525, 500], float),
    }


def _save(fig, out_base):
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    print(f"  saved: {out_base}.png/.pdf")


# ===================================================================
# Main figure: RT1 left, RT2 right
# ===================================================================

def plot_main(
    data,
    out_base,
    scale=1.15,
    add_pashler=False,
    rt1_ylim=None,
    rt2_ylim=None,
):
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))
    rt2_key = _rt_key(data, "rt_task2_from_stim")
    rt1_key = _rt_key(data, "rt_task1")
    star = soa_star_ms(data)
    tag = data.get("tag", "")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12 * scale, 4.8 * scale),
    )

    # ---------------------------------------------------------------
    # Task 1 RT
    # ---------------------------------------------------------------
    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(_get(data, cond, rt1_key))
        se = sim_seconds_to_ms(
            data["avg"][cond].get(
                rt1_key + "_se",
                np.zeros(len(soa_ms)),
            )
        )

        ax1.errorbar(
            soa_ms,
            mean,
            yerr=se,
            fmt="o-",
            color=COLORS[cond],
            capsize=3,
            label=LABELS[cond],
        )

    ax1.set_xlabel("SOA (ms)")
    ax1.set_ylabel("RT1 (ms)")

    if _SHOW_TITLES:
        ax1.set_title(f"Task 1 RT  (p = {p:.2f})")

    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(True, linestyle=":", alpha=0.4)

    if rt1_ylim:
        ax1.set_ylim(*rt1_ylim)
    else:
        lo, hi = ax1.get_ylim()
        pad = max(40.0, 0.15 * (hi - lo))
        ax1.set_ylim(lo - pad, hi + pad)

    # ---------------------------------------------------------------
    # Task 2 RT
    # ---------------------------------------------------------------
    console = [
        f"[{tag}] SOA* = {star:.0f} ms "
        f"(factor {SOA_STAR_FACTOR})"
    ]

    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(_get(data, cond, rt2_key))
        se = sim_seconds_to_ms(
            data["avg"][cond].get(
                rt2_key + "_se",
                np.zeros(len(soa_ms)),
            )
        )

        head_two = two_shortest_slope(soa_ms, mean)
        tail_two = two_longest_slope(soa_ms, mean)

        if np.isfinite(star):
            full_head, full_tail = head_tail_slopes(
                soa_ms,
                mean,
                star,
            )
        else:
            full_head, full_tail = np.nan, np.nan

        ax2.errorbar(
            soa_ms,
            mean,
            yerr=se,
            fmt="o-",
            color=COLORS[cond],
            capsize=3,
            label=(
                f"{LABELS[cond]}, "
                f"head slope {head_two:.2f}"
            ),
        )

        console.append(
            f"  {cond}: head {head_two:.2f} "
            f"| full-head {full_head:.2f} "
            f"| tail {full_tail:.2f}"
        )
        console.append(
            f"  {cond}: 2-longest {tail_two:.2f}"
        )

    if np.isfinite(star):
        ax2.axvline(
            star,
            color="gray",
            linestyle=":",
            linewidth=1.2,
        )
        ax2.text(
            star,
            ax2.get_ylim()[0],
            " SOA*",
            color="gray",
            va="bottom",
            ha="left",
        )

    if add_pashler:
        pashler = get_pashler_curve()
        ax2.plot(
            pashler["soa_ms"],
            pashler["rt2_ms"],
            "ko-",
            alpha=0.5,
            label="Pashler (1994) Fig. 1 (schematic)",
        )

    ax2.set_xlabel("SOA (ms)")
    ax2.set_ylabel("RT2 (ms)")

    if _SHOW_TITLES:
        ax2.set_title(f"Task 2 RT  (p = {p:.2f})")

    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, linestyle=":", alpha=0.4)

    if rt2_ylim:
        ax2.set_ylim(*rt2_ylim)

    print("\n".join(console))
    _save(fig, out_base)


# ===================================================================
# Error-rate figure
# ===================================================================

def plot_error_rates(data, out_base, scale=1.15):
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))

    fig, ax = plt.subplots(
        figsize=(8 * scale, 4.8 * scale)
    )

    for cond in ("dep", "ind"):
        for task, alpha in (
            ("acc_task2", 1.0),
            ("acc_task1", 0.45),
        ):
            acc = _get(data, cond, task)
            se = np.asarray(
                data["avg"][cond].get(
                    task + "_se",
                    np.zeros(len(soa_ms)),
                ),
                float,
            )

            error_rate = 1.0 - acc
            task_label = (
                "Task 2"
                if task == "acc_task2"
                else "Task 1"
            )

            ax.errorbar(
                soa_ms,
                error_rate,
                yerr=se,
                fmt="o-",
                color=COLORS[cond],
                alpha=alpha,
                capsize=3,
                label=f"{task_label} | {LABELS[cond]}",
            )

    ax.set_xlabel("SOA (ms)")
    ax.set_ylabel("Error rate")

    if _SHOW_TITLES:
        ax.set_title(f"Error rates  (p = {p:.2f})")

    ax.set_ylim(-0.005, 0.10)
    ax.legend(ncol=2, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4)

    _save(fig, out_base)


# ===================================================================
# Greedy-engagement summary: RT2 left, error rates right
# ===================================================================

def plot_greedy_summary(
    data,
    out_base,
    scale=1.15,
    rt2_ylim=None,
    error_ylim=(-0.005, 0.10),
    show_task1_errors=True,
    show_titles=False,
):
    """
    Plot the joint greedy-engagement signature.

    Left:
        Task-2 RT against SOA for dependent and independent conditions.

    Right:
        Task-2 error rates against SOA. Task-1 error rates are included
        with reduced opacity by default.

    The console additionally reports:
        - two-shortest-SOA Task-2 head slopes,
        - short-SOA Task-2 errors,
        - short-SOA Task-1 errors,
        - Task-1 SOA effect from shortest to longest SOA.
    """
    p = data["params"]["persistence"]
    soa_ms = steps_to_ms(np.asarray(data["soa"], float))
    rt2_key = _rt_key(data, "rt_task2_from_stim")
    rt1_key = _rt_key(data, "rt_task1")
    star = soa_star_ms(data)
    tag = data.get("tag", "")

    if data["params"].get("optimize_onset", False):
        print(
            "Warning: greedy-summary requested for a run with "
            "optimize_onset=True."
        )

    fig, (ax_rt, ax_er) = plt.subplots(
        1,
        2,
        figsize=(12 * scale, 4.8 * scale),
    )

    console = [
        f"[{tag}] Greedy-engagement summary",
        f"  SOA* = {star:.0f} ms "
        f"(factor {SOA_STAR_FACTOR})",
    ]

    # ---------------------------------------------------------------
    # Left: Task 2 RT
    # ---------------------------------------------------------------
    for cond in ("dep", "ind"):
        mean = sim_seconds_to_ms(
            _get(data, cond, rt2_key)
        )
        se = sim_seconds_to_ms(
            data["avg"][cond].get(
                rt2_key + "_se",
                np.zeros(len(soa_ms)),
            )
        )
        head_two = two_shortest_slope(
            soa_ms,
            mean,
        )

        ax_rt.errorbar(
            soa_ms,
            mean,
            yerr=se,
            fmt="o-",
            color=COLORS[cond],
            capsize=3,
            label=(
                f"{LABELS[cond]}, "
                f"head slope {head_two:.2f}"
            ),
        )

        console.append(
            f"  {cond} Task-2 head slope: "
            f"{head_two:.2f}"
        )

    if np.isfinite(star):
        ax_rt.axvline(
            star,
            color="gray",
            linestyle=":",
            linewidth=1.2,
        )

    ax_rt.set_xlabel("SOA (ms)")
    ax_rt.set_ylabel("RT2 (ms)")

    if show_titles:
        ax_rt.set_title(f"Task 2 RT  (p = {p:.2f})")

    ax_rt.legend(
        loc="upper right",
        frameon=False,
        fontsize=11,
    )
    ax_rt.grid(True, linestyle=":", alpha=0.4)

    if rt2_ylim:
        ax_rt.set_ylim(*rt2_ylim)

    if np.isfinite(star):
        ax_rt.text(
            star,
            ax_rt.get_ylim()[0],
            " SOA*",
            color="gray",
            va="bottom",
            ha="left",
        )

    # ---------------------------------------------------------------
    # Right: Task 2 and optional Task 1 error rates
    # ---------------------------------------------------------------
    for cond in ("dep", "ind"):
        acc_task2 = _get(
            data,
            cond,
            "acc_task2",
        )
        se_task2 = np.asarray(
            data["avg"][cond].get(
                "acc_task2_se",
                np.zeros(len(soa_ms)),
            ),
            float,
        )
        error_task2 = 1.0 - acc_task2

        ax_er.errorbar(
            soa_ms,
            error_task2,
            yerr=se_task2,
            fmt="o-",
            color=COLORS[cond],
            alpha=1.0,
            capsize=3,
            label=f"Task 2 | {LABELS[cond]}",
        )

        if show_task1_errors:
            acc_task1 = _get(
                data,
                cond,
                "acc_task1",
            )
            se_task1 = np.asarray(
                data["avg"][cond].get(
                    "acc_task1_se",
                    np.zeros(len(soa_ms)),
                ),
                float,
            )
            error_task1 = 1.0 - acc_task1

            ax_er.errorbar(
                soa_ms,
                error_task1,
                yerr=se_task1,
                fmt="o-",
                color=COLORS[cond],
                alpha=0.45,
                capsize=3,
                label=f"Task 1 | {LABELS[cond]}",
            )

    ax_er.set_xlabel("SOA (ms)")
    ax_er.set_ylabel("Error rate")

    if show_titles:
        ax_er.set_title(f"Error rates  (p = {p:.2f})")

    ax_er.set_ylim(*error_ylim)
    ax_er.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    ax_er.grid(True, linestyle=":", alpha=0.4)

    # ---------------------------------------------------------------
    # Console values for thesis text/caption
    # ---------------------------------------------------------------
    for cond in ("dep", "ind"):
        rt1_ms = sim_seconds_to_ms(
            _get(data, cond, rt1_key)
        )
        acc_task2 = _get(
            data,
            cond,
            "acc_task2",
        )
        acc_task1 = _get(
            data,
            cond,
            "acc_task1",
        )

        task2_error_short = 1.0 - acc_task2[0]
        task1_error_short = 1.0 - acc_task1[0]
        task1_soa_effect = rt1_ms[0] - rt1_ms[-1]

        console.append(
            f"  {cond} shortest-SOA Task-2 error: "
            f"{100 * task2_error_short:.1f}%"
        )
        console.append(
            f"  {cond} shortest-SOA Task-1 error: "
            f"{100 * task1_error_short:.1f}%"
        )
        console.append(
            f"  {cond} Task-1 SOA effect "
            f"(shortest - longest): "
            f"{task1_soa_effect:.1f} ms"
        )

    print("\n".join(console))
    _save(fig, out_base)


# ===================================================================
# Strategic-deferment figure
# ===================================================================

def plot_onset_delay(data, out_base, scale=1.15):
    if not data["params"].get("optimize_onset", False):
        return

    p = data["params"]["persistence"]
    soa_steps = np.asarray(data["soa"], float)
    soa_ms = steps_to_ms(soa_steps)

    fig, ax = plt.subplots(
        figsize=(8 * scale, 4.8 * scale)
    )

    for cond in ("dep", "ind"):
        onset = _get(data, cond, "onset2")
        se = np.asarray(
            data["avg"][cond].get(
                "onset2_se",
                np.zeros(len(soa_ms)),
            ),
            float,
        )

        delay_ms = (
            onset - soa_steps
        ) * _DEFAULTS["dt"] * 1000
        se_ms = se * _DEFAULTS["dt"] * 1000

        ax.errorbar(
            soa_ms,
            delay_ms,
            yerr=se_ms,
            fmt="o-",
            color=COLORS[cond],
            capsize=3,
            label=LABELS[cond],
        )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.8,
    )
    ax.set_xlabel("SOA (ms)")
    ax.set_ylabel("Strategic onset delay (ms)")

    if _SHOW_TITLES:
        ax.set_title(
            f"Task 2 engagement delay  (p = {p:.2f})"
        )

    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4)

    _save(fig, out_base)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot PRP sweep results (thesis-quality)."
    )

    parser.add_argument(
        "--json",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--context",
        type=str,
        choices=list(CONTEXTS),
        default="talk",
        help="Font/figure size preset.",
    )
    parser.add_argument(
        "--rt2_ylim",
        type=float,
        nargs=2,
        default=None,
        help="Shared RT2 y-limits, e.g. 440 700.",
    )
    parser.add_argument(
        "--rt1_ylim",
        type=float,
        nargs=2,
        default=None,
        help="Shared RT1 y-limits, e.g. 240 340.",
    )
    parser.add_argument(
        "--pashler",
        action="store_true",
        help=(
            "Overlay Pashler (1994) Fig. 1 schematic "
            "on the main RT2 panel."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/plots/ensemble",
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "main",
            "error",
            "onset",
            "greedy_summary",
        ),
        default="all",
        help=(
            "Generate only the selected figure type. "
            "'all' preserves the original behavior and "
            "does not generate greedy_summary."
        ),
    )

    args = parser.parse_args()
    scale = set_context(args.context)

    json_paths = []
    for pattern in args.json:
        expanded = sorted(glob.glob(pattern))
        json_paths.extend(
            expanded if expanded else [pattern]
        )

    for json_path in json_paths:
        print(f"\nProcessing: {json_path}")

        with open(json_path, "r") as file:
            data = json.load(file)

        tag = data.get(
            "tag",
            Path(json_path).stem,
        )

        if args.only in ("all", "main"):
            plot_main(
                data,
                os.path.join(
                    args.out_dir,
                    f"{tag}_main",
                ),
                scale=scale,
                add_pashler=args.pashler,
                rt1_ylim=args.rt1_ylim,
                rt2_ylim=args.rt2_ylim,
            )

        if args.only in ("all", "error"):
            plot_error_rates(
                data,
                os.path.join(
                    args.out_dir,
                    "ER",
                    f"{tag}_er",
                ),
                scale=scale,
            )

        if args.only in ("all", "onset"):
            plot_onset_delay(
                data,
                os.path.join(
                    args.out_dir,
                    "onset",
                    f"{tag}_onset",
                ),
                scale=scale,
            )

        if args.only == "greedy_summary":
            plot_greedy_summary(
                data,
                os.path.join(
                    args.out_dir,
                    "greedy_summary",
                    f"{tag}_greedy_summary",
                ),
                scale=scale,
                rt2_ylim=args.rt2_ylim,
                error_ylim=(-0.005, 0.10),
                show_task1_errors=True,
                show_titles=False,
            )


if __name__ == "__main__":
    main()
