"""Small-multiples figure: 54 primary RT2-SOA curves across 11 studies.

De Jong (1993) is split into go and nogo panels, yielding 12 panels in
a 3 x 4 grid.

Visual encoding
---------------
- Light blue: full extracted RT2-SOA curves.
- Gray: full extracted RT2-SOA curves.
- Dashed burgundy: slope = -1 reference segment for each evaluated head segment.
- Dark blue: empirical head segment used to evaluate the slope = -1 prediction,
  including the documented Van Selst and Lien exceptions.
- Orange: clean empirical tail segment used to evaluate the slope = 0 prediction.
- Light gray band: range of SOA* values across conditions in that panel.

Usage
-----
    python -m scripts.plot_empirical_curves [--context thesis|talk]
"""

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA = os.path.join(PROJECT_ROOT, "output", "all76.json")
OUTDIR = os.path.join(PROJECT_ROOT, "output", "plots", "ensemble")

FULL_CURVE = "#b5b5b5"
HEAD = "#2b5f8a"
TAIL = "#c46b2b"
REF = "#9b2226"
BOUNDARY = "0.90"


def primary(d):
    """Return True for conditions included in the primary 54-curve set."""
    return d["primary"]


PANELS = [
    (
        "Pashler & Johnston\n(1989)",
        lambda d: d["study"].startswith("Pashler & Johnston"),
    ),
    ("Pashler (1990)", lambda d: d["study"] == "Pashler (1990)"),
    (
        "McCann & Johnston\n(1992)",
        lambda d: d["study"].startswith("McCann"),
    ),
    (
        "Osman & Moore\n(1993)",
        lambda d: d["study"].startswith("Osman"),
    ),
    (
        "De Jong (1993)\ngo trials",
        lambda d: (
            d["study"].startswith("De Jong")
            and "nogo" not in set(d["flags"])
        ),
    ),
    (
        "De Jong (1993)\nnogo trials",
        lambda d: (
            d["study"].startswith("De Jong")
            and "nogo" in set(d["flags"])
        ),
    ),
    (
        "Van Selst et al.\n(1999)",
        lambda d: d["study"].startswith("Van Selst"),
    ),
    ("Schubert (1999)", lambda d: d["study"].startswith("Schubert")),
    ("Lien et al. (2005)", lambda d: d["study"].startswith("Lien")),
    (
        "Sigman & Dehaene\n(2008)",
        lambda d: d["study"].startswith("Sigman"),
    ),
    (
        "Halvorson et al.\n(2013)",
        lambda d: d["study"].startswith("Halvorson"),
    ),
    ("Rau & Zheng (2020)", lambda d: d["study"].startswith("Rau")),
]


def head_segment_indices(condition):
    """Index range used for the head evaluation.

    Computed in generate_extraction_data.py, including the documented
    Van Selst and Lien exceptions, so the figure and the tables cannot
    diverge.
    """
    start, end = condition["head_idx"]
    return start, end



def panel_boundary_stars(panel_conditions, primary_conditions):
    """Return SOA* values for a panel.

    The De Jong nogo conditions do not carry separate RT1 estimates, so the
    corresponding go-condition values are used for the boundary band, matching
    the original plotting logic.
    """
    stars = [
        float(d["soa_star"])
        for d in panel_conditions
        if d.get("soa_star") is not None
    ]
    if stars:
        return stars

    is_dejong_nogo = (
        panel_conditions
        and panel_conditions[0]["study"].startswith("De Jong")
        and "nogo" in set(panel_conditions[0]["flags"])
    )
    if is_dejong_nogo:
        return [
            float(d["soa_star"])
            for d in primary_conditions
            if d["study"].startswith("De Jong")
            and "nogo" not in set(d["flags"])
            and d.get("soa_star") is not None
        ]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        default="thesis",
        choices=["thesis", "talk"],
    )
    args = parser.parse_args()

    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)

    primary_conditions = [d for d in data if primary(d)]
    print(f"Primary set: {len(primary_conditions)} conditions")

    if args.context == "thesis":
        mpl.rcParams.update(
            {
                "font.size": 7.5,
                "axes.linewidth": 0.6,
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
                "font.family": "sans-serif",
            }
        )
        figsize = (7.2, 5.8)
        title_fs = 7
        label_fs = 7.5
        marker_size = 2.0
        curve_lw = 0.75
        emphasis_lw = 1.20
        ref_lw = 1.05
    else:
        mpl.rcParams.update(
            {
                "font.size": 11,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "font.family": "sans-serif",
            }
        )
        figsize = (12, 8.5)
        title_fs = 10
        label_fs = 11
        marker_size = 3.3
        curve_lw = 1.05
        emphasis_lw = 1.95
        ref_lw = 1.40

    fig, axes = plt.subplots(3, 4, figsize=figsize)

    for ax, (title, selector) in zip(axes.flat, PANELS):
        conditions = [d for d in primary_conditions if selector(d)]

        stars = panel_boundary_stars(conditions, primary_conditions)
        if stars:
            ax.axvspan(
                min(stars),
                max(stars),
                color=BOUNDARY,
                zorder=0,
            )

        # Full empirical curves.
        for d in conditions:
            soas = np.asarray(d["soas"], dtype=float)
            rt2 = np.asarray(d["rt2"], dtype=float)

            ax.plot(
                soas,
                rt2,
                color=FULL_CURVE,
                lw=curve_lw,
                marker="o",
                ms=marker_size,
                mfc=FULL_CURVE,
                mec="none",
                alpha=0.82,
                zorder=1,
            )

            # Head-evaluation segment, including documented study-specific
            # exceptions for Van Selst et al. (1999) and Lien et al. (2005).
            if len(soas) >= 2:
                head_start, head_end = head_segment_indices(d)

                # Condition-specific slope = -1 reference over exactly the same
                # SOA extent as the evaluated head segment. The line starts at
                # the first empirical point and shows where RT2 would end if the
                # segment followed the theoretical -1 prediction.
                x0 = soas[head_start]
                x1 = soas[head_end]
                y0 = rt2[head_start]
                y1_ref = y0 - (x1 - x0)
                ax.plot(
                    [x0, x1],
                    [y0, y1_ref],
                    color=REF,
                    lw=ref_lw,
                    ls=(0, (3.6, 2.0)),
                    alpha=0.90,
                    zorder=1.6,
                )

                ax.plot(
                    soas[head_start : head_end + 1],
                    rt2[head_start : head_end + 1],
                    color=HEAD,
                    lw=emphasis_lw,
                    marker="o",
                    ms=marker_size + 0.2,
                    mfc=HEAD,
                    mec="none",
                    alpha=0.92,
                    zorder=2,
                )

            # Tail-evaluation segment: only when the final adjacent segment is
            # cleanly beyond SOA* according to the extraction table.
            if d.get("tail_clean", False) and len(soas) >= 2:
                ax.plot(
                    soas[-2:],
                    rt2[-2:],
                    color=TAIL,
                    lw=emphasis_lw,
                    marker="o",
                    ms=marker_size + 0.2,
                    mfc=TAIL,
                    mec="none",
                    alpha=0.95,
                    zorder=3,
                )

        ax.set_title(
            f"{title}  (n={len(conditions)})",
            fontsize=title_fs,
            pad=3,
        )
        ax.tick_params(length=2, pad=1)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("SOA (ms)", fontsize=label_fs)
    for row in axes:
        row[0].set_ylabel("RT2 (ms)", fontsize=label_fs)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=FULL_CURVE,
            lw=curve_lw,
            marker="o",
            ms=marker_size,
            mfc=FULL_CURVE,
            mec="none",
            label="Extracted RT2-SOA curve",
        ),
        Line2D(
            [0],
            [0],
            color=REF,
            lw=ref_lw,
            ls=(0, (3.6, 2.0)),
            alpha=0.95,
            label=r"Slope $= -1$ reference",
        ),
        Line2D(
            [0],
            [0],
            color=HEAD,
            lw=emphasis_lw,
            marker="o",
            ms=marker_size,
            mfc=HEAD,
            mec="none",
            label="Head-evaluation segment",
        ),
        Line2D(
            [0],
            [0],
            color=TAIL,
            lw=emphasis_lw,
            marker="o",
            ms=marker_size,
            mfc=TAIL,
            mec="none",
            label="Tail-evaluation segment",
        ),
        Patch(
            facecolor=BOUNDARY,
            edgecolor="none",
            label=r"$SOA^{*}$ boundary region",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=label_fs,
        frameon=False,
        bbox_to_anchor=(0.5, -0.028),
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.tight_layout(h_pad=1.0, w_pad=0.65)
    fig.subplots_adjust(bottom=0.18)

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUTDIR, f"fig_empirical_curves.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
