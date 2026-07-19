"""Small-multiples figure: 54 primary RT2-SOA curves across 11 studies.
De Jong split go/nogo -> 12 panels (3x4 grid).
Elements: extracted RT2 points, -1 reference gradient (red dashed),
SOA* shaded band across each panel's conditions.

Usage:
    python -m scripts.plot_empirical_curves [--context thesis|talk]
"""
import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

DATA = os.path.join(os.path.dirname(__file__), '..', 'output', 'all76.json')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots', 'ensemble')

def primary(d):
    return not (set(d['flags']) & {'grouped', 'unknown_order', 'simple_rt',
                                    'practice_phase', 'suborder'})

PANELS = [
    ("Pashler & Johnston\n(1989)",
     lambda d: d['study'].startswith('Pashler & Johnston')),
    ("Pashler (1990)",
     lambda d: d['study'] == 'Pashler (1990)'),
    ("McCann & Johnston\n(1992)",
     lambda d: d['study'].startswith('McCann')),
    ("Osman & Moore\n(1993)",
     lambda d: d['study'].startswith('Osman')),
    ("De Jong (1993)\ngo trials",
     lambda d: d['study'].startswith('De Jong') and 'nogo' not in set(d['flags'])),
    ("De Jong (1993)\nnogo trials",
     lambda d: d['study'].startswith('De Jong') and 'nogo' in set(d['flags'])),
    ("Van Selst et al.\n(1999)",
     lambda d: d['study'].startswith('Van Selst')),
    ("Schubert (1999)",
     lambda d: d['study'].startswith('Schubert')),
    ("Lien et al. (2005)",
     lambda d: d['study'].startswith('Lien')),
    ("Sigman & Dehaene\n(2008)",
     lambda d: d['study'].startswith('Sigman')),
    ("Halvorson et al.\n(2013)",
     lambda d: d['study'].startswith('Halvorson')),
    ("Rau & Zheng (2020)",
     lambda d: d['study'].startswith('Rau')),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', default='thesis', choices=['thesis', 'talk'])
    args = parser.parse_args()

    D = json.load(open(DATA))
    P = [d for d in D if primary(d)]
    print(f"Primary set: {len(P)} conditions")

    # Style
    if args.context == 'thesis':
        mpl.rcParams.update({
            'font.size': 7.5, 'axes.linewidth': 0.6,
            'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
            'font.family': 'sans-serif',
        })
        figsize = (7.2, 5.6)
        title_fs, label_fs, ms = 7, 7.5, 2.2
    else:
        mpl.rcParams.update({
            'font.size': 11, 'axes.linewidth': 0.8,
            'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
            'font.family': 'sans-serif',
        })
        figsize = (12, 8)
        title_fs, label_fs, ms = 10, 11, 4

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    line_kw = dict(color='#2b5f8a', lw=0.9, marker='o', ms=ms,
                   mfc='#2b5f8a', mec='none', alpha=0.85)

    for ax, (title, sel) in zip(axes.flat, PANELS):
        conds = [d for d in P if sel(d)]
        all_soas = sorted({s for d in conds for s in d['soas']})

        # SOA* shaded range
        stars = [d['soa_star'] for d in conds if d.get('soa_star')]
        if stars:
            ax.axvspan(min(stars), max(stars), color='0.88', zorder=0)

        # Plot each condition
        for d in conds:
            ax.plot(d['soas'], d['rt2'], **line_kw)

        # -1 reference gradient anchored at mean shortest-SOA RT2
        s0 = min(all_soas)
        y0 = np.mean([d['rt2'][0] for d in conds])
        # For nogo panel (no RT1 → no SOA*), borrow from go conditions
        if not stars:
            go_stars = [d['soa_star'] for d in P
                        if d['study'].startswith('De Jong')
                        and 'nogo' not in set(d['flags'])
                        and d.get('soa_star')]
            x_end = min(go_stars) if go_stars else max(all_soas)
        else:
            x_end = min(stars)
        ax.plot([s0, x_end], [y0, y0 - (x_end - s0)],
                color='#c0392b', lw=1.0, ls='--', zorder=3)

        ax.set_title(f"{title}  (n={len(conds)})", fontsize=title_fs, pad=3)
        ax.tick_params(length=2, pad=1)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel('SOA (ms)', fontsize=label_fs)
    for row in axes:
        row[0].set_ylabel('RT2 (ms)', fontsize=label_fs)
    
    # ── Shared legend (bottom of figure) ──
    legend_elements = [
        Line2D([0], [0], color='#2b5f8a', lw=0.9, marker='o', ms=ms,
               mfc='#2b5f8a', mec='none', label='Extracted RT2-SOA curve'),
        Line2D([0], [0], color='#c0392b', lw=1.0, ls='--',
               label='Slope of −1 (predicted)'),
        mpl.patches.Patch(facecolor='0.88', edgecolor='none',
                          label='SOA* boundary region'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=3, fontsize=label_fs, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    fig.subplots_adjust(bottom=0.08)  # make room for legend

    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig_empirical_curves.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved {path}")

if __name__ == '__main__':
    main()