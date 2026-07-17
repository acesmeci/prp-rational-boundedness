#!/usr/bin/env python3
"""Extended summary table of the final runs: thresholds, slopes, errors.
Prints a markdown table and writes it to output/results_summary.md. Usage:
    python -m scripts.make_results_table --json "output/results/E20_*.json"
"""
import json, glob, argparse
from pathlib import Path
import numpy as np
from prp_model.utils import steps_to_ms, sim_seconds_to_ms

SOA_STAR_FACTOR = 0.80


def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() > 1 else np.nan


def two_shortest(soa, rt):
    o = np.argsort(soa)
    s, r = np.asarray(soa)[o], np.asarray(rt)[o]
    if len(s) < 2 or not (np.isfinite(r[0]) and np.isfinite(r[1])):
        return np.nan
    return float((r[1] - r[0]) / (s[1] - s[0]))


ap = argparse.ArgumentParser()
ap.add_argument("--json", nargs="+", required=True)
ap.add_argument("--out", default="output/results_summary.md")
ap.add_argument("--legacy", action="store_true",
                help="Pre-clock JSONs: 1 step = 0.1 dt-sec = 50 ms (x500 conversion)")
args = ap.parse_args()
if args.legacy:
    def steps_to_ms(steps):
        return np.asarray(steps, float) * 50.0
    def sim_seconds_to_ms(seconds):
        return np.asarray(seconds, float) * 500.0
    
paths = sorted(sum([glob.glob(p) or [p] for p in args.json], []))
rows, lines = [], []
for jp in paths:
    d = json.load(open(jp))
    P = d["params"]; p = P["persistence"]
    oo = P.get("optimize_onset", False)
    iti = P.get("ITI", np.nan)
    soa = steps_to_ms(np.asarray(d["soa"], float))
    rt1 = np.asarray(d["avg"]["dep"].get("rt_task1_correct",
                                         d["avg"]["dep"]["rt_task1"]), float)
    rt1_ms = float(sim_seconds_to_ms(rt1)[-1])   # RT1 at longest SOA (Ch2 convention)
    star = SOA_STAR_FACTOR * rt1_ms

    # Per-network thresholds (z entries saved per net: {"dep": {...}, "ind": {...}})
    znets = d.get("z_per_net", d.get("z", []))
    z_summary = {}
    for cond in ("dep", "ind"):
        z1s = [n[cond]["z1"] for n in znets if cond in n] if znets else []
        z2s = [n[cond]["z2"] for n in znets if cond in n] if znets else []
        z_summary[cond] = (
            float(np.median(z1s)) if z1s else np.nan,
            float(np.median(z2s)) if z2s else np.nan,
            (min(z1s), max(z1s)) if z1s else (np.nan, np.nan),
            (min(z2s), max(z2s)) if z2s else (np.nan, np.nan),
        )

    for cond in ("dep", "ind"):
        a = d["avg"][cond]
        rt2 = sim_seconds_to_ms(np.asarray(
            a.get("rt_task2_from_stim_correct", a["rt_task2_from_stim"]), float))
        err = 1 - np.asarray(a["acc_task2"], float)
        z1m, z2m, z1r, z2r = z_summary[cond]
        rows.append((
            p, "strat" if oo else "greedy", iti, cond,
            z1m, z1r, z2m, z2r,
            two_shortest(soa, rt2),
            ols(soa[soa <= star], rt2[soa <= star]),
            ols(soa[soa >= star], rt2[soa >= star]),
            100 * np.nanmax(err), 100 * err[-1],
            rt1_ms, star,
        ))

rows.sort(key=lambda r: (r[1], r[2], r[0], r[3]))

hdr = ("| p | regime | ITI | cond | z1 med (min-max) | z2 med (min-max) "
       "| head | fullhead | tail | peakE% | longE% | RT1 | SOA* |")
sep = "|" + "---|" * 13
lines.append(hdr); lines.append(sep)
for r in rows:
    (p, reg, iti, cond, z1m, z1r, z2m, z2r,
     s2, head, tail, pe, le, rt1_ms, star) = r
    z1s = f"{z1m:.2f} ({z1r[0]:.1f}-{z1r[1]:.1f})" if np.isfinite(z1m) else "—"
    z2s = f"{z2m:.2f} ({z2r[0]:.1f}-{z2r[1]:.1f})" if np.isfinite(z2m) else "—"
    lines.append(
        f"| {p:.2f} | {reg} | {iti:.1f} | {cond} | {z1s} | {z2s} "
        f"| {s2:.2f} | {head:.2f} | {tail:.2f} "
        f"| {pe:.1f} | {le:.1f} | {rt1_ms:.0f} | {star:.0f} |")

out = "\n".join(lines)
print(out)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(out + "\n")
print(f"\nsaved: {args.out}")