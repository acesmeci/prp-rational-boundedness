#!/usr/bin/env python3
"""Summary table (markdown + LaTeX rows) of head/tail slopes and peak/floor
T2 error rates across all final runs. Usage:
    python -m scripts.make_results_table --json "output/results/E20_*_ITI18_*zcD*.json"
"""
import json, glob, argparse
import numpy as np
from prp_model.utils import steps_to_ms, sim_seconds_to_ms

SOA_STAR_FACTOR = 0.80

def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() > 1 else np.nan

ap = argparse.ArgumentParser()
ap.add_argument("--json", nargs="+", required=True)
args = ap.parse_args()

paths = sorted(sum([glob.glob(p) or [p] for p in args.json], []))
rows = []
for jp in paths:
    d = json.load(open(jp))
    P = d["params"]; p = P["persistence"]; oo = P.get("optimize_onset", False)
    soa = steps_to_ms(np.asarray(d["soa"], float))
    rt1 = np.asarray(d["avg"]["dep"].get("rt_task1_correct",
                                         d["avg"]["dep"]["rt_task1"]), float)
    star = SOA_STAR_FACTOR * np.nanmean(sim_seconds_to_ms(rt1))
    for cond in ("dep", "ind"):
        a = d["avg"][cond]
        rt2 = sim_seconds_to_ms(np.asarray(
            a.get("rt_task2_from_stim_correct", a["rt_task2_from_stim"]), float))
        err = 1 - np.asarray(a["acc_task2"], float)
        rows.append((p, "strategic" if oo else "greedy", cond,
                     ols(soa[soa <= star], rt2[soa <= star]),
                     ols(soa[soa >= star], rt2[soa >= star]),
                     100 * np.nanmax(err), 100 * err[-1]))

rows.sort()
hdr = "| p | regime | condition | head | tail | peak T2 err (%) | long-SOA T2 err (%) |"
print(hdr); print("|" + "---|" * 7)
for r in rows:
    print(f"| {r[0]:.2f} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.2f} "
          f"| {r[5]:.1f} | {r[6]:.1f} |")
print("\n% LaTeX rows:")
for r in rows:
    print(f"{r[0]:.2f} & {r[1]} & {r[2]} & {r[3]:.2f} & {r[4]:.2f} "
          f"& {r[5]:.1f} & {r[6]:.1f} \\\\")