"""Generate the raw-values longtable from output/all76.json.

Writes output/AppendixB_raw_table.tex, meant to be \\input inside the
appendix chapter alongside the derived-values table.

Row numbering matches make_appendix_table.py, so the two tables can be
read against each other.

Usage
-----
    python -m scripts.chapter2.make_appendix_raw_table
"""
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(PROJECT_ROOT, 'output', 'all76.json')
OUT = os.path.join(PROJECT_ROOT, 'output', 'AppendixB_raw_table.tex')

STUDY_ORDER = [
    "Pashler & Johnston (1989)",
    "Pashler (1990)",
    "McCann & Johnston (1992)",
    "Osman & Moore (1993)",
    "De Jong (1993)",
    "Van Selst et al. (1999)",
    "Schubert (1999)",
    "Lien et al. (2005)",
    "Sigman & Dehaene (2008)",
    "Halvorson et al. (2013)",
    "Rau & Zheng (2020)",
]

CAPTION = (
    r"Raw extracted values for all 76 RT2--SOA conditions. RT2 values are "
    r"listed in SOA order, with the SOA levels given in the subheading above "
    r"each block. RT1 is the Task~1 reaction time at the longest SOA, used as "
    r"the proxy for single-task RT1 when computing $SOA^{*}$; for De~Jong's "
    r"nogo conditions, where no Task~1 response is produced, the value is "
    r"inherited from the corresponding go condition. \emph{Src}: T = "
    r"text-reported, F = figure-extracted."
)


def esc(s):
    return str(s).replace('&', r'\&')


def main():
    with open(DATA, encoding='utf-8') as f:
        data = json.load(f)

    by_study = {s: [] for s in STUDY_ORDER}
    for d in data:
        by_study[d['study']].append(d)

    # Number in the same sequence as make_appendix_table.py.
    n = 0
    for study in STUDY_ORDER:
        for d in by_study[study]:
            n += 1
            d['_n'] = n

    L = []
    w = L.append
    hdr = r'\# & Exp & Condition & RT2 per SOA (ms) & RT1 & Src \\'

    w(r'\footnotesize')
    w(r'\begin{longtable}{@{}r l l l r c@{}}')
    w(rf'\caption[Raw RT2--SOA extraction values]{{{CAPTION}}}'
      r'\label{tab:extraction_raw}\\')
    w(r'\toprule')
    w(hdr)
    w(r'\midrule')
    w(r'\endfirsthead')
    w(r'\multicolumn{6}{@{}l}{\itshape Table \thetable\ (continued)}\\')
    w(r'\toprule')
    w(hdr)
    w(r'\midrule')
    w(r'\endhead')
    w(r'\midrule')
    w(r'\multicolumn{6}{r@{}}{\itshape continued on next page}\\')
    w(r'\endfoot')
    w(r'\bottomrule')
    w(r'\endlastfoot')

    for i, study in enumerate(STUDY_ORDER):
        if i:
            w(r'\addlinespace')
        w(rf'\multicolumn{{6}}{{@{{}}l}}{{\bfseries {esc(study)}}}\\')

        soa_sets = []
        for d in by_study[study]:
            key = tuple(d['soas'])
            if key not in soa_sets:
                soa_sets.append(key)

        for key in soa_sets:
            soa_txt = ', '.join(str(int(s)) for s in key)
            w(rf'\multicolumn{{6}}{{@{{}}l}}'
              rf'{{\itshape SOAs: {soa_txt}\,ms}}\\')
            for d in by_study[study]:
                if tuple(d['soas']) != key:
                    continue
                rt2_txt = ', '.join(str(int(v)) for v in d['rt2'])
                rt1_txt = '---' if d['rt1'] is None else f"{int(d['rt1'])}"
                w(' & '.join([
                    str(d['_n']),
                    esc(d['exp']),
                    esc(d['cond']),
                    rt2_txt,
                    rt1_txt,
                    d['src'],
                ]) + r' \\')

    w(r'\end{longtable}')
    w(r'\normalsize')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    print(f"Written {n} conditions to {OUT}")


if __name__ == '__main__':
    main()