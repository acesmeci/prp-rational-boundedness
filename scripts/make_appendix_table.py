"""Generate the Appendix B longtable from output/all76.json.

Writes output/AppendixB_table.tex, meant to be \\input inside the
appendix chapter.

Usage
-----
    python -m scripts.make_appendix_table
"""
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(PROJECT_ROOT, 'output', 'all76.json')
OUT = os.path.join(PROJECT_ROOT, 'output', 'AppendixB_table.tex')

REASON = {
    'unknown_order': 'unknown task order',
    'simple_rt': 'simple-RT T2',
    'grouped': 'response grouping',
    'suborder': 'sub-condition',
    'practice_phase': 'practice phase',
}
ORDER = ['unknown_order', 'simple_rt', 'grouped', 'suborder', 'practice_phase']

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
    r"Complete extraction of all 76 RT2--SOA conditions across the eleven "
    r"reviewed studies. Head slopes are computed between the two shortest "
    r"SOAs except for the documented exceptions in Van~Selst et~al.\ (1999) "
    r"and Lien et~al.\ (2005). Tail slopes are computed between the two "
    r"longest SOAs; \emph{Clean} indicates whether both lie beyond "
    r"$SOA^{*}$. \emph{Src}: T = text-reported, F = figure-extracted. "
    r"\emph{Status} gives the reason for exclusion from the 54-condition "
    r"primary evaluation set."
)


def esc(s):
    return s.replace('&', r'\&')


def status(d):
    flags = set(d['flags']) - {'nogo'}
    if not flags:
        return 'primary'
    return '; '.join(REASON[f] for f in ORDER if f in flags)


def num(x, nd=2):
    return '---' if x is None else f'${x:.{nd}f}$'


def main():
    with open(DATA, encoding='utf-8') as f:
        data = json.load(f)

    by_study = {s: [] for s in STUDY_ORDER}
    for d in data:
        by_study[d['study']].append(d)

    L = []
    w = L.append
    hdr = (r'\# & Exp & Condition & Head & Tail & Clean & $SOA^{*}$ & Src '
           r'& Status \\')

    w(r'\footnotesize')
    w(r'\begin{longtable}{@{}r l l r r c r c l@{}}')
    w(rf'\caption[Complete RT2--SOA extraction]{{{CAPTION}}}'
      r'\label{tab:extraction_full}\\')
    w(r'\toprule')
    w(hdr)
    w(r'\midrule')
    w(r'\endfirsthead')
    w(r'\multicolumn{9}{@{}l}{\itshape Table \thetable\ (continued)}\\')
    w(r'\toprule')
    w(hdr)
    w(r'\midrule')
    w(r'\endhead')
    w(r'\midrule')
    w(r'\multicolumn{9}{r@{}}{\itshape continued on next page}\\')
    w(r'\endfoot')
    w(r'\bottomrule')
    w(r'\endlastfoot')

    n = 0
    for i, study in enumerate(STUDY_ORDER):
        if i:
            w(r'\addlinespace')
        w(rf'\multicolumn{{9}}{{@{{}}l}}{{\bfseries {esc(study)}}}\\')
        for d in by_study[study]:
            n += 1
            w(' & '.join([
                str(n),
                esc(str(d['exp'])),
                esc(d['cond']),
                num(d['head']),
                num(d['tail']),
                'Y' if d['tail_clean'] else 'N',
                '---' if d['soa_star'] is None else f"{d['soa_star']:.0f}",
                d['src'],
                status(d),
            ]) + r' \\')

    w(r'\end{longtable}')
    w(r'\normalsize')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    n_primary = sum(1 for d in data if d['primary'])
    n_clean = sum(1 for d in data if d['primary'] and d['tail_clean'])
    n_flat = sum(1 for d in data
                 if d['primary'] and d['tail_clean'] and abs(d['tail']) < 0.10)
    print(f"Written {n} conditions to {OUT}")
    print(f"Primary: {n_primary}   Clean tails: {n_clean}   Flat: {n_flat}")


if __name__ == '__main__':
    main()