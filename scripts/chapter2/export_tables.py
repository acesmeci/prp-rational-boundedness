"""Export the extraction data as CSV and Markdown for reuse outside LaTeX.

Usage
-----
    python -m scripts.chapter2.export_tables
"""
import csv
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(PROJECT_ROOT, 'output', 'all76.json')
OUTDIR = os.path.join(PROJECT_ROOT, 'output', 'exports')

REASON = {'unknown_order': 'unknown task order', 'simple_rt': 'simple-RT T2',
          'grouped': 'response grouping', 'suborder': 'sub-condition',
          'practice_phase': 'practice phase'}
ORDER = ['unknown_order', 'simple_rt', 'grouped', 'suborder', 'practice_phase']

COLS = ['n', 'study', 'exp', 'cond', 'soas', 'rt2', 'rt1', 'head', 'tail',
        'tail_clean', 'soa_star', 'src', 'status']


def status(d):
    flags = set(d['flags']) - {'nogo'}
    return 'primary' if not flags else '; '.join(
        REASON[f] for f in ORDER if f in flags)


def rows(data):
    for n, d in enumerate(data, 1):
        yield {
            'n': n, 'study': d['study'], 'exp': d['exp'], 'cond': d['cond'],
            'soas': ' '.join(str(int(s)) for s in d['soas']),
            'rt2': ' '.join(str(int(v)) for v in d['rt2']),
            'rt1': '' if d['rt1'] is None else int(d['rt1']),
            'head': round(d['head'], 2),
            'tail': '' if d['tail'] is None else round(d['tail'], 2),
            'tail_clean': 'Y' if d['tail_clean'] else 'N',
            'soa_star': '' if d['soa_star'] is None else round(d['soa_star']),
            'src': d['src'], 'status': status(d),
        }


def main():
    with open(DATA, encoding='utf-8') as f:
        data = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    R = list(rows(data))

    csv_path = os.path.join(OUTDIR, 'rt2_soa_extraction.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(R)

    md_path = os.path.join(OUTDIR, 'rt2_soa_extraction.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# RT2-SOA extraction: all 76 conditions\n\n')
        f.write('| ' + ' | '.join(COLS) + ' |\n')
        f.write('|' + '---|' * len(COLS) + '\n')
        for r in R:
            f.write('| ' + ' | '.join(str(r[c]) for c in COLS) + ' |\n')

    print(f'Wrote {len(R)} rows to {csv_path} and {md_path}')


if __name__ == '__main__':
    main()