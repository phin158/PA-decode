"""Phase 1 - compare results/results.json of this repository with pa_asot_v6_results.json of the Kaggle notebook
(criterion R1: every score, selected configuration, difference and confidence interval equal up to rounding).
Pure Python, no GPU and no ASOT code needed.

    python scripts/compare_v6.py --v6 pa_asot_v6_results.json --new results/results.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# method names: notebook v6 -> repository
NAMES = {'ASOT gốc': 'ASOT', 'ASOT-λ': 'ASOT-lambda', 'PA-decode': 'PA-decode', 'PA-tích phân (1b)': 'PA-integral',
         'PA-minmax (2a)': 'PA-minmax', 'Oracle-video': 'Oracle-video', 'Oracle-tập': 'PA-true-proportions'}


def load(path):
    """json.load that tolerates text pasted after the JSON object (e.g. copied from a web page)."""
    s = open(path, encoding='utf-8').read()
    obj, end = json.JSONDecoder().raw_decode(s.lstrip('﻿'))
    rest = s[end:].strip()
    if rest:
        print(f'note: {path}: ignored {len(rest)} characters after the JSON object: {rest[:40]!r}...')
    return obj


def rename_key(k):
    return '|'.join(NAMES.get(p, p) for p in k.split('|'))


KIND_V6 = {'1b': 'int', '2a': 'minmax'}           # notebook short names of the two variants


def norm_cfg(s):
    parts = s.replace('λ', 'lam').strip().split(' ')
    parts[0] = KIND_V6.get(parts[0], parts[0])
    return ' '.join(parts)


def flat(x, prefix=''):
    """Flatten nested lists / dicts of numbers into {path: float}."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f'{prefix}/{k}'))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(flat(v, f'{prefix}[{i}]'))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out[prefix] = float(x)
    return out


def compare_numbers(old, new, section, tol):
    rows, worst = [], []
    n_cmp = n_miss = 0
    max_d = 0.
    for k, v in old.items():
        k2 = rename_key(k)
        if k2 not in new:
            n_miss += 1
            continue
        a, b = flat(v), flat(new[k2])
        for p, x in a.items():
            if p not in b:
                n_miss += 1
                continue
            n_cmp += 1
            d = abs(x - b[p]) if not (np.isnan(x) and np.isnan(b[p])) else 0.
            if d > max_d:
                max_d = d
            if d > tol:
                worst.append((d, k2 + p, x, b[p]))
    worst.sort(reverse=True)
    rows.append(dict(section=section, compared=n_cmp, missing=n_miss, max_abs_diff=max_d, n_over_tol=len(worst),
                     ok=(len(worst) == 0 and n_cmp > 0)))
    return rows, worst[:10]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v6', required=True)
    p.add_argument('--new', default='results/results.json')
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--out', default=None, help='CSV with the summary (default: next to --new)')
    a = p.parse_args()
    old, new = load(a.v6), load(a.new)
    rows, details = [], {}
    for sec in ('point', 'delta', 'D'):
        if sec in old and sec in new and old[sec] is not None and new[sec] is not None:
            r, w = compare_numbers(old[sec], new[sec], sec, a.tol)
            rows += r
            details[sec] = w
    # selected configurations
    n_cmp, bad = 0, []
    for k, v in old.get('chosen', {}).items():
        k2 = rename_key(k)
        if k2 in new.get('chosen', {}):
            n_cmp += 1
            if [norm_cfg(x) for x in v] != [norm_cfg(x) for x in new['chosen'][k2]]:
                bad.append((k2, v, new['chosen'][k2]))
    rows.append(dict(section='chosen (configurations)', compared=n_cmp, missing=len(old.get('chosen', {})) - n_cmp,
                     max_abs_diff=np.nan, n_over_tol=len(bad), ok=(n_cmp > 0 and not bad)))
    # pre-registered verdicts
    n_cmp, badv = 0, []
    for ds, vv in old.get('verdict', {}).items():
        for m, crit in vv.items():
            m2 = NAMES.get(m, m)
            nv = new.get('verdict', {}).get(ds, {}).get(m2)
            if nv is None:
                continue
            for c, val in crit.items():
                if isinstance(val, bool) and c in nv:
                    n_cmp += 1
                    if nv[c] != val:
                        badv.append((ds, m2, c, val, nv[c]))
    rows.append(dict(section='verdict (G1/K1/K2/H)', compared=n_cmp, missing=np.nan, max_abs_diff=np.nan,
                     n_over_tol=len(badv), ok=(n_cmp > 0 and not badv)))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
    for sec, w in details.items():
        if w:
            print(f'\nLargest differences in {sec}:')
            for d, k, x, y in w:
                print(f'  {d:.3g}  {k}: v6 {x:.6f}  new {y:.6f}')
    if bad:
        print('\nDifferent configurations:', *bad[:10], sep='\n  ')
    if badv:
        print('\nDifferent verdicts:', *badv, sep='\n  ')
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.new)), 'compare_v6.csv')
    df.to_csv(out, index=False)
    print('\nsaved', out)
    ok = bool(df.ok.all())
    print('\nR1:', 'MET - the repository reproduces v6 exactly.' if ok else 'NOT MET - see the differences above.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
