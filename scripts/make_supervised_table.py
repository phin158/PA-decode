"""Build the LaTeX table of the supervised experiment (Table 3 of ASOT + PA-decode) from
results/supervised/supervised_table_A.csv and supervised_results.json (written by scripts/run_supervised.py).

    python scripts/make_supervised_table.py --res-dir results/supervised --out results/supervised/table_supervised.tex

Rows: the numbers reported in Table 3 of ASOT, then our runs (MS, MS + ASOT, MS + PA-decode, SS, SS + ASOT,
SS + PA-decode), averaged over the 5 splits of 50 Salads and the seeds of run_supervised.py.
Bold: best of our three rows for each base model (MS or SS) and metric.
"""
import argparse
import json
import os

import pandas as pd

METRICS = ['Acc', 'Edit', 'F1@10', 'F1@25', 'F1@50']
HEAD = ['Acc.', 'Edit', 'F1@10', 'F1@25', 'F1@50']


def fmt(x):
    return '--' if pd.isna(x) else f'{x:.1f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--res-dir', default='results/supervised')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or os.path.join(a.res_dir, 'table_supervised.tex')
    T = pd.read_csv(os.path.join(a.res_dir, 'supervised_table_A.csv')).set_index('method')
    jpath = os.path.join(a.res_dir, 'supervised_results.json')
    if os.path.exists(jpath):
        with open(jpath, encoding='utf-8') as f:
            res = json.load(f)
        print('Pre-registered verdicts:', res.get('verdict'))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\caption{Post-processing of the supervised MS-TCN++~\cite{li2023mstcn} on 50 Salads, presented as Table 3 '
                 r'of~\cite{xu2024}. MS: three refinement stages; SS: no refinement stage. Acc.\ is the MoF accuracy, Edit the '
                 r'segmental edit score and F1@$N$ the F1 score at an overlap threshold of $N$\%. Our results are averaged over '
                 r'the 5 splits; PA-decode uses $\hat{\boldsymbol\pi}$ from the training labels of each split, $w=1$, $\lambda=0.05$. '
                 r'Bold: best of our three rows for each base model}')
    lines.append(r'\label{tab:supervised}')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{l' + 'c' * len(METRICS) + '}')
    lines.append(r'\toprule')
    lines.append('Method & ' + ' & '.join(HEAD) + r'\\')
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{6}{l}{\textit{Reported in~\cite{xu2024}}}\\')
    for k in ['MS', 'SS', 'MS + ASOT', 'SS + ASOT']:
        key = f'[ASOT paper] {k}'
        if key in T.index:
            lines.append(f'{k} & ' + ' & '.join(fmt(T.loc[key, m]) for m in METRICS) + r'\\')
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{6}{l}{\textit{Our runs}}\\')
    for base in ['MS', 'SS']:
        rows = [base, f'{base} + ASOT', f'{base} + PA-decode']
        rows = [r for r in rows if r in T.index]
        best = {m: max(round(float(T.loc[r, m]), 1) for r in rows) for m in METRICS}  # ties after rounding are all bold
        for r in rows:
            cells = []
            for m in METRICS:
                v = fmt(T.loc[r, m])
                cells.append(rf'\textbf{{{v}}}' if round(float(T.loc[r, m]), 1) == best[m] else v)
            name = r.replace('PA-decode', r'\textcolor{ours}{PA-decode}')
            lines.append(f'{name} & ' + ' & '.join(cells) + r'\\')
        if base == 'MS':
            lines.append(r'\midrule')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nWritten:', out)


if __name__ == '__main__':
    main()
