"""Tables of the supervised experiment.
A: Table-3-style results (MS, SS, + ASOT, + PA-decode) next to the numbers of the ASOT paper.
B: G1, PA-decode (pihat from training labels, lambda = 0.05, w = 1) - ASOT (lambda = 0.05), t-CI over the 5 splits.
C: sensitivity to lambda.   D: other pihat sources.   E: 2-fold selection on the test videos.   F: per split."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils import t_ci as _t_ci, show, save_json  # noqa: E402
from .config import ASOT_SUP, BASES, LAMBDAS, LAM_PAPER, METRICS, MSTCN, PAPER_T3, TARGETS  # noqa: E402
from .metrics import agg_metrics  # noqa: E402


def tci(x):
    x = np.asarray(x, float)
    lo, hi = _t_ci(x)
    return x.mean(), lo, hi


def run_metrics(df):
    """Metrics of every (base, split, seed) run of one configuration."""
    return df.groupby(['base', 'split', 'seed']).apply(lambda g: pd.Series(agg_metrics(g))).reset_index()


def make_tables(R, out_dir, seeds, extra=None):
    os.makedirs(out_dir, exist_ok=True)

    def sel(target, lam=LAM_PAPER):
        if target == 'none':
            return R[R.target == 'none']
        return R[(R.target == target) & np.isclose(R.lam, lam)]

    def split_level(df):
        """Mean over seeds within each split -> 5 values per base model."""
        return run_metrics(df).groupby(['base', 'split'])[METRICS].mean().reset_index()

    # ---------- A ----------
    TA = []
    for base in BASES:
        for lab, tgt in [('', 'none'), (' + ASOT', 'uniform'), (' + PA-decode', 'train_w1')]:
            sl = split_level(sel(tgt))
            sl = sl[sl.base == base]
            rm = run_metrics(sel(tgt))
            rm = rm[rm.base == base]
            r = {'method': base + lab}
            for mt in METRICS:
                r[mt] = sl[mt].mean()
                r[mt + ' (sd seed)'] = rm.groupby('split')[mt].std().mean()
            TA.append(r)
    for k_, v_ in PAPER_T3.items():
        TA.append({'method': f'[ASOT paper] {k_}', **dict(zip(METRICS, v_))})
    TA = pd.DataFrame(TA)
    show(TA, f'Table A (mean over 5 splits x {len(seeds)} seeds):')

    # ---------- R: reproduction of the direction of Table 3 ----------
    R_OK = {}
    for base in BASES:
        a = split_level(sel('none'))
        a = a[a.base == base].set_index('split')
        b = split_level(sel('uniform'))
        b = b[b.base == base].set_index('split')
        d = (b[METRICS] - a[METRICS])
        R_OK[base] = dict(dEdit=d['Edit'].mean(), dF1_50=d['F1@50'].mean(), ok=bool(d['Edit'].mean() > 0 and d['F1@50'].mean() > 0))
    verdict = {'R reproduces the direction of Table 3': 'MET' if all(v['ok'] for v in R_OK.values()) else 'not met'}
    print('R:', R_OK, '->', verdict['R reproduces the direction of Table 3'])

    # ---------- B ----------
    def delta_table(tgt_new, tgt_ref='uniform', lam_new=LAM_PAPER, lam_ref=LAM_PAPER):
        out = []
        for base in BASES:
            a = split_level(sel(tgt_ref, lam_ref))
            a = a[a.base == base].set_index('split')
            b = split_level(sel(tgt_new, lam_new))
            b = b[b.base == base].set_index('split')
            d = b[METRICS] - a[METRICS]
            ra = run_metrics(sel(tgt_ref, lam_ref))
            ra = ra[ra.base == base].set_index(['split', 'seed'])
            rb = run_metrics(sel(tgt_new, lam_new))
            rb = rb[rb.base == base].set_index(['split', 'seed'])
            dr = rb[METRICS] - ra[METRICS]
            r = {'base': base}
            for mt in METRICS:
                m, lo, hi = tci(d[mt])
                r[mt] = m
                r[mt + ' CI'] = f'[{lo:.2f}, {hi:.2f}]'
            r['splits ΔF1@50>0'] = int((d['F1@50'] > 0).sum())
            r['runs ΔAcc>0'] = f"{int((dr['Acc'] > 0).sum())}/{len(dr)}"
            c1 = r['Acc'] > 0 and r['F1@50'] > 0
            c2 = r['Edit'] >= -0.5 and r['F1@10'] >= -0.5
            c3 = r['splits ΔF1@50>0'] >= 4
            r['G1'] = 'MET' if (c1 and c2 and c3) else f'not met (1:{c1} 2:{c2} 3:{c3})'
            out.append(r)
        return pd.DataFrame(out)

    TB = delta_table('train_w1')
    show(TB, 'Table B: PA-decode (pihat from training labels, lambda=0.05, w=1) - ASOT (lambda=0.05), mean of 5 splits, t-CI:')
    for _, r in TB.iterrows():
        verdict[f'G1 {r.base}'] = r['G1']

    # ---------- C ----------
    TC = []
    for base in BASES:
        for tgt in ['uniform', 'train_w1']:
            for lam in LAMBDAS:
                sl = split_level(sel(tgt, lam))
                sl = sl[sl.base == base]
                TC.append({'base': base, 'target': 'uniform (ASOT)' if tgt == 'uniform' else 'pihat train (PA)', 'lambda': lam,
                           **sl[METRICS].mean().to_dict()})
    TC = pd.DataFrame(TC)
    show(TC, 'Table C: sensitivity to lambda (no selection):')

    # ---------- D ----------
    TD = pd.concat([delta_table(t).assign(source=t) for t in ['train_w05', 'codes_w1', 'plan_w1', 'oracle_w1']], ignore_index=True)
    show(TD[['source', 'base'] + METRICS], 'Table D: other pihat sources - ASOT (lambda=0.05); oracle USES test labels:')
    TV = R.groupby('base')[[c for c in R.columns if c.startswith('tv_')]].mean()
    show(TV, 'Distance (total variation) to the true proportions of the test videos:', 3)

    # ---------- E ----------
    CAND = {'ASOT-lambda': [(l, 'uniform') for l in LAMBDAS],
            'PA-lambda': [(l, t) for l in LAMBDAS for t in ['uniform', 'train_w05', 'train_w1']]}
    avg5 = lambda m: np.mean([m[k_] for k_ in METRICS])
    allsel = []

    def twofold(meth):
        rows = []
        for (base, split, seed), g in R[R.target != 'none'].groupby(['base', 'split', 'seed']):
            vids = sorted(g.video.unique())
            fold = {v: i % 2 for i, v in enumerate(vids)}
            g = g.assign(fold=g.video.map(fold))
            parts = []
            for f in (0, 1):
                tr = g[g.fold == f]
                best = max(CAND[meth], key=lambda c: avg5(agg_metrics(tr[(tr.target == c[1]) & np.isclose(tr.lam, c[0])])))
                te = g[(g.fold != f) & (g.target == best[1]) & np.isclose(g.lam, best[0])]
                parts.append(te)
                rows.append(dict(base=base, split=split, seed=seed, fold=f, meth=meth, lam=best[0], target=best[1]))
            allsel.append(pd.concat(parts).assign(meth=meth))
        return pd.DataFrame(rows)

    CH = pd.concat([twofold('ASOT-lambda'), twofold('PA-lambda')], ignore_index=True)
    S2 = pd.concat(allsel, ignore_index=True)
    TE = []
    for base in BASES:
        a = S2[(S2.meth == 'ASOT-lambda') & (S2.base == base)]
        b = S2[(S2.meth == 'PA-lambda') & (S2.base == base)]
        sa = run_metrics(a).groupby('split')[METRICS].mean()
        sb = run_metrics(b).groupby('split')[METRICS].mean()
        d = sb - sa
        r = {'base': base}
        for mt in METRICS:
            m, lo, hi = tci(d[mt])
            r[mt] = m
            r[mt + ' CI'] = f'[{lo:.2f}, {hi:.2f}]'
        TE.append(r)
    TE = pd.DataFrame(TE)
    show(TE, 'Table E: PA-lambda - ASOT-lambda, both selected by 2-fold CV on the test videos (criterion: mean of 5 metrics):')
    show(CH.groupby(['base', 'meth', 'target', 'lam']).size().rename('times').reset_index(), 'Selected configurations:')

    # ---------- F ----------
    TF = []
    for base in BASES:
        for lab, tgt in [('plain', 'none'), ('+ASOT', 'uniform'), ('+PA', 'train_w1')]:
            sl = split_level(sel(tgt))
            sl = sl[sl.base == base]
            for _, r in sl.iterrows():
                TF.append({'base': base, 'method': lab, 'split': int(r.split), **{m: r[m] for m in METRICS}})
    TF = pd.DataFrame(TF)

    # ---------- figure ----------
    fig, ax = plt.subplots(len(BASES), 3, figsize=(10, 3.0 * len(BASES)), squeeze=False)
    for i, base in enumerate(BASES):
        for j, mt in enumerate(['Acc', 'Edit', 'F1@50']):
            for tgt, sty, lab in [('uniform (ASOT)', 'k-o', 'ASOT (uniform target)'),
                                  ('pihat train (PA)', 'C1-s', 'PA-decode (pihat from training labels, w = 1)')]:
                q = TC[(TC.base == base) & (TC['target'] == tgt)]
                ax[i, j].plot(q['lambda'], q[mt], sty, ms=3, lw=1, label=lab)
            nb = split_level(sel('none'))
            ax[i, j].axhline(nb[nb.base == base][mt].mean(), c='gray', ls='--', lw=.8, label=f'{base} (no post-processing)')
            ax[i, j].set_xscale('log')
            ax[i, j].set_title(f'{base}: {mt}', fontsize=9)
            ax[i, j].set_xlabel('lambda')
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'supervised_sensitivity_lambda.png'), dpi=200)
    fig.savefig(os.path.join(out_dir, 'supervised_sensitivity_lambda.pdf'))
    plt.close(fig)

    for name, t in [('A', TA), ('B', TB), ('C', TC), ('D', TD), ('E', TE), ('F', TF)]:
        t.to_csv(os.path.join(out_dir, f'supervised_table_{name}.csv'), index=False, encoding='utf-8-sig')
    res = dict(config=dict(SEEDS=seeds, MSTCN=MSTCN, ASOT_SUP={k: (list(v) if isinstance(v, tuple) else v) for k, v in ASOT_SUP.items()},
                           LAM_PAPER=LAM_PAPER, LAMBDAS=LAMBDAS, TARGETS=TARGETS, **(extra or {})),
               verdict=verdict, R=R_OK,
               tables={n: t.to_dict(orient='records') for n, t in [('A', TA), ('B', TB), ('C', TC), ('D', TD), ('E', TE), ('F', TF)]},
               tv=TV.reset_index().to_dict(orient='records'), chosen=CH.to_dict(orient='records'),
               train_log=R.groupby(['base', 'split', 'seed'])[['train_loss', 'train_acc', 'minutes']].first().reset_index().to_dict(orient='records'))
    save_json(res, os.path.join(out_dir, 'supervised_results.json'))
    print('\nVERDICTS (pre-registered):')
    for k_, v_ in verdict.items():
        print(f'  {k_}: {v_}')
    return res
