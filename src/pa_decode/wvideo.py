"""Upper bound of choosing w separately for every video (USES LABELS; diagnostic only).

The lambda chosen by PA-decode for each video is kept. For each video, every w in {0, .25, .5, .75, 1} x source of pihat
is tried (w = 0 is ASOT-lambda with the same lambda) and the choice is kept if the MoF of the WHOLE unit increases
(shared Hungarian matching); hill climbing until no improvement.
  (i)  in-sample bound: choose and score on the same frames -> always >= PA-decode, even without real signal.
  (ii) cross-fitted bound: split every video into first / second half; choose w on one half, score on the other,
       then swap; both halves are pooled into one confusion matrix before the Hungarian matching.
"""
import collections
import time

import numpy as np
import pandas as pd

from . import config as cfg
from .decode import CAND, CFGS
from .metrics import hungarian_mof
from .selection import evaluate, ones_weights, seed_vals
from .utils import t_ci, show

PA_IDS = np.array(sorted(CAND[cfg.M_PA].tolist()))
POS = {int(i): j for j, i in enumerate(PA_IDS)}
OPTS = {l: [i for i in PA_IDS.tolist() if CFGS[i]['lam'] == l] for l in cfg.LAMBDAS}   # 1 base + 8 pa per lambda


def half_confs(exp, ds, a, seed):
    """Confusion matrices of every PA-decode configuration on the first (h=0) / second (h=1) half of every video."""
    run, info = exp.dec[(ds, a, seed)], exp.uinfo[(ds, a)]
    K, C = run['conf'].shape[2], run['conf'].shape[3]
    offs = run['offs']
    nv = len(info['gt'])
    out = np.zeros((2, len(PA_IDS), nv, K, C), dtype=np.int64)
    for vi, g in enumerate(info['gt']):
        P = run['segs'][PA_IDS, offs[vi]:offs[vi + 1]].astype(np.int64)
        n = len(g)
        hmask = np.arange(n) >= n // 2
        for h in (0, 1):
            k = (g >= 0) & (hmask == bool(h))
            idx = (np.arange(len(PA_IDS))[:, None] * K * C + P[:, k] * C + g[k][None]).ravel()
            out[h, :, vi] = np.bincount(idx, minlength=len(PA_IDS) * K * C).reshape(len(PA_IDS), K, C)
    return out


def hill_climb(conf_sel, start, max_pass=4):
    """conf_sel: [nPA, nv, K, C] (indexed by POS). Per-video choice of configuration with the same lambda as the start."""
    cur = start.copy()
    nv = len(cur)
    M = conf_sel[[POS[i] for i in cur], np.arange(nv)].sum(0)
    best = hungarian_mof(M) if M.sum() > 0 else 0.
    for _ in range(max_pass):
        improved = False
        for vi in range(nv):
            base_M = M - conf_sel[POS[cur[vi]], vi]
            for o in OPTS[CFGS[cur[vi]]['lam']]:
                if o == cur[vi]:
                    continue
                M2 = base_M + conf_sel[POS[o], vi]
                sc = hungarian_mof(M2) if M2.sum() > 0 else 0.
                if sc > best + 1e-12:
                    best, M, cur[vi], improved = sc, M2, o, True
                    base_M = M - conf_sel[POS[cur[vi]], vi]
        if not improved:
            break
    return cur


def wvideo_bound(exp, ds, seed):
    _, _, _, ch, _ = evaluate(exp, ds, seed, cfg.M_PA, 'mof', ones_weights(exp, ds))
    ins, cf, n_change, wsel = [], [], 0, []
    for a in exp.units_of(ds):
        start = ch[a].astype(int)
        nv = len(start)
        full = exp.dec[(ds, a, seed)]['conf'][PA_IDS].astype(np.int64)        # [nPA, nv, K, C]
        cur = hill_climb(full, start)                                          # (i) in-sample
        ins.append(100 * hungarian_mof(full[[POS[i] for i in cur], np.arange(nv)].sum(0)))
        n_change += int((cur != start).sum())
        wsel += [CFGS[i]['w'] for i in cur]
        H = half_confs(exp, ds, a, seed)                                       # (ii) cross-fitted
        c0, c1 = hill_climb(H[0], start), hill_climb(H[1], start)
        M = H[1][[POS[i] for i in c0], np.arange(nv)].sum(0) + H[0][[POS[i] for i in c1], np.arange(nv)].sum(0)
        cf.append(100 * hungarian_mof(M))
    return np.mean(ins), np.mean(cf), n_change, wsel


def per_video_w_table(exp, verbose=True):
    rows, dval = [], {}
    t0 = time.time()
    for ds in exp.datasets:
        b, bcf, nch, ws_all = [], [], 0, []
        for s in exp.seeds[ds]:
            v, vcf, n_, ws_ = wvideo_bound(exp, ds, s)
            b.append(v)
            bcf.append(vcf)
            nch += n_
            ws_all += ws_
        b, bcf = np.array(b), np.array(bcf)
        pa = seed_vals(exp, ds, cfg.M_PA)[:, 0]
        hw, hcf = b - pa, bcf - pa
        tl, th = t_ci(hcf)
        dval[ds] = dict(Hw=hw.mean(), Hw_cf=hcf.mean(), cf_lo=tl, cf_hi=th, bound=b.mean(), bound_cf=bcf.mean(), pa=pa.mean(),
                        per_seed=dict(bound=b.tolist(), bound_cf=bcf.tolist(), Hw=hw.tolist(), Hw_cf=hcf.tolist()))
        nvid = sum(len(exp.uinfo[(ds, a)]['names']) for a in exp.units_of(ds)) * len(exp.seeds[ds])
        hist = collections.Counter(ws_all)
        rows.append({'setting': exp.label(ds), 'group': 'main' if ds in cfg.MAIN else 'side', 'PA-decode MoF': pa.mean(),
                     'in-sample bound MoF': b.mean(), 'in-sample - PA': hw.mean(),
                     'cross-fit bound MoF': bcf.mean(), 'cross-fit - PA': hcf.mean(),
                     'CI95 t (seed) cross-fit - PA': f'[{tl:.2f}, {th:.2f}]',
                     '% videos changing w (in-sample)': 100 * nch / nvid,
                     'w in bound (0/.25/.5/.75/1, %)': ' / '.join(f'{100 * hist.get(w, 0) / len(ws_all):.0f}'
                                                                 for w in [0., .25, .5, .75, 1.]),
                     'Oracle-video MoF': seed_vals(exp, ds, cfg.M_ORV)[:, 0].mean(),
                     'PA-true-proportions MoF': seed_vals(exp, ds, cfg.M_ORD)[:, 0].mean()})
    df = pd.DataFrame(rows).set_index('setting')
    if verbose:
        show(df, f'Per-video w upper bound (MoF A1, mean over seeds; {(time.time() - t0) / 60:.1f} min):')
    df.to_csv(exp.out('per_video_w_bound.csv'))
    return df, dval
