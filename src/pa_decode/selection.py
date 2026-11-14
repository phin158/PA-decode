"""2-fold parameter selection, point estimates and bootstrap.

For every method, ONE configuration is shared by all activities of a dataset: it is selected on fold A (criterion: mean
MoF over activities) and applied to fold B, and vice versa. Folds split videos by person/subject where available.
"""
import time

import numpy as np

from . import config as cfg
from .decode import CAND, CFGS, IDX_PAPER, cfg_str
from .metrics import seg_table, scores, f1_exact, hungarian_mof, unit_eval, aggregate


def select_configs(exp, ds, seed, method, rule, W):
    """2-fold selection only. Returns (chosen configuration per video {activity: array}, picked configuration per fold)."""
    acts, cand = exp.units_of(ds), CAND[method]
    chosen = {a: np.zeros(len(exp.uinfo[(ds, a)]['names']), dtype=int) for a in acts}
    picks = []
    for f in (0, 1):
        if len(cand) == 1:
            best = cand[0]
        else:
            tot, cntu = np.zeros(len(cand)), 0
            for a in acts:
                run, st, fold = exp.dec[(ds, a, seed)], exp.stab[(ds, a, seed)], exp.uinfo[(ds, a)]['fold']
                wsel = W[a] * (fold == f)
                if wsel.sum() == 0:
                    continue
                Ms = np.tensordot(wsel, run['conf'][cand], axes=([0], [1]))
                if rule == 'mof':
                    sc = [hungarian_mof(Mc) if Mc.sum() > 0 else 0. for Mc in Ms]
                else:   # 'avg3': mean of MoF, F1, mIoU
                    sw = wsel[st['vid']]
                    sc = [scores(Mc, st['cnt'][c], st['gt'], st['len'], sw, wsel.sum()).mean() if Mc.sum() > 0 else 0.
                          for Mc, c in zip(Ms, cand)]
                tot += np.array(sc)
                cntu += 1
            best = cand[int(np.argmax(tot / max(cntu, 1)))]
        for a in acts:
            chosen[a][exp.uinfo[(ds, a)]['fold'] != f] = best
        picks.append(best)
    return chosen, picks


def evaluate(exp, ds, seed, method, rule, W, exact=False):
    """Returns (A1, A2, per-activity scores, chosen configuration per video, picked configuration per fold)."""
    acts = exp.units_of(ds)
    chosen, picks = select_configs(exp, ds, seed, method, rule, W)
    vals, nfr, per = [], [], {}
    for a in acts:
        v, n = unit_eval(exp.dec[(ds, a, seed)], exp.stab[(ds, a, seed)], chosen[a], W[a], exp.uinfo[(ds, a)] if exact else None)
        vals.append(v)
        nfr.append(n)
        per[a] = v
    a1, a2 = aggregate(vals, nfr)
    return a1, a2, per, chosen, picks


def ones_weights(exp, ds):
    return {a: np.ones(len(exp.uinfo[(ds, a)]['names'])) for a in exp.units_of(ds)}


def build_segment_tables(exp):
    for ds in exp.datasets:
        for seed in exp.seeds[ds]:
            for a in exp.units_of(ds):
                exp.stab[(ds, a, seed)] = seg_table(exp.uinfo[(ds, a)], exp.dec[(ds, a, seed)])


def check_metrics(exp, ds, a, seed, first):
    """Our MoF/mIoU == the authors' metrics.py and == trainer.test; exact F1 ~ the authors' Monte-Carlo F1."""
    amet = exp.amet
    info, run, st = exp.uinfo[(ds, a)], exp.dec[(ds, a, seed)], exp.stab[(ds, a, seed)]
    nv = len(info['names'])
    ones = np.ones(nv)
    ex = exp.paper[ds]['exclude']
    gt_orig = np.concatenate([np.where(g >= 0, np.array(info['cls'])[np.maximum(g, 0)], ex if ex is not None else -999)
                              for g in info['gt']])
    pred = run['segs'][IDX_PAPER].astype(np.int64)
    mof_a, p2g = amet.eval_mof(pred, gt_orig, nv, ex)
    miou_a, _ = amet.eval_miou(pred, gt_orig, nv, ex, p2g)
    mine = scores(run['conf'][IDX_PAPER].sum(0), st['cnt'][IDX_PAPER], st['gt'], st['len'], ones[st['vid']], nv)
    assert abs(mine[0] - 100 * mof_a) < 1e-9 and abs(mine[2] - 100 * miou_a) < 1e-9, (ds, a, mine, mof_a, miou_a)
    tt = exp.trained[(ds, a, seed)]['test']
    chk = dict(ds=ds, act=a, seed=seed, d_mof=mine[0] - 100 * tt['test_mof_full'], d_miou=mine[2] - 100 * tt['test_miou_full'])
    assert abs(chk['d_mof']) < 0.2, chk
    if first:
        _, mp_loc = scores(run['conf'][IDX_PAPER].sum(0), st['cnt'][IDX_PAPER], st['gt'], st['len'], ones[st['vid']], nv, True)
        mp = {k: info['cls'][g] for k, g in mp_loc.items()}
        mapk = np.full(info['C'], -1)
        mapk[list(mp_loc.values())] = list(mp_loc.keys())
        f1_ex = f1_exact(run, info, np.full(nv, IDX_PAPER), mapk)
        np.random.seed(cfg.F1_SEED)
        f1_mc = 100 * amet.eval_f1(pred, gt_orig, nv, ex, mp, n_exper=300 if exp.quick else 1000)[0]
        chk.update(f1_exact_minus_mc=f1_ex - f1_mc, f1_pervideo_minus_mc=mine[1] - f1_mc)
        assert abs(f1_ex - f1_mc) < 0.6, (ds, a, f1_ex, f1_mc)
    return chk


def point_estimates(exp, check=True, verbose=True):
    """Fills exp.point[(ds, seed, method, rule)] = {A1, A2, per} and exp.chosen (configurations picked per fold)."""
    if not exp.stab:
        build_segment_tables(exp)
    for ds in exp.datasets:
        for si, seed in enumerate(exp.seeds[ds]):
            if check:
                for a in exp.units_of(ds):
                    exp.checks.append(check_metrics(exp, ds, a, seed, first=(si == 0 and a == exp.units_of(ds)[0])))
            W = ones_weights(exp, ds)
            for meth in cfg.METHODS:
                for rule in cfg.RULES:
                    a1, a2, per, ch, picks = evaluate(exp, ds, seed, meth, rule, W, exact=True)
                    exp.point[(ds, seed, meth, rule)] = dict(A1=a1, A2=a2, per=per)
                    exp.chosen[(ds, seed, meth, rule)] = [cfg_str(i) for i in picks]
        if verbose:
            print(f'{exp.label(ds)}: point estimates done')
    if check and verbose and exp.checks:
        print('OK: metrics match metrics.py; max |our MoF - trainer.test| =',
              f"{max(abs(c['d_mof']) for c in exp.checks):.4f};",
              'exact F1 (concatenated) - Monte-Carlo F1:',
              [round(c['f1_exact_minus_mc'], 3) for c in exp.checks if 'f1_exact_minus_mc' in c])


def seed_vals(exp, ds, meth, rule='mof', agg='A1'):
    return np.array([exp.point[(ds, s, meth, rule)][agg] for s in exp.seeds[ds]])


def bootstrap(exp, ds, rule, n_boot, seed=0):
    """Resample videos within each fold (and activity), redo the selection, average over seeds.
    Returns D [n_boot, n_pairs, 2 (A1/A2), 3 metrics] of differences for cfg.PAIRS."""
    rng = np.random.default_rng(seed)
    meths = sorted({m for p in cfg.PAIRS for m in p})
    D = np.zeros((n_boot, len(cfg.PAIRS), 2, 3))
    for b in range(n_boot):
        W = {}
        for a in exp.units_of(ds):
            fold = exp.uinfo[(ds, a)]['fold']
            w = np.zeros(len(fold))
            for f in (0, 1):
                ix = np.where(fold == f)[0]
                if len(ix):
                    np.add.at(w, rng.choice(ix, len(ix), replace=True), 1.)
            W[a] = w
        acc = {m: np.zeros((2, 3)) for m in meths}
        for s in exp.seeds[ds]:
            for m in meths:
                a1, a2, _, _, _ = evaluate(exp, ds, s, m, rule, W)
                acc[m] += np.stack([a1, a2]) / len(exp.seeds[ds])
        D[b] = [acc[x] - acc[y] for x, y in cfg.PAIRS]
    return D


def run_bootstrap(exp, verbose=True):
    t0 = time.time()
    for ds in exp.datasets:
        for rule in cfg.RULES:
            exp.boot[(ds, rule)] = bootstrap(exp, ds, rule, exp.n_boot[ds] if rule == 'mof' else max(exp.n_boot[ds] // 4, 20))
        if verbose:
            print(f'{exp.label(ds)}: bootstrap done ({(time.time() - t0) / 60:.1f} min)')
