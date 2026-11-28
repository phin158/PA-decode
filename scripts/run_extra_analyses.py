"""Additional analyses E1-E4 (pre-registered in docs/preregistration_extra.md), computed from the cache of the main
experiment (trained models + decoded grid) without retraining.

  E1  segmental metrics (Edit, F1@10/25/50, MS-TCN eval.py) of ASOT, ASOT-lambda and PA-decode
  E2  the selected configurations re-solved with 50 / 100 iterations and with the initialisation T0 = p nu^T
  E3  post-hoc baseline: ASOT plan reweighted by nu/q after decoding (no re-solving), same grid and 2-fold selection
  E4  label-free selection: one configuration chosen by leave-one-dataset-out

    python scripts/run_extra_analyses.py --data-dir data --cache-dir cache --out-dir results --parts E1,E2,E3,E4
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats

import _path  # noqa: F401
from pa_decode import config as cfg
from pa_decode.data import make_ds
from pa_decode.decode import CAND, CFGS, IDX_PAPER, KIND, cfg_str
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.metrics import aggregate, conf_mat, hungarian, seg_table, unit_eval
from pa_decode.pipeline import run_until
from pa_decode.selection import evaluate, ones_weights, select_configs
from pa_decode.solver import asot, video_cost
from pa_decode.supervised.metrics import agg_metrics, video_stats
from pa_decode.train import load_model

G1_SETTINGS = ['Breakfast', 'desktop_assembly', 'FSeval']
SOLVER_VARIANTS = [('it50', 50, 'uniform'), ('it100', 100, 'uniform'), ('it25_init', 25, 'target'),
                   ('it100_init', 100, 'target')]
PA_IDX = np.where((KIND == 'base') | (KIND == 'pa'))[0]
LOG = {}


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float('nan'), float('nan')
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def delta_table(df, a, b, metrics, key=('ds',)):
    """Per-seed differences a - b: mean, per-seed CI95, number of seeds > 0 / < 0."""
    rows = []
    for k, q in df.groupby(list(key)):
        pa, pb = q[q.method == a].set_index('seed'), q[q.method == b].set_index('seed')
        seeds = pa.index.intersection(pb.index)
        r = dict(zip(key, k if isinstance(k, tuple) else (k,)))
        r['pair'] = f'{a} - {b}'
        r['n_seeds'] = len(seeds)
        for m in metrics:
            d = pa.loc[seeds, m] - pb.loc[seeds, m]
            mu, lo, hi = ci(d)
            r.update({f'd{m}': mu, f'd{m}_lo': lo, f'd{m}_hi': hi, f'd{m}_pos': int((d > 1e-9).sum()),
                      f'd{m}_neg': int((d < -1e-9).sum())})
        rows.append(r)
    return pd.DataFrame(rows)


def chosen_for(exp, ds, s, method):
    ch, picks = select_configs(exp, ds, s, method, 'mof', ones_weights(exp, ds))
    return ch, picks


def target_of(c, run, K):
    if c['kind'] == 'base':
        return np.full(K, 1. / K)
    return (1 - c['w']) / K + c['w'] * np.asarray(run[f"pihat_{c['src']}"], dtype=np.float64)


def solve(cost, dy, lam, kw, n_iter, init):
    """Mirror descent of the ASOT solver (same as solver.segment_asot_pa) with a configurable initialisation.
    cost: [B,N,K]; dy: [B,K]; lam: [B]."""
    B, N, K = cost.shape
    dev, dt = cost.device, cost.dtype
    dx = torch.ones((B, N, 1), device=dev, dtype=dt) / N
    dyt = dy.reshape(B, K, 1).to(dev, dt)
    lamt = torch.as_tensor(lam, device=dev, dtype=dt).reshape(-1, 1, 1)
    T = dx * (dyt.transpose(1, 2) if init == 'target' else torch.full((B, 1, K), 1. / K, device=dev, dtype=dt))
    Cv = asot.construct_Cv_filter(N, kw['radius'], dev)
    step = None
    for it in range(n_iter):
        g = asot.grad_fgw(T, cost, kw['alpha'], Cv) - asot.grad_entropy(T, kw['eps'])
        g = g + asot.grad_kld(T, dyt.transpose(1, 2), lamt, 1)
        if it == 0:
            step = 4. / g.amax(dim=(1, 2), keepdim=True)
        T = T * torch.exp(-step * g)
        T /= T.sum(dim=2, keepdim=True)
        T = torch.nan_to_num(T, posinf=0., neginf=0.)
        T = T * dx
    return T * N


def pseudo_eval(info, segs_1d, offs, K):
    """Scores (MoF, F1, mIoU) of one segmentation of a unit, through the metric code of the paper."""
    nv = len(info['names'])
    conf = np.stack([conf_mat(segs_1d[offs[v]:offs[v + 1]], info['gt'][v], K, info['C']) for v in range(nv)])[None]
    pr = dict(segs=segs_1d[None].astype(np.int8), conf=conf.astype(np.int32), offs=offs)
    v, n = unit_eval(pr, seg_table(info, pr), np.zeros(nv, dtype=int), np.ones(nv), info)
    return v, n


# ----------------------------------------------------------------------------- E1
def segmental(exp, ds, s, chosen):
    """Acc / Edit / F1@k of one method (chosen config per video), Full protocol mapping, mean over activities."""
    per = []
    for a in exp.units_of(ds):
        run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
        nv, offs, C = len(info['names']), run['offs'], info['C']
        ch = chosen[a]
        M = run['conf'][ch, np.arange(nv)].sum(0)
        pr, gl = hungarian(M)
        lab = np.full(M.shape[0], C + 7)
        lab[pr] = gl
        rows = []
        for v in range(nv):
            g = np.asarray(info['gt'][v]).astype(int)
            p = lab[run['segs'][ch[v], offs[v]:offs[v + 1]].astype(int)]
            keep = g >= 0
            rows.append(video_stats(p[keep], g[keep]))
        per.append(agg_metrics(pd.DataFrame(rows)))
    return {k: float(np.mean([d[k] for d in per])) for k in per[0]}


def run_e1(exp, out):
    rows = []
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            for m in [cfg.M_ASOT, cfg.M_ASOTL, cfg.M_PA]:
                ch, _ = chosen_for(exp, ds, s, m)
                rows.append(dict(ds=ds, seed=s, method=m, **segmental(exp, ds, s, ch)))
        print(f'  E1 {ds}: done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'E1_segmental_per_seed.csv'), index=False)
    mets = ['Acc', 'Edit', 'F1@10', 'F1@25', 'F1@50']
    summ = df.groupby(['ds', 'method'])[mets].mean().reset_index()
    summ.to_csv(os.path.join(out, 'E1_segmental_mean.csv'), index=False)
    d = pd.concat([delta_table(df, cfg.M_PA, cfg.M_ASOTL, mets), delta_table(df, cfg.M_PA, cfg.M_ASOT, mets)])
    d.to_csv(os.path.join(out, 'E1_segmental_delta.csv'), index=False)
    v = {}
    for ds in G1_SETTINGS:
        r = d[(d.ds == ds) & (d.pair == f'{cfg.M_PA} - {cfg.M_ASOTL}')]
        if len(r):
            r = r.iloc[0]
            v[ds] = bool(r['dEdit'] >= -0.5 and r['dF1@50'] >= -0.5)
    LOG['E1'] = dict(verdict_per_setting=v, met=bool(v) and all(v.values()))
    print('  E1 verdict:', LOG['E1'], flush=True)


# ----------------------------------------------------------------------------- E2
@torch.no_grad()
def run_e2(exp, out):
    rows, check = [], {}
    for ds in exp.datasets:
        kw = exp.eval_kw(ds)
        for s in exp.seeds[ds]:
            ch = {m: chosen_for(exp, ds, s, m)[0] for m in [cfg.M_ASOTL, cfg.M_PA]}
            res = {(m, vn): ([], []) for m in ch for vn, _, _ in SOLVER_VARIANTS}
            base = {m: ([], []) for m in ch}
            for a in exp.units_of(ds):
                run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
                K, offs, nv = run['conf'].shape[2], run['offs'], len(info['names'])
                mdl, dobj = load_model(exp, ds, a, s), make_ds(exp, ds, a, None, False)
                segs = {(m, vn): np.zeros(offs[-1], dtype=np.int64) for m in ch for vn, _, _ in SOLVER_VARIANTS}
                for v in range(nv):
                    cost, _ = video_cost(mdl, dobj[v][0], exp.device)
                    cs = [CFGS[int(ch[m][a][v])] for m in ch]
                    dy = torch.tensor(np.stack([target_of(c, run, K) for c in cs]), dtype=cost.dtype, device=cost.device)
                    lam = [c['lam'] for c in cs]
                    if not check.get(ds) and v < 3:      # the 25-iteration solve must reproduce the cache
                        T = solve(cost.expand(len(cs), -1, -1), dy, lam, kw, kw['n_iters'][0], 'uniform')
                        for j, m in enumerate(ch):
                            agree = (T[j].argmax(1).cpu().numpy() == run['segs'][ch[m][a][v], offs[v]:offs[v + 1]]).mean()
                            check.setdefault(f'{ds}_agree', []).append(float(agree))
                    for vn, it, init in SOLVER_VARIANTS:
                        T = solve(cost.expand(len(cs), -1, -1), dy, lam, kw, it, init)
                        p = T.argmax(2).cpu().numpy()
                        for j, m in enumerate(ch):
                            segs[(m, vn)][offs[v]:offs[v + 1]] = p[j]
                check[ds] = True
                for m in ch:
                    vv, n = unit_eval(run, exp.stab[(ds, a, s)], ch[m][a], np.ones(nv), info)
                    base[m][0].append(vv)
                    base[m][1].append(n)
                    for vn, _, _ in SOLVER_VARIANTS:
                        vv, n = pseudo_eval(info, segs[(m, vn)], offs, K)
                        res[(m, vn)][0].append(vv)
                        res[(m, vn)][1].append(n)
                del mdl
            for m in ch:
                a1, _ = aggregate(*base[m])
                rows.append(dict(ds=ds, seed=s, method=m, variant='it25', MoF=a1[0], F1=a1[1], mIoU=a1[2]))
                for vn, _, _ in SOLVER_VARIANTS:
                    a1, _ = aggregate(*res[(m, vn)])
                    rows.append(dict(ds=ds, seed=s, method=m, variant=vn, MoF=a1[0], F1=a1[1], mIoU=a1[2]))
            print(f'  E2 {ds} seed {s}: done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'E2_solver_per_seed.csv'), index=False)
    d = delta_table(df, cfg.M_PA, cfg.M_ASOTL, ['MoF', 'F1', 'mIoU'], key=('ds', 'variant'))
    d.to_csv(os.path.join(out, 'E2_solver_delta.csv'), index=False)
    v = {}
    for ds in G1_SETTINGS:
        q = d[d.ds == ds].set_index('variant')
        if 'it25' not in q.index:
            continue
        ok = True
        for vn, _, _ in SOLVER_VARIANTS:
            r = q.loc[vn]
            ok &= bool(r['dMoF'] > 0 and r['dMoF_lo'] > 0 and abs(r['dMoF'] - q.loc['it25', 'dMoF']) < 1.0)
        v[ds] = ok
    LOG['E2'] = dict(verdict_per_setting=v, met=bool(v) and all(v.values()),
                     cache_check_agreement={k: float(np.mean(x)) for k, x in check.items() if k.endswith('_agree')})
    print('  E2 verdict:', LOG['E2'], flush=True)


# ----------------------------------------------------------------------------- E3
@torch.no_grad()
def posthoc_run(exp, ds, a, s):
    """Pseudo decoding run with the grid indices of CFGS: base = cached ASOT; pa = argmax_j T_lambda * nu_j / q_j."""
    run, info, kw = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)], exp.eval_kw(ds)
    K, C, offs, nv = run['conf'].shape[2], info['C'], run['offs'], len(info['names'])
    segs = run['segs'].copy()
    conf = run['conf'].copy()
    mdl, dobj = load_model(exp, ds, a, s), make_ds(exp, ds, a, None, False)
    base = [i for i in range(len(CFGS)) if CFGS[i]['kind'] == 'base']
    lam_b = [CFGS[i]['lam'] for i in base]
    pa = [i for i in range(len(CFGS)) if CFGS[i]['kind'] == 'pa']
    for v in range(nv):
        cost, _ = video_cost(mdl, dobj[v][0], exp.device)
        dy = torch.full((len(base), K), 1. / K, dtype=cost.dtype, device=cost.device)
        T = solve(cost.expand(len(base), -1, -1), dy, lam_b, kw, kw['n_iters'][0], 'uniform')   # [L,N,K]
        g = info['gt'][v]
        for i in pa:
            c = CFGS[i]
            li = lam_b.index(c['lam'])
            nu = torch.tensor(target_of(c, run, K) * K, dtype=T.dtype, device=T.device)         # nu_j / q_j
            p = (T[li] * nu[None]).argmax(1).cpu().numpy()
            segs[i, offs[v]:offs[v + 1]] = p
            conf[i, v] = conf_mat(p, g, K, C)
    del mdl
    return dict(segs=segs, conf=conf, offs=offs)


def run_e3(exp, out):
    rows = []
    for ds in exp.datasets:
        W = ones_weights(exp, ds)
        for s in exp.seeds[ds]:
            view = SimpleNamespace(units_of=exp.units_of, uinfo=exp.uinfo, dec={}, stab={})
            for a in exp.units_of(ds):
                pr = posthoc_run(exp, ds, a, s)
                view.dec[(ds, a, s)] = pr
                view.stab[(ds, a, s)] = seg_table(exp.uinfo[(ds, a)], pr)
            a1, _, _, _, picks = evaluate(view, ds, s, cfg.M_PA, 'mof', W, exact=True)
            rows.append(dict(ds=ds, seed=s, method='Post-hoc', MoF=a1[0], F1=a1[1], mIoU=a1[2],
                             picks='; '.join(cfg_str(i) for i in picks)))
            for m in [cfg.M_ASOTL, cfg.M_PA]:
                a1, _, _, _, picks = evaluate(exp, ds, s, m, 'mof', W, exact=True)
                rows.append(dict(ds=ds, seed=s, method=m, MoF=a1[0], F1=a1[1], mIoU=a1[2],
                                 picks='; '.join(cfg_str(i) for i in picks)))
            print(f'  E3 {ds} seed {s}: done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'E3_posthoc_per_seed.csv'), index=False)
    d = pd.concat([delta_table(df, cfg.M_PA, 'Post-hoc', ['MoF', 'F1', 'mIoU']),
                   delta_table(df, 'Post-hoc', cfg.M_ASOTL, ['MoF', 'F1', 'mIoU'])])
    d.to_csv(os.path.join(out, 'E3_posthoc_delta.csv'), index=False)
    v = {}
    for ds in G1_SETTINGS:
        r = d[(d.ds == ds) & (d.pair == f'{cfg.M_PA} - Post-hoc')]
        if len(r):
            v[ds] = bool(r.iloc[0]['dMoF'] > 0 and r.iloc[0]['dMoF_lo'] > 0)
    LOG['E3'] = dict(verdict_per_setting=v, met=sum(v.values()) >= 2)
    print('  E3 verdict:', LOG['E3'], flush=True)


# ----------------------------------------------------------------------------- E4
def run_e4(exp, out):
    # score of every configuration of the PA-decode grid, used for all videos (no selection; MoF is exact, it is the
    # only score used to choose the configuration)
    sc = {}
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            acts = exp.units_of(ds)
            for i in PA_IDX:
                vals, nfr = [], []
                for a in acts:
                    run, nv = exp.dec[(ds, a, s)], len(exp.uinfo[(ds, a)]['names'])
                    vv, n = unit_eval(run, exp.stab[(ds, a, s)], np.full(nv, i), np.ones(nv), None)
                    vals.append(vv)
                    nfr.append(n)
                sc[(ds, s, i)] = aggregate(vals, nfr)[0]
        print(f'  E4 grid scores {ds}: done', flush=True)
    grid = pd.DataFrame([dict(ds=k[0], seed=k[1], cfg=k[2], MoF=v[0], F1=v[1], mIoU=v[2]) for k, v in sc.items()])
    grid.to_csv(os.path.join(out, 'E4_grid_scores.csv'), index=False)
    mean_mof = grid.groupby(['ds', 'cfg']).MoF.mean().unstack(0)        # cfg x ds
    rows, picks = [], {}
    fs = {'FS', 'FSeval'}
    for ds in exp.datasets:
        others = [d for d in exp.datasets if d != ds and not (ds in fs and d in fs)]
        if not others:
            continue
        crit = mean_mof[others].mean(1)
        pa_pick = int(crit.loc[PA_IDX].idxmax())
        base_pick = int(crit.loc[[i for i in PA_IDX if KIND[i] == 'base']].idxmax())
        picks[ds] = dict(others=others, PA_LODO=cfg_str(pa_pick), ASOT_LODO=cfg_str(base_pick))
        W = ones_weights(exp, ds)
        for s in exp.seeds[ds]:
            for name, i in [('PA-LODO', pa_pick), ('ASOT-LODO', base_pick), (cfg.M_ASOT, IDX_PAPER)]:
                vals, nfr = [], []                       # exact metrics for the reported configurations
                for a in exp.units_of(ds):
                    info = exp.uinfo[(ds, a)]
                    vv, n = unit_eval(exp.dec[(ds, a, s)], exp.stab[(ds, a, s)], np.full(len(info['names']), i),
                                      np.ones(len(info['names'])), info)
                    vals.append(vv)
                    nfr.append(n)
                v = aggregate(vals, nfr)[0]
                rows.append(dict(ds=ds, seed=s, method=name, MoF=v[0], F1=v[1], mIoU=v[2]))
            for m in [cfg.M_ASOTL, cfg.M_PA]:
                a1 = evaluate(exp, ds, s, m, 'mof', W, exact=True)[0]
                rows.append(dict(ds=ds, seed=s, method=m, MoF=a1[0], F1=a1[1], mIoU=a1[2]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'E4_lodo_per_seed.csv'), index=False)
    d = pd.concat([delta_table(df, 'PA-LODO', cfg.M_ASOT, ['MoF', 'F1', 'mIoU']),
                   delta_table(df, 'PA-LODO', 'ASOT-LODO', ['MoF', 'F1', 'mIoU'])])
    d.to_csv(os.path.join(out, 'E4_lodo_delta.csv'), index=False)
    v = {}
    for ds in G1_SETTINGS:
        r = d[(d.ds == ds) & (d.pair == f'PA-LODO - {cfg.M_ASOT}')]
        if len(r):
            v[ds] = bool(r.iloc[0]['dMoF'] > 0 and r.iloc[0]['dMoF_lo'] > 0)
    LOG['E4'] = dict(picks=picks, verdict_per_setting=v, met=sum(v.values()) >= 2)
    print('  E4 verdict:', json.dumps(LOG['E4'], indent=1), flush=True)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--parts', default='E1,E2,E3,E4')
    a = p.parse_args()
    exp = experiment_from_args(a)
    run_until(exp, 'point', checks=False)
    out = exp.out('extra')
    os.makedirs(out, exist_ok=True)
    parts = a.parts.split(',')
    jf = os.path.join(out, 'extra_results.json')
    if os.path.exists(jf):                       # keep the verdicts of parts run in an earlier session
        with open(jf, encoding='utf-8') as f:
            LOG.update(json.load(f))
    t0 = time.time()
    for name, fn in [('E1', run_e1), ('E4', run_e4), ('E3', run_e3), ('E2', run_e2)]:   # cheapest first
        if name in parts:
            print(f'=== {name} ===', flush=True)
            fn(exp, out)
            LOG.setdefault('time_min', {})[name] = round((time.time() - t0) / 60, 1)
            with open(jf, 'w', encoding='utf-8') as f:
                json.dump(LOG, f, indent=1, default=str)
    print(json.dumps(LOG, indent=1, default=str))


if __name__ == '__main__':
    main()
