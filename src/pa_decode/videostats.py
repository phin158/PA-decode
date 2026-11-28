"""Per-video statistics for the next phase (CAM-UOT), and a re-solve check of the cached decoding.

For every (dataset, activity, seed) and every video, without labels:
  codes_mean   [nv, K]  mean soft assignment softmax(f·Cᵀ/temp)                 (local estimate from the codes)
  plan_mass    [nv, K]  column mass of the ASOT plan, lambda = 0.01, uniform    (local estimate from the plan)
  entropy      [nv]     mean normalised entropy of the codes, in [0, 1]
  js_cp        [nv]     Jensen-Shannon divergence between codes_mean and plan_mass (natural log)
  plan_nseg    [nv]     number of segments of the ASOT segmentation
  n_frames     [nv]     number of frames (all), n_scored [nv]: frames used by the metrics
With labels (prefix gt_, diagnostics only, never used by a method):
  gt_share     [nv, C]  true proportions of the GT classes of the unit
  gt_share_k   [nv, K]  the same mapped to clusters with the Hungarian match of ASOT (as the oracle targets)
  gt_nseg      [nv]     number of GT segments

Re-solve check (reproducibility of the solver on real data): the configurations picked for ASOT, ASOT-lambda and
PA-decode in this seed are solved again on the current GPU and compared with the cached segmentation (frame agreement).
"""
import time

import numpy as np
import pandas as pd
import torch

from . import config as cfg
from .data import make_ds
from .decode import CFGS, norm_floor
from .metrics import conf_mat, hungarian
from .selection import select_configs, ones_weights
from .solver import segment_asot_pa, video_cost, nu_int
from .train import load_model


def _js(p, q, eps=1e-12):
    p, q = np.maximum(p, eps), np.maximum(q, eps)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum()


def _nseg(x):
    return int(1 + (np.diff(x) != 0).sum()) if len(x) else 0


def stats_path(exp, ds, act, seed):
    return exp.path('videostats', f'{ds}_{act}_s{seed}.npz')


def _target(c, pihat, K):
    if c['kind'] == 'base':
        return np.full(K, 1. / K)
    if c['kind'] == 'pa':
        return (1 - c['w']) / K + c['w'] * pihat[c['src']]
    if c['kind'] == 'int':
        return nu_int(pihat[c['src']], K)
    return None                                   # minmax / oracles: not re-solved here


@torch.no_grad()
def video_stats(exp, ds, act, seed, check_cfgs=(), save_plans=False):
    """Computes and caches the statistics of one unit. Returns (stats dict, re-solve rows)."""
    path = stats_path(exp, ds, act, seed)
    dev = exp.device
    info, kw = exp.uinfo[(ds, act)], exp.eval_kw(ds)
    run = exp.dec.get((ds, act, seed))
    m = load_model(exp, ds, act, seed)
    K, C = m.n_clusters, info['C']
    dobj = make_ds(exp, ds, act, None, False)
    uni = torch.full((1, K), 1. / K, device=dev)
    nv = len(info['names'])
    codes_mean, plan_mass, ent, js, pnseg, nfr, plans, base_pred = [], [], [], [], [], [], [], []
    pihat = None
    if run is not None:
        pihat = {'codes': run['pihat_codes'], 'plan': run['pihat_plan']}
    rows = []
    check_cfgs = [i for i in check_cfgs if run is not None and _target(CFGS[i], pihat, K) is not None]
    for vi in range(nv):
        cost, codes = video_cost(m, dobj[vi][0], dev)
        T = segment_asot_pa(cost, uni, cfg.LAM_PAPER, **kw)[0]
        P = codes.double().cpu().numpy()
        cm, pm = P.mean(0), T.mean(0).double().cpu().numpy()
        H = -(P * np.log(P + 1e-12)).sum(1) / np.log(K)
        pred = T.argmax(1).cpu().numpy()
        codes_mean.append(cm)
        plan_mass.append(pm)
        ent.append(H.mean())
        js.append(_js(cm, pm))
        pnseg.append(_nseg(pred))
        nfr.append(len(pred))
        base_pred.append(pred)
        if save_plans:
            plans.append((T / T.shape[0]).cpu().numpy().astype(np.float16))
        if check_cfgs:                                   # re-solve the picked configurations and compare with the cache
            lo, hi = run['offs'][vi], run['offs'][vi + 1]
            dys = torch.tensor(np.stack([_target(CFGS[i], pihat, K) for i in check_cfgs]), dtype=torch.float32, device=dev)
            Tb = segment_asot_pa(cost.expand(len(check_cfgs), -1, -1), dys, [CFGS[i]['lam'] for i in check_cfgs], **kw)
            for j, i in enumerate(check_cfgs):
                agree = float((Tb[j].argmax(1).cpu().numpy() == run['segs'][i, lo:hi]).mean())
                rows.append(dict(dataset=ds, activity=act, seed=seed, video=info['names'][vi], config=i,
                                 n_frames=hi - lo, agree=agree))
    codes_mean, plan_mass = np.array(codes_mean), np.array(plan_mass)
    # global estimates: must equal the pihat stored in the decoding cache (same model, same solver)
    g_codes = norm_floor(codes_mean.mean(0), cfg.PIHAT_FLOOR)
    g_plan = norm_floor(plan_mass.mean(0), cfg.PIHAT_FLOOR)
    pihat_ok = None
    if run is not None:
        pihat_ok = float(max(np.abs(g_codes - run['pihat_codes']).max(), np.abs(g_plan - run['pihat_plan']).max()))
    # labels (diagnostics only)
    gshare = np.array([np.bincount(g[g >= 0], minlength=C) / max((g >= 0).sum(), 1) for g in info['gt']])
    Mb = sum(conf_mat(p, g, K, C) for p, g in zip(base_pred, info['gt']))
    pr, gl = hungarian(Mb)
    gk = np.zeros((nv, K))
    gk[:, pr] = gshare[:, gl]
    out = dict(names=np.array(info['names']), fold=info['fold'], n_frames=np.array(nfr), n_scored=info['nfr'],
               codes_mean=codes_mean, plan_mass=plan_mass, entropy=np.array(ent), js_cp=np.array(js),
               plan_nseg=np.array(pnseg), pihat_codes=g_codes, pihat_plan=g_plan,
               gt_share=gshare, gt_share_k=gk, gt_match_pred=pr, gt_match_gt=gl,
               gt_nseg=np.array([_nseg(g[g >= 0]) for g in info['gt']]))
    if save_plans:
        out['plans'] = np.concatenate(plans, 0)
        out['plan_offs'] = np.concatenate([[0], np.cumsum(nfr)])
    np.savez_compressed(path, **out)
    return out, rows, pihat_ok


def dump_all(exp, save_plans=False, check=True, verbose=True):
    """Video statistics for every unit and seed, and the re-solve check of the picked configurations."""
    t0 = time.time()
    rows, prow = [], []
    for ds in exp.datasets:
        W = ones_weights(exp, ds)
        for seed in exp.seeds[ds]:
            picks = set()
            if check and exp.dec:
                for meth in (cfg.M_ASOT, cfg.M_ASOTL, cfg.M_PA):
                    picks.update(int(i) for i in select_configs(exp, ds, seed, meth, 'mof', W)[1])
            for a in exp.units_of(ds):
                _, r, pok = video_stats(exp, ds, a, seed, sorted(picks), save_plans)
                rows += r
                prow.append(dict(dataset=ds, activity=a, seed=seed, max_abs_diff_pihat=pok))
        if verbose:
            print(f'{exp.label(ds)}: video statistics done ({(time.time() - t0) / 60:.1f} min)')
    df = pd.DataFrame(rows)
    pdf = pd.DataFrame(prow)
    pdf.to_csv(exp.out('resolve_check_pihat.csv'), index=False)
    if len(df):
        df.to_csv(exp.out('resolve_check_frames.csv'), index=False)
        tot = (df.agree * df.n_frames).sum() / df.n_frames.sum()
        by = (df.assign(w=df.agree * df.n_frames).groupby(['dataset', 'seed', 'config'])[['w', 'n_frames']].sum())
        by = (by.w / by.n_frames).rename('agree').reset_index()
        worst = by.agree.min()
        ok = worst >= cfg.R3_AGREE
        if verbose:
            print(f'{"OK" if ok else "!! CHECK"} [R3 re-solve]: frame agreement with the cached segmentation = {tot:.5f} '
                  f'overall, worst (dataset, seed, configuration) = {worst:.5f} (criterion >= {cfg.R3_AGREE}); '
                  f'max |pihat - cached pihat| = {pdf.max_abs_diff_pihat.max():.2e}')
        by.to_csv(exp.out('resolve_check_summary.csv'), index=False)
    return df, pdf
