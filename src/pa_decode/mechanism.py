"""Why does PA-decode help? (analyses reported in Section VI-E of the paper)

A. Without re-solving (from the decoding cache):
   - data characteristics of every (activity, seed): imbalance, heterogeneity between videos, error of pihat ...
     (quantities named with labels USE the ground truth -> explanation only, never used for selection);
   - selection ceilings, decomposition of the gain into a "lambda part" and a "target part",
     per-activity selection, and the 'avg3' selection rule.
B. Interventions on the target distribution (re-solved with the same solver; GPU recommended):
   keep the configuration (lambda, w, source) chosen by PA-decode for every video and only replace pihat by
     perm_j : a random permutation of pihat (same shape, wrong cluster order)
     dose_b : (1-b)*pihat + b*p*, p* = true dataset proportions (USES LABELS), b in BETAS.
   dose_0 equals PA-decode and is used to check that re-solving matches the cache.
"""
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch
from scipy import stats as sst

from . import config as cfg
from .data import make_ds
from .decode import BASE_IDX, CAND, CFGS, IDX_PAPER, norm_floor
from .metrics import conf_mat, hungarian, hungarian_mof, unit_eval
from .selection import evaluate, ones_weights
from .solver import segment_asot_pa, video_cost
from .train import load_model
from .utils import t_ci, show, save_json

TV = lambda x, y: 0.5 * float(np.abs(np.asarray(x, float) - np.asarray(y, float)).sum())
VAR = [f'perm{j}' for j in range(cfg.N_PERM)] + [f'dose{b}' for b in cfg.BETAS]
I_DOSE0 = VAR.index('dose0.0')


def chosen_of(exp, ds, seed, meth):
    """Configuration (index in CFGS) used for every video, exactly as in the main results (rule mof, 2-fold)."""
    cache = exp.__dict__.setdefault('_chosen_cache', {})
    k = (ds, seed, meth)
    if k not in cache:
        cache[k] = evaluate(exp, ds, seed, meth, 'mof', ones_weights(exp, ds))[3]
    return cache[k]


def unit_mof(exp, ds, a, seed, chosen):
    run = exp.dec[(ds, a, seed)]
    nv = len(chosen)
    return 100 * hungarian_mof(run['conf'][chosen, np.arange(nv)].sum(0))


# ------------------------------------------------------------------------------------- A1: characteristics
def characteristics(exp):
    rows = []
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            chl, chp = chosen_of(exp, ds, s, cfg.M_ASOTL), chosen_of(exp, ds, s, cfg.M_PA)
            for a in exp.units_of(ds):
                run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
                K, C = run['conf'].shape[2], run['conf'].shape[3]
                nv = len(info['names'])
                offs = run['offs']
                u = np.full(K, 1. / K)
                M = run['conf'][chl[a], np.arange(nv)].sum(0)        # cluster -> class matching of ASOT-lambda (labels)
                pr, gl = hungarian(M)
                c2k = np.full(C, -1)
                c2k[gl] = pr
                Pv, Mv, Mall, L = [], [], [], []
                for v, g in enumerate(info['gt']):
                    gk = g[g >= 0]
                    # (no labels) cluster proportions predicted by ASOT-lambda on ALL frames of the video
                    pv_all = np.bincount(run['segs'][chl[a][v], offs[v]:offs[v + 1]].astype(np.int64), minlength=K) / (offs[v + 1] - offs[v])
                    Mall.append(pv_all)
                    if len(gk) == 0:
                        continue
                    pc = np.bincount(gk, minlength=C) / len(gk)
                    pk = np.zeros(K)
                    ok = c2k >= 0
                    pk[c2k[ok]] = pc[ok]
                    cv = run['conf'][chl[a][v], v].sum(1).astype(float)
                    Pv.append(pk)
                    Mv.append(cv / max(cv.sum(), 1))
                    L.append(len(gk))
                Pv, Mv, Mall = np.array(Pv), np.array(Mv), np.array(Mall)
                pbar = Pv.mean(0)
                r = dict(ds=ds, act=a, seed=s, K=K, C=C, K_minus_C=K - C, n_video=nv, len_mean=float(np.mean(L)))
                # --- uses labels ---
                r['imb'] = TV(u, pbar)                                              # imbalance of the true proportions
                r['dom'] = float(pbar.max())                                        # share of the largest class
                r['het'] = float(np.mean([TV(p, pbar) for p in Pv]))                # heterogeneity between videos
                r['err_own'] = float(np.mean([TV(m, p) for m, p in zip(Mv, Pv)]))   # proportion error of ASOT-lambda
                r['err_unif'] = float(np.mean([TV(u, p) for p in Pv]))
                for src in ('plan', 'codes'):
                    ph = run[f'pihat_{src}']
                    r[f'err_pool_{src}'] = float(np.mean([TV(ph, p) for p in Pv]))   # error of a shared pihat
                    r[f'adv_pool_{src}'] = r['err_own'] - r[f'err_pool_{src}']      # > 0: shared pihat beats ASOT-lambda
                    r[f'info_{src}'] = r['err_unif'] - r[f'err_pool_{src}']         # > 0: pihat beats uniform
                # --- no labels ---
                r['nf_nonunif'] = TV(run['pihat_plan'], u)
                r['nf_disp'] = float(np.mean([TV(m, run['pihat_plan']) for m in Mall]))
                r['nf_src_gap'] = TV(run['pihat_plan'], run['pihat_codes'])
                # --- observed gain ---
                pl, pp = exp.point[(ds, s, cfg.M_ASOTL, 'mof')]['per'][a], exp.point[(ds, s, cfg.M_PA, 'mof')]['per'][a]
                r['gain'], r['gain_f1'], r['gain_miou'] = float(pp[0] - pl[0]), float(pp[1] - pl[1]), float(pp[2] - pl[2])
                r['w_pa'] = float(np.mean([CFGS[i]['w'] if CFGS[i]['kind'] == 'pa' else 0. for i in chp[a]]))
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- A2-A4: selection
def best_shared(exp, ds, s, meth):
    cand = CAND[meth]
    tot = np.zeros(len(cand))
    for a in exp.units_of(ds):
        Ms = exp.dec[(ds, a, s)]['conf'][cand].sum(1)
        tot += np.array([hungarian_mof(Mc) for Mc in Ms])
    return 100 * tot.max() / len(exp.units_of(ds))


def best_per_unit(exp, ds, s, meth):
    cand = CAND[meth]
    vals = []
    for a in exp.units_of(ds):
        Ms = exp.dec[(ds, a, s)]['conf'][cand].sum(1)
        vals.append(max(hungarian_mof(Mc) for Mc in Ms))
    return 100 * np.mean(vals)


def eval_unitwise(exp, ds, s, meth):
    """Same 2-fold selection, but one configuration per activity instead of per dataset."""
    vals = []
    for a in exp.units_of(ds):
        run, st, fold = exp.dec[(ds, a, s)], exp.stab[(ds, a, s)], exp.uinfo[(ds, a)]['fold']
        cand = CAND[meth]
        ch = np.zeros(len(fold), dtype=int)
        for f in (0, 1):
            wsel = (fold == f).astype(float)
            Ms = np.tensordot(wsel, run['conf'][cand], axes=([0], [1]))
            sc = [hungarian_mof(Mc) if Mc.sum() > 0 else 0. for Mc in Ms]
            ch[fold != f] = cand[int(np.argmax(sc))]
        vals.append(unit_eval(run, st, ch, np.ones(len(fold)), exp.uinfo[(ds, a)])[0])
    return np.mean(vals, 0)


def selection_analysis(exp, verbose=True):
    base_of = np.array([BASE_IDX[cfg.LAMBDAS.index(c['lam'])] for c in CFGS])     # w = 0 configuration with the same lambda
    rows = []
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            r = dict(ds=ds, seed=s)
            a1l, a1p = exp.point[(ds, s, cfg.M_ASOTL, 'mof')]['A1'], exp.point[(ds, s, cfg.M_PA, 'mof')]['A1']
            r['gain'] = float(a1p[0] - a1l[0])
            r['potential_shared'] = best_shared(exp, ds, s, cfg.M_PA) - best_shared(exp, ds, s, cfg.M_ASOTL)
            r['potential_unit'] = best_per_unit(exp, ds, s, cfg.M_PA) - best_per_unit(exp, ds, s, cfg.M_ASOTL)
            chp = chosen_of(exp, ds, s, cfg.M_PA)
            lam_only = np.mean([unit_mof(exp, ds, a, s, base_of[chp[a]]) for a in exp.units_of(ds)])
            r['part_lambda'] = float(lam_only - a1l[0])          # only lambda changes (target still uniform)
            r['part_target'] = float(a1p[0] - lam_only)          # target changes, same lambda
            if len(exp.units_of(ds)) > 1:
                ul, up = eval_unitwise(exp, ds, s, cfg.M_ASOTL), eval_unitwise(exp, ds, s, cfg.M_PA)
                r['gain_unitwise'], r['gain_unitwise_f1'], r['gain_unitwise_miou'] = [float(x) for x in (up - ul)]
            W = ones_weights(exp, ds)
            al = evaluate(exp, ds, s, cfg.M_ASOTL, 'avg3', W, exact=True)[0]
            ap = evaluate(exp, ds, s, cfg.M_PA, 'avg3', W, exact=True)[0]
            r['gain_avg3'], r['gain_avg3_f1'], r['gain_avg3_miou'] = [float(x) for x in (ap - al)]
            rows.append(r)
        if verbose:
            print(f'{exp.label(ds)}: selection analysis done')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- B: interventions
def _pstar(run, info):
    """Oracle dataset target, built exactly as PA-true-proportions (Hungarian matching of ASOT)."""
    K, C = run['conf'].shape[2], run['conf'].shape[3]
    pr, gl = hungarian(run['conf'][IDX_PAPER].sum(0))
    gshare = np.mean([np.bincount(g[g >= 0], minlength=C) / max((g >= 0).sum(), 1) for g in info['gt']], 0)
    d = np.zeros(K)
    d[pr] = gshare[gl]
    return norm_floor(d, cfg.ORACLE_FLOOR)


@torch.no_grad()
def intervene(exp, ds, a, s):
    path = exp.path('v7', f'{ds}_{a}_s{s}_p{cfg.N_PERM}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    run, info, kw = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)], exp.eval_kw(ds)
    K, C = run['conf'].shape[2], run['conf'].shape[3]
    nv = len(info['names'])
    offs = run['offs']
    chp = chosen_of(exp, ds, s, cfg.M_PA)[a]
    pstar = _pstar(run, info)
    rng = np.random.default_rng(zlib.crc32(f'{ds}|{a}|{s}'.encode()))
    perms = np.stack([rng.permutation(K) for _ in range(cfg.N_PERM)])
    conf = np.zeros((len(VAR), nv, K, C), dtype=np.int64)
    m = dobj = None
    agree = []
    for v in range(nv):
        c = CFGS[chp[v]]
        if c['kind'] != 'pa':        # PA-decode fell back to w = 0 for this fold: every variant = that configuration
            conf[:, v] = run['conf'][chp[v], v]
            continue
        if m is None:
            m, dobj = load_model(exp, ds, a, s), make_ds(exp, ds, a, None, False)
        ph = run[f"pihat_{c['src']}"]
        tg = [ph[p] for p in perms] + [(1 - b) * ph + b * pstar for b in cfg.BETAS]
        dys = torch.tensor(np.stack([(1 - c['w']) / K + c['w'] * t for t in tg]), dtype=torch.float32, device=exp.device)
        cost, _ = video_cost(m, dobj[v][0], exp.device)
        T = segment_asot_pa(cost.expand(len(VAR), -1, -1), dys, [c['lam']] * len(VAR), **kw)
        pred = T.argmax(2).cpu().numpy()
        agree.append((pred[I_DOSE0] == run['segs'][chp[v], offs[v]:offs[v + 1]]).mean())
        for j in range(len(VAR)):
            conf[j, v] = conf_mat(pred[j], info['gt'][v], K, C)
    if agree:   # re-solving with the original pihat must reproduce the cached PA-decode result
        assert np.mean(agree) >= 0.999, f'{ds}/{a}/s{s}: re-solving matches the cache only {np.mean(agree):.4f}'
    out = dict(conf=conf, pstar=pstar, perms=perms, agree=np.array(np.mean(agree) if agree else 1.),
               tv_plan=TV(run['pihat_plan'], pstar), tv_codes=TV(run['pihat_codes'], pstar), tv_unif=TV(np.full(K, 1. / K), pstar))
    np.savez_compressed(path, **out)
    del m
    if exp.device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def interventions(exp, verbose=True):
    rows = []
    t0 = time.time()
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            for a in exp.units_of(ds):
                z = intervene(exp, ds, a, s)
                base = exp.point[(ds, s, cfg.M_ASOTL, 'mof')]['per'][a][0]
                r = dict(ds=ds, act=a, seed=s, tv_plan=float(z['tv_plan']), tv_codes=float(z['tv_codes']), tv_unif=float(z['tv_unif']))
                for j, nm in enumerate(VAR):
                    r[f'g_{nm}'] = 100 * hungarian_mof(z['conf'][j].sum(0)) - base      # ΔMoF vs ASOT-lambda
                r['g_perm'] = float(np.mean([r[f'g_perm{j}'] for j in range(cfg.N_PERM)]))
                rows.append(r)
        if verbose:
            print(f'{exp.label(ds)}: interventions done ({(time.time() - t0) / 60:.1f} min)')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- summary
def seedlevel(df, col):
    """Mean over activities within each seed (A1) -> one value per seed."""
    return df.groupby('seed')[col].mean().values


FEATS_LAB = ['imb', 'dom', 'het', 'err_own', 'info_plan', 'adv_pool_plan', 'K_minus_C']
FEATS_NF = ['nf_nonunif', 'nf_disp', 'nf_src_gap']


def summarize(exp, feat, sel, inter, verbose=True):
    rows = []
    for ds in exp.datasets:
        Fd, S_, I_ = feat[feat.ds == ds], sel[sel.ds == ds], inter[inter.ds == ds]
        r = {'setting': exp.label(ds), 'ΔMoF PA': S_['gain'].mean()}
        for c in FEATS_LAB + FEATS_NF:
            r[c] = Fd[c].mean()
        r['ceiling shared selection'] = S_['potential_shared'].mean()
        r['ceiling per-activity selection'] = S_['potential_unit'].mean()
        r['lambda part'] = S_['part_lambda'].mean()
        r['target part'] = S_['part_target'].mean()
        r['PA - permuted'] = np.mean(seedlevel(I_, 'g_dose0.0') - seedlevel(I_, 'g_perm'))
        for b in cfg.BETAS:
            r[f'dose beta={b}'] = np.mean(seedlevel(I_, f'g_dose{b}'))
        r['sensitivity (beta=1 - beta=0)'] = r['dose beta=1.0'] - r['dose beta=0.0']
        if 'gain_unitwise' in S_ and S_['gain_unitwise'].notna().any():
            r['ΔMoF per-activity selection'] = S_['gain_unitwise'].mean()
        r['ΔMoF rule avg3'] = S_['gain_avg3'].mean()
        r['ΔmIoU rule avg3'] = S_['gain_avg3_miou'].mean()
        rows.append(r)
    tab = pd.DataFrame(rows).set_index('setting')

    def corr_block(col):
        out = {}
        x, y = feat[col].values, feat['gain'].values
        out['rho (activity x seed)'], out['p'] = sst.spearmanr(x, y)
        U = feat.groupby(['ds', 'act'])[[col, 'gain']].mean()
        out['rho (activity)'], out['p (activity)'] = sst.spearmanr(U[col], U['gain'])
        for ds in ['Breakfast', 'YTI']:
            if ds in exp.datasets:
                Fd = feat[feat.ds == ds]
                out[f'rho within {ds}'] = sst.spearmanr(Fd[col], Fd['gain'])[0]
        return out
    corr = pd.DataFrame({c: corr_block(c) for c in FEATS_LAB + FEATS_NF}).T
    if verbose:
        show(tab.T, 'Table M1. Characteristics and interventions per setting (mean over seeds):', 3)
        show(corr, 'Table M2. Spearman correlation between characteristics and ΔMoF (PA - ASOT-lambda):', 3)

    # ---------- pre-registered hypotheses ----------
    def h_corr(col, sign=+1):
        c = corr.loc[col]
        ok = (sign * c['rho (activity x seed)'] >= 0.3 and c['p'] < 0.01 and sign * c['rho (activity)'] >= 0.3
              and ('rho within Breakfast' not in c or sign * c['rho within Breakfast'] > 0))
        return 'SUPPORTED' if ok else 'not supported'

    main3 = [d for d in cfg.MAIN if d in exp.datasets]
    all_pos = lambda arrs: all(t_ci(v)[0] > 0 for v in arrs)
    V = {'H1 advantage of a shared pihat (adv_pool_plan, +)': h_corr('adv_pool_plan', +1),
         'H2 imbalance (imb, +)': h_corr('imb', +1),
         'H3 heterogeneity between videos (het, -)': h_corr('het', -1),
         'H4 pihat better than uniform (info_plan, +)': h_corr('info_plan', +1)}
    for c in FEATS_NF:
        V[f'Label-free rule ({c})'] = ('candidate (to be checked on other data)'
                                       if corr.loc[c, 'p'] < 0.01 and abs(corr.loc[c, 'rho (activity x seed)']) >= 0.3 else 'no')
    V['H5 gain comes from the target, not from lambda'] = 'SUPPORTED' if all_pos(
        [sel[sel.ds == d]['part_target'].values for d in main3]) else 'not supported'
    V['H6 the cluster order of pihat matters (PA - permuted > 0)'] = 'SUPPORTED' if all_pos(
        [seedlevel(inter[inter.ds == d], 'g_dose0.0') - seedlevel(inter[inter.ds == d], 'g_perm') for d in main3]) else 'not supported'
    for d in [x for x in cfg.SIDE if x in exp.datasets]:
        sens = seedlevel(inter[inter.ds == d], 'g_dose1.0') - seedlevel(inter[inter.ds == d], 'g_dose0.0')
        pot = sel[sel.ds == d]['potential_shared'].values
        lo, hi = t_ci(sens)
        V[f'H7/H8 {exp.label(d)}'] = (f'sensitivity to the target {sens.mean():+.2f} [{lo:.2f}, {hi:.2f}]; shared-selection ceiling '
                                      f'{pot.mean():+.2f} -> ' + ('bottleneck is ESTIMATING pihat' if lo > 1.0 else
                                                                   'data is NOT very sensitive to the target' if hi < 1.0 else 'unclear'))
    for d in [x for x in ['Breakfast', 'YTI'] if x in exp.datasets]:
        dd = sel[sel.ds == d]['gain_unitwise'].values - sel[sel.ds == d]['gain'].values
        lo, hi = t_ci(dd)
        V[f'H9 per-activity selection on {exp.label(d)}'] = f'{dd.mean():+.2f} [{lo:.2f}, {hi:.2f}] -> ' + (
            'SUPPORTED' if lo > 0 else 'not supported')
    if verbose:
        print('================ MECHANISM: VERDICTS (pre-registered) ================')
        for k, v in V.items():
            print(f'  {k}: {v}')
    return tab, corr, V


def run_mechanism(exp, verbose=True):
    feat = characteristics(exp)
    sel = selection_analysis(exp, verbose)
    inter = interventions(exp, verbose)
    gap0 = (inter['g_dose0.0'] - feat.set_index(['ds', 'act', 'seed']).loc[list(zip(inter.ds, inter.act, inter.seed)), 'gain'].values).abs().max()
    print(f'Check: dose_0 (original pihat) - PA-decode of the main results, max deviation {gap0:.3f} MoF points (must be ~0)')
    tab, corr, V = summarize(exp, feat, sel, inter, verbose)
    feat.to_csv(exp.out('mechanism_characteristics.csv'), index=False)
    sel.to_csv(exp.out('mechanism_selection.csv'), index=False)
    inter.to_csv(exp.out('mechanism_interventions.csv'), index=False)
    save_json(dict(table=tab.reset_index().to_dict(orient='records'), corr=corr.reset_index().to_dict(orient='records'),
                   verdict=V, feat=feat.to_dict(orient='records'), sel=sel.to_dict(orient='records'),
                   inter=inter.to_dict(orient='records'), N_PERM=cfg.N_PERM, BETAS=cfg.BETAS),
              exp.out('mechanism_results.json'))
    return feat, sel, inter, tab, corr, V
