"""Decode every video with every configuration of the grid and cache the segmentations.

Grid (150 configurations):
  base        ASOT with a uniform target, lambda in LAMBDAS                                     (10)
  pa          PA-decode, target (1-w)/K + w*pihat, lambda x source x w                          (80)
  oracle_vid  true proportions of the video (uses labels; diagnostic)                           (10)
  oracle_ds   true proportions of the dataset (uses labels; diagnostic = PA-true-proportions)   (10)
  int         PA-integral (closed-form target, no w), lambda x source                           (20)
  mm          PA-minmax (smooth max of two KL terms, no w), lambda x source                     (20)
pihat is estimated WITHOUT labels from the codes (mean softmax) or from the ASOT plan (mean column mass) of the
same model, averaged over all videos of the unit.
"""
import hashlib
import json
import time

import numpy as np
import torch

from . import config as cfg
from .data import make_ds
from .metrics import conf_mat, hungarian
from .solver import segment_asot_pa, segment_asot_mm, nu_int, video_cost, check_against_asot, check_minmax, check_nu_int
from .train import load_model

CFGS = [dict(kind='base', lam=l, w=0., src='-') for l in cfg.LAMBDAS]
CFGS += [dict(kind='pa', lam=l, w=w, src=s) for l in cfg.LAMBDAS for s in cfg.SRCS for w in cfg.WS]
CFGS += [dict(kind=o, lam=l, w=1., src=o) for l in cfg.LAMBDAS for o in ['oracle_vid', 'oracle_ds']]
CFGS += [dict(kind='int', lam=l, w=-1., src=s) for l in cfg.LAMBDAS for s in cfg.SRCS]
CFGS += [dict(kind='mm', lam=l, w=-1., src=s) for l in cfg.LAMBDAS for s in cfg.SRCS]
KIND = np.array([c['kind'] for c in CFGS])
IDX_PAPER = next(i for i, c in enumerate(CFGS) if c['kind'] == 'base' and c['lam'] == cfg.LAM_PAPER)
BASE_IDX = np.where(KIND == 'base')[0]
assert list(BASE_IDX) == list(range(len(cfg.LAMBDAS)))
SOLVE_STD = np.where(np.isin(KIND, ['pa', 'oracle_vid', 'oracle_ds', 'int']))[0]   # solved with segment_asot_pa
SOLVE_MM = np.where(KIND == 'mm')[0]

# candidate configurations of every method (the selection picks one of them)
CAND = {cfg.M_ASOT: np.array([IDX_PAPER]),
        cfg.M_ASOTL: np.where(KIND == 'base')[0],
        cfg.M_PA: np.where((KIND == 'base') | (KIND == 'pa'))[0],          # includes w = 0 (= ASOT-lambda)
        cfg.M_INT: np.where((KIND == 'base') | (KIND == 'int'))[0],
        cfg.M_MM: np.where((KIND == 'base') | (KIND == 'mm'))[0],
        cfg.M_ORV: np.where(KIND == 'oracle_vid')[0],
        cfg.M_ORD: np.where(KIND == 'oracle_ds')[0]}


def cfg_str(i):
    c = CFGS[i]
    return {'pa': f"pa lam={c['lam']} w={c['w']} {c['src']}",
            'int': f"int lam={c['lam']} {c['src']}",
            'mm': f"minmax lam={c['lam']} {c['src']}"}.get(c['kind'], f"{c['kind']} lam={c['lam']}")


def norm_floor(p, floor):
    p = np.maximum(np.asarray(p, dtype=np.float64), floor)
    return p / p.sum()


def _gt_hash(exp, ds, act):
    return hashlib.md5(b''.join(g.astype(np.int16).tobytes() for g in exp.uinfo[(ds, act)]['gt'])).hexdigest()[:6]


def decode_path(exp, ds, act, seed):
    key = [CFGS, exp.eval_kw(ds), cfg.LAM_PAPER, _gt_hash(exp, ds, act), exp.paper[ds]['ne'], cfg.TAU_MM]
    if getattr(exp, 'backbone', 'asot') != 'asot':
        key.append(exp.backbone)
    tag = hashlib.md5(json.dumps(key).encode()).hexdigest()[:8]
    return exp.path('decode', f'{ds}_{act}_s{seed}_{tag}.npz')


def _solve_std(cost, ids, dys, kw, lam_all, chunk):
    out = {}
    for s0 in range(0, len(ids), chunk):
        sub = ids[s0:s0 + chunk]
        T = segment_asot_pa(cost.expand(len(sub), -1, -1), dys[s0:s0 + len(sub)], [lam_all[i] for i in sub], **kw)
        for j, i in enumerate(sub):
            out[i] = T[j].argmax(1)
    return out


@torch.no_grad()
def decode_run(exp, ds, act, seed):
    """Returns segs [ncfg, total_frames] (int8), conf [ncfg, n_videos, K, C], offs, pihat_codes, pihat_plan."""
    path = decode_path(exp, ds, act, seed)
    try:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    except FileNotFoundError:
        pass
    dev = exp.device
    chunk = 40 if dev.type == 'cuda' else 16
    info, kw = exp.uinfo[(ds, act)], exp.eval_kw(ds)
    m = load_model(exp, ds, act, seed)
    K, C = m.n_clusters, info['C']
    dobj = make_ds(exp, ds, act, None, False)
    uni = torch.full((1, K), 1. / K, device=dev)
    lens = np.array([len(g) for g in info['gt']])
    offs = np.concatenate([[0], np.cumsum(lens)])
    # Step 1 (no labels): pihat from the ASOT plan and from the codes
    plan_m, codes_m, base_pred = [], [], []
    for vi in range(len(lens)):
        cost, codes = video_cost(m, dobj[vi][0], dev)
        assert cost.shape[1] == lens[vi]
        T = segment_asot_pa(cost, uni, cfg.LAM_PAPER, **kw)[0]
        plan_m.append(T.mean(0).double().cpu().numpy())
        codes_m.append(codes.mean(0).double().cpu().numpy())
        base_pred.append(T.argmax(1).cpu().numpy())
    pihat = {'codes': norm_floor(np.mean(codes_m, 0), cfg.PIHAT_FLOOR), 'plan': norm_floor(np.mean(plan_m, 0), cfg.PIHAT_FLOOR)}
    # Oracle targets (USE LABELS, diagnostic only): true proportions mapped to clusters by the Hungarian match of ASOT
    Mb = sum(conf_mat(p, g, K, C) for p, g in zip(base_pred, info['gt']))
    pr, gl = hungarian(Mb)

    def oracle_dy(share):
        d = np.zeros(K)
        d[pr] = share[gl]
        return norm_floor(d, cfg.ORACLE_FLOOR)

    gshare = [np.bincount(g[g >= 0], minlength=C) / max((g >= 0).sum(), 1) for g in info['gt']]
    or_ds = oracle_dy(np.mean(gshare, 0))
    segs = np.zeros((len(CFGS), offs[-1]), dtype=np.int8)
    conf = np.zeros((len(CFGS), len(lens), K, C), dtype=np.int32)
    lam_all = [c['lam'] for c in CFGS]
    todo_std, todo_mm = list(SOLVE_STD), list(SOLVE_MM)
    targets = {}
    for i in todo_std:
        c = CFGS[i]
        targets[i] = ((1 - c['w']) / K + c['w'] * pihat[c['src']] if c['kind'] == 'pa' else
                      nu_int(pihat[c['src']], K) if c['kind'] == 'int' else
                      or_ds if c['kind'] == 'oracle_ds' else None)      # oracle_vid: per video
    for vi in range(len(lens)):
        cost, codes = video_cost(m, dobj[vi][0], dev)
        or_v = oracle_dy(gshare[vi])
        preds = {}
        Tb = segment_asot_pa(cost.expand(len(BASE_IDX), -1, -1), uni.expand(len(BASE_IDX), -1),
                             [lam_all[i] for i in BASE_IDX], **kw)          # ASOT (uniform target) for every lambda
        for j, i in enumerate(BASE_IDX):
            preds[i] = Tb[j].argmax(1)
        dys = torch.tensor(np.stack([targets[i] if targets.get(i) is not None else or_v for i in todo_std]),
                           dtype=torch.float32, device=dev)
        preds.update(_solve_std(cost, todo_std, dys, kw, lam_all, chunk))
        for s0 in range(0, len(todo_mm), chunk):                           # PA-minmax
            sub = todo_mm[s0:s0 + chunk]
            pb = torch.tensor(np.stack([pihat[CFGS[i]['src']] for i in sub]), dtype=torch.float32, device=dev)
            T = segment_asot_mm(cost.expand(len(sub), -1, -1), uni.expand(len(sub), -1), pb, [lam_all[i] for i in sub], **kw)
            for j, i in enumerate(sub):
                preds[i] = T[j].argmax(1)
        assert len(preds) == len(CFGS), (len(preds), len(CFGS))
        for i, pr_ in preds.items():
            pr_ = pr_.cpu().numpy()
            segs[i, offs[vi]:offs[vi + 1]] = pr_
            conf[i, vi] = conf_mat(pr_, info['gt'][vi], K, C)
    out = dict(segs=segs, conf=conf, offs=offs, pihat_codes=pihat['codes'], pihat_plan=pihat['plan'])
    np.savez_compressed(path, **out)
    return out


def run_solver_checks(exp):
    """Sanity checks of the solvers on one real video per dataset."""
    check_nu_int()
    for ds in exp.datasets:
        act = exp.units_of(ds)[0]
        m = load_model(exp, ds, act, exp.seeds[ds][0])
        feat = make_ds(exp, ds, act, None, False)[0][0]
        if getattr(exp, 'backbone', 'asot') == 'peot':
            from .peot import check_embed
            print(f'{ds}: PEOT GCN tridiagonal vs dense max diff {check_embed(m, feat, exp.device):.2e}')
        cost, codes = video_cost(m, feat, exp.device)
        kw = exp.eval_kw(ds)
        check_against_asot(cost, kw, cfg.COMMON['lfe'], exp.device, tag=ds)
        check_minmax(cost, codes, kw, exp.device, tag=ds)
        del m, cost


def decode_all(exp, verbose=True):
    t0 = time.time()
    for ds in exp.datasets:
        for seed in exp.seeds[ds]:
            for act in exp.units_of(ds):
                exp.dec[(ds, act, seed)] = decode_run(exp, ds, act, seed)
        if verbose:
            print(f'{exp.label(ds)}: decoding done ({(time.time() - t0) / 60:.1f} min)')
