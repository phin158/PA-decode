"""ASOT / PA-decode as post-processing of MS-TCN++ logits (Appendix C of ASOT):
cost C = 2(1 - (L - Lmin)/(Lmax - Lmin)), solved per video with lambda = 0.05, alpha = 0.4, eps = 0.06, r = 0.01,
25 iterations, no temporal prior. PA-decode replaces the uniform target by (1-w)/K + w*pihat."""
import os

import numpy as np
import pandas as pd
import torch

from ..asot_bridge import asot
from ..solver import segment_asot_pa
from .config import ASOT_SUP, CFGS, FLOOR, LAM_PAPER, SPLITS
from .metrics import video_stats
from .train import train_run

KW = dict(eps=ASOT_SUP['eps'], alpha=ASOT_SUP['alpha'], radius=ASOT_SUP['radius'], n_iters=ASOT_SUP['n_iters'])


def logits_to_cost(L, device):
    Lt = torch.from_numpy(np.ascontiguousarray(L.T)).float().to(device)          # [n, K]
    lo, hi = Lt.min(), Lt.max()
    return (2. * (1. - (Lt - lo) / (hi - lo + 1e-12)))[None]


def _norm(p):
    p = np.maximum(np.asarray(p, dtype=np.float64), FLOOR)
    return p / p.sum()


def check_vs_author(cost, K, device):
    """PA-decode with a uniform target must equal the authors' segment_asot (lambda = 0.05 and two other values)."""
    mask = torch.ones(cost.shape[:2], dtype=bool, device=device)
    uni = torch.full((1, K), 1. / K, device=device)
    for lam in [LAM_PAPER, 0.0125, 0.4]:
        ref = asot.segment_asot(cost, mask, eps=KW['eps'], alpha=KW['alpha'], radius=KW['radius'], ub_frames=False,
                                ub_actions=True, lambda_frames=0.1, lambda_actions=lam, n_iters=KW['n_iters'])[0]
        mine = segment_asot_pa(cost, uni, lam, **KW)
        assert torch.allclose(mine, ref, atol=1e-5, rtol=1e-4) and (mine.argmax(2) == ref.argmax(2)).float().mean() > 0.9999, lam
    print('OK: PA-decode with a uniform target == asot.segment_asot (alpha=0.4, eps=0.06, r=0.01)')


_checked = [False]


def postprocess_run(D, base, split, seed, cache_dir, device, epochs, extra_cache_roots=()):
    """Per-video statistics of: the plain model ('none') and every (lambda, target) post-processing configuration."""
    path = os.path.join(cache_dir, 'runs', f'{base}_split{split}_seed{seed}_stats.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    L, log = train_run(D, base, split, seed, cache_dir, device, epochs, extra_cache_roots)
    K, SR = D.K, D.sr
    vids = D.split_test[split]
    costs = {v: logits_to_cost(L[v], device) for v in vids}
    if not _checked[0]:
        check_vs_author(costs[vids[0]], K, device)
        _checked[0] = True
    uni = np.full(K, 1. / K)
    pi = {'train': D.class_props(D.split_train[split], FLOOR),     # TRAINING labels (no test labels)
          'oracle': D.class_props(vids, FLOOR)}                    # TEST labels: upper bound only
    sm = [torch.softmax(torch.from_numpy(L[v]).float(), 0).mean(1).numpy() for v in vids]
    pi['codes'] = _norm(np.mean(sm, 0))
    with torch.no_grad():
        pl = [segment_asot_pa(costs[v], torch.tensor(uni[None]), LAM_PAPER, **KW)[0].mean(0).cpu().numpy() for v in vids]
    pi['plan'] = _norm(np.mean(pl, 0))
    tg = {'uniform': uni, 'train_w1': pi['train'], 'train_w05': 0.5 * uni + 0.5 * pi['train'],
          'codes_w1': pi['codes'], 'plan_w1': pi['plan'], 'oracle_w1': pi['oracle']}
    dys = torch.tensor(np.stack([tg[t] for _, t in CFGS]), dtype=torch.float32, device=device)
    lams = [l for l, _ in CFGS]
    rows = []
    for v in vids:
        gt = D.gt[v]
        base_pred = np.repeat(L[v].argmax(0), SR)                      # as predict(): repeat x sample_rate
        rows.append(dict(base=base, split=split, seed=seed, video=v, lam=np.nan, target='none', **video_stats(base_pred, gt)))
        with torch.no_grad():
            T = segment_asot_pa(costs[v].expand(len(CFGS), -1, -1), dys, lams, **KW)
        preds = T.argmax(2).cpu().numpy()
        for (lam, t), p in zip(CFGS, preds):
            rows.append(dict(base=base, split=split, seed=seed, video=v, lam=lam, target=t, **video_stats(np.repeat(p, SR), gt)))
        del T
    df = pd.DataFrame(rows)
    for k_, val in log.items():
        df[k_] = val
    for name, p_ in pi.items():
        df[f'tv_{name}'] = 0.5 * np.abs(p_ - pi['oracle']).sum()     # distance to the true test proportions
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df
