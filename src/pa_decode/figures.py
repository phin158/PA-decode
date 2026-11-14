"""Figures of the paper, generated from the real decoding results.

Video selection rule (fixed in advance, no hand picking): for every dataset, the first seed and the first activity;
the video where PA-decode gains the most MoF over ASOT-lambda and the video where it loses the most.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import config as cfg  # noqa: E402
from .data import make_ds  # noqa: E402
from .decode import CFGS, cfg_str  # noqa: E402
from .mechanism import chosen_of, seedlevel  # noqa: E402
from .metrics import hungarian, hungarian_mof  # noqa: E402
from .solver import segment_asot_pa, video_cost  # noqa: E402
from .train import load_model  # noqa: E402

SHORT = {'FSeval': '50 Salads (Eval)', 'FS': '50 Salads (Mid)', 'desktop_assembly': 'Desktop Assembly',
         'Breakfast': 'Breakfast', 'YTI': 'YouTube Instr.'}
PAPER_RC = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix', 'font.size': 7}


def _save(fig, path_noext, dpi=400):
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    fig.savefig(path_noext + '.pdf')
    fig.savefig(path_noext + '.png', dpi=dpi)
    plt.close(fig)
    print('saved', path_noext + '.pdf/.png')


def per_video_mof(exp, ds, a, s, chosen):
    run = exp.dec[(ds, a, s)]
    nv = len(chosen)
    M = run['conf'][chosen, np.arange(nv)].sum(0)
    pr, gl = hungarian(M)
    acc = np.array([run['conf'][chosen[v], v][pr, gl].sum() / max(run['conf'][chosen[v], v].sum(), 1) for v in range(nv)])
    return acc, (pr, gl)


def to_gt_colors(pred, gt, pr, gl, C):
    """cluster -> index of the matched GT class (its colour); unmatched cluster = C (grey); excluded frame = C+1 (white)."""
    k2c = {int(k): int(c) for k, c in zip(pr, gl)}
    out = np.array([k2c.get(int(p), C) for p in np.asarray(pred).astype(int)])
    out[np.asarray(gt) < 0] = C + 1
    return out


# ----------------------------------------------------------------------------- qualitative segmentation bars
def collect_qualitative(exp):
    rec = []
    for ds in exp.datasets:
        s = exp.seeds[ds][0]
        a = exp.units_of(ds)[0]
        run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
        offs, C = run['offs'], info['C']
        chl, chp = chosen_of(exp, ds, s, cfg.M_ASOTL)[a], chosen_of(exp, ds, s, cfg.M_PA)[a]
        ml, (prl, gll) = per_video_mof(exp, ds, a, s, chl)
        mp, (prp, glp) = per_video_mof(exp, ds, a, s, chp)
        d = mp - ml
        for kind, v in [('gain', int(np.argmax(d))), ('drop', int(np.argmin(d)))]:
            g = np.asarray(info['gt'][v]).astype(int)
            gt = g.copy()
            gt[gt < 0] = C + 1
            rec.append(dict(ds=ds, act=a, seed=s, kind=kind, video=info['names'][v], C=C,
                            mof_l=100 * ml[v], mof_p=100 * mp[v], gt=gt,
                            asotl=to_gt_colors(run['segs'][chl[v], offs[v]:offs[v + 1]], g, prl, gll, C),
                            pa=to_gt_colors(run['segs'][chp[v], offs[v]:offs[v + 1]], g, prp, glp, C)))
    return rec


def save_qualitative(rec, path):
    np.savez_compressed(path, meta=np.array([json.dumps({k: (float(r[k]) if k.startswith('mof') else r[k])
                                                         for k in ['ds', 'act', 'seed', 'kind', 'video', 'C', 'mof_l', 'mof_p']})
                                             for r in rec]),
                        **{f'{i}_{k}': rec[i][k] for i in range(len(rec)) for k in ['gt', 'asotl', 'pa']})


def load_qualitative(path):
    z = np.load(path)
    rec = []
    for i, m in enumerate(z['meta']):
        r = json.loads(str(m))
        r.update({k: z[f'{i}_{k}'] for k in ['gt', 'asotl', 'pa']})
        rec.append(r)
    return rec


def draw_qual(rec, path_noext):
    plt.rcParams.update(PAPER_RC)
    dss = list(dict.fromkeys(r['ds'] for r in rec))
    n = len(dss)
    # sizes in inches so that text never overlaps the bars
    H_HEAD, H_ROW, L, R = 0.26, 0.80, 0.62, 0.05
    H = H_HEAD + n * H_ROW
    W = 7.16
    fig = plt.figure(figsize=(W, H))
    outer = fig.add_gridspec(n, 2, hspace=0.55, wspace=0.06, left=L / W, right=1 - R / W,
                             top=1 - (H_HEAD + 0.17) / H, bottom=0.02 / H)
    pal = list(plt.get_cmap('tab20').colors)
    for i, ds in enumerate(dss):
        for j, kind in enumerate(['gain', 'drop']):
            r = next(x for x in rec if x['ds'] == ds and x['kind'] == kind)
            cmap = ListedColormap([pal[c % 20] for c in range(r['C'])] + [(0.62, 0.62, 0.62), (1, 1, 1)])
            inner = outer[i, j].subgridspec(3, 1, hspace=0.15)
            for k, (lab, seq) in enumerate([('GT', r['gt']), ('ASOT-$\\lambda$', r['asotl']), ('PA-decode', r['pa'])]):
                ax = fig.add_subplot(inner[k])
                ax.imshow(np.asarray(seq)[None], aspect='auto', cmap=cmap, vmin=-0.5, vmax=r['C'] + 1.5, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.4)
                if j == 0:
                    ax.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=7, labelpad=3)
                if k == 0:
                    ax.set_title(f"{SHORT.get(ds, ds)}: {r['video']}   MoF {r['mof_l']:.1f} $\\rightarrow$ {r['mof_p']:.1f}",
                                 fontsize=7, loc='left', pad=2)
    cw = (W - L - R) / W
    for j, t in enumerate(['Largest MoF gain of PA-decode', 'Largest MoF drop of PA-decode']):
        fig.text(L / W + cw * (0.25 + 0.5 * j), 1 - 0.04 / H, t, ha='center', va='top', fontsize=8, weight='bold')
    _save(fig, path_noext)


# ----------------------------------------------------------------------------- cost and transport plans
@torch.no_grad()
def plan_matrices(exp, ds, a, s, v):
    run, kw = exp.dec[(ds, a, s)], exp.eval_kw(ds)
    K = run['conf'].shape[2]
    m, dobj = load_model(exp, ds, a, s), make_ds(exp, ds, a, None, False)
    cost, _ = video_cost(m, dobj[v][0], exp.device)
    out = {}
    for meth in [cfg.M_ASOTL, cfg.M_PA]:
        idx = int(chosen_of(exp, ds, s, meth)[a][v])
        c = CFGS[idx]
        tgt = ((1 - c['w']) / K + c['w'] * np.asarray(run[f"pihat_{c['src']}"], dtype=np.float64) if c['kind'] == 'pa'
               else np.full(K, 1. / K))
        dy = torch.tensor(tgt[None], dtype=torch.float32, device=cost.device)
        T = segment_asot_pa(cost, dy, c['lam'], **kw)[0].detach().cpu().numpy()
        out[meth] = (T, cfg_str(idx))
    return 1 - cost[0].detach().cpu().numpy(), out


def draw_matrices(exp, rec, path_noext):
    q0 = rec[0]
    ds0, a0, s0 = q0['ds'], q0['act'], q0['seed']
    v0 = list(exp.uinfo[(ds0, a0)]['names']).index(q0['video'])
    aff, mats = plan_matrices(exp, ds0, a0, s0, v0)
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.9))
    panels = [('(a) affinity $1-\\mathbf{C}$', aff), ('(b) plan of ASOT-$\\lambda$', mats[cfg.M_ASOTL][0]),
              ('(c) plan of PA-decode', mats[cfg.M_PA][0])]
    for i, (title, M) in enumerate(panels):
        ax[i].imshow(M.T, aspect='auto', interpolation='nearest')
        ax[i].set_title(title)
        ax[i].set_xlabel('frame')
        ax[i].set_ylabel('action')
    fig.tight_layout()
    _save(fig, path_noext)
    print(f'  matrices: {exp.label(ds0)}, {q0["video"]}, seed {s0}; ASOT-lambda [{mats[cfg.M_ASOTL][1]}], '
          f'PA-decode [{mats[cfg.M_PA][1]}]')


# ----------------------------------------------------------------------------- sensitivity to lambda
def sensitivity_table(exp):
    """MoF of ONE configuration shared by all videos (no 2-fold selection), for every lambda."""
    rows = []
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            for i, c in enumerate(CFGS):
                if c['kind'] not in ('base', 'pa'):
                    continue
                mof = np.mean([100 * hungarian_mof(exp.dec[(ds, a, s)]['conf'][i].sum(0)) for a in exp.units_of(ds)])
                rows.append(dict(ds=ds, seed=s, lam=float(c['lam']), w=float(c['w']), src=str(c['src']), mof=float(mof)))
    return pd.DataFrame(rows)


def draw_sensitivity(exp, sens, path_noext):
    plt.rcParams.update(PAPER_RC)
    n = len(exp.datasets)
    fig, ax = plt.subplots(1, n, figsize=(1.45 * n + 0.3, 1.9), squeeze=False)
    curves = [((0.0, '-'), 'k-o', 'ASOT (uniform target)'), ((0.5, 'plan'), 'C1--s', 'PA, w=0.5, plan'),
              ((1.0, 'plan'), 'C1-^', 'PA, w=1, plan'), ((1.0, 'codes'), 'C0-v', 'PA, w=1, codes')]
    for j, ds in enumerate(exp.datasets):
        q = sens[sens.ds == ds]
        for (w, src), sty, lab in curves:
            z = q[(np.isclose(q.w, w)) & (q.src == src)].groupby('lam')['mof'].mean()
            if len(z):
                ax[0, j].plot(z.index.values, z.values, sty, ms=2.5, lw=0.9, label=lab)
        ax[0, j].set_xscale('log')
        ax[0, j].set_title(SHORT.get(ds, ds))
        ax[0, j].set_xlabel('$\\lambda$')
    ax[0, 0].set_ylabel('MoF (%)')
    ax[0, 0].legend(fontsize=5)
    fig.tight_layout()
    _save(fig, path_noext)


# ----------------------------------------------------------------------------- dose-response
def draw_dose(exp, inter, path_noext):
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, ds in enumerate(exp.datasets):
        I_ = inter[inter.ds == ds]
        ax.plot(cfg.BETAS, [np.mean(seedlevel(I_, f'g_dose{b}')) for b in cfg.BETAS], 'o-', c=f'C{k}', ms=3, lw=1,
                label=exp.label(ds))
        ax.scatter([-0.1], [np.mean(seedlevel(I_, 'g_perm'))], marker='x', c=f'C{k}', s=18)
    ax.axhline(0, c='gray', lw=.6)
    ax.set_xticks([-0.1] + cfg.BETAS)
    ax.set_xticklabels(['perm.'] + [f'{b:g}' for b in cfg.BETAS])
    ax.set_xlabel('$\\beta$  (0: $\\hat\\pi$ of PA-decode; 1: true proportions)')
    ax.set_ylabel('$\\Delta$MoF vs. ASOT-$\\lambda$ (points)')
    ax.legend(fontsize=5.5, loc='lower right')
    fig.tight_layout()
    _save(fig, path_noext)


def draw_correlations(exp, feat, path_noext):
    U = feat.groupby(['ds', 'act']).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for i, c in enumerate(['adv_pool_plan', 'imb', 'het']):
        for k, d in enumerate(exp.datasets):
            q = U[U.ds == d]
            ax[i].scatter(q[c], q['gain'], c=f'C{k}', label=exp.label(d), s=25)
        ax[i].axhline(0, c='gray', lw=.8)
        ax[i].set_xlabel(c)
        ax[i].set_ylabel('ΔMoF PA - ASOT-lambda')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, path_noext, dpi=200)
