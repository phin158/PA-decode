"""Figures in the layout of the ASOT paper (Xu & Gould, CVPR 2024), with PA-decode added.

  Fig. 2     one affinity matrix decoded in four ways (2 x 2, as Fig. 2 of ASOT):
             (a) affinity 1 - C, (b) ASOT, (c) PA-decode, (d) PA-decode with a shuffled pihat (wrong action order)
  Fig. 4     GT / ASOT / PA-decode on the videos of Fig. 4 of ASOT (Breakfast "salat", YTI "repot")
  Fig. 5     sensitivity of the decoding to the parameters of PA-decode (2 x 2, layout of Fig. 5 of ASOT), MoF:
             (a) lambda, (b) w, (c) quality of pihat beta, (d) source of pihat;
             Desktop Assembly, 50 Salads (Mid), 50 Salads (Eval)
  Fig. 6     GT / ASOT / PA-decode on the videos of Fig. 6 of ASOT (Desktop Assembly, 50 Salads Eval)
             + action proportions (true, uniform 1/K, pihat)
  Fig. 7, 8  Fig. 5 with F1 and mIoU (appendix, as Fig. 7 and 8 of ASOT)

Example videos (Fig. 2, 4, 6): by default (--pick gain) the videos where PA-decode improves most over ASOT
(per-video MoF, Full protocol; outputs collapsed to a single action are skipped), all candidates are written to
video_gain_ranking.csv; --pick paper uses the videos of the ASOT paper instead.

ASOT      = the authors' decoding (lambda = 0.01, uniform target) of the same trained model.
PA-decode = the configuration selected by the 2-fold procedure of the main experiment.
One run per dataset (RUN below = the run of Table III). The cache of the main experiment (trained models and
decoded segmentations) is reused; missing runs are trained and decoded first.

    python scripts/make_asot_style_figures.py --data-dir data --cache-dir cache --out-dir results
"""
import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _path  # noqa: F401,E402
from pa_decode import config as cfg  # noqa: E402
from pa_decode.data import make_ds  # noqa: E402
from pa_decode.decode import CFGS, IDX_PAPER, cfg_str, norm_floor  # noqa: E402
from pa_decode.experiment import Experiment  # noqa: E402
from pa_decode.figures import PAPER_RC, per_video_mof, to_gt_colors  # noqa: E402
from pa_decode.mechanism import _pstar, chosen_of  # noqa: E402
from pa_decode.metrics import conf_mat, hungarian, seg_table, unit_eval  # noqa: E402
from pa_decode.pipeline import run_until  # noqa: E402
from pa_decode.selection import ones_weights, select_configs  # noqa: E402
from pa_decode.solver import segment_asot_pa, video_cost  # noqa: E402
from pa_decode.train import load_model  # noqa: E402

RUN = {'Breakfast': 5, 'YTI': 3, 'FS': 0, 'FSeval': 0, 'desktop_assembly': 6}
# videos shown in the ASOT paper
VID_FIG2 = ('FSeval', 'rgb-18-2')
VID_FIG4 = [('Breakfast', 'P36_webcam01_P36_salat'), ('Breakfast', 'P17_stereo01_P17_salat'),
            ('YTI', 'repot_0027'), ('YTI', 'repot_0025')]
VID_FIG6 = [('desktop_assembly', '2020-04-19_14-03-25'), ('FSeval', 'rgb-18-2')]
SENS_DS = ['desktop_assembly', 'FS', 'FSeval']
DS_NAME = {'desktop_assembly': 'DA', 'FS': 'FS (Mid)', 'FSeval': 'FS (Eval)', 'Breakfast': 'Breakfast', 'YTI': 'YTI'}
DS_COLOR = {'desktop_assembly': '#4285F4', 'FS': '#EA4335', 'FSeval': '#FBBC04'}     # colours of Fig. 5 of ASOT
METRIC_IDX = {'MoF': 0, 'F1': 1, 'mIoU': 2}
LOG = {'videos': {}, 'configs': {}, 'warnings': []}


def warn(msg):
    print('WARNING:', msg)
    LOG['warnings'].append(msg)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(out, name + '.pdf/.png'))


# ----------------------------------------------------------------------------- helpers
def find_video(exp, ds, pattern, used=()):
    """(activity, index, name) of the video whose name contains `pattern`.
    If it is missing: first unused video of the activity named in `pattern` (e.g. 'salat'), else of the first activity."""
    for a in exp.units_of(ds):
        names = [os.path.basename(str(n)) for n in exp.uinfo[(ds, a)]['names']]
        hits = [i for i, n in enumerate(names) if pattern in n]
        if hits:
            LOG['videos'][f'{ds}:{pattern}'] = names[hits[0]]
            return a, hits[0], names[hits[0]]
    acts = exp.units_of(ds)
    a = next((x for x in acts if x in pattern), acts[0])
    names = [os.path.basename(str(n)) for n in exp.uinfo[(ds, a)]['names']]
    i = next((k for k, n in enumerate(names) if n not in used), 0)
    warn(f'{ds}: video "{pattern}" not found, using "{names[i]}" instead (first names: {names[:5]})')
    LOG['videos'][f'{ds}:{pattern}'] = names[i]
    return a, i, names[i]


def class_names(exp, ds, a):
    """local class index -> action name (falls back to the label id)."""
    info = exp.uinfo[(ds, a)]
    inv = {}
    try:
        dobj = make_ds(exp, ds, a, None, False)
        inv = {v: k for k, v in getattr(dobj, 'action_mapping', {}).items()}
    except Exception as e:  # noqa: BLE001
        warn(f'{ds}/{a}: could not read action names ({e})')
    return [str(inv.get(c, c)) for c in info['cls']]


def selected_config(exp, ds, s):
    """Configuration of PA-decode used as the fixed point of Fig. 5 (the one picked on fold A)."""
    _, picks = select_configs(exp, ds, s, cfg.M_PA, 'mof', ones_weights(exp, ds))
    c = dict(CFGS[picks[0]])
    LOG['configs'][ds] = dict(seed=s, fold_picks=[cfg_str(i) for i in picks], used=cfg_str(picks[0]))
    if picks[0] != picks[1]:
        print(f'  {ds}: the two folds picked {cfg_str(picks[0])} / {cfg_str(picks[1])}; fold A is used for Fig. 5')
    if c['kind'] != 'pa':
        warn(f'{ds}: PA-decode picked w = 0 ({cfg_str(picks[0])}); Fig. 5 uses w = 1 and source "plan" with that lambda')
        c = dict(kind='pa', lam=c['lam'], w=1.0, src='plan')
    return c


def cfg_index(kind, lam, w=0., src='-'):
    for i, c in enumerate(CFGS):
        if c['kind'] == kind and np.isclose(c['lam'], lam) and np.isclose(c['w'], w) and c['src'] == src:
            return i
    raise KeyError((kind, lam, w, src))


def scores_cfg(exp, ds, s, idx):
    """[MoF, F1, mIoU] (%) of ONE configuration used for every video, Full protocol, mean over activities."""
    vals = []
    for a in exp.units_of(ds):
        run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
        nv = len(info['names'])
        v, _ = unit_eval(run, exp.stab[(ds, a, s)], np.full(nv, idx), np.ones(nv), info)
        vals.append(v)
    return np.mean(vals, 0)


@torch.no_grad()
def solve_targets(exp, ds, a, s, lam, targets):
    """Re-solve every video of a unit with each target in `targets` (list of K-vectors). Returns a pseudo decoding
    run (segs, conf, offs) with one 'configuration' per target, so that the metrics of the paper can be reused."""
    run, info, kw = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)], exp.eval_kw(ds)
    K, C = run['conf'].shape[2], run['conf'].shape[3]
    offs, nv, nt = run['offs'], len(info['names']), len(targets)
    m, dobj = load_model(exp, ds, a, s), make_ds(exp, ds, a, None, False)
    segs = np.zeros((nt, offs[-1]), dtype=np.int8)
    conf = np.zeros((nt, nv, K, C), dtype=np.int32)
    dys = torch.tensor(np.stack(targets), dtype=torch.float32, device=exp.device)
    for v in range(nv):
        cost, _ = video_cost(m, dobj[v][0], exp.device)
        T = segment_asot_pa(cost.expand(nt, -1, -1), dys, [lam] * nt, **kw)
        pred = T.argmax(2).cpu().numpy()
        for j in range(nt):
            segs[j, offs[v]:offs[v + 1]] = pred[j]
            conf[j, v] = conf_mat(pred[j], info['gt'][v], K, C)
    del m
    if exp.device.type == 'cuda':
        torch.cuda.empty_cache()
    return dict(segs=segs, conf=conf, offs=offs)


def scores_pseudo(exp, ds, s, pseudo_by_unit, j):
    vals = []
    for a in exp.units_of(ds):
        info, pr = exp.uinfo[(ds, a)], pseudo_by_unit[a]
        nv = len(info['names'])
        v, _ = unit_eval(pr, seg_table(info, pr), np.full(nv, j), np.ones(nv), info)
        vals.append(v)
    return np.mean(vals, 0)


def method_rows(exp, ds, a, s, v):
    """Coloured sequences of one video: GT, ASOT, PA-decode (+ per-video MoF)."""
    run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
    nv, C, offs = len(info['names']), info['C'], run['offs']
    g = np.asarray(info['gt'][v]).astype(int)
    gt = g.copy()
    gt[gt < 0] = C + 1
    rows = [('GT', gt, None)]
    for lab, ch in [('ASOT', np.full(nv, IDX_PAPER)), ('PA-decode', chosen_of(exp, ds, s, cfg.M_PA)[a])]:
        acc, (pr, gl) = per_video_mof(exp, ds, a, s, ch)
        rows.append((lab, to_gt_colors(run['segs'][ch[v], offs[v]:offs[v + 1]], g, pr, gl, C), 100 * acc[v]))
    return rows


def draw_bars(fig, spec, rows, C, title, show_labels=True):
    pal = list(plt.get_cmap('tab20').colors)
    cmap = ListedColormap([pal[c % 20] for c in range(C)] + [(0.62, 0.62, 0.62), (1, 1, 1)])
    inner = spec.subgridspec(len(rows), 1, hspace=0.12)
    for k, (lab, seq, mof) in enumerate(rows):
        ax = fig.add_subplot(inner[k])
        ax.imshow(np.asarray(seq)[None], aspect='auto', cmap=cmap, vmin=-0.5, vmax=C + 1.5, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        if show_labels:
            ax.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=6, labelpad=3)
        if mof is not None:
            ax.text(1.01, 0.5, f'{mof:.1f}', transform=ax.transAxes, fontsize=5, va='center')
        if k == 0:
            ax.set_title(title, fontsize=6.5, pad=2)
    return pal


PICK = {'mode': 'gain'}   # 'gain': videos where PA-decode improves most over ASOT; 'paper': the videos of the ASOT paper
GAIN_TABLES = {}


def gain_table(exp, ds):
    """Per-video MoF (Full protocol) of ASOT and PA-decode for the run of Table III, sorted by gain."""
    if ds in GAIN_TABLES:
        return GAIN_TABLES[ds]
    s, rows = exp.seeds[ds][0], []
    for a in exp.units_of(ds):
        run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
        nv, offs = len(info['names']), run['offs']
        acc_as, _ = per_video_mof(exp, ds, a, s, np.full(nv, IDX_PAPER))
        ch = chosen_of(exp, ds, s, cfg.M_PA)[a]
        acc_pa, _ = per_video_mof(exp, ds, a, s, ch)
        for v in range(nv):
            g = np.asarray(info['gt'][v]).astype(int)
            n_gt = len(np.unique(g[g >= 0]))
            n_pa = len(np.unique(run['segs'][ch[v], offs[v]:offs[v + 1]]))
            rows.append(dict(ds=ds, act=a, v=v, name=os.path.basename(str(info['names'][v])), ASOT=100 * acc_as[v],
                             PA=100 * acc_pa[v], gain=100 * (acc_pa[v] - acc_as[v]),
                             collapsed=bool(n_pa < min(3, n_gt))))   # PA output (almost) one action: not a useful example
    df = pd.DataFrame(rows).sort_values('gain', ascending=False).reset_index(drop=True)
    GAIN_TABLES[ds] = df
    return df


def videos_for(exp, ds, k, paper_patterns):
    """k videos of one activity: the videos of the ASOT paper ('paper') or the k largest gains of PA-decode over ASOT
    within one activity, excluding outputs collapsed to one action and taking at most one camera view per recording ('gain')."""
    if PICK['mode'] == 'paper':
        out, used = [], set()
        for pat in paper_patterns:
            a, v, name = find_video(exp, ds, pat, used)
            used.add(name)
            out.append((a, v, name))
        return out
    df = gain_table(exp, ds)
    ok = df[(~df.collapsed) & (df.gain > 0)].copy()
    # one video per recording: Breakfast films each person with several cameras (P16_cam01, P16_webcam01, ...)
    ok['rec'] = ok.name.str.split('_').str[0] if ds == 'Breakfast' else ok.name
    best = None
    for a, q in ok.groupby('act'):
        q = q.drop_duplicates('rec').head(k)
        if len(q) == k and (best is None or q.gain.min() > best.gain.min()):
            best = q
    if best is None:
        warn(f'{ds}: fewer than {k} non-collapsed videos with a gain in one activity; using the largest gains overall')
        best = df[~df.collapsed].head(k)
    return [(r.act, int(r.v), r.name) for r in best.itertuples()]


# ----------------------------------------------------------------------------- Fig. 2
@torch.no_grad()
def fig2(exp, out):
    ds, pat = VID_FIG2
    s = exp.seeds[ds][0]
    a, v, name = videos_for(exp, ds, 1, [pat])[0]
    run, info, kw = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)], exp.eval_kw(ds)
    K = run['conf'].shape[2]
    m, dobj = load_model(exp, ds, a, s), make_ds(exp, ds, a, None, False)
    cost, _ = video_cost(m, dobj[v][0], exp.device)
    c = CFGS[int(chosen_of(exp, ds, s, cfg.M_PA)[a][v])]
    if c['kind'] == 'pa':
        ph = np.asarray(run[f"pihat_{c['src']}"], dtype=np.float64)
        lam, w = c['lam'], c['w']
    else:
        warn(f'Fig. 2: PA-decode uses w = 0 on this video ({cfg_str(CFGS.index(c))}); panels (c), (d) use w = 1, source plan')
        ph, lam, w = np.asarray(run['pihat_plan'], dtype=np.float64), c['lam'], 1.0
        c = dict(c, src='plan')
    perm = np.random.default_rng(zlib.crc32(f'{ds}|{a}|{s}'.encode())).permutation(K)     # = first permutation of Table VII
    tg = [np.full(K, 1. / K), (1 - w) / K + w * ph, (1 - w) / K + w * ph[perm]]
    lams = [cfg.LAM_PAPER, lam, lam]
    mats = []
    for t, l in zip(tg, lams):
        dy = torch.tensor(t[None], dtype=torch.float32, device=cost.device)
        mats.append(segment_asot_pa(cost, dy, l, **kw)[0].detach().cpu().numpy())
    g = np.asarray(info['gt'][v]).astype(int)
    mofs = []
    for T in mats:
        M = conf_mat(T.argmax(1), g, K, info['C'])
        mofs.append(100 * M[hungarian(M)].sum() / max(M.sum(), 1))
    LOG['fig2'] = dict(dataset=ds, video=name, seed=s, pa_config=f"lam={lam}, w={w}, src={c['src']}", video_mof=dict(
        zip(['ASOT', 'PA-decode', 'PA-decode shuffled'], [round(x, 2) for x in mofs])))
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(2, 2, figsize=(3.4, 2.6))
    panels = [('(a) Affinity matrix $1-\\mathbf{C}^k$.', 1 - cost[0].detach().cpu().numpy()),
              ('(b) ASOT solution.', mats[0]), ('(c) PA-decode solution.', mats[1]),
              ('(d) PA-decode w/ shuffled $\\hat{\\boldsymbol{\\pi}}$.', mats[2])]
    for axi, (title, M) in zip(ax.ravel(), panels):
        axi.imshow(M.T, aspect='auto', interpolation='nearest')
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlabel(title, fontsize=6.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, out, 'fig2_matrices')


# ----------------------------------------------------------------------------- Fig. 4 and Fig. 6
def fig4(exp, out):
    plt.rcParams.update(PAPER_RC)
    groups = [('Breakfast', '(a) Example videos from the "{act}" activity of Breakfast.'),
              ('YTI', '(b) Example videos from YTI ("{act}" category).')]
    W, H = 3.4, 3.9
    fig = plt.figure(figsize=(W, H))
    outer = fig.add_gridspec(2, 1, hspace=0.35, left=0.2, right=0.92, top=0.95, bottom=0.04)
    LOG['fig4'] = []
    for gi, (ds, cap) in enumerate(groups):
        pats = [p for d, p in VID_FIG4 if d == ds]
        sel = videos_for(exp, ds, len(pats), pats)
        s = exp.seeds[ds][0]
        sub = outer[gi].subgridspec(len(sel) + 1, 1, hspace=0.55, height_ratios=[1] * len(sel) + [0.35])
        used = set()
        for k, (a, v, name) in enumerate(sel):
            rows = method_rows(exp, ds, a, s, v)
            C = exp.uinfo[(ds, a)]['C']
            pal = draw_bars(fig, sub[k], rows, C, name)
            used |= {int(x) for r in rows for x in np.unique(r[1]) if x < C}
            LOG['fig4'].append(dict(dataset=ds, activity=a, video=name, seed=s, mof={r[0]: r[2] for r in rows[1:]}))
        a0 = sel[0][0]
        names = class_names(exp, ds, a0)
        ax = fig.add_subplot(sub[len(sel)])
        ax.axis('off')
        hs = [Patch(color=pal[c % 20], label=names[c]) for c in sorted(used)]
        ax.legend(handles=hs, ncol=4, fontsize=4.5, loc='center', frameon=False, handlelength=1, columnspacing=0.8)
        ax.set_title(cap.format(act=a0), fontsize=6, y=-1.2)
    save(fig, out, 'fig4_examples')


def fig6(exp, out):
    plt.rcParams.update(PAPER_RC)
    fig = plt.figure(figsize=(7.16, 2.3))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], hspace=0.9, wspace=0.25, left=0.08, right=0.98, top=0.9, bottom=0.12)
    LOG['fig6'] = []
    for i, (ds, pat) in enumerate(VID_FIG6):
        s = exp.seeds[ds][0]
        a, v, name = videos_for(exp, ds, 1, [pat])[0]
        run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
        C, K = info['C'], run['conf'].shape[2]
        rows = method_rows(exp, ds, a, s, v)
        draw_bars(fig, outer[i, 0], rows, C, f"({'ab'[i]}) {'Desktop Assembly' if ds == 'desktop_assembly' else '50 Salads (Eval)'}: {name}")
        # proportions per GT class: true (dataset), pihat (no labels, mapped to classes by the ASOT matching), uniform
        true = np.mean([np.bincount(g[g >= 0], minlength=C) / max((g >= 0).sum(), 1) for g in info['gt']], 0)
        src = selected_config(exp, ds, s)['src']
        ph = np.asarray(run[f'pihat_{src}'], dtype=np.float64)
        pr, gl = hungarian(run['conf'][IDX_PAPER].sum(0))
        ph_c = np.zeros(C)
        ph_c[gl] = ph[pr]
        order = np.argsort(-true)
        ax = fig.add_subplot(outer[i, 1])
        x = np.arange(C)
        ax.bar(x - 0.2, true[order], 0.4, color='0.35', label='true (uses labels)')
        ax.bar(x + 0.2, ph_c[order], 0.4, color='C2', label='$\\hat{\\pi}$ (no labels)')
        ax.axhline(1. / K, color='k', ls='--', lw=0.7, label='uniform $1/K$ (ASOT)')
        ax.set_xticks([])
        ax.set_xlabel('action (sorted by true proportion)', fontsize=5.5)
        ax.set_ylabel('proportion', fontsize=5.5)
        ax.tick_params(labelsize=5)
        if i == 0:
            ax.legend(fontsize=4.5, frameon=False)
        LOG['fig6'].append(dict(dataset=ds, video=name, seed=s, pihat_source=src, mof={r[0]: r[2] for r in rows[1:]},
                                tv_uniform=0.5 * float(np.abs(true - 1. / C).sum()), tv_pihat=0.5 * float(np.abs(true - ph_c).sum())))
    save(fig, out, 'fig6_examples_proportions')


# ----------------------------------------------------------------------------- Fig. 5, 7, 8
def sensitivity(exp):
    rows = []
    for ds in SENS_DS:
        s = exp.seeds[ds][0]
        c = selected_config(exp, ds, s)
        lam, w, src = c['lam'], c['w'], c['src']

        def add(panel, x, method, idx=None, val=None):
            v = scores_cfg(exp, ds, s, idx) if val is None else val
            rows.append(dict(ds=ds, seed=s, panel=panel, x=x, method=method, MoF=v[0], F1=v[1], mIoU=v[2],
                             fixed=f'lam={lam}, w={w}, src={src}'))
        for l in cfg.LAMBDAS:                                         # (a) lambda
            add('lambda', l, 'PA-decode', cfg_index('pa', l, w, src))
            add('lambda', l, 'ASOT', cfg_index('base', l))
        add('w', 0.0, 'PA-decode', cfg_index('base', lam))             # (b) w (w = 0: uniform target)
        for ww in cfg.WS:
            add('w', ww, 'PA-decode', cfg_index('pa', lam, ww, src))
        for sr in cfg.SRCS:                                           # (d) source of pihat
            add('source', sr, 'PA-decode', cfg_index('pa', lam, w, sr))
        pseudo = {}                                                   # (c) quality of pihat (re-solved)
        for a in exp.units_of(ds):
            run, info = exp.dec[(ds, a, s)], exp.uinfo[(ds, a)]
            K = run['conf'].shape[2]
            ph, ps = np.asarray(run[f'pihat_{src}'], dtype=np.float64), _pstar(run, info)
            tg = [(1 - w) / K + w * ((1 - b) * ph + b * ps) for b in cfg.BETAS]
            pseudo[a] = solve_targets(exp, ds, a, s, lam, tg)
            i_pa = cfg_index('pa', lam, w, src)
            agree = (pseudo[a]['segs'][0] == run['segs'][i_pa]).mean()
            if agree < 0.999:
                warn(f'{ds}/{a}: re-solving beta = 0 matches the cached PA-decode only {agree:.4f}')
        for j, b in enumerate(cfg.BETAS):
            add('beta', b, 'PA-decode', val=scores_pseudo(exp, ds, s, pseudo, j))
        print(f'  sensitivity {ds}: done (fixed lam={lam}, w={w}, src={src})')
    return pd.DataFrame(rows)


def draw_sensitivity(df, metric, out, name):
    plt.rcParams.update(PAPER_RC)
    j = metric
    fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.1))
    caps = {'lambda': '(a) Balanc. assign. weight $\\lambda$ (test)', 'w': '(b) Target weight $w$',
            'beta': '(c) Quality of $\\hat{\\pi}$: $\\beta$', 'source': '(d) Source of $\\hat{\\pi}$'}
    xl = {'lambda': '$\\lambda$', 'w': '$w$', 'beta': '$\\beta$', 'source': 'source'}
    top = max(0.8, float(np.ceil((df[j].max() / 100 + 0.05) * 10) / 10))
    for axi, panel in zip(ax.ravel(), ['lambda', 'w', 'beta', 'source']):
        for ds in SENS_DS:
            q = df[(df.ds == ds) & (df.panel == panel)]
            for meth, ls in [('PA-decode', '-'), ('ASOT', '--')]:
                z = q[q.method == meth]
                if z.empty:
                    continue
                xs = np.arange(len(z)) if panel == 'source' else z.x.astype(float).values
                axi.plot(xs, z[j].values / 100, ls, marker='o', ms=2.2, lw=0.9, color=DS_COLOR[ds],
                         label=DS_NAME[ds] if meth == 'PA-decode' else None)
        if panel == 'lambda':
            axi.set_xscale('log')
            axi.axvline(cfg.LAM_PAPER, color='0.6', lw=0.5, ls=':')
        if panel == 'source':
            axi.set_xticks([0, 1])
            axi.set_xticklabels(cfg.SRCS)
            axi.set_xlim(-0.4, 1.4)
        axi.set_ylim(0, top)
        axi.set_ylabel(metric, fontsize=6)
        axi.set_xlabel(xl[panel], fontsize=6, labelpad=1)
        axi.text(0.5, -0.42, caps[panel], transform=axi.transAxes, ha='center', va='top', fontsize=6)
        axi.grid(axis='y', lw=0.3, color='0.85')
        axi.tick_params(labelsize=5)
        hs, ls_ = axi.get_legend_handles_labels()
        if panel == 'lambda':
            hs.append(plt.Line2D([], [], color='0.4', ls='--', lw=0.9))
            ls_.append('ASOT')
        axi.legend(hs, ls_, fontsize=4.5, ncol=4 if panel == 'lambda' else 3, loc='upper center',
                   bbox_to_anchor=(0.5, 1.2), frameon=False, handlelength=1.2, columnspacing=0.6)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, out, name)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--cache-dir', default='cache')
    p.add_argument('--out-dir', default='results')
    p.add_argument('--runs', default=None, help='override, e.g. "Breakfast=5,YTI=3,FS=0,FSeval=0,desktop_assembly=6"')
    p.add_argument('--quick', action='store_true', help='smoke test only (models trained for 2 epochs)')
    p.add_argument('--device', default=None)
    p.add_argument('--pick', choices=['gain', 'paper'], default='gain',
                   help="example videos: largest gain of PA-decode over ASOT (default) or the videos of the ASOT paper")
    a = p.parse_args()
    run = dict(RUN)
    if a.runs:
        run.update({k: int(v) for k, v in (x.split('=') for x in a.runs.split(','))})
    exp = Experiment(a.data_dir, a.cache_dir, a.out_dir, datasets=list(run), quick=a.quick, device=a.device)
    exp.seeds = {ds: [run[ds]] for ds in cfg.ALL_DATASETS if ds in run}
    run_until(exp, 'point', checks=False)
    out = exp.out('figures_asot_style')
    os.makedirs(out, exist_ok=True)
    PICK['mode'] = a.pick
    LOG['pick'] = a.pick
    fig2(exp, out)
    fig4(exp, out)
    fig6(exp, out)
    if GAIN_TABLES:
        pd.concat(GAIN_TABLES.values()).to_csv(os.path.join(out, 'video_gain_ranking.csv'), index=False)
    df = sensitivity(exp)
    df.to_csv(os.path.join(out, 'sensitivity_pa_params.csv'), index=False)
    draw_sensitivity(df, 'MoF', out, 'fig5_sensitivity_mof')
    draw_sensitivity(df, 'F1', out, 'fig7_sensitivity_f1')
    draw_sensitivity(df, 'mIoU', out, 'fig8_sensitivity_miou')
    LOG['runs'] = run
    with open(os.path.join(out, 'figures_log.json'), 'w', encoding='utf-8') as f:
        json.dump(LOG, f, indent=1, default=float)
    print('\nlog:', json.dumps(LOG, indent=1, default=float)[:3000])


if __name__ == '__main__':
    main()
