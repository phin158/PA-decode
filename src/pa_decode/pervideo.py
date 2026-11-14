"""Per-Video protocol (the second block of Table 1 of ASOT).

The Hungarian matching is done inside every video, and the scores are averaged over the videos of an activity (then over
the activities, A1). This is the same computation as the authors' ``indep_eval_metrics`` in ``test_step``
(``test_mof_per``, ``test_f1_per``, ``test_miou_per``): the matching is found with MoF and reused for F1 and mIoU.
F1 is the exact expectation of the authors' Monte-Carlo F1, as in the Full protocol.

No new parameter selection is made: every method uses the configurations picked by the 2-fold selection of the Full
protocol (criterion: MoF, Full protocol). The decision rule for this protocol was fixed before running it (config.PV_*).
"""
import time

import numpy as np
import pandas as pd

from . import config as cfg
from .decode import IDX_PAPER, decode_path
from .metrics import scores
from .selection import select_configs, ones_weights
from .utils import t_ci, show, save_json, key_str


# ---------------------------------------------------------------------------- per-video scores of every configuration
def video_scores(run, st):
    """S [n_configs, n_videos, 3]: MoF, F1, mIoU (%) of every video under every decoding configuration.
    NaN for a video without scored frames (cannot happen with the public datasets; counted and reported)."""
    conf = run['conf']
    ncfg, nv = conf.shape[:2]
    S = np.full((ncfg, nv, 3), np.nan)
    vid = st['vid']
    for v in range(nv):
        ix = np.where(vid == v)[0]
        if len(ix) == 0:
            continue
        ones = np.ones(len(ix))
        gt_, len_ = st['gt'][ix], st['len'][ix]
        for i in range(ncfg):
            M = conf[i, v]
            if M.sum() > 0:
                S[i, v] = scores(M, st['cnt'][i, ix], gt_, len_, ones, 1)
    return S


def build_video_scores(exp, verbose=True):
    """Fills exp.pv[(ds, act, seed)] = S. Cached next to the decoding cache (same hash)."""
    if not hasattr(exp, 'pv'):
        exp.pv = {}
    t0 = time.time()
    for ds in exp.datasets:
        for seed in exp.seeds[ds]:
            for a in exp.units_of(ds):
                dp = decode_path(exp, ds, a, seed)
                path = dp.replace('/decode/', '/pervideo/').replace('.npz', '_pv.npy')
                exp.path('pervideo', 'x')            # create the folder
                try:
                    S = np.load(path)
                except FileNotFoundError:
                    S = video_scores(exp.dec[(ds, a, seed)], exp.stab[(ds, a, seed)])
                    np.save(path, S)
                exp.pv[(ds, a, seed)] = S
        if verbose:
            n_nan = sum(int(np.isnan(exp.pv[(ds, a, s)][IDX_PAPER, :, 0]).sum())
                        for s in exp.seeds[ds] for a in exp.units_of(ds))
            print(f'{exp.label(ds)}: per-video scores ready ({(time.time() - t0) / 60:.1f} min); '
                  f'videos without scored frames: {n_nan}')


# ---------------------------------------------------------------------------- point estimates
def pv_evaluate(exp, ds, seed, method, W, rule='mof'):
    """Per-Video scores (A1 = mean over activities of the weighted mean over videos) with the Full-protocol selection."""
    chosen, _ = select_configs(exp, ds, seed, method, rule, W)
    per = {}
    for a in exp.units_of(ds):
        S = exp.pv[(ds, a, seed)]
        s = S[chosen[a], np.arange(len(chosen[a]))]
        w = W[a] * ~np.isnan(s[:, 0])
        per[a] = np.nansum(w[:, None] * s, 0) / w.sum()
    return np.mean(list(per.values()), 0), per


def pv_point_estimates(exp):
    exp.pv_point = {}
    for ds in exp.datasets:
        W = ones_weights(exp, ds)
        for s in exp.seeds[ds]:
            for m in cfg.PV_METHODS:
                a1, per = pv_evaluate(exp, ds, s, m, W)
                exp.pv_point[(ds, s, m)] = dict(A1=a1, per=per)


def pv_seed_vals(exp, ds, m):
    return np.array([exp.pv_point[(ds, s, m)]['A1'] for s in exp.seeds[ds]])


# ---------------------------------------------------------------------------- check against the authors' trainer.test
def check_against_trainer(exp, verbose=True):
    """For ASOT (lambda = 0.01, uniform target), our Per-Video MoF / mIoU must equal test_mof_per / test_miou_per logged by
    the authors' trainer.test. F1 is compared with their Monte-Carlo F1 (50 draws per video, so a small difference is
    expected)."""
    rows = []
    for ds in exp.datasets:
        for s in exp.seeds[ds]:
            for a in exp.units_of(ds):
                t = exp.trained[(ds, a, s)]['test']
                if 'test_mof_per' not in t:
                    continue
                mine = np.nanmean(exp.pv[(ds, a, s)][IDX_PAPER], 0)
                rows.append(dict(dataset=ds, activity=a, seed=s,
                                 d_mof=mine[0] - 100 * t['test_mof_per'], d_miou=mine[2] - 100 * t['test_miou_per'],
                                 f1_exact_minus_mc=mine[1] - 100 * t['test_f1_per']))
    df = pd.DataFrame(rows)
    df.to_csv(exp.out('pervideo_check_trainer.csv'), index=False)
    if len(df):
        mx_m, mx_i = df.d_mof.abs().max(), df.d_miou.abs().max()
        ok = mx_m < cfg.PV_CHECK_TOL and mx_i < cfg.PV_CHECK_TOL
        if verbose:
            print(f'{"OK" if ok else "!! MISMATCH"} [Per-Video]: max |our MoF - test_mof_per| = {mx_m:.5f}, '
                  f'max |our mIoU - test_miou_per| = {mx_i:.5f} points ({len(df)} models); '
                  f'exact F1 - Monte-Carlo F1: mean {df.f1_exact_minus_mc.mean():+.3f}, '
                  f'max |.| {df.f1_exact_minus_mc.abs().max():.3f}')
        assert ok, 'Per-Video protocol does not match the authors\' trainer.test; see pervideo_check_trainer.csv'
    elif verbose:
        print('!! trainer.test per-video values not found in the cached models: check skipped')
    return df


# ---------------------------------------------------------------------------- bootstrap (same resampling as the Full protocol)
def pv_bootstrap(exp, ds, n_boot, seed=0, rule='mof'):
    """Resample videos within each fold (and activity), redo the 2-fold selection, average over seeds.
    Uses the same random generator and order of draws as selection.bootstrap, so the resampled sets are identical.
    Returns D [n_boot, n_pairs, 3 metrics]."""
    rng = np.random.default_rng(seed)
    meths = sorted({m for p in cfg.PV_PAIRS for m in p})
    D = np.zeros((n_boot, len(cfg.PV_PAIRS), 3))
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
        acc = {m: np.zeros(3) for m in meths}
        for s in exp.seeds[ds]:
            for m in meths:
                acc[m] += pv_evaluate(exp, ds, s, m, W, rule)[0] / len(exp.seeds[ds])
        D[b] = [acc[x] - acc[y] for x, y in cfg.PV_PAIRS]
    return D


# ---------------------------------------------------------------------------- tables and verdicts
def pv_tables(exp, n_boot=None, verbose=True):
    from scipy import stats
    n_boot = n_boot or exp.n_boot
    main_rows, delta_rows, seed_rows = [], [], []
    t0 = time.time()
    for ds in exp.datasets:
        V = {m: pv_seed_vals(exp, ds, m) for m in cfg.PV_METHODS}
        for m in cfg.PV_METHODS:
            main_rows.append({'dataset': exp.label(ds), 'method': m,
                              **{met: V[m][:, j].mean() for j, met in enumerate(cfg.METRICS)},
                              'MoF std (seed)': V[m][:, 0].std(ddof=1) if len(V[m]) > 1 else np.nan})
        D = pv_bootstrap(exp, ds, n_boot[ds])
        for pi, (x, y) in enumerate(cfg.PV_PAIRS):
            d = V[x] - V[y]
            r = {'dataset': exp.label(ds), 'comparison': f'{x} - {y}'}
            for j, met in enumerate(cfg.METRICS):
                lo, hi = np.percentile(D[:, pi, j], [2.5, 97.5])
                r[f'Δ{met}'] = d[:, j].mean()
                r[f'Δ{met} CI95 bootstrap'] = f'[{lo:.2f}, {hi:.2f}]'
                r[f'_lo_{met}'] = lo
            dm = d[:, 0]
            nz = dm[np.abs(dm) > 1e-9]
            lo_t, hi_t = t_ci(dm)
            r['ΔMoF CI95 t (seed)'] = f'[{lo_t:.2f}, {hi_t:.2f}]'
            r['seeds >0/<0/=0'] = f'{(dm > 1e-9).sum()}/{(dm < -1e-9).sum()}/{(np.abs(dm) <= 1e-9).sum()}'
            r['_npos'] = int((dm > 1e-9).sum())
            r['p_t'] = stats.ttest_1samp(dm, 0).pvalue if len(dm) > 1 else np.nan
            r['p_Wilcoxon'] = stats.wilcoxon(nz).pvalue if len(nz) > 0 else np.nan
            delta_rows.append(r)
        for i, s in enumerate(exp.seeds[ds]):
            r = {'dataset': exp.label(ds), 'seed': s}
            for m in cfg.PV_METHODS:
                for j, met in enumerate(cfg.METRICS):
                    r[f'{m} {met}'] = V[m][i, j]
            for j, met in enumerate(cfg.METRICS):
                r[f'Δ{met} {cfg.M_PA} - {cfg.M_ASOTL}'] = V[cfg.M_PA][i, j] - V[cfg.M_ASOTL][i, j]
            seed_rows.append(r)
        if verbose:
            print(f'{exp.label(ds)}: Per-Video bootstrap done ({(time.time() - t0) / 60:.1f} min)')
    main = pd.DataFrame(main_rows)
    delta = pd.DataFrame(delta_rows)
    # Holm correction of the t-test p-values over the settings, for the main pair only
    mp = delta[delta.comparison == f'{cfg.M_PA} - {cfg.M_ASOTL}']
    order = mp.sort_values('p_t').index.tolist()
    delta['significant (Holm, 0.05)'] = pd.Series([None] * len(delta), index=delta.index, dtype=object)
    for i, ix in enumerate(order):
        delta.loc[ix, 'significant (Holm, 0.05)'] = bool(delta.loc[ix, 'p_t'] <= 0.05 / (len(order) - i))
    # Descriptive verdict with the G1 definition of the Full protocol (fixed before running: config.PV_*)
    n_seeds = {exp.label(ds): len(exp.seeds[ds]) for ds in exp.datasets}
    g1 = []
    for _, r in delta.iterrows():
        g1.append(bool(r['ΔMoF'] > 0 and r['_lo_MoF'] > 0 and r['_npos'] >= 0.6 * n_seeds[r['dataset']]
                       and r['ΔF1'] >= cfg.PV_G1_TOL and r['ΔmIoU'] >= cfg.PV_G1_TOL))
    delta['G1 (Per-Video)'] = g1
    seeds = pd.DataFrame(seed_rows)
    main.to_csv(exp.out('pervideo_main.csv'), index=False, encoding='utf-8-sig')
    delta.drop(columns=[c for c in delta.columns if c.startswith('_')]).to_csv(exp.out('pervideo_delta.csv'), index=False,
                                                                              encoding='utf-8-sig')
    seeds.to_csv(exp.out('pervideo_per_seed.csv'), index=False, encoding='utf-8-sig')
    if verbose:
        wide = main.pivot(index='method', columns='dataset', values='MoF').reindex(cfg.PV_METHODS)
        show(wide, 'Per-Video protocol, MoF (A1, mean over seeds):')
        show(delta[['dataset', 'comparison', 'ΔMoF', 'ΔMoF CI95 bootstrap', 'ΔMoF CI95 t (seed)', 'seeds >0/<0/=0',
                    'ΔF1', 'ΔmIoU', 'p_t', 'p_Wilcoxon', 'significant (Holm, 0.05)', 'G1 (Per-Video)']].set_index('dataset'),
             'Per-Video protocol, differences (points):', 3)
    return main, delta, seeds


def save_pv_results(exp, main, delta, seeds, check):
    res = dict(protocol='Per-Video (Hungarian per video; configurations from the Full-protocol 2-fold selection)',
               seeds={d: exp.seeds[d] for d in exp.datasets},
               constants=dict(PV_G1_TOL=cfg.PV_G1_TOL, PV_CHECK_TOL=cfg.PV_CHECK_TOL, PV_PAIRS=cfg.PV_PAIRS),
               point={key_str(k): dict(A1=v['A1'].tolist(), per={a: x.tolist() for a, x in v['per'].items()})
                      for k, v in exp.pv_point.items()},
               main=main.to_dict(orient='records'),
               delta=delta.drop(columns=[c for c in delta.columns if c.startswith('_')]).to_dict(orient='records'),
               per_seed=seeds.to_dict(orient='records'),
               check_trainer=check.to_dict(orient='records'))
    save_json(res, exp.out('pervideo_results.json'))
