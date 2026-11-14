"""Result tables: reproduction of ASOT, main comparison with confidence intervals, per-activity and per-seed
results, class diagnostics, and the pre-registered verdicts."""
import numpy as np
import pandas as pd

from . import config as cfg
from .decode import CFGS
from .metrics import hungarian
from .selection import evaluate, ones_weights, seed_vals
from .utils import t_ci, show, save_json, key_str

MET = cfg.METRICS
SHORT = {cfg.M_ASOT: 'ASOT', cfg.M_ASOTL: 'ASOT-lambda', cfg.M_PA: 'PA', cfg.M_INT: 'PA-int', cfg.M_MM: 'PA-mm',
         cfg.M_ORD: 'PA-true'}


def reproduction_table(exp, verbose=True):
    rows = []
    for ds in exp.datasets:
        pub = cfg.PUBLISHED[ds]
        v = seed_vals(exp, ds, cfg.M_ASOT)
        se = v[:, 0].std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        r = {'dataset': exp.label(ds), 'MoF per seed': ' '.join(f'{x:.1f}' for x in v[:, 0]), 'MoF mean': round(v[:, 0].mean(), 2),
             'se': round(se, 2), 'F1 mean': round(v[:, 1].mean(), 2), 'mIoU mean': round(v[:, 2].mean(), 2),
             'paper 1-run (MoF/F1/mIoU)': f"{pub['mof1']} / {pub['f11']} / {pub['miou1']}"}
        if pub['mof5'] is not None:
            r['paper 5-run MoF (se)'] = f"{pub['mof5']} ({pub['se5']})"
            r['z vs 5-run'] = round((v[:, 0].mean() - pub['mof5']) / np.sqrt(se ** 2 + pub['se5'] ** 2), 2)
        rows.append(r)
    df = pd.DataFrame(rows).set_index('dataset')
    if verbose:
        show(df, 'Reproduction of ASOT (A1; lambda_eval = 0.01, alpha/eps/r as in Appendix B; KMeans n_init=10):')
        if exp.checks:
            print('Pipeline check: max |our decoded MoF - trainer.test| =', f"{max(abs(c['d_mof']) for c in exp.checks):.4f} points.\n")
    df.to_csv(exp.out('reproduction_asot.csv'))
    return df


def main_tables(exp, verbose=True):
    """Mean ± sd over seeds of every method, and paired differences with bootstrap and t confidence intervals."""
    out = {}
    for rule in cfg.RULES:
        for ds in exp.datasets:
            pub, n = cfg.PUBLISHED[ds], len(exp.seeds[ds])
            tab = []
            for meth in cfg.METHODS:
                v1, v2 = seed_vals(exp, ds, meth, rule, 'A1'), seed_vals(exp, ds, meth, rule, 'A2')
                tab.append({'method': meth,
                            **{f'{m} (A1)': f'{v1[:, i].mean():.2f} ± {v1[:, i].std(ddof=1) if n > 1 else 0:.2f}' for i, m in enumerate(MET)},
                            **{f'{m} (A2)': f'{v2[:, i].mean():.2f}' for i, m in enumerate(MET)}})
            tab.append({'method': 'Paper 1-run', 'MoF (A1)': pub['mof1'], 'F1 (A1)': pub['f11'], 'mIoU (A1)': pub['miou1']})
            if pub['mof5'] is not None:
                tab.append({'method': 'Paper 5-run (mean ± se)', 'MoF (A1)': f"{pub['mof5']} ± {pub['se5']}"})
            T = pd.DataFrame(tab).set_index('method').fillna('')
            D = exp.boot[(ds, rule)]
            drows = []
            for pi, (a, c) in enumerate(cfg.PAIRS):
                ps = seed_vals(exp, ds, a, rule)[:, :3] - seed_vals(exp, ds, c, rule)[:, :3]
                ps2 = seed_vals(exp, ds, a, rule, 'A2')[:, :3] - seed_vals(exp, ds, c, rule, 'A2')[:, :3]
                r = {'comparison': f'{a} - {c}'}
                for i, m in enumerate(MET):
                    lo, hi = np.percentile(D[:, pi, 0, i], [2.5, 97.5])
                    r[f'Δ{m}'] = round(ps[:, i].mean(), 2)
                    r[f'CI95 boot {m}'] = f'[{lo:.2f}, {hi:.2f}]'
                    if m == 'MoF':
                        tl, th = t_ci(ps[:, 0])
                        r['CI95 t (seed)'] = f'[{tl:.2f}, {th:.2f}]'
                        r['#seeds ΔMoF>0'] = f'{int((ps[:, 0] > 0).sum())}/{n}'
                        lo2, hi2 = np.percentile(D[:, pi, 1, 0], [2.5, 97.5])
                        r['ΔMoF (A2)'] = f'{ps2[:, 0].mean():.2f} [{lo2:.2f}, {hi2:.2f}]'
                    exp.delta[(ds, rule, a, c, m)] = dict(mean=ps[:, i].mean(), lo=lo, hi=hi, npos=int((ps[:, i] > 0).sum()))
                drows.append(r)
            Dt = pd.DataFrame(drows).set_index('comparison')
            out[(ds, rule)] = (T, Dt)
            T.to_csv(exp.out(f'main_{ds}.csv'))
            Dt.to_csv(exp.out(f'delta_{ds}.csv'))
            if verbose:
                show(T, f'--- {exp.label(ds)} (mean ± sd over {n} seeds; selection rule: {rule}) ---')
                show(Dt)
                print('Selected configuration (fold0 -> scored on fold1 | fold1 -> scored on fold0):')
                for meth in [cfg.M_ASOTL, cfg.M_PA] + cfg.NEW_METHODS:
                    print(f'  {meth:20s}', ' ; '.join(f's{s}: ' + ' | '.join(exp.chosen[(ds, s, meth, rule)]) for s in exp.seeds[ds]))
                print()
    return out


def significance_table(exp, a=cfg.M_PA, b=cfg.M_ASOTL, verbose=True):
    """Paired tests over seeds for ΔMoF (A1): one-sample t-test, Wilcoxon signed-rank on the non-zero differences,
    sign counts, and the Holm correction of the t-test p-values over the settings (as reported in the paper)."""
    from scipy import stats
    rows, pv = [], {}
    for ds in exp.datasets:
        d = seed_vals(exp, ds, a)[:, 0] - seed_vals(exp, ds, b)[:, 0]
        nz = d[np.abs(d) > 1e-9]
        t = stats.ttest_1samp(d, 0)
        w = stats.wilcoxon(nz).pvalue if len(nz) > 0 else np.nan
        pv[ds] = t.pvalue
        lo, hi = t_ci(d)
        rows.append({'setting': exp.label(ds), 'ΔMoF': d.mean(), 'CI95 t (seed)': f'[{lo:.2f}, {hi:.2f}]',
                     'seeds >0/<0/=0': f'{(d > 1e-9).sum()}/{(d < -1e-9).sum()}/{(np.abs(d) <= 1e-9).sum()}',
                     'p_t': t.pvalue, 'p_Wilcoxon': w})
    order = sorted(pv, key=pv.get)
    holm = {ds: pv[ds] <= 0.05 / (len(order) - i) for i, ds in enumerate(order)}
    for r, ds in zip(rows, exp.datasets):
        r['significant (Holm, 0.05)'] = bool(holm[ds])
    df = pd.DataFrame(rows).set_index('setting')
    df.to_csv(exp.out('significance.csv'))
    if verbose:
        show(df, f'Paired tests over seeds, {a} - {b} (MoF A1):', 4)
    return df


def per_activity_table(exp, verbose=True):
    res = {}
    for ds in exp.datasets:
        if len(exp.units_of(ds)) < 2:
            continue
        rows = []
        for a in exp.units_of(ds):
            V = {m: np.array([exp.point[(ds, s, m, 'mof')]['per'][a] for s in exp.seeds[ds]])
                 for m in [cfg.M_ASOTL, cfg.M_PA] + cfg.NEW_METHODS}
            r = {'activity': a, '#videos': len(exp.uinfo[(ds, a)]['names'])}
            r.update({f'MoF {m}': V[m][:, 0].mean() for m in V})
            r.update({f'Δ {m} - PA': (V[m] - V[cfg.M_PA])[:, 0].mean() for m in cfg.NEW_METHODS})
            rows.append(r)
        df = pd.DataFrame(rows).set_index('activity')
        res[ds] = df
        df.to_csv(exp.out(f'per_activity_{ds}.csv'))
        if verbose:
            show(df, f'--- {exp.label(ds)}: per activity (rule mof, mean over {len(exp.seeds[ds])} seeds) ---')
    return res


def class_diag(exp, ds, meth):
    """Share of the largest GT class vs. its predicted share, its IoU, and the mean IoU of small classes."""
    big_gt, big_pred, small_iou, big_iou = [], [], [], []
    W = ones_weights(exp, ds)
    for s in exp.seeds[ds]:
        _, _, _, ch, _ = evaluate(exp, ds, s, meth, 'mof', W)
        for a in exp.units_of(ds):
            run = exp.dec[(ds, a, s)]
            nv = len(ch[a])
            M = run['conf'][ch[a], np.arange(nv)].sum(0).astype(float)
            pr, gl = hungarian(M)
            g = M.sum(0) / M.sum()
            sp = np.zeros_like(g)
            io = np.zeros_like(g)
            sp[gl] = M[pr].sum(1) / M.sum()
            io[gl] = M[pr, gl] / (M[pr].sum(1) + M[:, gl].sum(0) - M[pr, gl])
            present = g > 0
            j = np.argmax(g)
            big_gt.append(g[j])
            big_pred.append(sp[j])
            big_iou.append(io[j])
            small = present & (g < np.median(g[present]))
            small_iou.append(io[small].mean() if small.any() else np.nan)
    return 100 * np.mean(big_gt), 100 * np.mean(big_pred), 100 * np.mean(big_iou), 100 * np.nanmean(small_iou)


def class_diag_table(exp, verbose=True):
    rows = []
    for ds in exp.datasets:
        for meth in [cfg.M_ASOTL, cfg.M_PA] + cfg.NEW_METHODS:
            bg, bp, bi, si = class_diag(exp, ds, meth)
            rows.append({'dataset': exp.label(ds), 'method': meth, 'largest class: GT share %': bg, 'predicted share %': bp,
                         'IoU largest class': bi, 'mean IoU small classes (< median)': si})
    df = pd.DataFrame(rows).set_index(['dataset', 'method'])
    df.to_csv(exp.out('class_diagnostics.csv'))
    if verbose:
        show(df, '--- Long-class inflation (mean over activities and seeds) ---')
    return df


def per_seed_tables(exp, dval, verbose=True):
    """One row per seed (one independent ASOT training each); configuration selected by 2-fold within each seed."""
    show_m = list(SHORT)
    per_seed, long_rows = {}, []
    for ds in exp.datasets:
        rows = []
        for i, s in enumerate(exp.seeds[ds]):
            r = {'seed': s}
            V = {m: np.array(exp.point[(ds, s, m, 'mof')]['A1']) for m in show_m}
            for m in show_m:
                for j, met in enumerate(MET):
                    r[f'{SHORT[m]} {met}'] = V[m][j]
            for j, met in enumerate(MET):
                r[f'Δ{met} PA - ASOT-lambda'] = V[cfg.M_PA][j] - V[cfg.M_ASOTL][j]
            for m in cfg.NEW_METHODS:
                r[f'ΔMoF {SHORT[m]} - ASOT-lambda'] = V[m][0] - V[cfg.M_ASOTL][0]
                r[f'ΔMoF {SHORT[m]} - PA'] = V[m][0] - V[cfg.M_PA][0]
            if dval:
                r['in-sample per-video-w bound - PA'] = dval[ds]['per_seed']['Hw'][i]
                r['cross-fit per-video-w bound - PA'] = dval[ds]['per_seed']['Hw_cf'][i]
            for m in [cfg.M_ASOTL, cfg.M_PA] + cfg.NEW_METHODS:
                r[f'config {SHORT[m]} (f0->f1 | f1->f0)'] = ' | '.join(exp.chosen[(ds, s, m, 'mof')])
            rows.append(r)
            for a in exp.units_of(ds):
                for m in show_m:
                    v = exp.point[(ds, s, m, 'mof')]['per'][a]
                    long_rows.append({'dataset': exp.label(ds), 'activity': a, 'seed': s, 'method': m,
                                      'MoF': v[0], 'F1': v[1], 'mIoU': v[2],
                                      'config (f0->f1 | f1->f0)': ' | '.join(exp.chosen[(ds, s, m, 'mof')])})
        df = pd.DataFrame(rows).set_index('seed')
        per_seed[ds] = df
        d_cols = [c for c in df.columns if c.startswith('Δ') or 'bound' in c]
        if verbose:
            n_ = len(exp.seeds[ds])
            print(f'==================== {exp.label(ds)}: {n_} seeds, one row per seed ====================')
            show(df[[c for c in df.columns if not c.startswith('Δ') and 'bound' not in c and not c.startswith('config')]],
                 '(a) Scores (A1, %):')
            show(df[d_cols], '(b) Differences per seed (points):')
            cnt = pd.DataFrame({'#seeds > 0': [f'{int(x)}/{n_}' for x in (df[d_cols] > 0).sum()],
                                '#seeds < 0': [f'{int(x)}/{n_}' for x in (df[d_cols] < 0).sum()],
                                'min': [f'{x:+.2f}' for x in df[d_cols].min()], 'max': [f'{x:+.2f}' for x in df[d_cols].max()]},
                               index=d_cols).T
            show(cnt, '(c) Counts over seeds:')
    wide = pd.concat([per_seed[ds].reset_index().assign(dataset=exp.label(ds)) for ds in exp.datasets], ignore_index=True)
    wide = wide[['dataset'] + [c for c in wide.columns if c != 'dataset']]
    wide.to_csv(exp.out('per_seed.csv'), index=False, encoding='utf-8-sig')
    longdf = pd.DataFrame(long_rows)
    longdf.to_csv(exp.out('per_activity_seed.csv'), index=False, encoding='utf-8-sig')
    return per_seed, longdf


def verdicts(exp, dval, verbose=True):
    """Pre-registered criteria.
    G1 (method vs ASOT-lambda): ΔMoF > 0, bootstrap CI of ΔMoF > 0, ΔMoF > 0 on >= 60% of seeds, ΔF1 >= -0.5, ΔmIoU >= -0.5.
    K1 (variant without w can replace PA-decode): G1 and variant - PA-decode >= K1_TOL on all main settings.
    K2: variant better than PA-decode with CI > 0.   H (side settings): variant - ASOT-lambda >= H_TOL.
    D (per-video w): in-sample bound < D_GO on all main settings -> NO-GO."""
    delta = lambda ds, a, b, m='MoF': exp.delta[(ds, 'mof', a, b, m)]

    def g1(ds, meth, ref=cfg.M_ASOTL):
        n = len(exp.seeds[ds])
        d = {m: delta(ds, meth, ref, m) for m in MET}
        checks = {'ΔMoF > 0': d['MoF']['mean'] > 0, 'boot CI95 ΔMoF > 0': d['MoF']['lo'] > 0,
                  f'>= {int(np.ceil(0.6 * n))}/{n} seeds ΔMoF > 0': d['MoF']['npos'] >= int(np.ceil(0.6 * n)),
                  'ΔF1 >= -0.5': d['F1']['mean'] >= -0.5, 'ΔmIoU >= -0.5': d['mIoU']['mean'] >= -0.5}
        return all(checks.values()), checks, d

    fmt = lambda d: f"{d['mean']:+.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    verdict, summary = {}, []
    for ds in exp.datasets:
        gp, cp, dp = g1(ds, cfg.M_PA)
        if verbose:
            print(f'\n### {exp.label(ds)}  ({len(exp.seeds[ds])} seeds; {"main" if ds in cfg.MAIN else "side"} setting)')
            print(f"  PA-decode - ASOT-lambda: ΔMoF {fmt(dp['MoF'])} | G1 {'MET' if gp else 'not met'} ("
                  + ', '.join(f"{k}: {'yes' if v else 'no'}" for k, v in cp.items()) + ')')
        verdict[ds] = {cfg.M_PA: dict(G1=gp, dmof=dp['MoF']['mean'])}
        row = {'setting': exp.label(ds), '#seeds': len(exp.seeds[ds]),
               cfg.M_ASOTL: seed_vals(exp, ds, cfg.M_ASOTL)[:, 0].mean(), cfg.M_PA: seed_vals(exp, ds, cfg.M_PA)[:, 0].mean()}
        for m in cfg.NEW_METHODS:
            gm, cm, dm = g1(ds, m)
            dvp = {k: delta(ds, m, cfg.M_PA, k) for k in MET}
            k1 = gm and dvp['MoF']['mean'] >= cfg.K1_TOL
            k2 = dvp['MoF']['mean'] > 0 and dvp['MoF']['lo'] > 0
            h = dm['MoF']['mean'] >= cfg.H_TOL
            verdict[ds][m] = dict(G1=gm, K1=k1, K2=k2, H=h, d_asot=dm['MoF']['mean'], d_pa=dvp['MoF']['mean'],
                                  d_pa_lo=dvp['MoF']['lo'], d_pa_hi=dvp['MoF']['hi'])
            if verbose:
                print(f"  {m}:")
                print(f"     - ASOT-lambda: ΔMoF {fmt(dm['MoF'])} | ΔF1 {dm['F1']['mean']:+.2f} | ΔmIoU {dm['mIoU']['mean']:+.2f}"
                      f" | G1 {'MET' if gm else 'not met'}")
                print(f"     - PA-decode  : ΔMoF {fmt(dvp['MoF'])} | ΔF1 {dvp['F1']['mean']:+.2f} | ΔmIoU {dvp['mIoU']['mean']:+.2f}")
                if ds in cfg.MAIN:
                    print(f"     K1 (G1 and Δ vs PA >= {cfg.K1_TOL}): {'MET' if k1 else 'not met'} | K2 (better than PA, CI > 0): "
                          f"{'MET' if k2 else 'no'}")
                else:
                    print(f"     H (Δ vs ASOT-lambda >= {cfg.H_TOL}): {'MET' if h else 'not met'}")
            row[m] = seed_vals(exp, ds, m)[:, 0].mean()
            row[f'{m} - PA [CI]'] = fmt(dvp['MoF'])
        if dval:
            row['per-video-w in-sample'] = dval[ds]['Hw']
            row['per-video-w cross-fit'] = dval[ds]['Hw_cf']
        row[cfg.M_ORD] = seed_vals(exp, ds, cfg.M_ORD)[:, 0].mean()
        summary.append(row)
    sdf = pd.DataFrame(summary).set_index('setting')
    sdf.to_csv(exp.out('summary.csv'))
    if verbose:
        show(sdf, '\n================  SUMMARY (MoF A1, mean over seeds) ================')

    main = [d for d in cfg.MAIN if d in exp.datasets]
    side = [d for d in cfg.SIDE if d in exp.datasets]
    go = {}
    lines = []
    if len(main) < len(cfg.MAIN):
        lines.append(f'!! Missing main settings {set(cfg.MAIN) - set(main)}: K1/D verdicts are not conclusive.')
    for m in cfg.NEW_METHODS:
        k1_all = all(verdict[d][m]['K1'] for d in main) and len(main) == len(cfg.MAIN)
        h_all = all(verdict[d][m]['H'] for d in side)
        k2_list = [exp.label(d) for d in main if verdict[d][m]['K2']]
        go[m] = k1_all
        lines.append(f"{m}: K1 {'MET' if k1_all else 'not met'} on the main settings "
                     f"({', '.join(exp.label(d) + (' yes' if verdict[d][m]['K1'] else ' no') for d in main)}) | "
                     f"H {'met' if h_all else 'not met'} on {', '.join(exp.label(d) for d in side)} | "
                     f"K2 (better than PA): {', '.join(k2_list) if k2_list else 'none'}")
    if go[cfg.M_INT] and go[cfg.M_MM]:
        gap = np.mean([verdict[d][cfg.M_MM]['d_pa'] - verdict[d][cfg.M_INT]['d_pa'] for d in main])
        pick = cfg.M_MM if gap >= cfg.PREF_2A else cfg.M_INT
        lines.append(f'Both variants meet K1; PA-minmax - PA-integral = {gap:+.2f} (threshold {cfg.PREF_2A}) -> {pick}.')
    elif go[cfg.M_INT] or go[cfg.M_MM]:
        pick = cfg.M_INT if go[cfg.M_INT] else cfg.M_MM
        lines.append(f'Only {pick} meets K1 -> it can replace PA-decode (w removed).')
    else:
        pick = None
        lines.append('No variant meets K1 -> keep PA-decode (w selected by 2-fold) as the main method; '
                     'report PA-integral / PA-minmax as negative results.')
    d_verdict = None
    if dval and main:
        hw_main = {d: dval[d]['Hw'] for d in main}
        hcf_main = {d: dval[d]['Hw_cf'] for d in main}
        d_verdict = ('NO-GO: learning w per video is not worth it (low in-sample bound)'
                     if all(v < cfg.D_GO for v in hw_main.values()) else
                     'GO: worth trying' if sum(v >= cfg.D_GO for v in hcf_main.values()) >= 2 else 'grey zone: report only')
        lines.append(f"D: in-sample bound - PA = {', '.join(f'{exp.label(d)} {v:+.2f}' for d, v in hw_main.items())}; "
                     f"cross-fit = {', '.join(f'{exp.label(d)} {v:+.2f}' for d, v in hcf_main.items())} "
                     f"(threshold {cfg.D_GO}) -> {d_verdict}")
    if verbose:
        print('================  VERDICTS (pre-registered criteria)  ================')
        for l in lines:
            print('  ' + l)
    return verdict, sdf, pick, d_verdict, lines


def save_results(exp, verdict, summary, pick, d_verdict, dval, repro, wdf, lines):
    import torch, sklearn, pytorch_lightning as pl
    res = dict(seeds={d: exp.seeds[d] for d in exp.datasets}, verdict=verdict, verdict_lines=lines,
               summary=summary.reset_index().to_dict(orient='records'), pick=pick, D=dval, d_verdict=d_verdict,
               constants=dict(TAU_MM=cfg.TAU_MM, K1_TOL=cfg.K1_TOL, H_TOL=cfg.H_TOL, PREF_2A=cfg.PREF_2A, D_GO=cfg.D_GO),
               point={key_str(k): {kk: (vv.tolist() if hasattr(vv, 'tolist') else {a: x.tolist() for a, x in vv.items()})
                                   for kk, vv in v.items()} for k, v in exp.point.items()},
               chosen={key_str(k): v for k, v in exp.chosen.items()},
               delta={key_str(k): v for k, v in exp.delta.items()},
               wvideo=wdf.reset_index().to_dict(orient='records') if wdf is not None else None,
               repro=repro.reset_index().to_dict(orient='records'), checks=exp.checks,
               versions=dict(torch=torch.__version__, pl=pl.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                             asot_commit=exp.asot_head))
    save_json(res, exp.out('results.json'))
