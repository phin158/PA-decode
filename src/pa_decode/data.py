"""Datasets: locate / download the raw releases, build the folder layout expected by the authors' loader
(``data/<dataset>/{features,groundTruth,mapping}``), and describe every evaluation unit (dataset, activity).
"""
import collections
import glob
import os
import re
import shutil
import time

import numpy as np

# ------------------------------------------------------------------ raw data: locate or download
DL_IDS = {'Breakfast': ['1DbYnU2GBb68CxEt2I50QZm17KGYKNR1L', '1RO8lrvLy4bVaxZ7C62R0jVQtclXibLXU'],
          'YTI': ['1HyF3_bwWgz1QNgzLvN4J66TJVsQTYFTa', '1ENgdHvwHj2vFwflVXosCkCVP9mfLL5lP'],
          '50Salads': ['17o0WfF970cVnazrRuOWE92-OiYHEXTT3', '1mzcN9pz1tKygklQOiWI7iEvcJ1vJfU3R']}   # HOWTO of Annusha/unsup_temp_embed
DA_FOLDER = '1m7ljnnnd5kJ_Hi4Ir-sdNZRDFSqT1sHd'                                                  # Desktop Assembly (ASOT README)


class RawLocator:
    def __init__(self, raw_roots, download_dir, data_dir):
        self.download_dir = os.path.abspath(download_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.roots = [os.path.abspath(r) for r in raw_roots if r] + [self.download_dir]

    def _roots(self):
        return [x for x in self.roots if os.path.isdir(x)]

    def _find(self, pattern, up):
        for root in self._roots():
            hits = sorted(glob.glob(os.path.join(root, '**', pattern), recursive=True))
            hits = [h for h in hits if not h.startswith(self.data_dir)]
            if hits:
                p = hits[0]
                for _ in range(up):
                    p = os.path.dirname(p)
                return p
        return None

    def _find_fs(self):
        """50 Salads: folder with groundTruth/ and mapping/mappingeval.txt (features may live in a sibling folder)."""
        for root in self._roots():
            for h in sorted(glob.glob(os.path.join(root, '**', 'mapping', 'mappingeval.txt'), recursive=True)):
                d = os.path.dirname(os.path.dirname(h))
                if (os.path.isdir(os.path.join(d, 'groundTruth')) and 'desktop_assembly' not in h
                        and not h.startswith(self.data_dir)):
                    return d
        return None

    def locate(self, ds):
        if ds in ('FSeval', 'FS'):
            return self._find_fs()
        if ds == 'Breakfast':
            return self._find(os.path.join('BF_gt_mapping', 'groundTruth'), 2)
        if ds == 'YTI':
            return self._find(os.path.join('YTI', 'mapping', 'gtidx2labels.txt'), 2)
        if ds == 'desktop_assembly':
            return self._find(os.path.join('desktop_assembly', 'groundTruth', 'fixed_order'), 2)
        raise ValueError(ds)

    def download(self, ds):
        try:
            import gdown
        except ImportError:
            from .utils import pip_install
            pip_install('-U', 'gdown')
            import gdown
        out = self.download_dir

        def retry(fn, what, n=6, wait=60):
            for k in range(n):
                try:
                    r = fn()
                    if r:
                        return r
                except Exception as e:
                    print(f'  attempt {k + 1} failed for {what}:', str(e).strip().splitlines()[0])
                time.sleep(wait)
            raise RuntimeError(f'Could not download {what}')

        key = '50Salads' if ds in ('FSeval', 'FS') else ds
        if key in DL_IDS:
            os.makedirs(f'{out}/{key}', exist_ok=True)
            for fid in DL_IDS[key]:
                mark = f'{out}/{key}/.done_{fid}'
                if os.path.exists(mark):
                    continue
                p = retry(lambda: gdown.download(id=fid, output=f'{out}/{key}/', quiet=False), fid)
                shutil.unpack_archive(p, os.path.dirname(p))
                os.remove(p)
                open(mark, 'w').close()
        else:
            da = f'{out}/desktop_assembly'
            items = retry(lambda: gdown.download_folder(id=DA_FOLDER, output=da, quiet=True, skip_download=True), 'DA listing')
            for it in [it for it in items if '/videos/' not in it.local_path and not os.path.exists(it.local_path)]:
                os.makedirs(os.path.dirname(it.local_path), exist_ok=True)
                retry(lambda: gdown.download(id=it.id, output=it.local_path, quiet=True) and os.path.exists(it.local_path),
                      it.local_path, wait=20)
        for p in glob.glob(f'{out}/**/__MACOSX', recursive=True):
            shutil.rmtree(p, ignore_errors=True)


def _link(src, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(os.path.abspath(src), dst)


def prepare_datasets(datasets, raw_roots, data_dir, download_dir, allow_download=True):
    """Build ``data_dir/<FS|desktop_assembly|Breakfast|YTI>`` for the requested datasets.
    Returns the list of datasets that are ready (datasets that could not be found/downloaded are skipped)."""
    loc = RawLocator(raw_roots, download_dir, data_dir)
    data_dir, download_dir = loc.data_dir, loc.download_dir
    ready, base = [], {}
    for ds in datasets:
        base[ds] = loc.locate(ds)
        if base[ds] is None and allow_download:
            print(f'{ds} not found in {loc.roots} -> downloading from Google Drive')
            try:
                loc.download(ds)
                base[ds] = loc.locate(ds)
            except Exception as e:
                print(f'  !! {ds}: download failed ({str(e).splitlines()[0]})')
        if base[ds] is None:
            print(f'  !! SKIP {ds}: raw data not found (Google Drive may limit downloads; retry later or pass --raw-root).')
            continue
        print(f'{ds}: {base[ds]}')
        ready.append(ds)

    def _convert(src_txt, dst_npy, dtype):
        """np.loadtxt exactly as the authors' loader; the .txt is deleted if it was downloaded by us (saves disk)."""
        if not os.path.exists(dst_npy):
            os.makedirs(os.path.dirname(dst_npy), exist_ok=True)
            np.save(dst_npy, np.loadtxt(src_txt).astype(dtype))
        if src_txt.startswith(download_dir):
            os.remove(src_txt)

    def _prebuilt_feats(ds):
        """Already converted .npy features (…/data/<ds>/features) inside one of the raw roots."""
        for root in loc._roots():
            for d in sorted(glob.glob(os.path.join(root, '**', 'data', ds, 'features'), recursive=True)):
                if not d.startswith(data_dir) and glob.glob(os.path.join(d, '**', '*.npy'), recursive=True):
                    return d
        return None

    t0 = time.time()
    for ds in ready:
        b, dst = base[ds], os.path.join(data_dir, 'FS' if ds in ('FSeval', 'FS') else ds)
        if ds in ('Breakfast', 'YTI') and not glob.glob(os.path.join(b, 'features', '**', '*.txt'), recursive=True):
            pre = _prebuilt_feats(ds)
            if pre:
                print(f'  {ds}: no .txt features in {b}; using converted .npy in {pre}')
                _link(pre, os.path.join(dst, 'features'))
            elif not os.path.isdir(os.path.join(dst, 'features')):
                print(f'  {ds}: features not found -> downloading again')
                loc.download(ds)
                pat = (os.path.join('BF_gt_mapping', 'groundTruth') if ds == 'Breakfast'
                       else os.path.join('YTI', 'mapping', 'gtidx2labels.txt'))
                b = base[ds] = os.path.dirname(os.path.dirname(
                    glob.glob(os.path.join(download_dir, '**', pat), recursive=True)[0]))
        if ds in ('FSeval', 'FS'):
            _prepare_fs(b, dst, loc)
        elif ds == 'Breakfast':
            _link(os.path.join(b, 'BF_gt_mapping', 'groundTruth'), os.path.join(dst, 'groundTruth'))
            _link(os.path.join(b, 'BF_gt_mapping', 'mapping'), os.path.join(dst, 'mapping'))
            txts = sorted(glob.glob(os.path.join(b, 'features', '**', '*.txt'), recursive=True))
            if txts:   # check one file before converting (loader: np.loadtxt float64, then standardised -> keep float64)
                out0 = os.path.join(dst, 'features', os.path.basename(os.path.dirname(txts[0])), os.path.basename(txts[0])[:-4] + '.npy')
                ref = np.loadtxt(txts[0])
                _convert(txts[0], out0, np.float64)
                assert np.array_equal(np.load(out0), ref)
            for p in txts[1:]:
                _convert(p, os.path.join(dst, 'features', os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-4] + '.npy'),
                         np.float64)
        elif ds == 'YTI':
            for p in sorted(glob.glob(os.path.join(b, 'features', '**', '*.txt'), recursive=True)):
                # not standardised -> the loader casts to float32 immediately; storing float32 is identical and saves ~5 GB
                _convert(p, os.path.join(dst, 'features', os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-4] + '.npy'),
                         np.float32)
            fix_yti_ground_truth(b, dst)
        else:
            _link(os.path.join(b, 'groundTruth', 'fixed_order'), os.path.join(dst, 'groundTruth'))
            _link(os.path.join(b, 'features', 'fixed_order'), os.path.join(dst, 'features'))
            os.makedirs(os.path.join(dst, 'mapping'), exist_ok=True)
            shutil.copy(os.path.join(b, 'mapping', 'fixed_order', 'mapping.txt'), os.path.join(dst, 'mapping', 'mapping.txt'))
    print(f'Data folders ready after {(time.time() - t0) / 60:.1f} min')
    return ready


def _prepare_fs(b, dst, loc):
    if os.path.exists(os.path.join(dst, '.done')):      # FSeval and FS share data/FS
        return
    _link(os.path.join(b, 'groundTruth'), os.path.join(dst, 'groundTruth'))
    _link(os.path.join(b, 'mapping'), os.path.join(dst, 'mapping'))
    names = sorted(f for f in os.listdir(os.path.join(b, 'groundTruth')) if len(f.split('_')) > 1 or len(f.split('-')) > 1)
    idx = {}
    for root in [os.path.dirname(b)] + loc._roots():
        for ext in ('npy', 'txt'):
            for pth in glob.glob(os.path.join(root, '**', f'*.{ext}'), recursive=True):
                st_ = os.path.basename(pth)[:-len(ext) - 1]
                if st_ in names and f'{os.sep}mapping{os.sep}' not in pth and not pth.startswith(loc.data_dir):
                    idx.setdefault(st_, pth)
        if len(idx) == len(names):
            break
    miss = sorted(set(names) - set(idx))
    assert not miss, f'50 Salads: missing features {miss[:5]}'
    os.makedirs(os.path.join(dst, 'features'), exist_ok=True)
    for n_ in names:   # not standardised -> the loader casts to float32 immediately; storing float32 is identical
        out_ = os.path.join(dst, 'features', n_ + '.npy')
        if not os.path.exists(out_):
            x_ = np.load(idx[n_]) if idx[n_].endswith('.npy') else np.loadtxt(idx[n_])
            np.save(out_, x_.astype(np.float32))
    open(os.path.join(dst, '.done'), 'w').close()
    print(f'  50 Salads: {len(names)} videos, features from {os.path.dirname(next(iter(idx.values())))}')


def fix_yti_ground_truth(b, dst):
    """YTI ground truth is inconsistent in the public release: some files store action NAMES ('bg', 'GIVE_COMPRESSION'),
    others store NUMBERS ('0', '15', '-1'), and the two id->name tables shipped with it disagree. We align the
    numeric/name versions of the same video (X and X_idt) by relative position, choose the table supported by the data,
    and rewrite every ground-truth file with names."""
    src_gt = os.path.join(b, 'groundTruth')
    rd = lambda f: [l.rstrip() for l in open(os.path.join(src_gt, f)) if l.strip()]
    is_num = lambda seq: len(seq) > 0 and all(re.fullmatch(r'-?\d+', x) for x in seq)
    bases = sorted(f for f in os.listdir(src_gt) if not f.endswith('_idt') and not f.startswith('.'))
    seqs = {f: (rd(f), rd(f + '_idt') if os.path.exists(os.path.join(src_gt, f + '_idt')) else None) for f in bases}
    # X (numbers, same length as features) and X_idt (names) may differ in frame rate -> align by RELATIVE POSITION:
    # frame i/n of the numeric version <-> frame floor((i+0.5)*m/n) of the name version; majority vote per numeric code.
    i2n = collections.defaultdict(collections.Counter)
    act_names = collections.defaultdict(set)
    n_eq = n_resamp = 0
    for f, (x, y) in seqs.items():
        for u, v in [(x, y), (y, x)]:
            if u and v and is_num(u) and not is_num(v):
                idx = np.minimum(((np.arange(len(u)) + 0.5) * len(v) / len(u)).astype(int), len(v) - 1)
                for a_, j in zip(u, idx):
                    i2n[a_][v[j]] += 1
                act_names['_'.join(f.split('_')[:-1])].update(v)
                n_eq += len(u) == len(v)
                n_resamp += len(u) != len(v)
    n_pair = sum(sum(c.values()) for c in i2n.values())
    assert n_pair > 0, 'YTI: no numeric/name pair found'
    vote = {u: c.most_common(1)[0][0] for u, c in i2n.items()}
    purity = 1 - sum(sum(c.values()) - c.most_common(1)[0][1] for c in i2n.values()) / n_pair
    print(f'  YTI: {n_eq} pairs of equal length, {n_resamp} pairs of different length (aligned by relative position); '
          f'vote purity = {purity:.3f}')
    # The two tables are two complete 1-1 hypotheses; pick the one supported by the aligned frames.
    tables = {}
    for tf in ['gtidx2labels.txt', 'mapping_idxlabel.txt']:
        tp = os.path.join(b, 'mapping', tf)
        if os.path.exists(tp):
            tables[tf] = {l.split(' ')[0]: l.rstrip().split(' ')[-1] for l in open(tp) if l.strip()}
    fg = [u for u in i2n if u != '-1']          # ignore background ('-1' -> 'bg' in every table; 63% of frames)
    n_fg = sum(sum(i2n[u].values()) for u in fg)
    agree = {tf: sum(i2n[u][t.get(u)] for u in fg) / n_fg for tf, t in tables.items()}
    vote_agree = {tf: float(np.mean([t.get(u) == vote[u] for u in fg])) for tf, t in tables.items()}
    print('  YTI: share of aligned non-background frames agreeing with each table:', {k: round(v, 3) for k, v in agree.items()},
          '| share of codes agreeing with the vote:', {k: round(v, 3) for k, v in vote_agree.items()})
    best_tab = max(agree, key=agree.get)
    other = max([v for k, v in agree.items() if k != best_tab], default=0.)
    ceiling = sum(i2n[u].most_common(1)[0][1] for u in fg) / n_fg      # best possible agreement given alignment noise
    print(f'  YTI: alignment ceiling = {ceiling:.3f}; best table reaches {agree[best_tab] / ceiling:.1%} of it; other table {other:.3f}')
    assert agree[best_tab] >= 0.9 * ceiling and other <= 0.2, \
        f'YTI: no table is clearly supported by the data {agree} (ceiling {ceiling:.3f}) -> check manually'
    learned = tables[best_tab]
    seen = [u for u in learned if u in i2n]
    wrong = [(u, learned[u], vote[u], round(i2n[u][learned[u]] / sum(i2n[u].values()), 2)) for u in seen if vote[u] != learned[u]]
    print(f'  YTI: USING {best_tab} (1-1, {len(learned)} codes). Codes where the vote differs from the table ({len(wrong)}):', wrong[:8])
    viol = []   # every code must map to a name of the SAME activity (names taken from the _idt files of that activity)
    for f, (x, y) in seqs.items():
        a_ = '_'.join(f.split('_')[:-1])
        num = x if is_num(x) else (y if y and is_num(y) else None)
        if num:
            viol += [(a_, u, learned.get(u)) for u in set(num) if learned.get(u) not in act_names[a_] and learned.get(u) != 'bg']
    assert not viol, f'YTI: numeric code maps to an action of another activity: {sorted(set(viol))[:8]}'
    print('  YTI: every code maps to an action of the video\'s own activity')
    name_set = {l.rstrip().split(' ')[-1] for l in open(os.path.join(b, 'mapping', 'gtidx2labels.txt')) if l.strip()}
    gt_dst = os.path.join(dst, 'groundTruth')
    if os.path.islink(gt_dst):
        os.unlink(gt_dst)
    os.makedirs(gt_dst, exist_ok=True)
    stat, cnt = collections.Counter(), collections.Counter()
    for f, (x, y) in seqs.items():
        if not is_num(x):
            seq, how = x, 'names'
        else:
            miss = sorted({u for u in x if u not in learned})
            assert not miss, f'YTI {f}: unmapped numeric codes {miss[:10]}'
            seq, how = [learned[u] for u in x], 'numbers->names'
        seq = ['bg' if v == '-1' else v for v in seq]          # remaining '-1' = background
        stat[how] += 1
        cnt.update(seq)
        act_ = '_'.join(f.split('_')[:-1])
        npy = os.path.join(dst, 'features', act_, f + '.npy')
        if os.path.exists(npy):
            n_feat = np.load(npy, mmap_mode='r').shape[0]
            assert n_feat == len(seq), f'YTI {f}: {n_feat} feature frames but {len(seq)} label frames'
        with open(os.path.join(gt_dst, f), 'w') as fh:
            fh.write('\n'.join(seq) + '\n')
    miss = sorted(set(cnt) - name_set)
    assert not miss, f'YTI: labels not in gtidx2labels: {miss[:10]}'
    os.makedirs(os.path.join(dst, 'mapping'), exist_ok=True)
    shutil.copy(os.path.join(b, 'mapping', 'gtidx2labels.txt'), os.path.join(dst, 'mapping', 'mapping.txt'))   # '-1 bg'
    print(f'  YTI: {len(seqs)} ground-truth files, source: {dict(stat)}; {len(cnt)} labels; background frames = {cnt["bg"]} '
          f'({100 * cnt["bg"] / sum(cnt.values()):.1f}%, excluded at evaluation as with -x -1)')


# ------------------------------------------------------------------ evaluation units
def make_ds(exp, ds, act, n_frames, random):
    return exp.VideoDataset(exp.data_dir, ds, n_frames, standardise=exp.paper[ds]['std'], random=random, action_class=[act])


def read_gt(dobj, fname):
    """Same as the authors' __getitem__ with n_frames=None, without loading the features."""
    return np.array([dobj.action_mapping[l.rstrip()] for l in open(os.path.join(dobj.data_dir, 'groundTruth', fname))])


def fold_of(ds, names):
    """2-fold split used for parameter selection (by person / subject where available)."""
    if ds == 'Breakfast':
        persons = sorted({n.split('_')[0] for n in names})
        return np.array([persons.index(n.split('_')[0]) % 2 for n in names])
    if ds == 'YTI':
        return np.array([int(n.split('_')[-1]) % 2 for n in names])
    if ds in ('FSeval', 'FS'):
        subj = lambda n: (re.search(r'rgb-(\d+)-', n).group(1) if re.search(r'rgb-(\d+)-', n) else n)
        S = sorted({subj(n) for n in names})
        return np.array([S.index(subj(n)) % 2 for n in names])
    return np.arange(len(names)) % 2


def build_units(exp, verbose=True):
    """Fill ``exp.units`` (list of (dataset, activity)) and ``exp.uinfo`` (names, local GT, folds, ...)."""
    import pandas as pd
    exp.units, exp.uinfo = [], {}
    for ds in exp.datasets:
        P = exp.paper[ds]
        for act, K in P['acts'].items():
            dobj = make_ds(exp, ds, act, None, False)
            names = list(dobj.video_fnames)
            assert names, f'{ds}/{act}: no videos'
            gts = [read_gt(dobj, n) for n in names]
            ex = P['exclude']
            keep = [g != ex if ex is not None else np.ones(len(g), bool) for g in gts]
            cls = sorted(set(np.concatenate([g[k] for g, k in zip(gts, keep)]).tolist()))
            loc = {c: i for i, c in enumerate(cls)}
            gloc = [np.array([loc.get(x, -1) for x in g]) for g in gts]       # -1 = excluded frame (YTI background)
            f0 = dobj[0][0]
            assert f0.shape[1] == P['layers'][0], (ds, act, f0.shape)
            exp.units.append((ds, act))
            exp.uinfo[(ds, act)] = dict(names=names, gt=gloc, keep=keep, C=len(cls), cls=cls, K=K,
                                        fold=None, nfr=np.array([k.sum() for k in keep]))
        # folds (Breakfast: persons are numbered over the whole dataset)
        allnames = [n for (d, a) in exp.units if d == ds for n in exp.uinfo[(d, a)]['names']]
        persons = sorted({n.split('_')[0] for n in allnames})
        for (d, a) in [u for u in exp.units if u[0] == ds]:
            nm = exp.uinfo[(d, a)]['names']
            exp.uinfo[(d, a)]['fold'] = (np.array([persons.index(n.split('_')[0]) % 2 for n in nm]) if ds == 'Breakfast'
                                         else fold_of(ds, nm))
        if verbose:
            rows = [(a, len(exp.uinfo[(ds, a)]['names']), exp.uinfo[(ds, a)]['C'], exp.uinfo[(ds, a)]['K'],
                     int(exp.uinfo[(ds, a)]['nfr'].sum()), int(sum((~k).sum() for k in exp.uinfo[(ds, a)]['keep'])),
                     np.bincount(exp.uinfo[(ds, a)]['fold']).tolist())
                    for (d, a) in exp.units if d == ds]
            print(f'\n{exp.label(ds)}:')
            print(pd.DataFrame(rows, columns=['activity', '#videos', '#GT classes', '#clusters', '#scored frames',
                                              '#excluded frames', 'videos/fold']).set_index('activity').to_string())
