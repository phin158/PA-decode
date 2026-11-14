"""50 Salads in the MS-TCN format: features/<video>.npy [2048, T], groundTruth/<video>.txt, splits/, mapping.txt."""
import glob
import os
import time

import numpy as np
import torch

from .config import MSTCN, SPLITS, ZENODO_URL


def is_50salads(d):
    return (all(os.path.exists(os.path.join(d, x)) for x in ['features', 'groundTruth', 'splits', 'mapping.txt'])
            and len(glob.glob(os.path.join(d, 'splits', 'test.split*.bundle'))) >= 5)


def find_or_download(data_dir, search_roots=(), download_dir='data_raw/mstcn', allow_download=True):
    if data_dir:
        assert is_50salads(data_dir), f'Not a 50salads folder in MS-TCN format: {data_dir}'
        return data_dir
    cands = []
    for root in list(search_roots) + [download_dir]:
        if root and os.path.exists(root):
            cands += [os.path.dirname(p) for p in glob.glob(os.path.join(root, '**', 'mapping.txt'), recursive=True)]
    cands = [c for c in cands if '50salads' in c.lower() and is_50salads(c)]
    if cands:
        return sorted(cands, key=len)[0]
    if not allow_download:
        raise FileNotFoundError('50salads (MS-TCN format) not found; pass --data-dir')
    print('50salads not found -> downloading only data/50salads/ from data.zip on Zenodo (record 3625992)')
    try:
        from remotezip import RemoteZip
    except ImportError:
        from ..utils import pip_install
        pip_install('remotezip')
        from remotezip import RemoteZip
    t0 = time.time()
    with RemoteZip(ZENODO_URL) as z:
        names = [i for i in z.infolist() if i.filename.startswith('data/50salads/') and not i.is_dir()]
        print(f'{len(names)} files, {sum(i.file_size for i in names) / 1e9:.2f} GB')
        for k, i in enumerate(names):
            z.extract(i, download_dir)
            if k % 20 == 0:
                print(f'  {k}/{len(names)}  ({(time.time() - t0) / 60:.1f} min)')
    return os.path.join(download_dir, 'data', '50salads')


def _lines(p):
    return open(p).read().split('\n')[:-1]          # same reading as batch_gen.py / eval.py


class Salads:
    """Loads all videos into memory (features at the training sample rate, labels at full rate)."""

    def __init__(self, root, strict=True):
        self.root = root
        actions = [l.split() for l in _lines(os.path.join(root, 'mapping.txt')) if l.strip()]
        self.a2i = {a[1]: int(a[0]) for a in actions}
        self.K = len(self.a2i)
        self.split_train = {s: [v for v in _lines(os.path.join(root, 'splits', f'train.split{s}.bundle')) if v.strip()] for s in SPLITS}
        self.split_test = {s: [v for v in _lines(os.path.join(root, 'splits', f'test.split{s}.bundle')) if v.strip()] for s in SPLITS}
        self.vids = sorted(set(sum(self.split_train.values(), []) + sum(self.split_test.values(), [])))
        self.sr = MSTCN['sample_rate']
        self.feat, self.gt, self.tgt = {}, {}, {}
        t0 = time.time()
        for v in self.vids:
            f = np.load(os.path.join(root, 'features', v.split('.')[0] + '.npy'))        # [D, T]
            gt = np.array([self.a2i[c] for c in _lines(os.path.join(root, 'groundTruth', v))], dtype=np.int64)
            self.gt[v] = gt                                                              # full rate (for scoring)
            self.feat[v] = torch.from_numpy(np.ascontiguousarray(f[:, ::self.sr], dtype=np.float32))
            n_tr = min(f.shape[1], len(gt))                                              # as in batch_gen.py
            self.tgt[v] = torch.from_numpy(gt[:n_tr][::self.sr].copy())
        self.feat_dim = next(iter(self.feat.values())).shape[0]
        print(f'{len(self.vids)} videos, K = {self.K} classes, {self.feat_dim}-d features, loaded in {time.time() - t0:.0f}s')
        for s in SPLITS:
            assert not set(self.split_train[s]) & set(self.split_test[s]), f'split {s}: train and test overlap'
        if strict:
            assert self.feat_dim == 2048 and self.K == 19 and len(self.vids) == 50, \
                'Not the MS-TCN 50 Salads release (2048-d, 19 classes, 50 videos)'

    def class_props(self, vids, floor):
        """Frame share of every class, averaged over videos (same definition as pihat in the main experiments)."""
        p = np.mean([np.bincount(self.gt[v], minlength=self.K) / len(self.gt[v]) for v in vids], 0)
        p = np.maximum(p, floor)
        return p / p.sum()
