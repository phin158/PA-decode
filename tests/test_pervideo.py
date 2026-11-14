"""The Per-Video protocol equals the authors' indep_eval_metrics (MoF, mIoU exactly; F1 = expectation of their
Monte-Carlo F1). No dataset needed."""
import numpy as np
import torch

import _path  # noqa: F401
from pa_decode.asot_bridge import asot  # noqa: F401  (puts the authors' src/ on sys.path)
from pa_decode.metrics import conf_mat, seg_table
from pa_decode.pervideo import video_scores
from metrics import indep_eval_metrics  # authors' code


def _toy(rng, n_videos=6, K=5, C=5, exclude=False):
    gts, preds, keep = [], [], []
    for _ in range(n_videos):
        n = int(rng.integers(40, 90))
        g = np.repeat(rng.permutation(C), np.diff(np.r_[0, np.sort(rng.choice(np.arange(1, n), C - 1, replace=False)), n]))
        if exclude:
            g = np.where(rng.random(n) < 0.2, -1, g)
        p = np.where(rng.random(n) < 0.7, np.where(g >= 0, g, 0), rng.integers(0, K, n))
        gts.append(g)
        preds.append(p.astype(np.int8))
        keep.append(g >= 0)
    offs = np.r_[0, np.cumsum([len(g) for g in gts])]
    segs = np.concatenate(preds)[None]
    conf = np.stack([conf_mat(p, g, K, C) for p, g in zip(preds, gts)])[None]
    info = dict(gt=gts, keep=keep)
    run = dict(segs=segs, conf=conf, offs=offs)
    return info, run, preds, gts


def test_pervideo_matches_authors():
    rng = np.random.default_rng(0)
    for exclude in (False, True):
        info, run, preds, gts = _toy(rng, exclude=exclude)
        S = video_scores(run, seg_table(info, run))[0]
        for v, (p, g) in enumerate(zip(preds, gts)):
            ex = -1 if exclude else None
            mask = torch.ones(1, len(p), dtype=bool)
            ref = indep_eval_metrics([torch.tensor(p.astype(np.int64))], [torch.tensor(g)], mask, ['mof', 'miou'],
                                     exclude_cls=ex)
            assert abs(S[v, 0] - 100 * ref['mof']) < 1e-9 and abs(S[v, 2] - 100 * ref['miou']) < 1e-9
            np.random.seed(1)
            mc = indep_eval_metrics([torch.tensor(p.astype(np.int64))], [torch.tensor(g)], mask, ['mof', 'f1'],
                                    exclude_cls=ex)['f1']
            assert abs(S[v, 1] - 100 * mc) < 3.0          # Monte-Carlo noise of the authors' F1 (50 draws)
