"""Unsupervised evaluation metrics, identical to the authors' ``metrics.py`` (Hungarian matching per activity,
MoF, mIoU, F1). F1 in the authors' code is a Monte-Carlo estimate (15 frames sampled with replacement per segment,
a segment is a TP if > 50% of them are correct); here we compute its exact expectation.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom

P_TP = lambda p: binom.sf(7, 15, p)        # P(segment is TP) given the fraction p of correct frames


def conf_mat(pred, gt, K, C):
    """gt holds local labels of the unit; frames with gt = -1 (excluded) are ignored, as in filter_exclusions."""
    k = gt >= 0
    return np.bincount(pred[k].astype(np.int64) * C + gt[k], minlength=K * C).reshape(K, C)


def hungarian(M):
    rows, cols = np.where(M.sum(1) > 0)[0], np.where(M.sum(0) > 0)[0]
    r, c = linear_sum_assignment(M[np.ix_(rows, cols)], maximize=True)
    return rows[r], cols[c]


def hungarian_mof(M):
    pr, gl = hungarian(M)
    return M[pr, gl].sum() / M.sum()


def seg_table(info, run):
    """GT segments of every video after removing excluded frames (as before eval_f1); segment [lo, up] includes up.
    For every decoding configuration, counts how many frames of each segment are assigned to each cluster."""
    offs, segs = run['offs'], run['segs']
    K = run['conf'].shape[2]
    sv, sg, sl, cnt = [], [], [], []
    for vi, (g, k) in enumerate(zip(info['gt'], info['keep'])):
        g = g[k]
        N = len(g)
        if N == 0:
            continue
        b = np.concatenate([[0], np.where(g[1:] != g[:-1])[0] + 1, [N - 1]])
        lo, up = b[:-1], b[1:]
        P = segs[:, offs[vi]:offs[vi + 1]][:, k].astype(np.int64)
        oh = np.zeros((P.shape[0], N + 1, K), dtype=np.int32)
        np.put_along_axis(oh[:, 1:], P[..., None], 1, axis=2)
        cs = oh.cumsum(1)
        cnt.append(cs[:, up + 1] - cs[:, lo])
        sv.append(np.full(len(lo), vi))
        sg.append(g[lo])
        sl.append(up - lo + 1)
    return dict(vid=np.concatenate(sv), gt=np.concatenate(sg), len=np.concatenate(sl).astype(float),
                cnt=np.concatenate(cnt, axis=1))


def scores(M, cnt, sgt, slen, sw, n_videos, with_map=False):
    """MoF, F1 (per-video segment approximation) and mIoU, in %, from a confusion matrix M [K, C]."""
    C = M.shape[1]
    pr, gl = hungarian(M)
    tp = M[pr, gl].astype(float)
    n_act = int((M.sum(0) > 0).sum())
    mof = tp.sum() / M.sum()
    miou = (tp / (M[pr].sum(1) + M[:, gl].sum(0) - tp)).sum() / n_act
    mapk = np.full(C, -1)
    mapk[gl] = pr
    k = mapk[sgt]
    ok = k >= 0
    p = np.zeros(len(sgt))
    p[ok] = cnt[np.where(ok)[0], k[ok]] / slen[ok]
    tp_tot = 50. * (P_TP(p) * ok * sw).sum()
    prec = tp_tot / (n_videos * n_act * 50.)
    rec = tp_tot / (sw.sum() * 50. + 1e-8)
    out = np.array([mof, 2. * prec * rec / (prec + rec + 1e-8), miou]) * 100
    return (out, dict(zip(pr.tolist(), gl.tolist()))) if with_map else out


def f1_exact(run, info, chosen, mapk):
    """EXACT expectation of the authors' eval_f1: all videos are concatenated (original order, excluded frames removed)
    before cutting segments, so the last segment of video i and the first of video i+1 merge as in their code."""
    offs = run['offs']
    G = np.concatenate([g[k] for g, k in zip(info['gt'], info['keep'])])
    P = np.concatenate([run['segs'][chosen[vi], offs[vi]:offs[vi + 1]][k] for vi, k in enumerate(info['keep'])]).astype(np.int64)
    N, K = len(G), run['conf'].shape[2]
    b = np.concatenate([[0], np.where(G[1:] != G[:-1])[0] + 1, [N - 1]])
    lo, up = b[:-1], b[1:]
    oh = np.zeros((N + 1, K), dtype=np.int32)
    oh[np.arange(1, N + 1), P] = 1
    cs = oh.cumsum(0)
    m = mapk[G[lo]]
    ok = m >= 0
    p = np.zeros(len(lo))
    p[ok] = (cs[up[ok] + 1, m[ok]] - cs[lo[ok], m[ok]]) / (up[ok] - lo[ok] + 1)
    tp = 50. * P_TP(p)[ok].sum()
    n_act = len(np.unique(G))
    prec = tp / (len(info['names']) * n_act * 50.)
    rec = tp / (len(lo) * 50. + 1e-8)
    return 100 * 2. * prec * rec / (prec + rec + 1e-8)


def unit_eval(run, st, chosen, wv, info=None):
    """Scores of one unit when video v is decoded with configuration chosen[v] and has weight wv[v].
    info != None (only when all weights are 1): F1 is the exact expectation on the concatenated sequence.
    Otherwise (bootstrap, selection): per-video segment approximation of F1."""
    nv = len(chosen)
    M = np.tensordot(wv, run['conf'][chosen, np.arange(nv)], axes=([0], [0]))
    cnt = st['cnt'][chosen[st['vid']], np.arange(len(st['vid']))]
    v = scores(M, cnt, st['gt'], st['len'], wv[st['vid']], wv.sum())
    if info is not None:
        pr, gl = hungarian(M)
        mapk = np.full(M.shape[1], -1)
        mapk[gl] = pr
        v[1] = f1_exact(run, info, chosen, mapk)
    return v, M.sum()


def aggregate(vals, nfr):
    """A1 = mean over activities; A2 = frame-weighted (MoF pooled over frames)."""
    vals, nfr = np.array(vals), np.array(nfr, float)
    a1 = vals.mean(0)
    a2 = (vals * nfr[:, None]).sum(0) / nfr.sum()
    a2[0] = sum(v[0] * n for v, n in zip(vals, nfr)) / nfr.sum()
    return a1, a2
