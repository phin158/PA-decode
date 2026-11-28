"""Metrics of MS-TCN eval.py (Acc, Edit, F1@{10,25,50}), vectorised and checked against a verbatim copy."""
import numpy as np

from .config import OVERLAPS

BG = set()      # 50 Salads has no background class -> no segment is removed (as in eval.py)


def segments(lab):
    """= get_labels_start_end_time of eval.py (same convention: the last segment ends at index len-1)."""
    lab = np.asarray(lab)
    n = len(lab)
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate([[0], ch])
    ends = np.concatenate([ch, [n - 1]])
    labels = lab[starts]
    keep = np.array([l not in BG for l in labels], dtype=bool)
    return labels[keep], starts[keep], ends[keep]


def levenstein_norm(p, y):
    """= levenstein(p, y, norm=True) of eval.py; loop over y, vectorised over p (cumulative minimum)."""
    m, n = len(p), len(y)
    if max(m, n) == 0:
        return 100.
    prev = np.arange(m + 1, dtype=np.float64)
    idx = np.arange(m + 1, dtype=np.float64)
    for j in range(1, n + 1):
        x = np.empty(m + 1)
        x[0] = j
        sub = prev[:-1] + (p != y[j - 1])
        x[1:] = np.minimum(prev[1:] + 1, sub)
        prev = np.minimum.accumulate(x - idx) + idx           # D[i,j] = min_{k<=i} (x[k] + i - k)
    return (1 - prev[-1] / max(m, n)) * 100


def f_counts(pl, ps, pe, yl, ys, ye, overlap):
    """= f_score of eval.py."""
    tp = fp = 0
    hits = np.zeros(len(yl))
    for j in range(len(pl)):
        inter = np.minimum(pe[j], ye) - np.maximum(ps[j], ys)
        union = np.maximum(pe[j], ye) - np.minimum(ps[j], ys)
        iou = (1.0 * inter / union) * (yl == pl[j])
        idx = int(np.array(iou).argmax())
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    return float(tp), float(fp), float(len(yl) - hits.sum())


def video_stats(pred_full, gt):
    n = min(len(pred_full), len(gt))
    pr, g = np.asarray(pred_full)[:n], gt[:n]
    P, Y = segments(pr), segments(g)
    r = dict(correct=int((pr == g).sum()), total=int(n), edit=levenstein_norm(P[0], Y[0]))
    for o in OVERLAPS:
        tp, fp, fn = f_counts(*P, *Y, o)
        r[f'tp{o}'], r[f'fp{o}'], r[f'fn{o}'] = tp, fp, fn
    return r


def agg_metrics(df):
    out = {'Acc': 100. * df['correct'].sum() / df['total'].sum(), 'Edit': df['edit'].mean()}
    for o, nm in zip(OVERLAPS, ['F1@10', 'F1@25', 'F1@50']):
        tp, fp, fn = df[f'tp{o}'].sum(), df[f'fp{o}'].sum(), df[f'fn{o}'].sum()
        pr = tp / max(tp + fp, 1e-12)
        rc = tp / max(tp + fn, 1e-12)
        out[nm] = 100. * np.nan_to_num(2 * pr * rc / max(pr + rc, 1e-12))
    return out


# ---- verbatim (pure Python) copy of eval.py, used only for the self-check ----
def _ref_segments(fl, bg):
    labels, starts, ends = [], [], []
    last = fl[0]
    if fl[0] not in bg:
        labels.append(fl[0])
        starts.append(0)
    for i in range(len(fl)):
        if fl[i] != last:
            if fl[i] not in bg:
                labels.append(fl[i])
                starts.append(i)
            if last not in bg:
                ends.append(i)
            last = fl[i]
    if last not in bg:
        ends.append(i)
    return labels, starts, ends


def _ref_lev(p, y):
    D = np.zeros([len(p) + 1, len(y) + 1])
    D[:, 0] = np.arange(len(p) + 1)
    D[0, :] = np.arange(len(y) + 1)
    for j in range(1, len(y) + 1):
        for i in range(1, len(p) + 1):
            D[i, j] = D[i - 1, j - 1] if y[j - 1] == p[i - 1] else min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    return (1 - D[-1, -1] / max(len(p), len(y))) * 100


def self_check(n=300, seed=0):
    r = np.random.default_rng(seed)
    for _ in range(n):
        g = np.repeat(r.integers(0, 6, r.integers(1, 15)), r.integers(1, 30, 1))
        p = np.repeat(r.integers(0, 6, r.integers(1, 25)), r.integers(1, 20, 1))
        L_ = min(len(g), len(p))
        g, p = g[:L_], p[:L_]
        a, b = segments(p), _ref_segments(list(p), set())
        assert list(a[0]) == b[0] and list(a[1]) == b[1] and list(a[2]) == b[2]
        assert abs(levenstein_norm(segments(p)[0], segments(g)[0]) - _ref_lev(b[0], _ref_segments(list(g), set())[0])) < 1e-9
    print(f'OK: segments + Edit identical to eval.py on {n} random sequences')
