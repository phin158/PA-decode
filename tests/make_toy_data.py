"""Create a tiny synthetic copy of the five settings in the layout of the ASOT loader, for smoke tests only.

    python tests/make_toy_data.py --out toy_data
    python scripts/train_asot.py --data-dir toy_data --cache-dir toy_cache --out-dir toy_results --quick
"""
import argparse
import os

import numpy as np

import _path  # noqa: F401
from pa_decode.config import PAPER


def video(rng, n_cls, dim, n, proto):
    props = rng.dirichlet(np.full(n_cls, 1.0))
    order = rng.permutation(n_cls)
    g = np.repeat(order, np.maximum(1, (props[order] * n).astype(int)))[:n]
    g = np.concatenate([g, np.full(n - len(g), g[-1])])
    x = proto[g] + 0.8 * rng.standard_normal((n, dim))
    return g, x.astype(np.float32)


def write(root, ds, acts, dim, names_fn, n_videos, rng, bg=False, mapping_extra=None):
    d = os.path.join(root, ds)
    os.makedirs(os.path.join(d, 'groundTruth'), exist_ok=True)
    os.makedirs(os.path.join(d, 'mapping'), exist_ok=True)
    labels = (['bg'] if bg else []) + [f'{a}_step{c}' for a, K in acts.items() for c in range(K)]
    with open(os.path.join(d, 'mapping', 'mapping.txt'), 'w') as f:
        for i, l in enumerate(labels):
            f.write(f'{i - 1 if bg else i} {l}\n')
    for extra in (mapping_extra or []):
        with open(os.path.join(d, 'mapping', extra), 'w') as f:
            for i, l in enumerate(labels):
                f.write(f'{i} {l}\n')
    for a, K in acts.items():
        proto = 3 * rng.standard_normal((K, dim))
        for v in range(n_videos):
            name = names_fn(a, v)
            g, x = video(rng, K, dim, int(rng.integers(60, 140)), proto)
            lab = [f'{a}_step{c}' for c in g]
            if bg:
                lab = ['bg' if rng.random() < 0.3 else l for l in lab]
            fdir = os.path.join(d, 'features', a if ds in ('Breakfast', 'YTI') else '')
            os.makedirs(fdir, exist_ok=True)
            np.save(os.path.join(fdir, name + '.npy'), x)
            with open(os.path.join(d, 'groundTruth', name), 'w') as f:
                f.write('\n'.join(lab) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='toy_data')
    p.add_argument('--videos', type=int, default=12)   # >= 10: the authors' validation_step needs >= 5 batches
    a = p.parse_args()
    rng = np.random.default_rng(0)
    nv = a.videos
    # 50 Salads: 'eval' (12 classes) and 'mid' (19) granularities share the same videos -> use 12 classes for both
    write(a.out, 'FS', {'': 12}, 64, lambda _, v: f'rgb-{v // 2 + 1:02d}-{v % 2 + 1}', nv, rng, mapping_extra=['mappingeval.txt'])
    write(a.out, 'desktop_assembly', {'': 22}, 512, lambda _, v: f'{v // 2 + 1}_{v % 2 + 1}', nv, rng)
    write(a.out, 'Breakfast', {k: v for k, v in PAPER['Breakfast']['acts'].items()}, 64,
          lambda act, v: f'P{v + 3:02d}_cam01_P{v + 3:02d}_{act}', nv, rng)
    write(a.out, 'YTI', {k: v for k, v in PAPER['YTI']['acts'].items()}, 3000, lambda act, v: f'{act}_{v + 1:04d}', nv, rng, bg=True)
    print('toy data written to', a.out)


if __name__ == '__main__':
    main()
