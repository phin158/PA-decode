"""Supervised experiment: MS-TCN++ (MS: 3 refinement stages, SS: none) on the 5 splits of 50 Salads x seeds, then ASOT
and PA-decode as post-processing (Table 3 of the ASOT paper + PA-decode). Training runs are cached in <cache-dir>/runs.

    python scripts/run_supervised.py --data-dir /path/to/data/50salads
"""
import argparse
import os
import time

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch

from pa_decode.asot_bridge import ASOT_HEAD
from pa_decode.supervised import config as scfg
from pa_decode.supervised.data import Salads, find_or_download
from pa_decode.supervised.metrics import agg_metrics, self_check
from pa_decode.supervised.postprocess import postprocess_run
from pa_decode.supervised.report import make_tables


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default=None, help='50salads folder in MS-TCN format (downloaded from Zenodo if omitted)')
    p.add_argument('--search-root', action='append', default=[], help='folders searched for an existing 50salads copy')
    p.add_argument('--download-dir', default='data_raw/mstcn')
    p.add_argument('--cache-dir', default='cache_supervised')
    p.add_argument('--out-dir', default='results/supervised')
    p.add_argument('--seeds', default=','.join(map(str, scfg.SEEDS)))
    p.add_argument('--epochs', type=int, default=scfg.MSTCN['epochs'])
    p.add_argument('--device', default=None)
    p.add_argument('--no-strict', action='store_true', help='do not require the 2048-d / 19-class / 50-video release (tests)')
    a = p.parse_args()
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seeds = [int(s) for s in a.seeds.split(',')]
    print('torch', torch.__version__, '| numpy', np.__version__, '| device', device, '| seeds', seeds, '| epochs', a.epochs)
    self_check()
    root = find_or_download(a.data_dir, a.search_root, a.download_dir)
    D = Salads(root, strict=not a.no_strict)
    rows = []
    t_all = time.time()
    for base in scfg.BASES:
        for split in scfg.SPLITS:
            for seed in seeds:
                t0 = time.time()
                df = postprocess_run(D, base, split, seed, a.cache_dir, device, a.epochs)
                rows.append(df)
                m0 = agg_metrics(df[df.target == 'none'])
                m1 = agg_metrics(df[(df.target == 'uniform') & np.isclose(df.lam, scfg.LAM_PAPER)])
                m2 = agg_metrics(df[(df.target == 'train_w1') & np.isclose(df.lam, scfg.LAM_PAPER)])
                print(f'{base} split{split} seed{seed}: Acc/F1@50  plain {m0["Acc"]:.1f}/{m0["F1@50"]:.1f} | '
                      f'+ASOT {m1["Acc"]:.1f}/{m1["F1@50"]:.1f} | +PA {m2["Acc"]:.1f}/{m2["F1@50"]:.1f}   '
                      f'({(time.time() - t0) / 60:.1f} min, total {(time.time() - t_all) / 60:.0f} min)')
    R = pd.concat(rows, ignore_index=True)
    os.makedirs(a.out_dir, exist_ok=True)
    R.to_csv(os.path.join(a.out_dir, 'supervised_per_video.csv'), index=False)
    make_tables(R, a.out_dir, seeds, extra=dict(asot_commit=ASOT_HEAD, data=root, K=D.K, epochs=a.epochs))


if __name__ == '__main__':
    main()
