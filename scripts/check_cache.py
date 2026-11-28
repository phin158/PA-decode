"""Phase 1 - before running anything: check that the trained models and the decoding files copied from the Kaggle v6
output are found under the names this repository expects. A missing decoding file would be decoded again from
scratch (several hours on a T4), so run this first.

    python scripts/check_cache.py --cache-dir cache
"""
import argparse
import os
import sys

import _path  # noqa: F401
from pa_decode.data import build_units
from pa_decode.decode import decode_path
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.train import model_path


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    exp = experiment_from_args(p.parse_args())
    build_units(exp, verbose=False)
    miss_m, miss_d, n = [], [], 0
    for ds in exp.datasets:
        for seed in exp.seeds[ds]:
            for a in exp.units_of(ds):
                n += 1
                if not os.path.exists(model_path(exp, ds, a, seed)):
                    miss_m.append(os.path.basename(model_path(exp, ds, a, seed)))
                if not os.path.exists(decode_path(exp, ds, a, seed)):
                    miss_d.append(os.path.basename(decode_path(exp, ds, a, seed)))
        nd = len(exp.units_of(ds)) * len(exp.seeds[ds])
        print(f'{exp.label(ds):22s} expected {nd:3d} | models missing {sum(m.startswith(ds + "_") for m in miss_m):3d} '
              f'| decoding files missing {sum(d.startswith(ds + "_") for d in miss_d):3d}')
    print(f'\nTotal: {n} (dataset, activity, seed). Models missing: {len(miss_m)}. Decoding files missing: {len(miss_d)}.')
    if miss_m or miss_d:
        print('First missing files:', (miss_m + miss_d)[:6])
        dec_dir = os.path.join(exp.cache_dir, 'decode')
        have = sorted(os.listdir(dec_dir))[:6] if os.path.isdir(dec_dir) else []
        print('Files present in', dec_dir, ':', have)
        print('If the names differ only in the last 8 characters (hash), the configuration differs from v6: do NOT run '
              'decode.py; send the two lists above.')
        sys.exit(1)
    print('OK: all models and decoding files are in place; nothing will be trained or decoded again.')


if __name__ == '__main__':
    main()
