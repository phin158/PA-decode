"""Step 1 - train ASOT (authors' code and hyper-parameters) for every dataset / activity / seed.
Models are cached in <cache-dir>/models; already trained models are skipped.

    python scripts/train_asot.py --datasets Breakfast --seeds 0,1,2
"""
import argparse

import _path  # noqa: F401
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.pipeline import run_until


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    exp = experiment_from_args(p.parse_args())
    run_until(exp, 'train')


if __name__ == '__main__':
    main()
