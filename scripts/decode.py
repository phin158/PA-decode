"""Step 2 - decode every video with the full grid (ASOT for every lambda, PA-decode, PA-integral, PA-minmax and the
label-using oracles). Segmentations are cached in <cache-dir>/decode. Runs the solver self-checks first.

    python scripts/decode.py --datasets Breakfast
"""
import argparse

import _path  # noqa: F401
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.pipeline import run_until


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    exp = experiment_from_args(p.parse_args())
    run_until(exp, 'decode')


if __name__ == '__main__':
    main()
