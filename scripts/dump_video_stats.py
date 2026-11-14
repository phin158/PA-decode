"""Phase 1 - save per-video statistics for CAM-UOT (local proportion estimates from the codes and from the ASOT plan,
entropy, disagreement, true proportions for diagnostics) and re-solve the picked configurations to check that the solver
reproduces the cached segmentation on this GPU (criterion R3). GPU recommended.

    python scripts/dump_video_stats.py                 # -> <cache-dir>/videostats/*.npz, <out-dir>/resolve_check_*.csv
    python scripts/dump_video_stats.py --save-plans    # also store the ASOT plans (float16; large for Breakfast)
"""
import argparse

import _path  # noqa: F401
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.pipeline import run_until
from pa_decode.videostats import dump_all


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--save-plans', action='store_true', help='store the ASOT transport plans (float16)')
    p.add_argument('--no-check', action='store_true', help='skip the re-solve check')
    args = p.parse_args()
    exp = experiment_from_args(args)
    run_until(exp, 'point', checks=False)
    dump_all(exp, save_plans=args.save_plans, check=not args.no_check)


if __name__ == '__main__':
    main()
