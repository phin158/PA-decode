"""Phase 1 - Per-Video protocol (second block of Table 1 of ASOT) for ASOT, ASOT-lambda and PA-decode.

No new selection: the configurations picked by the 2-fold selection of the Full protocol are reused. The Per-Video MoF /
mIoU of ASOT are first checked against the values logged by the authors' trainer.test (test_mof_per, test_miou_per).
CPU only; needs the model and decoding caches.

    python scripts/evaluate_pervideo.py
"""
import argparse

import _path  # noqa: F401
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.pervideo import build_video_scores, pv_point_estimates, check_against_trainer, pv_tables, save_pv_results
from pa_decode.pipeline import run_until


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--n-boot', type=int, default=None, help='bootstrap rounds for every dataset (default: as in the paper)')
    args = p.parse_args()
    exp = experiment_from_args(args)
    run_until(exp, 'point', checks=False)
    build_video_scores(exp)
    check = check_against_trainer(exp)
    pv_point_estimates(exp)
    nb = {ds: args.n_boot for ds in exp.datasets} if args.n_boot else None
    main_, delta, seeds = pv_tables(exp, nb)
    save_pv_results(exp, main_, delta, seeds, check)


if __name__ == '__main__':
    main()
