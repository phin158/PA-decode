"""Step 3 - 2-fold selection, metrics, bootstrap confidence intervals, per-seed / per-activity tables, the per-video-w
upper bound and the pre-registered verdicts. Writes CSV tables and results.json to <out-dir>.

    python scripts/evaluate.py
"""
import argparse

import _path  # noqa: F401
from pa_decode import report
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.pipeline import run_until
from pa_decode.selection import run_bootstrap
from pa_decode.wvideo import per_video_w_table


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--skip-wvideo', action='store_true', help='skip the per-video w upper bound (slow on Breakfast)')
    p.add_argument('--no-checks', action='store_true', help='skip the comparisons with the authors\' metric code')
    args = p.parse_args()
    exp = experiment_from_args(args)
    run_until(exp, 'point', checks=not args.no_checks)
    repro = report.reproduction_table(exp)
    run_bootstrap(exp)
    report.main_tables(exp)
    report.significance_table(exp)
    report.per_activity_table(exp)
    report.class_diag_table(exp)
    wdf, dval = (None, None) if args.skip_wvideo else per_video_w_table(exp)
    report.per_seed_tables(exp, dval)
    verdict, summary, pick, d_verdict, lines = report.verdicts(exp, dval)
    report.save_results(exp, verdict, summary, pick, d_verdict, dval, repro, wdf, lines)


if __name__ == '__main__':
    main()
