"""Step 5 - figures of the paper: qualitative segmentation bars, cost / transport-plan matrices and sensitivity to
lambda (the dose-response figure is produced by analyze_mechanism.py).

    python scripts/make_figures.py
"""
import argparse

import _path  # noqa: F401
from pa_decode import figures
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.pipeline import run_until


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    exp = experiment_from_args(p.parse_args())
    run_until(exp, 'point', checks=False)
    rec = figures.collect_qualitative(exp)
    figures.save_qualitative(rec, exp.out('figures/qual_seqs.npz'))
    figures.draw_qual(rec, exp.out('figures/qual_bars'))
    figures.draw_matrices(exp, rec, exp.out('figures/matrices'))
    sens = figures.sensitivity_table(exp)
    sens.to_csv(exp.out('sensitivity_lambda.csv'), index=False)
    figures.draw_sensitivity(exp, sens, exp.out('figures/sensitivity_lambda'))


if __name__ == '__main__':
    main()
