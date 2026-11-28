"""Step 4 - mechanism analysis: data characteristics, decomposition of the gain (lambda part vs target part),
permutation of pihat and dose-response towards the true proportions (re-solves; GPU recommended).

    python scripts/analyze_mechanism.py
"""
import argparse

import _path  # noqa: F401
from pa_decode import figures
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.mechanism import run_mechanism
from pa_decode.pipeline import run_until


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    exp = experiment_from_args(p.parse_args())
    run_until(exp, 'point', checks=False)
    feat, sel, inter, tab, corr, V = run_mechanism(exp)
    figures.draw_dose(exp, inter, exp.out('figures/dose'))
    figures.draw_correlations(exp, feat, exp.out('figures/correlations'))


if __name__ == '__main__':
    main()
