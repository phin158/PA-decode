"""Experiment E6 (docs/preregistration_extra.md): PA-decode on the PEOT backbone (Li, Vu & Gall, ECCV 2026).

Run after training, decoding and evaluation with ``--backbone peot`` (see run_peot.sh):
  1. cache check: the authors' own evaluation at the last epoch (printed by PEOT/train.py, Full protocol per activity)
     must equal our decoding of the same model with the authors' test setting (lambda = 0.01, uniform target);
  2. summary table PEOT / PEOT-lambda / PA-decode (mean over seeds) and the E6 verdict:
     G1 of PA-decode - PEOT-lambda on >= 2 of the 4 settings.
In the tables of evaluate.py the methods keep their generic names: "ASOT" = authors' decoding of the trained model
(here PEOT), "ASOT-lambda" = PEOT-lambda.

    python scripts/run_peot.py --backbone peot --datasets Breakfast,YTI,FSeval,desktop_assembly \
        --data-dir data --cache-dir cache_peot --out-dir results_peot
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _path  # noqa: F401
from pa_decode.decode import IDX_PAPER
from pa_decode.experiment import add_common_args, experiment_from_args
from pa_decode.metrics import unit_eval
from pa_decode.pipeline import run_until


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = p.parse_args()
    assert a.backbone == 'peot', 'run with --backbone peot'
    exp = experiment_from_args(a)
    run_until(exp, 'point', checks=False)
    out = exp.out('peot')
    os.makedirs(out, exist_ok=True)

    rows = []
    for (ds, act, s), r in exp.trained.items():
        run, info = exp.dec[(ds, act, s)], exp.uinfo[(ds, act)]
        nv = len(info['names'])
        v, _ = unit_eval(run, exp.stab[(ds, act, s)], np.full(nv, IDX_PAPER), np.ones(nv), info)
        rows.append(dict(ds=ds, act=act, seed=s, authors_MoF=100 * r['test']['test_mof_full'], ours_MoF=float(v[0]),
                         authors_F1=100 * r['test']['test_f1_full'], ours_F1=float(v[1]),
                         train_minutes=r.get('minutes')))
    chk = pd.DataFrame(rows)
    chk['abs_diff_MoF'] = (chk.authors_MoF - chk.ours_MoF).abs()
    chk.to_csv(os.path.join(out, 'cache_check_vs_authors_eval.csv'), index=False)

    res = json.load(open(exp.out('results.json'), encoding='utf-8'))
    per = pd.read_csv(exp.out('per_seed.csv'))
    cols = {'ASOT': 'PEOT', 'ASOT-lambda': 'PEOT-lambda', 'PA': 'PA-decode'}
    tab = []
    for ds_label, q in per.groupby('dataset', sort=False):
        r = dict(dataset=ds_label, n_seeds=len(q))
        for c, name in cols.items():
            for m in ['MoF', 'F1', 'mIoU']:
                r[f'{name} {m}'] = q[f'{c} {m}'].mean()
        tab.append(r)
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(out, 'E6_summary.csv'), index=False)
    g1 = {ds: bool(res['verdict'][ds]['PA-decode']['G1']) for ds in exp.datasets}
    log = dict(G1_PA_vs_PEOT_lambda=g1, n_met=sum(g1.values()), E6_met=sum(g1.values()) >= 2,
               cache_check=dict(max_abs_diff_MoF=float(chk.abs_diff_MoF.max()),
                                mean_abs_diff_MoF=float(chk.abs_diff_MoF.mean()), n_units=len(chk)),
               train_hours=float(chk.train_minutes.sum() / 60))
    with open(os.path.join(out, 'E6_results.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=1)
    pd.set_option('display.width', 200)
    print(tab.round(2).to_string(index=False))
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
