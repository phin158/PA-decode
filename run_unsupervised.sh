#!/bin/bash
# Full unsupervised pipeline (5 settings x 10 seeds). Every step is cached: re-running skips finished work.
# Usage:  bash run_unsupervised.sh                      # everything
#         bash run_unsupervised.sh --seeds 0,1,2        # extra arguments go to every step after data preparation
#         RAW_ROOT=/path/to/downloads bash run_unsupervised.sh   # use datasets you already downloaded
set -e
DATASETS=${DATASETS:-FSeval,FS,desktop_assembly,Breakfast,YTI}
ARGS="--datasets $DATASETS --data-dir data --cache-dir cache --out-dir results"

python scripts/prepare_data.py --datasets $DATASETS --data-dir data --download-dir data_raw ${RAW_ROOT:+--raw-root $RAW_ROOT}
python scripts/train_asot.py        $ARGS "$@"    # ASOT training with the authors' code
python scripts/decode.py            $ARGS "$@"    # full decoding grid (ASOT-lambda, PA-decode, variants, oracles)
python scripts/evaluate.py          $ARGS "$@"    # main tables, CIs, per-seed results, per-video-w bound, verdicts
python scripts/analyze_mechanism.py $ARGS "$@"    # mechanism table, dose-response figure
python scripts/make_figures.py      $ARGS "$@"    # qualitative bars, plan matrices, sensitivity to lambda
