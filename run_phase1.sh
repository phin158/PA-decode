#!/bin/bash
# Phase 1: reproduce v6 from this repository, add the Per-Video protocol and save per-video statistics for CAM-UOT.
# Nothing is trained or decoded again: the models and decoding files of the Kaggle v6 output must be in $CACHE.
# Usage:  V6_JSON=/path/pa_asot_v6_results.json bash run_phase1.sh
set -e
DATASETS=${DATASETS:-FSeval,FS,desktop_assembly,Breakfast,YTI}
CACHE=${CACHE:-cache}
OUT=${OUT:-results}
DATA=${DATA:-data}
ARGS="--datasets $DATASETS --data-dir $DATA --cache-dir $CACHE --out-dir $OUT"

python scripts/check_cache.py        $ARGS "$@"      # step 1: stop if a model or a decoding file is missing
python scripts/evaluate.py           $ARGS "$@"      # step 2: v6 tables from the cached decoding
python scripts/compare_v6.py --v6 "$V6_JSON" --new $OUT/results.json   # step 3: criterion R1
python scripts/evaluate_pervideo.py  $ARGS "$@"      # step 4: Per-Video protocol (CPU)
python scripts/dump_video_stats.py   $ARGS "$@"      # step 5: per-video statistics + re-solve check R3 (GPU)
