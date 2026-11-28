#!/bin/bash
# Additional experiments pre-registered in docs/preregistration_extra.md (run after run_unsupervised.sh).
# Usage: bash run_extra.sh E1,E2,E3,E4      # analyses from the cache of the main experiment
#        bash run_extra.sh E6               # PEOT backbone (needs internet once to clone PEOT)
set -e
PARTS=${1:-E1,E2,E3,E4}
ARGS="--data-dir data --cache-dir cache"
if [[ $PARTS == *E1* || $PARTS == *E2* || $PARTS == *E3* || $PARTS == *E4* ]]; then
    python scripts/run_extra_analyses.py $ARGS --out-dir results --parts $(echo $PARTS | tr ',' '\n' | grep -E '^E[1-4]$' | paste -sd,)
fi
if [[ $PARTS == *E6* ]]; then
    P="--backbone peot --datasets FSeval,desktop_assembly,YTI,Breakfast --data-dir data --cache-dir cache_peot --out-dir results_peot"
    python scripts/train_asot.py $P
    python scripts/decode.py     $P
    python scripts/evaluate.py   $P
    python scripts/run_peot.py   $P
fi
