#!/bin/bash
# Supervised experiment: MS-TCN++ (MS / SS) on 50 Salads, 5 splits x 3 seeds, + ASOT / PA-decode post-processing.
# Usage: bash run_supervised.sh [--data-dir /path/to/data/50salads]   (downloads 50salads from Zenodo if omitted)
set -e
python scripts/run_supervised.py --cache-dir cache_supervised --out-dir results/supervised "$@"
