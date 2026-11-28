# From the Kaggle notebooks to this repository

The experiments were first run as Kaggle notebooks (v6: main results, v7: mechanism analysis, v8: supervised
experiment). This repository contains the same code, split into modules. Decoding, selection, bootstrap and the
mechanism analyses were checked on synthetic data: they give **identical** outputs to the notebook cells.

| Notebook cell | Module / script |
|---|---|
| v6 c01 configuration | `src/pa_decode/config.py`, `src/pa_decode/asot_bridge.py` |
| v6 c02 data | `src/pa_decode/data.py`, `scripts/prepare_data.py` |
| v6 c03 ASOT training | `src/pa_decode/train.py`, `scripts/train_asot.py` |
| v6 c04 solvers + checks | `src/pa_decode/solver.py` |
| v6 c05 decoding grid | `src/pa_decode/decode.py`, `scripts/decode.py` |
| v6 c06 metrics, selection, bootstrap | `src/pa_decode/metrics.py`, `src/pa_decode/selection.py` |
| v6 c07, c08, c10, c11 tables and verdicts | `src/pa_decode/report.py`, `scripts/evaluate.py` |
| v6 c09 per-video w bound | `src/pa_decode/wvideo.py` |
| v7 c12–c14 mechanism | `src/pa_decode/mechanism.py`, `scripts/analyze_mechanism.py` |
| v7 c15–c16 figures | `src/pa_decode/figures.py`, `scripts/make_figures.py` |
| v8 c01–c06 supervised | `src/pa_decode/supervised/`, `scripts/run_supervised.py` |

Differences:

- The notebooks reused models and decoding caches from earlier notebook versions (v4/v5). This reuse logic was
  removed, and the scripts train and decode from scratch.
- Cache file names and hashes are unchanged. Model files (`models/*.pt`) and decoding files (`decode/*.npz`) copied
  from a notebook output folder into `cache/` are therefore picked up directly.
- Method names are in English: `ASOT gốc` → `ASOT`, `ASOT-λ` → `ASOT-lambda`, `PA-tích phân (1b)` → `PA-integral`,
  `PA-minmax (2a)` → `PA-minmax`, `Oracle-tập` → `PA-true-proportions`.
- The paired t-test, Wilcoxon test and Holm correction of Table IV are computed in `report.significance_table`.
  In the notebook workflow, they were computed afterwards from the saved JSON.
