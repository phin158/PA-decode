# Phase 1: reproduction, Per-Video protocol, per-video statistics

Fixed on 2026-11-14, **before** running on the real data. The constants are in `src/pa_decode/config.py`
(`PV_*`, `R1_TOL`, `R3_AGREE`).

## What is run

| Step | Script | Device | Output |
|---|---|---|---|
| 1 | `check_cache.py` | CPU | stops if a v6 model / decoding file is not found under the expected name |
| 2 | `evaluate.py` | CPU | `results/results.json` and all v6 tables, from the cached decoding |
| 3 | `compare_v6.py` | CPU | `results/compare_v6.csv` (criterion R1) |
| 4 | `evaluate_pervideo.py` | CPU | `results/pervideo_*.csv`, `results/pervideo_results.json` |
| 5 | `dump_video_stats.py` | GPU | `cache/videostats/*.npz`, `results/resolve_check_*.csv` (criterion R3) |

## Criteria

- **R1 (the repository reproduces v6).** With the v6 models and decoding files, every score (A1, A2, per activity),
  difference, bootstrap interval, per-video-`w` bound, selected configuration and verdict in `results.json` equals
  `pa_asot_v6_results.json` (absolute difference ≤ 1e-6).
- **R2 (Per-Video protocol is the authors' one).** For ASOT (λ = 0.01), our Per-Video MoF and mIoU equal
  `test_mof_per` / `test_miou_per` logged by the authors' `trainer.test` for every model (difference < 0.01 points).
- **R3 (the solver reproduces the cached segmentation on real data).** The configurations picked for ASOT, ASOT-λ
  and PA-decode are solved again; the frames agree with the cached segmentation on ≥ 99.9 % for every
  (dataset, seed, configuration), and the proportion estimates π̂ recomputed from the per-video statistics equal the
  cached ones.
- **Per-Video results.** No new selection: the configurations chosen by the 2-fold selection under the Full protocol
  are reused. Reported for ASOT, ASOT-λ and PA-decode with the same statistics as Table IV (bootstrap with the same
  resampled sets, t-interval over seeds, Wilcoxon, Holm). The G1 definition is applied **only descriptively**; the
  main claim of the paper stays on the Full protocol. Negative results are reported as they are.

If R1, R2 or R3 is not met, the cause is found and reported before Phase 2 starts.

## Per-video statistics (`cache/videostats/<dataset>_<activity>_s<seed>.npz`)

Without labels: `codes_mean`, `plan_mass` (ASOT plan, λ = 0.01, uniform target), `entropy` (mean normalised entropy of the
codes), `js_cp` (Jensen–Shannon divergence between the two estimates), `plan_nseg`, `n_frames`, `fold`, and the global
`pihat_codes`, `pihat_plan`. With labels (`gt_*`, diagnostics only): `gt_share`, `gt_share_k` (mapped to clusters by the
Hungarian match of ASOT), `gt_nseg`. With `--save-plans`: the ASOT plans (float16) and `plan_offs`.
