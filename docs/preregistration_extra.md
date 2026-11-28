# Pre-registration of the additional experiments (E1–E4, E6)

Written on 2026-11-28, **before any of these experiments was run**. Commit this file to the repository before running
the notebooks, so that its date is recorded by git. The criteria below must not be changed after the results are seen;
every result is reported whether the criterion is met or not.

E5 (20 more seeds) was planned in a draft of this document and dropped before anything was run, because of the GPU
budget; the numbering of the other parts is kept.

Common definitions (same as the main experiment):

- Settings: Breakfast, YTI, 50 Salads Mid (FS), 50 Salads Eval (FSeval), Desktop Assembly (DA).
- **G1 settings** = the three settings on which PA-decode met criterion G1 in the main experiment: Breakfast, DA, FSeval.
- Full protocol (Hungarian matching over all videos of an activity), mean over activities.
- ASOT = authors' decoding (λ = 0.01, uniform target); ASOT-λ = λ selected by 2-fold CV; PA-decode = (λ, w, source)
  selected by 2-fold CV with labels, exactly as in the main experiment.
- Per-seed CI95 = t interval over the per-seed differences.

## E1 – Segmental metrics (Edit, F1@10/25/50) — from cache

Metrics of MS-TCN `eval.py` (same code as the supervised experiment), computed on the frames with a ground-truth label
(YTI background frames, label −1, are removed before computing segments). Predicted clusters are mapped to classes by
the Full-protocol Hungarian matching; unmatched clusters get a label that never matches.

- Report for ASOT, ASOT-λ and PA-decode, 5 settings, 10 seeds.
- **Criterion E1:** on each G1 setting, mean ΔEdit(PA-decode − ASOT-λ) ≥ −0.5 and mean ΔF1@50 ≥ −0.5.

## E2 – Solver iterations and initialization — from cache

The configurations selected for ASOT-λ and PA-decode are re-solved with: 50 and 100 mirror-descent iterations
(uniform initialization T⁰ = p qᵀ, as in the authors' code), and with initialization T⁰ = p νᵀ (25 and 100 iterations).
The 25-iteration uniform-initialization result is the cached one (checked by re-solving the first unit).

- **Criterion E2:** on each G1 setting and for every solver variant, ΔMoF(PA-decode − ASOT-λ) > 0 with per-seed CI95
  lower bound > 0, and |ΔMoF(variant) − ΔMoF(25 iterations)| < 1.0.

## E3 – Post-hoc proportion adjustment without re-solving — from cache

Baseline "Post-hoc": the ASOT plan T(λ) (uniform target) is reweighted column-wise, label_i = argmax_j T_ij · ν_j / q_j
with ν = (1 − w)q + w π̂ — i.e. the proportions are applied after decoding instead of inside the UOT problem.
Same grid (λ, w, source), same 2-fold selection with labels as PA-decode (w = 0 gives ASOT-λ).

- **Criterion E3 (re-solving matters):** ΔMoF(PA-decode − Post-hoc) > 0 with per-seed CI95 lower bound > 0 on at least
  2 of the 3 G1 settings. Post-hoc − ASOT-λ is also reported.

## E4 – Label-free parameter selection (leave-one-dataset-out) — from cache

For a target setting, one configuration (λ, w, source) is chosen by the mean MoF on the **other** settings (10 seeds),
using no label of the target setting; when the target is FS or FSeval, both 50 Salads settings are excluded (same videos).
The chosen configuration is applied to every video of the target setting.

- Reported: PA-LODO (candidates = PA-decode grid incl. w = 0), ASOT-LODO (λ only), ASOT, ASOT-λ, PA-decode.
- **Criterion E4:** ΔMoF(PA-LODO − ASOT) > 0 with per-seed CI95 lower bound > 0 on at least 2 of the 3 G1 settings.
  (ASOT, λ = 0.01, is the label-free reference.)

## E6 – PA-decode on PEOT (Li, Vu & Gall, ECCV 2026)

PEOT is trained with the authors' code and the commands of their README for probabilistic embeddings, on Breakfast,
YTI, FSeval and DA (PEOT does not report 50 Salads Mid), 10 seeds, using the **last epoch** (no checkpoint selected on
test results). The PEOT cost matrix (1 − features·clustersᵀ + temporal prior) is decoded with the same grid and the
same 2-fold selection as in the main experiment: PEOT (authors' decoding, λ = 0.01), PEOT-λ and PA-decode.

- **Criterion E6:** PA-decode − PEOT-λ meets G1 (as defined in the main experiment) on at least 2 of the 4 settings.

If PEOT cannot be trained within the Kaggle limits, the reason is reported instead of a result.

## Implementation notes (fixed before running, 2026-11-28)

- E2/E3 re-solve with a copy of the mirror-descent solver of `solver.segment_asot_pa`; the copy must reproduce the
  cached 25-iteration segmentation (agreement is written to `extra_results.json`).
- E6, PEOT code: repository derkbreeze/PEOT at commit fbc9c11. `train.py` is run with text patches that only change
  I/O (save the last-epoch checkpoint; train every activity — the script contains debugging filters that train only
  Breakfast/coffee and YTI/changing_tire; read our data folder) and run the per-epoch test evaluation only at the last
  epoch (the DataLoader iterator is still created, so the random stream is unchanged).
- E6, Desktop Assembly: `train.py` uses K = 23 clusters while `test.py` uses 22, which does not match the trained
  codebook; the trained K = 23 is used.
- E6, decoding: the GCN of `test.py` on the chain graph is computed in an equivalent tridiagonal form (checked against
  the authors' dense form); our decoding with λ = 0.01 must reproduce the evaluation printed by `train.py`.
