"""Experiment configuration.

All hyper-parameters of ASOT are taken from the authors' README, ``run_*.sh`` scripts and Appendix B of
Xu & Gould (CVPR 2024). The decoding grid and the decision criteria were fixed before running the experiments.
"""
import copy

ASOT_REPO_URL = 'https://github.com/mingu6/action_seg_ot.git'
ASOT_COMMIT = '0c4c86b1'          # pinned commit of the authors' code

ALL_DATASETS = ['FSeval', 'FS', 'desktop_assembly', 'Breakfast', 'YTI']

LABEL = {'FSeval': '50 Salads Eval', 'FS': '50 Salads Mid', 'desktop_assembly': 'Desktop Assembly',
         'Breakfast': 'Breakfast', 'YTI': 'YouTube Instructions'}

# ---------------------------------------------------------------- ASOT training (per dataset)
# acts: activity -> number of clusters K;  ne: epochs;  rho: temporal prior weight;  lat: lambda_actions_train;
# radius: GW radius r;  alpha_*: FGW mixing weight;  std: standardise features;  layers: MLP sizes;
# exclude: label id ignored at evaluation (YTI background)
PAPER = {
    'FSeval': dict(acts={'all': 12}, ne=30, rho=0.15, lat=0.11, radius=0.04, alpha_train=0.3, alpha_eval=0.6,
                   std=False, layers=[64, 128, 40], exclude=None),
    'FS': dict(acts={'all': 19}, ne=30, rho=0.15, lat=0.15, radius=0.04, alpha_train=0.3, alpha_eval=0.6,
               std=False, layers=[64, 128, 40], exclude=None),
    'desktop_assembly': dict(acts={'all': 22}, ne=30, rho=0.25, lat=0.16, radius=0.02, alpha_train=0.3, alpha_eval=0.6,
                             std=False, layers=[512, 128, 40], exclude=None),
    'Breakfast': dict(acts=dict(pancake=14, salat=8, friedegg=9, scrambledegg=12, sandwich=9, juice=8, milk=5, tea=7,
                                cereals=5, coffee=7),
                      ne=15, rho=0.2, lat=0.1, radius=0.04, alpha_train=0.4, alpha_eval=0.7, std=True,
                      layers=[64, 128, 40], exclude=None),
    'YTI': dict(acts=dict(changing_tire=11, coffee=10, cpr=7, jump_car=12, repot=8), ne=10, rho=0.15, lat=0.12,
                radius=0.04, alpha_train=0.3, alpha_eval=0.6, std=False, layers=[3000, 32, 32], exclude=-1),
}
COMMON = dict(lr=1e-3, wd=1e-4, vf=5, bs=2, n_frames=256, eps_train=0.07, eps_eval=0.04, lft=0.05, lfe=0.05,
              lae=0.01, n_ot=[25, 1])
KMEANS_NINIT = 10                 # sklearn KMeans n_init used to initialise the cluster embeddings

# Numbers reported in the ASOT paper (Table 1, one run; Table 5 of the appendix: 5 runs, mean and std. err.)
PUBLISHED = {'FSeval': dict(mof1=59.3, f11=53.6, miou1=30.1, mof5=56.4, se5=2.8),
             'FS': dict(mof1=46.2, f11=37.4, miou1=24.9, mof5=43.0, se5=1.9),
             'desktop_assembly': dict(mof1=70.4, f11=68.0, miou1=45.9, mof5=62.9, se5=2.8),
             'Breakfast': dict(mof1=56.1, f11=38.3, miou1=18.6, mof5=None, se5=None),
             'YTI': dict(mof1=52.9, f11=35.1, miou1=24.7, mof5=None, se5=None)}

# ---------------------------------------------------------------- decoding grid
LAMBDAS = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
WS = [0.25, 0.5, 0.75, 1.0]
SRCS = ['codes', 'plan']          # label-free sources of the proportion estimate pi_hat
LAM_PAPER = COMMON['lae']         # lambda used by the authors at test time (0.01)
TAU_MM = 0.01                     # temperature of the smooth max in PA-minmax
PIHAT_FLOOR = 1e-4                # floor applied to pi_hat before renormalising
ORACLE_FLOOR = 1e-3               # floor applied to the oracle targets
F1_SEED = 12345                   # RNG seed of the Monte-Carlo F1 check

# ---------------------------------------------------------------- evaluation / statistics
SEEDS = {ds: list(range(10)) for ds in ALL_DATASETS}
N_BOOT = {'FSeval': 1000, 'FS': 1000, 'desktop_assembly': 1000, 'Breakfast': 400, 'YTI': 500}

# Method names used in all tables
M_ASOT = 'ASOT'                   # authors' decoding (lambda = 0.01, uniform target)
M_ASOTL = 'ASOT-lambda'           # ASOT with lambda selected by 2-fold CV (main baseline)
M_PA = 'PA-decode'                # ours: (lambda, w, source) selected by 2-fold CV
M_INT = 'PA-integral'             # variant without w: closed-form target integrated over w
M_MM = 'PA-minmax'                # variant without w: smooth max of the two KL terms
M_ORV = 'Oracle-video'            # uses true per-video proportions (labels; diagnostic only)
M_ORD = 'PA-true-proportions'     # uses true dataset proportions (labels; diagnostic only)
METHODS = [M_ASOT, M_ASOTL, M_PA, M_INT, M_MM, M_ORV, M_ORD]
NEW_METHODS = [M_INT, M_MM]
PAIRS = [(M_PA, M_ASOTL)] + [(m, r) for m in NEW_METHODS for r in [M_ASOTL, M_PA]] + [(M_MM, M_INT)]
METRICS = ['MoF', 'F1', 'mIoU']
RULES = ['mof']                   # selection criterion used by the 2-fold procedure

# Pre-registered decision criteria
MAIN = ['FSeval', 'desktop_assembly', 'Breakfast']
SIDE = ['FS', 'YTI']
K1_TOL = -0.3                     # variant - PA-decode >= -0.3 MoF on all main settings (and G1 vs ASOT-lambda)
H_TOL = -0.5                      # variant - ASOT-lambda >= -0.5 MoF on FS and YTI
PREF_2A = 0.3                     # prefer PA-minmax over PA-integral only if >= 0.3 MoF better on average
D_GO = 1.0                        # per-video-w upper bound threshold

# ---------------------------------------------------------------- Phase 1 (fixed before running, 2026-11-14)
# Per-Video protocol: Hungarian matching inside every video. No new selection: every method keeps the configurations
# picked by the 2-fold selection of the Full protocol. Reported as a secondary result; the G1 definition of the Full
# protocol is applied to it only descriptively.
PV_METHODS = [M_ASOT, M_ASOTL, M_PA]
PV_PAIRS = [(M_PA, M_ASOTL), (M_PA, M_ASOT)]
PV_G1_TOL = -0.5                  # same F1 / mIoU tolerance as G1
PV_CHECK_TOL = 0.01               # |our Per-Video MoF/mIoU - authors' trainer.test| must be < 0.01 points
# Reproduction of v6 from this repository (scripts/compare_v6.py and scripts/dump_video_stats.py)
R1_TOL = 1e-6                     # with the v6 caches: every score / CI equal to v6 up to rounding
R3_AGREE = 0.999                  # re-solved segmentations (fresh solve on GPU) agree with the cached ones on >= 99.9% of frames

# Mechanism analysis (intervention on the target distribution)
N_PERM = 5
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def quick_overrides():
    """Small settings for a smoke test (numbers are NOT meaningful)."""
    paper = copy.deepcopy(PAPER)
    for d in paper:
        paper[d]['ne'] = 2
    return dict(paper=paper, seeds={k: [0, 1, 2] for k in ALL_DATASETS}, n_boot={k: 30 for k in N_BOOT})
