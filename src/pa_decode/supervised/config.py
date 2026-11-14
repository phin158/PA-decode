"""Configuration of the supervised experiment (fixed before running)."""
SPLITS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
SEED_BASE = 1538574472                      # fixed seed of MS-TCN2 main.py; seed s uses SEED_BASE + s
BASES = {'MS': 3, 'SS': 0}                  # number of refinement stages (num_R)
MSTCN = dict(num_layers_PG=11, num_layers_R=10, num_f_maps=64, lr=0.0005, epochs=100, sample_rate=2, mse_w=0.15)
ASOT_SUP = dict(eps=0.06, alpha=0.4, radius=0.01, n_iters=(25, 1))   # Appendix C of ASOT
LAM_PAPER = 0.05
LAMBDAS = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
# targets: uniform = ASOT; train_* = pihat from TRAINING labels (w = 1 or 0.5); codes / plan = label-free from the test
# videos; oracle = true proportions of the TEST videos (uses test labels, upper bound only)
TARGETS = ['uniform', 'train_w1', 'train_w05', 'codes_w1', 'plan_w1', 'oracle_w1']
CFGS = [(lam, tg) for tg in TARGETS for lam in LAMBDAS]
FLOOR = 1e-4
METRICS = ['Acc', 'Edit', 'F1@10', 'F1@25', 'F1@50']
OVERLAPS = [0.10, 0.25, 0.50]
# Table 3 of the ASOT paper (50 Salads)
PAPER_T3 = {'MS': (83.5, 72.6, 80.1, 77.2, 69.4), 'SS': (76.0, 36.3, 44.8, 42.3, 34.9),
            'MS + ASOT': (83.6, 77.6, 84.4, 83.1, 74.9), 'SS + ASOT': (78.3, 72.5, 80.3, 78.5, 69.9)}
ZENODO_URL = 'https://zenodo.org/records/3625992/files/data.zip?download=1'   # MS-TCN data (I3D features)
