"""Experiment container shared by all pipeline stages.

Every stage is cached on disk, so each script can rebuild the state it needs by calling the earlier stages again:
``load_experiment -> build_units -> train_all -> decode_all -> point_estimates -> ...``. Stages that are already
cached are only loaded.
"""
import copy
import os
import warnings

import torch

from . import config as cfg

warnings.filterwarnings('ignore')


class Experiment:
    def __init__(self, data_dir, cache_dir, out_dir, datasets=None, seeds=None, quick=False, device=None):
        from .asot_bridge import load_training_modules, ASOT_HEAD
        self.amet, self.atrain, self.VideoDataset = load_training_modules()
        self.asot_head = ASOT_HEAD
        self.data_dir = os.path.abspath(data_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.out_dir = os.path.abspath(out_dir)
        for d in (self.cache_dir, self.out_dir):
            os.makedirs(d, exist_ok=True)
        self.quick = quick
        self.datasets = list(datasets or cfg.ALL_DATASETS)
        self.paper = copy.deepcopy(cfg.PAPER)
        self.seeds = copy.deepcopy(cfg.SEEDS)
        self.n_boot = dict(cfg.N_BOOT)
        if quick:
            q = cfg.quick_overrides()
            self.paper, self.seeds, self.n_boot = q['paper'], q['seeds'], q['n_boot']
        if seeds is not None:
            self.seeds = {ds: list(seeds) for ds in cfg.ALL_DATASETS}
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        # filled by the stages
        self.units, self.uinfo = [], {}
        self.trained, self.dec, self.stab = {}, {}, {}
        self.point, self.chosen, self.checks, self.boot, self.delta = {}, {}, [], {}, {}

    # ------------------------------------------------------------------ helpers
    def units_of(self, ds):
        return [a for (d, a) in self.units if d == ds]

    @staticmethod
    def label(ds):
        return cfg.LABEL.get(ds, ds)

    def eval_kw(self, ds):
        return dict(eps=cfg.COMMON['eps_eval'], alpha=self.paper[ds]['alpha_eval'], radius=self.paper[ds]['radius'],
                    n_iters=tuple(cfg.COMMON['n_ot']))

    def path(self, *parts):
        p = os.path.join(self.cache_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def out(self, name):
        return os.path.join(self.out_dir, name)


def add_common_args(parser):
    parser.add_argument('--data-dir', default='data', help='folder with data/<dataset>/{features,groundTruth,mapping}')
    parser.add_argument('--cache-dir', default='cache', help='trained models and decoded segmentations')
    parser.add_argument('--out-dir', default='results', help='tables, JSON and figures')
    parser.add_argument('--datasets', default=','.join(cfg.ALL_DATASETS),
                        help='comma-separated subset of ' + ','.join(cfg.ALL_DATASETS))
    parser.add_argument('--seeds', default=None, help='comma-separated seeds (default: 0..9)')
    parser.add_argument('--quick', action='store_true', help='smoke test: 2 epochs, 3 seeds, 30 bootstrap rounds')
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    return parser


def experiment_from_args(args):
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    return Experiment(args.data_dir, args.cache_dir, args.out_dir, datasets=args.datasets.split(','),
                      seeds=seeds, quick=args.quick, device=args.device)
