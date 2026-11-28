"""PEOT backbone (Li, Vu & Gall, "Learning Probabilistic Embeddings for Unsupervised Action Segmentation", ECCV 2026)
for experiment E6 (docs/preregistration_extra.md).

Training uses the authors' ``PEOT/train.py`` (repository derkbreeze/PEOT, pinned commit) with the README commands for
probabilistic embeddings (``-pr``). The script is run unchanged except for these patches, all of them I/O only:
  * checkpoints are saved (the script has ``save = False``) into a folder given by us, last epoch only;
  * the two debugging filters that train only Breakfast/coffee and YTI/changing_tire are replaced by a filter on one
    activity per call (so every activity is trained, one process per (dataset, activity, seed));
  * the data folder '/data' is replaced by ours;
  * the per-epoch test evaluation is run at the last epoch only. The DataLoader iterator is still created at the
    other epochs, so the random stream of the training (and therefore the trained model) is unchanged.
The last epoch is used (no checkpoint selection on the test labels, unlike the epochs hard-coded in test.py).

Decoding uses the cost of the authors' ``test.py``: frame embedding = normalised GCN mean, cost = 1 - f C^T + temporal
prior, with the test-time hyper-parameters of test.py. The GCN on the chain graph is computed with an equivalent
tridiagonal form (the dense T x T form of the authors' code is too slow for long videos); ``check_embed`` verifies
that both give the same embeddings.

Desktop Assembly: train.py uses K = 23 clusters (comment in the code: "but GT K is 22"), test.py uses 22, which does
not match the trained codebook. We use the trained K = 23.
"""
import os
import re
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PEOT_URL = 'https://github.com/derkbreeze/PEOT.git'
PEOT_COMMIT = 'fbc9c11'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PEOT_ROOT = os.environ.get('PEOT_ROOT', os.path.join(REPO_ROOT, 'third_party', 'PEOT'))
DATASETS = ['Breakfast', 'YTI', 'FSeval', 'desktop_assembly']

# README training commands, probabilistic embeddings
TRAIN_ARGS = {
    'Breakfast': '-pr -s --rho 0.2 -r 0.04 -at 0.4 -ae 0.7 -lat 0.1 -ua -lc 1e-4',
    'YTI': '-pr -s --rho 0.2 -r 0.02 -lat 0.12 -ua -lc 5e-4',
    'FSeval': '-pr -s --rho 0.05 -r 0.02 -lat 0.1 -ua -lc 1e-4',
    'desktop_assembly': '-pr -ns 1 -s --rho 0.25 -r 0.01 -lat 0.16 -ua -lr 5e-3 -lc 1e-3',
}
# Experiment.paper entries for PEOT (same keys as config.PAPER). K, layers, epochs: train.py; test-time rho / radius /
# alpha_eval: test.py "full model" lines (equal to the training command of each dataset). All use -s (standardised).
PEOT_PAPER = {
    'FSeval': dict(acts={'all': 12}, ne=70, rho=0.05, lat=0.1, radius=0.02, alpha_train=0.3, alpha_eval=0.6,
                   std=True, layers=[64, 128, 40], exclude=None),
    'desktop_assembly': dict(acts={'all': 23}, ne=15, rho=0.25, lat=0.16, radius=0.01, alpha_train=0.3,
                             alpha_eval=0.6, std=True, layers=[512, 128, 40], exclude=None),
    'Breakfast': dict(acts=dict(pancake=14, salat=8, friedegg=9, scrambledegg=12, sandwich=9, juice=8, milk=5, tea=7,
                                cereals=5, coffee=7),
                      ne=15, rho=0.2, lat=0.1, radius=0.04, alpha_train=0.4, alpha_eval=0.7, std=True,
                      layers=[64, 128, 40], exclude=None),
    'YTI': dict(acts=dict(changing_tire=11, coffee=10, cpr=7, jump_car=12, repot=8), ne=30, rho=0.2, lat=0.12,
                radius=0.02, alpha_train=0.3, alpha_eval=0.6, std=True, layers=[3000, 32, 32], exclude=-1),
}
TEMP = 0.1                       # softmax temperature of train.py (codes)

PATCHES = [  # (original text, replacement); every original must occur exactly as written
    ("save = False #whether or not to save training hyper-parameters and checkpoints",
     "save = True #whether or not to save training hyper-parameters and checkpoints"),
    ("    exp_id = 'exp_' + datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n"
     "    exp_dir = '{}_/{}'.format(args.dataset, exp_id) if args.prob else '{}/{}'.format(args.dataset, exp_id)",
     "    exp_id = 'pa_decode'\n    exp_dir = os.environ['PEOT_EXP_DIR']"),
    ("    if args.dataset == 'YTI' and args.activity != 'changing_tire':\n        continue\n\n"
     "    if args.dataset == 'Breakfast' and args.activity != 'coffee':\n        continue",
     "    if os.environ.get('PEOT_ACTS') and args.activity not in os.environ['PEOT_ACTS'].split(','):\n        continue"),
    ("    train_set = VideoDataset('/data',", "    train_set = VideoDataset(os.environ['PEOT_DATA'],"),
    ("    test_set = VideoDataset('/data',", "    test_set = VideoDataset(os.environ['PEOT_DATA'],"),
    ("        if save:\n            if not args.prob:\n                state_dict",
     "        if save and epoch == args.n_epochs - 1:\n            if not args.prob:\n                state_dict"),
    ("        mlp.eval()\n        for batch_idx, batch in enumerate(test_loader):",
     "        mlp.eval()\n"
     "        if epoch < args.n_epochs - 1 and not os.environ.get('PEOT_EVAL_EVERY'):\n"
     "            iter(test_loader)  # same random draw as the evaluation loop (DataLoader seed); no evaluation\n"
     "            continue\n"
     "        for batch_idx, batch in enumerate(test_loader):"),
    ("torch.set_printoptions(sci_mode=False)\n",
     "torch.set_printoptions(sci_mode=False)\n"
     "if os.environ.get('PEOT_CPU'):  # smoke test without GPU only\n"
     "    torch.Tensor.cuda = lambda self, *a, **k: self\n"
     "    nn.Module.cuda = lambda self, *a, **k: self\n"),
    ("print(args)\n", "if os.environ.get('PEOT_NE'):\n    args.n_epochs = int(os.environ['PEOT_NE'])  # smoke test only\n"
                      "print(args)\n"),
]


def ensure_code():
    if not os.path.isdir(os.path.join(PEOT_ROOT, 'PEOT', 'src')):
        subprocess.check_call(['git', 'clone', '-q', PEOT_URL, PEOT_ROOT])
    subprocess.check_call(['git', '-C', PEOT_ROOT, 'checkout', '-q', PEOT_COMMIT])
    head = subprocess.check_output(['git', '-C', PEOT_ROOT, 'rev-parse', 'HEAD']).decode().strip()
    assert head.startswith(PEOT_COMMIT), head
    code_dir = os.path.join(PEOT_ROOT, 'PEOT')
    src = open(os.path.join(code_dir, 'train.py'), encoding='utf-8').read()
    for old, new in PATCHES:
        assert src.count(old) == 1, f'PEOT train.py changed, patch not applicable: {old[:60]!r}'
        src = src.replace(old, new)
    path = os.path.join(code_dir, '_train_pa_decode.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    return code_dir, path


def model_path(exp, ds, act, seed):
    return exp.path('models', f'{ds}_{act}_s{seed}_peot_ne{exp.paper[ds]["ne"]}.pt')


_LINE = re.compile(r'PEOT \S+ (\S+) frames \d+ epoch (\d+) loss \S+ activity level MoF ([\d.]+) F1 ([\d.]+) mIoU ([\d.]+)')


def train_one_peot(exp, ds, act, seed):
    assert ds in DATASETS, f'PEOT experiment is defined for {DATASETS}, not {ds}'
    path = model_path(exp, ds, act, seed)
    if os.path.exists(path):
        return torch.load(path, map_location='cpu', weights_only=False)
    code_dir, script = ensure_code()
    out_dir = os.path.join(exp.cache_dir, 'peot_tmp', f'{ds}_s{seed}_{act}')
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, PEOT_EXP_DIR=out_dir, PEOT_ACTS=act, PEOT_DATA=exp.data_dir)
    if exp.quick:
        env['PEOT_NE'] = str(exp.paper[ds]['ne'])
    if exp.device.type != 'cuda':
        env['PEOT_CPU'] = '1'
    cmd = [sys.executable, os.path.basename(script)] + TRAIN_ARGS[ds].split() + ['-d', ds, '--seed', str(seed)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=code_dir, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'PEOT training failed ({ds}/{act}/seed {seed}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}')
    last = [m for m in _LINE.finditer(p.stdout) if m.group(1) == act]
    assert last, f'no evaluation line in the PEOT log:\n{p.stdout[-2000:]}'
    m = last[-1]
    ne = exp.paper[ds]['ne']
    assert int(m.group(2)) == ne - 1, (m.group(0), ne)
    ck = torch.load(os.path.join(out_dir, act, f'epoch_{ne - 1:02d}.pth'), map_location='cpu', weights_only=False)
    P = exp.paper[ds]
    r = dict(state=dict(mlp=ck['mlp'], gcn_mean=ck['gcn_mean'], gcn_logstd=ck['gcn_logstd'],
                        clusters=ck['clusters'].detach().cpu()),
             hparams=dict(layer_sizes=P['layers'], n_clusters=P['acts'][act], rho=P['rho'], cmd=' '.join(cmd[1:])),
             # authors' own evaluation at the last epoch (activity level = Full protocol), used as a cache check
             test=dict(test_mof_full=float(m.group(3)), test_f1_full=float(m.group(4)),
                       test_miou_full=float(m.group(5))),
             minutes=(time.time() - t0) / 60, peot_commit=PEOT_COMMIT)
    assert r['state']['clusters'].shape[0] == P['acts'][act], (ds, act, r['state']['clusters'].shape)
    torch.save(r, path + '.tmp')          # atomic: a run stopped by the time limit never leaves a broken file
    os.replace(path + '.tmp', path)
    return r


class PEOTNet(nn.Module):
    """Embedding of the authors' test.py (mean of the probabilistic embedding, normalised)."""

    def __init__(self, layer_sizes, n_clusters, rho, **_):
        super().__init__()
        L = layer_sizes
        layers = [nn.Sequential(nn.Linear(a, b), nn.ReLU()) for a, b in zip(L[:-2], L[1:-1])] + [nn.Linear(L[-2], L[-1])]
        self.mlp = nn.Sequential(*layers)
        self.gcn_mean = nn.Linear(L[-1], L[-1])
        self.register_buffer('clusters', torch.zeros(n_clusters, L[-1]))
        self.layer_sizes, self.n_clusters, self.rho, self.temp = list(L), n_clusters, rho, TEMP

    def embed(self, x):
        """x: [B, T, F] -> normalised GCN mean [B, T, D] (tridiagonal form of the chain-graph GCN)."""
        f = F.normalize(self.mlp(x), dim=-1)
        w_self = (torch.einsum('btd,btd->bt', f, f) + 1) / 2                              # diagonal (= 1)
        w_next = (torch.einsum('btd,btd->bt', f[:, :-1], f[:, 1:]) + 1) / 2               # (t, t+1) and (t+1, t)
        deg = w_self.clone()
        deg[:, :-1] += w_next
        deg[:, 1:] += w_next
        r = deg.pow(-0.5)
        g = self.gcn_mean(f)
        out = (w_self * r * r)[..., None] * g
        c = (w_next * r[:, :-1] * r[:, 1:])[..., None]
        out[:, :-1] += c * g[:, 1:]
        out[:, 1:] += c * g[:, :-1]
        return F.normalize(out, dim=-1)

    def embed_dense(self, x):
        """Authors' code (test.py lines 171-201), for checking."""
        f = F.normalize(self.mlp(x), dim=-1)
        B, T, _ = f.shape
        adj = torch.stack([torch.eye(T, device=x.device) for _ in range(B)])
        idx = torch.arange(T - 1)
        adj[:, idx, idx + 1] = 1
        adj[:, idx + 1, idx] = 1
        aw = adj * ((f @ f.transpose(1, 2)) + 1) / 2
        nd = torch.diag_embed(torch.pow(aw.sum(-1), -0.5))
        return F.normalize((nd @ aw @ nd).matmul(self.gcn_mean(f)), dim=-1)


def load_peot(exp, ds, act, seed):
    r = exp.trained[(ds, act, seed)]
    m = PEOTNet(**r['hparams'])
    m.mlp.load_state_dict(r['state']['mlp'])
    m.gcn_mean.load_state_dict(r['state']['gcn_mean'])
    m.clusters.copy_(r['state']['clusters'])
    return m.to(exp.device).eval()


@torch.no_grad()
def check_embed(m, feat, device, n=1500):
    x = feat[:n].to(device)[None]
    d = (m.embed(x) - m.embed_dense(x)).abs().max().item()
    assert d < 1e-4, f'tridiagonal GCN differs from the authors\' dense form: {d}'
    return d
