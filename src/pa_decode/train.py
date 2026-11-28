"""Train ASOT exactly as the authors' ``src/train.py`` (same call order, same hyper-parameters) for every
(dataset, activity, seed). Trained weights are cached in ``<cache>/models``.
"""
import functools
import time

import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans as SKKMeans

from . import config as cfg
from .data import make_ds


def model_path(exp, ds, act, seed):
    # the number of epochs is part of the file name, so smoke-test models never mix with real ones
    return exp.path('models', f'{ds}_{act}_s{seed}_ne{exp.paper[ds]["ne"]}.pt')


def check_hparams(exp, h, ds, act):
    P, K = exp.paper[ds], exp.paper[ds]['acts'][act]
    ok = (list(h['layer_sizes']) == P['layers'] and h['n_clusters'] == K and abs(h['lambda_actions_train'] - P['lat']) < 1e-12
          and abs(h['rho'] - P['rho']) < 1e-12 and abs(h['radius_gw'] - P['radius']) < 1e-12
          and abs(h['alpha_train'] - P['alpha_train']) < 1e-12)
    assert ok, f'Cached model {ds}/{act} was trained with a different configuration: {h}'


def train_one(exp, ds, act, seed):
    if getattr(exp, 'backbone', 'asot') == 'peot':
        from .peot import train_one_peot
        return train_one_peot(exp, ds, act, seed)
    path = model_path(exp, ds, act, seed)
    try:
        r = torch.load(path, map_location='cpu', weights_only=False)
        check_hparams(exp, r['hparams'], ds, act)
        return r
    except FileNotFoundError:
        pass
    P, c, K = exp.paper[ds], cfg.COMMON, exp.paper[ds]['acts'][act]
    atrain = exp.atrain
    orig_kmeans = atrain.KMeans
    atrain.KMeans = functools.partial(SKKMeans, n_init=cfg.KMEANS_NINIT)
    t0 = time.time()
    try:
        # ---- same order as train.py __main__ ----
        pl.seed_everything(seed)
        data_val = make_ds(exp, ds, act, c['n_frames'], False)
        data_train = make_ds(exp, ds, act, c['n_frames'], True)
        data_test = make_ds(exp, ds, act, None, False)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=c['bs'], shuffle=False)
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=c['bs'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False)
        ssl = atrain.VideoSSL(layer_sizes=P['layers'], n_clusters=K, alpha_train=P['alpha_train'], alpha_eval=P['alpha_eval'],
                              ub_frames=False, ub_actions=True, lambda_frames_train=c['lft'], lambda_frames_eval=c['lfe'],
                              lambda_actions_train=P['lat'], lambda_actions_eval=c['lae'], step_size=None,
                              train_eps=c['eps_train'], eval_eps=c['eps_eval'], radius_gw=P['radius'],
                              n_ot_train=c['n_ot'], n_ot_eval=c['n_ot'], n_frames=c['n_frames'], lr=c['lr'],
                              weight_decay=c['wd'], rho=P['rho'], exclude_cls=P['exclude'], visualize=False)
        trainer = pl.Trainer(accelerator='gpu' if exp.device.type == 'cuda' else 'cpu', devices=1,
                             check_val_every_n_epoch=c['vf'], max_epochs=P['ne'], log_every_n_steps=50,
                             logger=False, enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, inference_mode=False)
        ssl.fit_clusters(train_loader, K)
        trainer.validate(ssl, val_loader, verbose=False)
        trainer.fit(ssl, train_loader, val_loader)
        res = trainer.test(ssl, dataloaders=test_loader, verbose=False)[0]
    finally:
        atrain.KMeans = orig_kmeans
    out = dict(state_dict={k: v.detach().cpu() for k, v in ssl.state_dict().items()},
               hparams=dict(ssl.hparams), test={k: float(v) for k, v in res.items()}, minutes=(time.time() - t0) / 60)
    torch.save(out, path)
    return out


def train_all(exp, verbose=True):
    t_start = time.time()
    for ds in exp.datasets:
        for seed in exp.seeds[ds]:
            for act in exp.units_of(ds):
                r = train_one(exp, ds, act, seed)
                exp.trained[(ds, act, seed)] = r
                t = r['test']
                if verbose:
                    print(f'{exp.label(ds):20s} {act:14s} seed {seed}: trainer.test MoF {100 * t["test_mof_full"]:.2f} '
                          f'F1 {100 * t["test_f1_full"]:.2f} mIoU {100 * t["test_miou_full"]:.2f} '
                          f'({r["minutes"]:.1f} min; total {(time.time() - t_start) / 60:.0f})')


def load_model(exp, ds, act, seed):
    if getattr(exp, 'backbone', 'asot') == 'peot':
        from .peot import load_peot
        return load_peot(exp, ds, act, seed)
    r = exp.trained[(ds, act, seed)]
    m = exp.atrain.VideoSSL(**r['hparams'])
    m.load_state_dict(r['state_dict'])
    return m.to(exp.device).eval()
