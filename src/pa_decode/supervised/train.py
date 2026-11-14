"""Training of MS-TCN++ exactly as Trainer.train of sj-li/MS-TCN2 (batch size 1, CE + 0.15 * truncated MSE)."""
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BASES, MSTCN, SEED_BASE
from .model import MS_TCN2


def set_seed(s):
    random.seed(s)
    np.random.seed(s % 2 ** 32)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_path(cache_dir, base, split, seed):
    return os.path.join(cache_dir, 'runs', f'{base}_split{split}_seed{seed}.npz')


def train_run(D, base, split, seed, cache_dir, device, epochs=MSTCN['epochs'], extra_cache_roots=()):
    """Train MS (3 refinement stages) or SS (none) on a split; returns final-stage logits [K, n] of the test videos."""
    p = run_path(cache_dir, base, split, seed)
    if not os.path.exists(p):
        for root in extra_cache_roots:
            hits = glob.glob(os.path.join(root, '**', 'runs', os.path.basename(p)), recursive=True)
            if hits:
                p = hits[0]
                break
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return {v: z[f'L_{i}'] for i, v in enumerate(z['vids'])}, json.loads(str(z['log']))
    t0 = time.time()
    set_seed(SEED_BASE + seed)
    K = D.K
    model = MS_TCN2(MSTCN['num_layers_PG'], MSTCN['num_layers_R'], BASES[base], MSTCN['num_f_maps'], D.feat_dim, K).to(device)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    mse = nn.MSELoss(reduction='none')
    opt = torch.optim.Adam(model.parameters(), lr=MSTCN['lr'])
    order = list(D.split_train[split])
    random.shuffle(order)                                              # BatchGenerator.read_data
    hist = []
    for ep in range(epochs):
        model.train()
        ep_loss, correct, total = 0., 0, 0
        for v in order:
            y = D.tgt[v]
            n = len(y)
            x = D.feat[v][:, :n].unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            mask = torch.ones(1, K, n, device=device)
            opt.zero_grad()
            preds = model(x)
            loss = 0
            for pr in preds:
                loss += ce(pr.transpose(2, 1).contiguous().view(-1, K), y.view(-1))
                loss += MSTCN['mse_w'] * torch.mean(torch.clamp(
                    mse(F.log_softmax(pr[:, :, 1:], dim=1), F.log_softmax(pr.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:])
            ep_loss += loss.item()
            loss.backward()
            opt.step()
            correct += (preds[-1].argmax(1) == y).sum().item()
            total += n
        random.shuffle(order)                                          # BatchGenerator.reset
        hist.append((ep_loss / len(order), correct / total))
    model.eval()
    L = {}
    with torch.no_grad():
        for v in D.split_test[split]:
            L[v] = model(D.feat[v].unsqueeze(0).to(device))[-1][0].float().cpu().numpy()   # [K, n], as predict()
    log = dict(train_loss=hist[-1][0], train_acc=hist[-1][1], minutes=(time.time() - t0) / 60)
    vids = list(D.split_test[split])
    os.makedirs(os.path.dirname(run_path(cache_dir, base, split, seed)), exist_ok=True)
    np.savez_compressed(run_path(cache_dir, base, split, seed), vids=np.array(vids), log=json.dumps(log),
                        **{f'L_{i}': L[v] for i, v in enumerate(vids)})
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return L, log
