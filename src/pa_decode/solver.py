"""PA-decode solver and the two variants without ``w``.

``segment_asot_pa`` is a copy of ``asot.segment_asot`` (the branch used at test time: ub_actions=True,
ub_frames=False, full mask). The only changes are
  (1) the KL target of the action marginal is ``dy_target`` instead of the uniform 1/K
      (initialisation stays uniform, as in the authors' code), and
  (2) lambda and the step size may differ for every element of the batch (B = 1 is identical to the authors' code).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .asot_bridge import asot
from .config import TAU_MM


def segment_asot_pa(cost_matrix, dy_target, lambda_actions, eps, alpha, radius, n_iters=(25, 1)):
    B, N, K = cost_matrix.shape
    dev = cost_matrix.device
    dx = torch.ones((B, N, 1), device=dev) / N
    dy = dy_target.reshape(B, K, 1).to(dev, cost_matrix.dtype)
    lam = torch.as_tensor(lambda_actions, device=dev, dtype=cost_matrix.dtype).reshape(-1, 1, 1)
    T = dx * torch.full((B, 1, K), 1. / K, device=dev, dtype=cost_matrix.dtype)
    Cv = asot.construct_Cv_filter(N, radius, dev)
    step = None
    for it in range(n_iters[0]):
        fgw_cost_matrix = asot.grad_fgw(T, cost_matrix, alpha, Cv)
        grad_obj = fgw_cost_matrix - asot.grad_entropy(T, eps)
        grad_obj = grad_obj + asot.grad_kld(T, dy.transpose(1, 2), lam, 1)
        if it == 0:
            step = 4. / grad_obj.amax(dim=(1, 2), keepdim=True)
        T = T * torch.exp(-step * grad_obj)
        T /= T.sum(dim=2, keepdim=True)
        T = torch.nan_to_num(T, posinf=0., neginf=0.)
        T = T * dx
    return T * N


def nu_int(pihat, K):
    """PA-integral target: nu_k ∝ exp(∫_0^1 log((1-w)/K + w*pihat_k) dw), in closed form
    ∫_0^1 log(a + (b-a)w) dw = (b ln b - a ln a)/(b - a) - 1   (= ln a when b = a)."""
    a, b = 1. / K, np.asarray(pihat, dtype=np.float64)
    d = b - a
    safe = np.abs(d) > 1e-12
    I = np.where(safe, (b * np.log(b) - a * np.log(a)) / np.where(safe, d, 1.) - 1., np.log(a))
    v = np.exp(I)
    return v / v.sum()


def segment_asot_mm(cost_matrix, dy_a, dy_b, lambda_actions, eps, alpha, radius, n_iters=(25, 1), tau=TAU_MM,
                    return_alpha=False):
    """PA-minmax: identical to segment_asot_pa except that the KL gradient is the gradient of the smooth max of two KL
    terms, a*grad KL(m||nu_a) + (1-a)*grad KL(m||nu_b) with a = sigmoid((KL_a - KL_b)/tau), recomputed every
    iteration. nu_a, nu_b: [B, K]."""
    B, N, K = cost_matrix.shape
    dev = cost_matrix.device
    dx = torch.ones((B, N, 1), device=dev) / N
    da = dy_a.reshape(B, 1, K).to(dev, cost_matrix.dtype)
    db = dy_b.reshape(B, 1, K).to(dev, cost_matrix.dtype)
    lam = torch.as_tensor(lambda_actions, device=dev, dtype=cost_matrix.dtype).reshape(-1, 1, 1)
    T = dx * torch.full((B, 1, K), 1. / K, device=dev, dtype=cost_matrix.dtype)
    Cv = asot.construct_Cv_filter(N, radius, dev)
    step = None
    for it in range(n_iters[0]):
        fgw_cost_matrix = asot.grad_fgw(T, cost_matrix, alpha, Cv)
        grad_obj = fgw_cost_matrix - asot.grad_entropy(T, eps)
        m = T.sum(dim=1)                                                  # [B, K], sums to 1
        a_w = torch.sigmoid((asot.kld(m, da[:, 0]) - asot.kld(m, db[:, 0])) / tau).reshape(B, 1, 1)
        grad_obj = grad_obj + a_w * asot.grad_kld(T, da, lam, 1) + (1. - a_w) * asot.grad_kld(T, db, lam, 1)
        if it == 0:
            step = 4. / grad_obj.amax(dim=(1, 2), keepdim=True)
        T = T * torch.exp(-step * grad_obj)
        T /= T.sum(dim=2, keepdim=True)
        T = torch.nan_to_num(T, posinf=0., neginf=0.)
        T = T * dx
    return (T * N, a_w.reshape(B)) if return_alpha else T * N


@torch.no_grad()
def video_cost(m, feat, device):
    """Same as the authors' test_step: cost = 1 - f·Cᵀ + temporal prior. Also returns codes = softmax(f·Cᵀ / temp)."""
    D = m.layer_sizes[-1]
    x = feat.to(device)[None]
    B, T, _ = x.shape
    if hasattr(m, 'embed'):                  # PEOT: normalised GCN mean (pa_decode/peot.py)
        f = m.embed(x)
    else:
        f = F.normalize(m.mlp(x.reshape(-1, x.shape[-1])).reshape(B, T, D), dim=-1)
    tp = asot.temporal_prior(T, m.n_clusters, m.rho, device)
    cost = 1. - f @ m.clusters.T.unsqueeze(0)
    cost += tp
    codes = torch.exp(f @ m.clusters.T / m.temp)
    codes = codes / codes.sum(dim=-1, keepdim=True)
    return cost, codes[0]


# ---------------------------------------------------------------------------- self-checks
def check_nu_int(seed=0):
    """Closed form of ∫_0^1 log nu_w dw equals numerical integration; nu lies between 1/K and pihat; uniform -> uniform."""
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    rng = np.random.default_rng(seed)
    for K_ in [5, 12, 22]:
        for _ in range(20):
            ph = rng.dirichlet(np.full(K_, 0.3))
            ph = np.maximum(ph, 1e-4)
            ph /= ph.sum()
            ph[0] = 1. / K_                                  # one component with b = a (special branch)
            ww = np.linspace(0, 1, 200001)
            num = trapz(np.log((1 - ww)[:, None] / K_ + ww[:, None] * ph[None]), ww, axis=0)
            a_ = 1. / K_
            d_ = ph - a_
            sf = np.abs(d_) > 1e-12
            clo = np.where(sf, (ph * np.log(ph) - a_ * np.log(a_)) / np.where(sf, d_, 1.) - 1., np.log(a_))
            assert np.abs(num - clo).max() < 1e-6, np.abs(num - clo).max()
            raw = np.exp(clo)
            assert np.all(raw <= np.maximum(ph, a_) + 1e-12) and np.all(raw >= np.minimum(ph, a_) - 1e-12)
        assert np.allclose(nu_int(np.full(K_, 1. / K_), K_), np.full(K_, 1. / K_))
    print('OK [PA-integral]: closed form matches numerical integration (< 1e-6); uniform pihat -> uniform target')


def check_against_asot(cost, kw, lfe, device, lams=(0.005, 0.01, 0.04), tag=''):
    """With a uniform target, segment_asot_pa must reproduce asot.segment_asot (also in batch mode)."""
    K = cost.shape[2]
    mask = torch.ones(cost.shape[:2], dtype=bool, device=device)
    refs = [asot.segment_asot(cost, mask, eps=kw['eps'], alpha=kw['alpha'], radius=kw['radius'], ub_frames=False,
                              ub_actions=True, lambda_frames=lfe, lambda_actions=l, n_iters=kw['n_iters'],
                              step_size=None)[0] for l in lams]
    uni = torch.full((1, K), 1. / K, device=device)
    for l, ref in zip(lams, refs):
        mine = segment_asot_pa(cost, uni, l, **kw)
        assert torch.allclose(mine, ref, atol=1e-6, rtol=1e-5) and torch.equal(mine.argmax(2), ref.argmax(2)), (tag, l)
    bat = segment_asot_pa(cost.expand(len(lams), -1, -1), uni.expand(len(lams), -1), list(lams), **kw)
    for i, ref in enumerate(refs):
        assert (bat[i:i + 1].argmax(2) == ref.argmax(2)).float().mean().item() >= 0.9999, (tag, lams[i])
    print(f'OK [{tag}]: PA-decode with a uniform target == asot.segment_asot (alpha={kw["alpha"]}, r={kw["radius"]})')


def check_minmax(cost, codes, kw, device, lams=(0.01, 0.16, 1.28), tag=''):
    """PA-minmax with pihat = uniform must reproduce ASOT (float32 rounding only)."""
    K = cost.shape[2]
    uni = torch.full((1, K), 1. / K, device=device)
    for l in lams:
        ref = segment_asot_pa(cost, uni, l, **kw)
        mine = segment_asot_mm(cost, uni, uni, l, **kw)
        err, agr = (mine - ref).abs().max().item(), (mine.argmax(2) == ref.argmax(2)).float().mean().item()
        assert err < 1e-4 and agr >= 0.999, (tag, l, err, agr)
    n = len(lams)
    bat = segment_asot_mm(cost.expand(n, -1, -1), uni.expand(n, -1), uni.expand(n, -1), list(lams), **kw)
    for i, l in enumerate(lams):
        agr = (bat[i:i + 1].argmax(2) == segment_asot_pa(cost, uni, l, **kw).argmax(2)).float().mean().item()
        assert agr >= 0.999, (tag, l, agr)
    ph = torch.tensor(np.maximum(codes.mean(0).double().cpu().numpy(), 1e-4), dtype=torch.float32, device=device)[None]
    ph = ph / ph.sum()
    Tm, aw = segment_asot_mm(cost.expand(n, -1, -1), uni.expand(n, -1), ph.expand(n, -1), list(lams), **kw, return_alpha=True)
    mm_ = Tm.sum(1) / Tm.shape[1]
    kl_u, kl_p = asot.kld(mm_, uni.expand(n, -1)), asot.kld(mm_, ph.expand(n, -1))
    f3 = lambda x: '[' + ', '.join(f'{v:.3f}' for v in x.cpu().tolist()) + ']'
    print(f'OK [{tag}] PA-minmax: uniform pihat -> same as ASOT. Example (video 1, pihat = codes, lambda = {list(lams)}): '
          f'final a = {f3(aw)}, KL(m||uniform) = {f3(kl_u)}, KL(m||pihat) = {f3(kl_p)}')
