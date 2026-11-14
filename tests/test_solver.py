"""Unit tests that need no dataset: the solvers reproduce the authors' ASOT and the closed-form target is correct."""
import numpy as np
import torch

import _path  # noqa: F401
from pa_decode.asot_bridge import asot
from pa_decode.solver import check_against_asot, check_minmax, check_nu_int, nu_int, segment_asot_pa

KW = dict(eps=0.04, alpha=0.6, radius=0.04, n_iters=(25, 1))


def _toy_cost(n=80, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.repeat_interleave(torch.arange(K), n // K + 1)[:n]
    cost = 0.6 * torch.rand(n, K, generator=g)
    cost[torch.arange(n), labels] -= 0.35
    cost = cost[None] + asot.temporal_prior(n, K, 0.15, 'cpu')
    codes = torch.softmax(-cost[0] / 0.1, dim=1)
    return cost.float(), codes


def test_nu_int():
    check_nu_int()
    assert np.allclose(nu_int(np.full(7, 1 / 7), 7), np.full(7, 1 / 7))


def test_pa_equals_asot_with_uniform_target():
    cost, _ = _toy_cost()
    check_against_asot(cost, KW, 0.05, torch.device('cpu'), tag='toy')


def test_minmax_equals_asot_with_uniform_target():
    cost, codes = _toy_cost(seed=1)
    check_minmax(cost, codes, KW, torch.device('cpu'), tag='toy')


def test_target_moves_column_mass():
    """A larger target share for one cluster increases the mass of that cluster in the plan."""
    cost, _ = _toy_cost(seed=2)
    K = cost.shape[2]
    uni = torch.full((1, K), 1. / K)
    tgt = torch.full((1, K), 0.5 / (K - 1))
    tgt[0, 0] = 0.5
    m_u = segment_asot_pa(cost, uni, 0.5, **KW)[0].mean(0)
    m_t = segment_asot_pa(cost, tgt, 0.5, **KW)[0].mean(0)
    assert m_t[0] > m_u[0]
