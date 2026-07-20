"""Smoke test: MPPI reduces a quadratic toy cost."""

from __future__ import annotations

import numpy as np

from mppi.mppi_core import MPPIOptimizer
from mppi.sampler import GaussianSampler


def test_mppi_reduces_quadratic_cost() -> None:
    rng = np.random.default_rng(0)
    target = np.array([[1.0, -0.5], [0.5, 0.5], [0.0, 1.0]])
    nominal = np.zeros_like(target)

    def rollout_fn(_state, actions):
        return {"actions": actions}

    def cost_fn(actions, _info):
        # (K, H, 2)
        return np.sum((actions - target[None]) ** 2, axis=(1, 2))

    opt = MPPIOptimizer(
        n_samples=128,
        temperature=0.5,
        sigma=0.4,
        sampler=GaussianSampler(rng),
    )
    c0 = float(np.sum((nominal - target) ** 2))
    nom = nominal.copy()
    for _ in range(30):
        nom, _, _ = opt.solve(None, nom, rollout_fn, cost_fn)
    c1 = float(np.sum((nom - target) ** 2))
    assert c1 < c0 * 0.5
