"""Smoke tests for modular MPPIOptimizer."""

from __future__ import annotations

import numpy as np

from mppi.mppi_core import MPPIOptimizer, weighted_mean_aggregate
from mppi.sampler import GaussianSampler


def test_mppi_reduces_quadratic_cost() -> None:
    rng = np.random.default_rng(0)
    target = np.array([[1.0, -0.5], [0.5, 0.5], [0.0, 1.0]])
    nominal = np.zeros_like(target)

    def rollout_fn(actions):
        return {"actions": actions}

    def cost_fn(actions, _info):
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
        nom, _, _ = opt.solve(nom, rollout_fn, cost_fn)
    c1 = float(np.sum((nom - target) ** 2))
    assert c1 < c0 * 0.5


def test_custom_sample_fn_used() -> None:
    """Custom sampler path is exercised (object-style)."""
    rng = np.random.default_rng(1)
    nominal = np.zeros((4, 2))
    called = {"sample": False}

    def sample_fn(nom, k, scale):
        called["sample"] = True
        return nom[None] + scale * rng.standard_normal((k,) + nom.shape) * 0.1

    def rollout_fn(actions):
        return {}

    def cost_fn(actions, _info):
        return np.sum(actions**2, axis=(1, 2))

    opt = MPPIOptimizer(n_samples=16, temperature=1.0, sampler=GaussianSampler(rng))
    new_nom, weights, _ = opt.solve(
        nominal, rollout_fn, cost_fn, sample_fn=sample_fn, sigma_scale=1.0
    )
    assert called["sample"]
    assert new_nom.shape == nominal.shape
    assert np.isclose(weights.sum(), 1.0)


def test_weighted_mean_matches_eps_update() -> None:
    nominal = np.array([[1.0, 0.0], [0.0, 1.0]])
    eps = np.array([[[0.1, 0.0], [0.0, 0.2]], [[-0.1, 0.0], [0.0, -0.2]]])
    samples = nominal[None] + eps
    weights = np.array([0.75, 0.25])
    via_mean = weighted_mean_aggregate(nominal, samples, weights)
    via_eps = nominal + np.tensordot(weights, eps, axes=(0, 0))
    assert np.allclose(via_mean, via_eps)
