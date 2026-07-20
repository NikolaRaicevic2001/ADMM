"""WrenchConsensus unit tests."""

from __future__ import annotations

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from utils.math_utils import shift_horizon_zero_tail


def test_z_update_average() -> None:
    c = WrenchConsensus(horizon=4, rho=1.0, max_dual=5.0)
    w_o = np.ones((4, 3))
    w_r = np.full((4, 3), 3.0)
    z = c.z_update(w_o, w_r)
    assert np.allclose(z, 2.0)


def test_penalty_matches_formula() -> None:
    c = WrenchConsensus(horizon=2, rho=2.0, max_dual=10.0)
    actual = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    z = np.zeros((2, 3))
    dual = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected = 0.5 * 2.0 * np.sum((actual - z + dual) ** 2)
    assert abs(c.penalty_cost(actual, z, dual) - expected) < 1e-12


def test_dual_anti_windup() -> None:
    c = WrenchConsensus(horizon=1, rho=1.0, max_dual=1.0)
    dual = np.array([[0.8, 0.0, 0.0]])
    actual = np.array([[2.0, 0.0, 0.0]])
    z = np.zeros((1, 3))
    new = c.dual_update(actual, z, dual)
    assert np.all(new <= 1.0)
    assert np.all(new >= -1.0)
    assert abs(new[0, 0] - 1.0) < 1e-12


def test_horizon_shift_zeros_tail() -> None:
    gamma = np.arange(12, dtype=float).reshape(4, 3)
    shifted = shift_horizon_zero_tail(gamma)
    assert np.allclose(shifted[0], gamma[1])
    assert np.allclose(shifted[-1], 0.0)
