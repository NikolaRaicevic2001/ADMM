"""Tests for world CoM wrench map."""

from __future__ import annotations

import numpy as np

from contact.wrench_map import contact_force_to_com_wrench


def test_pure_force_no_offset() -> None:
    pose = np.array([1.0, 2.0, 0.0])
    p_c = np.array([1.0, 2.0])
    f = np.array([3.0, -1.0])
    w = contact_force_to_com_wrench(pose, p_c, f)
    assert np.allclose(w[:2], f)
    assert abs(w[2]) < 1e-12


def test_torque_from_offset() -> None:
    pose = np.array([0.0, 0.0, 0.0])
    p_c = np.array([1.0, 0.0])  # offset along +x
    f = np.array([0.0, 2.0])  # force along +y => tau = x*fy = 2
    w = contact_force_to_com_wrench(pose, p_c, f)
    assert np.allclose(w, [0.0, 2.0, 2.0])


def test_batched() -> None:
    pose = np.zeros((4, 3))
    p_c = np.tile(np.array([0.5, 0.0]), (4, 1))
    f = np.tile(np.array([0.0, 1.0]), (4, 1))
    w = contact_force_to_com_wrench(pose, p_c, f)
    assert w.shape == (4, 3)
    assert np.allclose(w[:, 2], 0.5)
