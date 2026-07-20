"""Small vector / SE(2) helpers used across geometry, dynamics, and ADMM."""

from __future__ import annotations

import numpy as np


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rotate(theta: np.ndarray | float, v: np.ndarray) -> np.ndarray:
    """Rotate 2D vector(s) v (..., 2) by angle(s) theta."""
    c, s = np.cos(theta), np.sin(theta)
    vx, vy = v[..., 0], v[..., 1]
    return np.stack([c * vx - s * vy, s * vx + c * vy], axis=-1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def goal_cost(
    poses: np.ndarray,
    goal: np.ndarray,
    w_pos: float,
    w_theta: float,
) -> np.ndarray:
    """Quadratic SE(2) goal cost; returns shape (N,)."""
    poses = np.atleast_2d(poses)
    diff_pos = poses[:, :2] - goal[:2]
    diff_theta = wrap_angle(poses[:, 2] - goal[2])
    return w_pos * np.einsum("ij,ij->i", diff_pos, diff_pos) + w_theta * diff_theta**2


def shift_horizon_zero_tail(seq: np.ndarray) -> np.ndarray:
    """Receding-horizon shift: seq[t] <- seq[t+1], last slot <- 0."""
    out = np.roll(seq, -1, axis=0).copy()
    out[-1] = 0.0
    return out
