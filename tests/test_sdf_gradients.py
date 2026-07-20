"""Finite-difference checks on SDF gradients."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.analytical_2d import BoxSDF, CircleSDF, PolygonSDF, t_shape_vertices


def _fd_grad(sdf_fn, p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    g = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = eps
        g[i] = (sdf_fn(p + e) - sdf_fn(p - e)) / (2.0 * eps)
    return g


@pytest.mark.parametrize(
    "shape,point",
    [
        (CircleSDF(np.array([0.0, 0.0]), 1.0), np.array([1.5, 0.3])),
        (BoxSDF(np.array([0.0, 0.0]), np.array([0.5, 0.3]), 0.2), np.array([1.0, 0.4])),
        (PolygonSDF(t_shape_vertices()), np.array([0.2, 0.1])),
    ],
)
def test_sdf_gradient_matches_fd(shape, point) -> None:
    d, grad = shape.sdf_and_grad(point[None, :])
    assert d.shape == (1,)
    assert grad.shape == (1, 2)
    fd = _fd_grad(lambda q: float(shape.sdf(q[None, :])[0]), point)
    # Normalize FD for comparison (eikonal ||grad||≈1 outside)
    fd_n = fd / (np.linalg.norm(fd) + 1e-12)
    g_n = grad[0] / (np.linalg.norm(grad[0]) + 1e-12)
    assert np.allclose(g_n, fd_n, atol=5e-2)


def test_circle_outside_positive() -> None:
    c = CircleSDF(np.zeros(2), 1.0)
    assert c.sdf(np.array([[2.0, 0.0]]))[0] > 0
    assert c.sdf(np.array([[0.0, 0.0]]))[0] < 0
