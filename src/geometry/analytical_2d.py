"""Analytical 2D SDFs (Inigo Quilez style) + polygon SDF from nearest-edge / winding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate


@dataclass
class CircleSDF(BaseSDF):
    center_xy: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.center_xy = np.asarray(self.center_xy, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return self.center_xy

    @property
    def bounding_radius(self) -> float:
        return float(self.radius)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        return np.linalg.norm(points - self.center_xy, axis=-1) - self.radius

    def sdf_and_grad(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(points)
        diff = points - self.center_xy
        dist = np.linalg.norm(diff, axis=-1)
        grad = diff / np.clip(dist, 1e-9, None)[..., None]
        return dist - self.radius, grad


@dataclass
class BoxSDF(BaseSDF):
    """Axis-aligned or rotated box; half_extents = [hx, hy], angle in radians."""

    center_xy: np.ndarray
    half_extents: np.ndarray
    angle: float = 0.0

    def __post_init__(self) -> None:
        self.center_xy = np.asarray(self.center_xy, dtype=float)
        self.half_extents = np.asarray(self.half_extents, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return self.center_xy

    @property
    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.half_extents))

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return self.sdf_and_grad(points)[0]

    def sdf_and_grad(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(points)
        local = rotate(-self.angle, points - self.center_xy)
        q = np.abs(local) - self.half_extents
        outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
        inside = np.clip(np.max(q, axis=-1), None, 0.0)

        is_outside = np.any(q > 0.0, axis=-1)
        clipped = np.clip(q, 0.0, None)
        grad_outside = (
            np.sign(local)
            * clipped
            / np.clip(np.linalg.norm(clipped, axis=-1, keepdims=True), 1e-9, None)
        )
        onehot = np.zeros_like(q)
        onehot[np.arange(len(q)), np.argmax(q, axis=-1)] = 1.0
        grad_inside = np.sign(local) * onehot
        grad_local = np.where(is_outside[..., None], grad_outside, grad_inside)
        return outside + inside, rotate(self.angle, grad_local)


class PolygonSDF(BaseSDF):
    """Closed polygon SDF via nearest-edge distance and winding-number inside test."""

    def __init__(self, vertices: np.ndarray, boundary_samples_per_edge: int = 4) -> None:
        self.vertices = np.asarray(vertices, dtype=float)
        self.boundary_samples = self._sample_boundary(boundary_samples_per_edge)
        self.edge_normals = self._edge_normals()
        self._center = self.vertices.mean(axis=0)
        self._bounding_radius = float(
            np.max(np.linalg.norm(self.vertices - self._center, axis=1))
        )

    def _sample_boundary(self, n: int) -> np.ndarray:
        v = self.vertices
        pts = [
            v[i] + (v[(i + 1) % len(v)] - v[i]) * k / n
            for i in range(len(v))
            for k in range(n)
        ]
        return np.asarray(pts, dtype=float)

    def _edge_normals(self) -> np.ndarray:
        v = self.vertices
        edge_vecs = np.roll(v, -1, axis=0) - v
        signed_area = 0.5 * np.sum(
            v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]
        )
        orientation = 1.0 if signed_area > 0 else -1.0
        edge_len = np.linalg.norm(edge_vecs, axis=1, keepdims=True)
        return orientation * np.stack([edge_vecs[:, 1], -edge_vecs[:, 0]], axis=1) / edge_len

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def bounding_radius(self) -> float:
        return self._bounding_radius

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return self.sdf_and_grad(points)[0]

    def sdf_and_grad(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        v = self.vertices
        n = len(v)
        edge_normals = self.edge_normals

        best_dist2 = np.full(len(points), np.inf)
        nearest = np.zeros_like(points)
        nearest_edge_normal = np.zeros_like(points)
        nearest_is_interior = np.zeros(len(points), dtype=bool)
        winding = np.zeros(len(points))

        for i in range(n):
            a, b = v[i], v[(i + 1) % n]
            ab = b - a
            raw_t = ((points - a) @ ab) / (ab @ ab)
            t = np.clip(raw_t, 0.0, 1.0)
            proj = a + t[:, None] * ab
            diff = points - proj
            dist2 = np.einsum("ij,ij->i", diff, diff)
            closer = dist2 < best_dist2
            best_dist2 = np.where(closer, dist2, best_dist2)
            nearest = np.where(closer[:, None], proj, nearest)
            nearest_edge_normal = np.where(closer[:, None], edge_normals[i], nearest_edge_normal)
            nearest_is_interior = np.where(
                closer, (raw_t > 0.0) & (raw_t < 1.0), nearest_is_interior
            )

            upward = (a[1] <= points[:, 1]) & (b[1] > points[:, 1])
            downward = (a[1] > points[:, 1]) & (b[1] <= points[:, 1])
            is_left = (b[0] - a[0]) * (points[:, 1] - a[1]) - (points[:, 0] - a[0]) * (
                b[1] - a[1]
            )
            winding += np.where(upward & (is_left > 0), 1, 0)
            winding += np.where(downward & (is_left < 0), -1, 0)

        sign = np.where(winding != 0, -1.0, 1.0)
        dist = np.sqrt(best_dist2)

        diff = points - nearest
        diff_norm = np.linalg.norm(diff, axis=1, keepdims=True)
        vertex_dir = sign[:, None] * diff / np.clip(diff_norm, 1e-9, None)
        grad = np.where(nearest_is_interior[:, None], nearest_edge_normal, vertex_dir)

        still_degenerate = (~nearest_is_interior) & (diff_norm[:, 0] < 1e-7)
        if np.any(still_degenerate):
            vertex_idx = np.argmin(
                np.linalg.norm(
                    points[still_degenerate][:, None, :] - v[None, :, :], axis=2
                ),
                axis=1,
            )
            prev_edge = (vertex_idx - 1) % n
            averaged = edge_normals[prev_edge] + edge_normals[vertex_idx]
            averaged /= np.clip(np.linalg.norm(averaged, axis=1, keepdims=True), 1e-9, None)
            grad[still_degenerate] = averaged

        return sign * dist, grad


def t_shape_vertices() -> np.ndarray:
    """Capital-T outline in body frame, origin near centroid."""
    return np.array(
        [
            [-0.090, 0.045],
            [0.090, 0.045],
            [0.090, 0.015],
            [0.015, 0.015],
            [0.015, -0.105],
            [-0.015, -0.105],
            [-0.015, 0.015],
            [-0.090, 0.015],
        ],
        dtype=float,
    )


@dataclass
class CapsuleSDF(BaseSDF):
    """Stadium / capsule between endpoints a,b with radius r (Quilez)."""

    a: np.ndarray
    b: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.a + self.b)

    @property
    def bounding_radius(self) -> float:
        return 0.5 * float(np.linalg.norm(self.b - self.a)) + self.radius

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return self.sdf_and_grad(points)[0]

    def sdf_and_grad(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(points)
        pa = points - self.a
        ba = self.b - self.a
        h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
        closest = self.a + h[:, None] * ba
        diff = points - closest
        dist = np.linalg.norm(diff, axis=-1)
        grad = diff / np.clip(dist, 1e-9, None)[..., None]
        return dist - self.radius, grad
