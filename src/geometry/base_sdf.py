"""Abstract signed-distance interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSDF(ABC):
    """Signed-distance field with optional analytic gradient."""

    @abstractmethod
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """Signed distance; positive outside. Shape (...,)."""

    @abstractmethod
    def sdf_and_grad(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (distance, unit outward gradient) with shapes (...,) and (..., 2)."""

    def project_to_boundary(self, points: np.ndarray) -> np.ndarray:
        """Pi(p) = p - d(p) grad d(p)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        d, grad = self.sdf_and_grad(points)
        return points - d[..., None] * grad

    @property
    @abstractmethod
    def center(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def bounding_radius(self) -> float:
        ...
