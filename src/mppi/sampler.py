"""Gaussian perturbation sampler for MPPI."""

from __future__ import annotations

import numpy as np


class GaussianSampler:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def sample(
        self,
        nominal: np.ndarray,
        sigma: np.ndarray | float,
        n_samples: int,
        low: np.ndarray | float | None = None,
        high: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Return (K, *nominal.shape) = nominal + N(0, sigma^2), optionally clipped."""
        noise = self.rng.standard_normal((n_samples,) + nominal.shape) * sigma
        samples = nominal[None] + noise
        if low is not None or high is not None:
            samples = np.clip(samples, low, high)
        return samples
