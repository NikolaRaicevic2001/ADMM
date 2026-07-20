"""Generic vectorized MPPI optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mppi.sampler import GaussianSampler
from utils.math_utils import softmax


class MPPIOptimizer:
    """Sampling-based optimizer with arbitrary rollout_fn and cost_fn.

    rollout_fn(initial_state, action_batch) -> info dict or trajectories
    cost_fn(action_batch, rollout_info) -> costs shape (K,)
    """

    def __init__(
        self,
        n_samples: int,
        temperature: float,
        sigma: np.ndarray | float,
        sampler: GaussianSampler | None = None,
        low: np.ndarray | float | None = None,
        high: np.ndarray | float | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.temperature = temperature
        self.sigma = sigma
        self.sampler = sampler or GaussianSampler()
        self.low = low
        self.high = high

    def solve(
        self,
        initial_state: Any,
        nominal: np.ndarray,
        rollout_fn: Callable,
        cost_fn: Callable,
        sigma_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Sample, weight, update nominal.

        Returns (new_nominal, weights, best_rollout_info_from_weighted_mean_resim optional).
        The returned new_nominal is the importance-weighted mean of samples.
        """
        sigma = np.asarray(self.sigma, dtype=float) * sigma_scale
        samples = self.sampler.sample(
            nominal, sigma, self.n_samples, low=self.low, high=self.high
        )
        # Also keep epsilon for mean-shift style updates if needed
        eps = samples - nominal[None]
        rollout_info = rollout_fn(initial_state, samples)
        costs = cost_fn(samples, rollout_info)
        weights = softmax(-costs / self.temperature)
        new_nominal = nominal + np.tensordot(weights, eps, axes=(0, 0))
        if self.low is not None or self.high is not None:
            new_nominal = np.clip(new_nominal, self.low, self.high)
        return new_nominal, weights, rollout_info
