"""Generic vectorized MPPI: one algorithm, arbitrary action spaces.

The optimizer only implements:
  sample -> rollout -> cost -> softmax weights -> aggregate nominal

Domain-specific logic (contact rejection sampling, friction cones, contact
simulation) lives in the caller's sample_fn / rollout_fn / cost_fn /
project_fn closures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mppi.sampler import GaussianSampler
from utils.math_utils import softmax

# sample_fn(nominal, n_samples, sigma_scale) -> samples with shape (K, *nominal.shape)
SampleFn = Callable[[np.ndarray, int, float], np.ndarray]
# rollout_fn(samples) -> opaque info used by cost_fn
RolloutFn = Callable[[np.ndarray], Any]
# cost_fn(samples, rollout_info) -> costs shape (K,)
CostFn = Callable[[np.ndarray, Any], np.ndarray]
# project_fn(nominal) -> feasible nominal (optional)
ProjectFn = Callable[[np.ndarray], np.ndarray]
# aggregate_fn(nominal, samples, weights) -> new_nominal (optional)
AggregateFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def weighted_mean_aggregate(
    nominal: np.ndarray, samples: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Importance-weighted mean of samples (equals u + sum w_k eps_k for additive noise)."""
    # samples: (K, *shape), weights: (K,)
    leading = samples.ndim
    # einsum over sample axis 0
    w = weights.reshape((weights.shape[0],) + (1,) * (leading - 1))
    return np.sum(w * samples, axis=0)


class MPPIOptimizer:
    """Reusable MPPI core shared by object, robot, or any other action vector."""

    def __init__(
        self,
        n_samples: int,
        temperature: float,
        sigma: np.ndarray | float | None = None,
        sampler: GaussianSampler | None = None,
        low: np.ndarray | float | None = None,
        high: np.ndarray | float | None = None,
    ) -> None:
        self.n_samples = int(n_samples)
        self.temperature = float(temperature)
        self.sigma = sigma
        self.sampler = sampler or GaussianSampler()
        self.low = low
        self.high = high

    def _default_sample(
        self, nominal: np.ndarray, n_samples: int, sigma_scale: float
    ) -> np.ndarray:
        if self.sigma is None:
            raise ValueError("sigma must be set for default Gaussian sampling")
        sigma = np.asarray(self.sigma, dtype=float) * sigma_scale
        return self.sampler.sample(
            nominal, sigma, n_samples, low=self.low, high=self.high
        )

    def solve(
        self,
        nominal: np.ndarray,
        rollout_fn: RolloutFn,
        cost_fn: CostFn,
        *,
        sample_fn: SampleFn | None = None,
        project_fn: ProjectFn | None = None,
        aggregate_fn: AggregateFn | None = None,
        sigma_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Run one MPPI update.

        Parameters
        ----------
        nominal:
            Current nominal action trajectory, shape (H, ...) or any array.
        rollout_fn:
            Maps action batch (K, *nominal.shape) -> rollout info.
        cost_fn:
            Maps (actions, rollout_info) -> costs (K,).
        sample_fn:
            Optional custom sampler. If None, uses Gaussian around ``nominal``.
        project_fn:
            Optional feasibility projection applied to the updated nominal.
        aggregate_fn:
            Optional custom aggregation. Default: weighted mean of samples.
        sigma_scale:
            Scales default Gaussian sigma (ignored if sample_fn is provided,
            unless the custom sampler uses it).

        Returns
        -------
        new_nominal, weights, rollout_info
        """
        if sample_fn is None:
            samples = self._default_sample(nominal, self.n_samples, sigma_scale)
        else:
            samples = sample_fn(nominal, self.n_samples, sigma_scale)

        rollout_info = rollout_fn(samples)
        costs = np.asarray(cost_fn(samples, rollout_info), dtype=float).reshape(-1)
        if costs.shape[0] != samples.shape[0]:
            raise ValueError(
                f"cost_fn returned shape {costs.shape}, expected ({samples.shape[0]},)"
            )

        weights = softmax(-costs / self.temperature)
        agg = aggregate_fn or weighted_mean_aggregate
        new_nominal = agg(nominal, samples, weights)

        if self.low is not None or self.high is not None:
            new_nominal = np.clip(new_nominal, self.low, self.high)
        if project_fn is not None:
            new_nominal = project_fn(new_nominal)

        return new_nominal, weights, rollout_info
