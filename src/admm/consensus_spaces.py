"""Consensus space interfaces. Phase 1: WrenchConsensus only.

ObjectStateConsensus can be swapped in later without changing ADMMSolver,
as long as it implements BaseConsensusSpace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseConsensusSpace(ABC):
    @abstractmethod
    def penalty_cost(
        self, actual_var: np.ndarray, z: np.ndarray, dual: np.ndarray
    ) -> float:
        """(rho / 2) * ||actual_var - z + dual||^2"""

    @abstractmethod
    def z_update(self, w_obj: np.ndarray, w_rob: np.ndarray) -> np.ndarray:
        """Consensus z = 0.5 * (w_obj + w_rob)."""

    @abstractmethod
    def dual_update(
        self, actual_var: np.ndarray, z: np.ndarray, dual: np.ndarray
    ) -> np.ndarray:
        """Dual step with anti-windup clamping."""


class WrenchConsensus(BaseConsensusSpace):
    """World-frame CoM wrench consensus; arrays shaped (H, 3)."""

    def __init__(self, horizon: int, rho: float, max_dual: float = 10.0) -> None:
        self.H = horizon
        self.dim = 3
        self.rho = float(rho)
        self.max_dual = float(max_dual)

    def penalty_cost(
        self, actual_w: np.ndarray, z: np.ndarray, dual: np.ndarray
    ) -> float:
        diff = actual_w - z + dual
        return float(0.5 * self.rho * np.sum(diff**2))

    def penalty_cost_batch(
        self, actual_w: np.ndarray, z: np.ndarray, dual: np.ndarray
    ) -> np.ndarray:
        """Batch over leading sample dim K: actual_w (K, H, 3) -> costs (K,)."""
        diff = actual_w - z[None] + dual[None]
        return 0.5 * self.rho * np.sum(diff**2, axis=(1, 2))

    def z_update(self, w_obj: np.ndarray, w_rob: np.ndarray) -> np.ndarray:
        return 0.5 * (w_obj + w_rob)

    def dual_update(
        self, actual_w: np.ndarray, z: np.ndarray, dual: np.ndarray
    ) -> np.ndarray:
        new_dual = dual + (actual_w - z)
        return np.clip(new_dual, -self.max_dual, self.max_dual)
