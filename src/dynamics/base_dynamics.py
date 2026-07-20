"""Abstract dynamics interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDynamics(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def control_dim(self) -> int:
        ...

    @abstractmethod
    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Advance one timestep: state' = f(state, control, dt)."""

    def rollout(
        self, state0: np.ndarray, controls: np.ndarray, dt: float
    ) -> np.ndarray:
        """Roll out controls of shape (H, nu) -> states (H+1, nx) including state0."""
        h = controls.shape[0]
        states = np.zeros((h + 1, self.state_dim))
        states[0] = state0
        x = state0.copy()
        for t in range(h):
            x = self.step(x, controls[t], dt)
            states[t + 1] = x
        return states
