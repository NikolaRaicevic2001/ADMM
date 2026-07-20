"""Kinematic velocity-controlled 2D robot (point agent)."""

from __future__ import annotations

import numpy as np

from dynamics.base_dynamics import BaseDynamics


class KinematicRobot2D(BaseDynamics):
    """State = [px, py]; control = [vx, vy]. Pure kinematics (contact handled elsewhere)."""

    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=float).copy()

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def control_dim(self) -> int:
        return 2

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        return state + dt * control
