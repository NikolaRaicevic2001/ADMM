"""Physics engine abstraction for contact rollouts and execution.

NumPy-only algorithm boundary. Backends: analytical | mjx.
Designed so Warp (later) implements the same three methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamics.object_2d import QuasiStaticObject2D
from geometry.base_sdf import BaseSDF


@dataclass
class EnginePair:
    """Planning (ref-pose object) + execution (coupled plant) worlds."""

    planning: "PhysicsEngine2D"
    execution: "PhysicsEngine2D"


class PhysicsEngine2D(ABC):
    """Backend-agnostic 2D contact physics for robot MPPI and MPC execution.

    All public I/O is ``numpy.ndarray``. JAX/MuJoCo types stay inside MJX.
    """

    @abstractmethod
    def seed(self, object_pose: np.ndarray, robot_pos: np.ndarray) -> None:
        """Set the execution-world base state (MPC plant). NumPy only."""

    @abstractmethod
    def step_execution(
        self, u_cmd: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Step the coupled execution world by one control timestep.

        Returns: object_pose (3,), robot_pos (2,).
        """

    @abstractmethod
    def rollout_batch(
        self,
        u_seq: np.ndarray,
        ref_poses: np.ndarray,
        robot_pos0: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Planning MPPI rollouts. Pure from the caller's perspective.

        Args:
            u_seq: (K, H, 2) velocity control sequences
            ref_poses: (H, 3) object poses for the planning world
            robot_pos0: (2,) robot start
            dt: control timestep

        Returns:
            wrenches: (K, H, 3) world CoM wrenches [fx, fy, tau]
            paths: (K, H, 2) robot position trajectories
        """


def build_engine_pair(
    cfg: dict[str, Any],
    object_: QuasiStaticObject2D,
    obstacles: list[BaseSDF],
) -> EnginePair:
    """Construct planning + execution engines from ``physics_backend``."""
    backend = str(cfg.get("physics_backend", "analytical")).lower().strip()
    if backend == "analytical":
        from dynamics.analytical_engine import build_analytical_engine_pair

        return build_analytical_engine_pair(cfg, object_, obstacles)
    if backend == "mjx":
        from dynamics.mjx_engine import build_mjx_engine_pair

        return build_mjx_engine_pair(cfg, object_, obstacles)
    raise ValueError(
        f"Unknown physics_backend '{backend}'. Choose from: analytical, mjx"
    )
