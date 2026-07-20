"""Quasi-static SE(2) object with limit-surface compliance."""

from __future__ import annotations

import numpy as np

from dynamics.base_dynamics import BaseDynamics
from dynamics.obstacles import push_object_out_of_obstacles
from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate, wrap_angle


class QuasiStaticObject2D(BaseDynamics):
    """State x = [px, py, theta]; control = world CoM wrench w in R^3."""

    def __init__(
        self,
        shape: BaseSDF,
        pose: np.ndarray,
        mu: float,
        mass: float,
        gravity: float,
        limit_surface_c: float,
        limit_surface_r: float,
        obstacles: list[BaseSDF] | None = None,
        obstacle_margin: float = 0.015,
        pushout_iters: int = 4,
    ) -> None:
        d_trans = 1.0 / (mu * mass * gravity)
        d_rot = 1.0 / (limit_surface_c * limit_surface_r * mu * mass * gravity)
        self.D = np.array([d_trans, d_trans, d_rot], dtype=float)
        self.shape = shape
        self.pose = np.asarray(pose, dtype=float).copy()
        self.obstacles = obstacles or []
        self.obstacle_margin = obstacle_margin
        self.pushout_iters = pushout_iters

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def control_dim(self) -> int:
        return 3

    def body_frame_point(self, world_point: np.ndarray, pose: np.ndarray) -> np.ndarray:
        return rotate(-pose[..., 2], world_point - pose[..., :2])

    def geometry(
        self, body_point: np.ndarray, theta: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """World inward normal/tangent and body moment arms at body_point."""
        _, grad = self.shape.sdf_and_grad(body_point)
        n_body = -grad
        t_body = np.stack([-n_body[..., 1], n_body[..., 0]], axis=-1)
        gamma_n = body_point[..., 0] * n_body[..., 1] - body_point[..., 1] * n_body[..., 0]
        gamma_t = body_point[..., 0] * t_body[..., 1] - body_point[..., 1] * t_body[..., 0]
        return rotate(theta, n_body), rotate(theta, t_body), gamma_n, gamma_t

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        new_pose = state + dt * self.D * control
        new_pose = np.asarray(new_pose, dtype=float)
        new_pose[..., 2] = wrap_angle(new_pose[..., 2])
        if self.obstacles:
            new_pose = push_object_out_of_obstacles(
                self.shape,
                new_pose,
                self.obstacles,
                self.obstacle_margin,
                self.pushout_iters,
            )
        return new_pose

    def propagate(self, pose: np.ndarray, w_o: np.ndarray, dt: float) -> np.ndarray:
        return self.step(pose, w_o, dt)

    def world_vertices(self, pose: np.ndarray | None = None) -> np.ndarray:
        pose = self.pose if pose is None else pose
        if not hasattr(self.shape, "vertices"):
            raise AttributeError("shape has no vertices")
        return pose[:2] + rotate(pose[2], self.shape.vertices)
