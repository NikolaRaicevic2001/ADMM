"""Analytical physics backend wrapping simulate_contact_step."""

from __future__ import annotations

from typing import Any

import numpy as np

from contact.resolution import simulate_contact_step
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import push_point_out_of_obstacles
from dynamics.physics_engine import EnginePair, PhysicsEngine2D
from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate


class AnalyticalPhysicsEngine(PhysicsEngine2D):
    """SDF contact adapter exposing the PhysicsEngine2D contract."""

    def __init__(
        self,
        object_: QuasiStaticObject2D,
        obstacles: list[BaseSDF],
        cfg: dict[str, Any],
        *,
        planning: bool,
    ) -> None:
        self.object_ = object_
        self.obstacles = obstacles
        self.cfg = cfg
        self.planning = planning
        self.robot_radius = float(cfg.get("robot_radius", 0.012))
        self._object_pose = object_.pose.copy()
        self._robot_pos = np.zeros(2)

    def _params(self, *, freeze_object: bool) -> dict[str, Any]:
        return dict(
            f_max=float(self.cfg["f_max"]),
            mu_c=float(self.cfg["mu_c"]),
            obstacles=self.obstacles,
            n_substeps=int(self.cfg["n_contact_substeps"]),
            contact_step_margin=float(self.cfg["contact_step_margin"]),
            max_contact_step=float(self.cfg["max_contact_step"]),
            obstacle_margin=float(self.cfg["obstacle_margin"]),
            pushout_iters=int(self.cfg["object_pushout_iters"]),
            freeze_object=freeze_object,
        )

    def _push_robot_clear(self) -> None:
        """CPU SDF push-out (analytical only; never used inside MJX JIT)."""
        pose = self._object_pose
        rob = self._robot_pos.copy()
        q = rotate(-pose[2], rob - pose[:2])
        d, grad = self.object_.shape.sdf_and_grad(q)
        d = float(np.asarray(d).reshape(()))
        grad = np.asarray(grad, dtype=float).reshape(2)
        if d < self.robot_radius:
            n_world = rotate(pose[2], grad)
            n_norm = float(np.linalg.norm(n_world))
            if n_norm > 1e-12:
                n_world /= n_norm
                rob = rob + (self.robot_radius - d + 1e-4) * n_world
        self._robot_pos = push_point_out_of_obstacles(rob, self.obstacles)

    def _push_robot_clear_batch(
        self, pose_batch: np.ndarray, rob_batch: np.ndarray
    ) -> np.ndarray:
        """Vectorized form of ``_push_robot_clear`` for a batch of (pose, robot_pos)
        pairs. Same math as the scalar version above, applied to all samples in
        one call instead of one Python-level call per sample (see
        CODE_CHANGES_LOG.md — this batching is what makes ``rollout_batch``
        fast; looping this per-sample was the root cause of a ~15-20x slowdown).
        """
        q = rotate(-pose_batch[:, 2], rob_batch - pose_batch[:, :2])
        d, grad = self.object_.shape.sdf_and_grad(q)
        n_world = rotate(pose_batch[:, 2], grad)
        n_norm = np.linalg.norm(n_world, axis=-1, keepdims=True)
        n_world = np.where(n_norm > 1e-12, n_world / np.clip(n_norm, 1e-12, None), 0.0)
        needs_push = (d < self.robot_radius)[:, None]
        push_dist = (self.robot_radius - d)[:, None] + 1e-4
        rob_pushed = rob_batch + np.where(needs_push, push_dist * n_world, 0.0)
        return push_point_out_of_obstacles(rob_pushed, self.obstacles)

    def seed(self, object_pose: np.ndarray, robot_pos: np.ndarray) -> None:
        self._object_pose = np.asarray(object_pose, dtype=float).reshape(3).copy()
        self._robot_pos = np.asarray(robot_pos, dtype=float).reshape(2).copy()

    def step_execution(
        self, u_cmd: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.planning:
            raise RuntimeError("step_execution is only valid on the execution engine")
        # Not calling _push_robot_clear() here: this step didn't exist in the
        # original (pre-MJX-refactor) analytical execution path, and adding
        # it changed the validated closed-loop behavior as an unintended
        # side effect of introducing robot_radius for MJX parity. See
        # CODE_CHANGES_LOG.md. simulate_contact_step's own internal
        # collision-safety logic (displacement capping, obstacle push-out)
        # is unaffected and still applies.
        new_pose, new_robot, _ = simulate_contact_step(
            self.object_,
            self._object_pose,
            self._robot_pos,
            np.asarray(u_cmd, dtype=float).reshape(2),
            float(dt),
            **self._params(freeze_object=False),
        )
        self._object_pose = np.asarray(new_pose, dtype=float).reshape(3).copy()
        self._robot_pos = np.asarray(new_robot, dtype=float).reshape(2).copy()
        return self._object_pose.copy(), self._robot_pos.copy()

    def rollout_batch(
        self,
        u_seq: np.ndarray,
        ref_poses: np.ndarray,
        robot_pos0: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.planning:
            raise RuntimeError("rollout_batch is only valid on the planning engine")
        u_seq = np.asarray(u_seq, dtype=float)
        ref_poses = np.asarray(ref_poses, dtype=float)
        robot_pos0 = np.asarray(robot_pos0, dtype=float).reshape(2)
        k, h, _ = u_seq.shape
        wrenches_out = np.zeros((k, h, 3))
        paths_out = np.zeros((k, h, 2))
        params = self._params(freeze_object=True)

        # Batched across all k samples per horizon step (not per-sample) —
        # simulate_contact_step and everything under it already support a
        # batched leading dimension. See CODE_CHANGES_LOG.md.
        rob = np.tile(robot_pos0, (k, 1))
        for t in range(h):
            pose_t = np.tile(np.asarray(ref_poses[t], dtype=float).reshape(3), (k, 1))
            # Not calling _push_robot_clear_batch() here -- see the matching
            # note in step_execution() above; this step didn't exist in the
            # original (pre-MJX-refactor) rollout and changed validated
            # behavior as an unintended side effect. Kept as a method (used
            # nowhere now) rather than deleted, in case robot_radius-aware
            # push-out is deliberately wanted back for the analytical
            # backend later.
            _, new_robot, wrench = simulate_contact_step(
                self.object_,
                pose_t,
                rob,
                u_seq[:, t],
                float(dt),
                **params,
            )
            wrenches_out[:, t] = np.asarray(wrench).reshape(k, 3)
            paths_out[:, t] = np.asarray(new_robot).reshape(k, 2)
            rob = paths_out[:, t].copy()
        return wrenches_out, paths_out


def build_analytical_engine_pair(
    cfg: dict[str, Any],
    object_: QuasiStaticObject2D,
    obstacles: list[BaseSDF],
) -> EnginePair:
    return EnginePair(
        planning=AnalyticalPhysicsEngine(object_, obstacles, cfg, planning=True),
        execution=AnalyticalPhysicsEngine(object_, obstacles, cfg, planning=False),
    )
