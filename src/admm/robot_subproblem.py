"""Robot-level subproblem: MPPI over velocity controls u^r in R^2."""

from __future__ import annotations

from typing import Any

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import robot_obstacle_cost
from dynamics.physics_engine import PhysicsEngine2D
from geometry.base_sdf import BaseSDF
from mppi.mppi_core import MPPIOptimizer
from mppi.sampler import GaussianSampler
from utils.math_utils import goal_cost


class RobotSubproblem:
    def __init__(
        self,
        object_: QuasiStaticObject2D,
        obstacles: list[BaseSDF],
        cfg: dict[str, Any],
        consensus: WrenchConsensus,
        rng: np.random.Generator,
        planning_engine: PhysicsEngine2D,
        goal: np.ndarray | None = None,
    ) -> None:
        self.object_ = object_
        self.obstacles = obstacles
        self.cfg = cfg
        self.consensus = consensus
        self.rng = rng
        self.planning_engine = planning_engine
        self.goal = (
            np.asarray(goal, dtype=float).reshape(3)
            if goal is not None
            else np.zeros(3)
        )
        self.horizon = int(cfg["horizon"])
        k_robot = int(cfg["k_robot"])
        self.robot_max_speed = float(
            cfg.get("robot_max_speed", cfg.get("seek_max_speed", 1.0))
        )
        self.ell_c_source = str(cfg.get("ell_c_source", "implied")).lower().strip()
        if self.ell_c_source not in ("implied", "live"):
            raise ValueError(
                f"Unknown ell_c_source '{self.ell_c_source}'. Choose from: implied, live"
            )
        self.mppi = MPPIOptimizer(
            n_samples=k_robot,
            temperature=float(cfg["nu_r"]),
            sigma=np.asarray(cfg["sigma_robot"], dtype=float),
            sampler=GaussianSampler(rng),
        )

    def _project_u(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).copy()
        speeds = np.linalg.norm(u, axis=-1, keepdims=True)
        scale = np.ones_like(speeds)
        mask = speeds[..., 0] > self.robot_max_speed
        scale[mask] = self.robot_max_speed / np.maximum(speeds[mask], 1e-12)
        return u * scale

    def _implied_poses(
        self, pose0: np.ndarray, wrenches: np.ndarray, dt: float
    ) -> np.ndarray:
        """Integrate quasi-static D under realized wrenches: (K,H,3) or (H,3)."""
        w = np.asarray(wrenches, dtype=float)
        single = w.ndim == 2
        if single:
            w = w[None]
        k, h, _ = w.shape
        poses = np.zeros((k, h, 3))
        pose = np.tile(np.asarray(pose0, dtype=float).reshape(3), (k, 1))
        for t in range(h):
            pose = self.object_.propagate(pose, w[:, t], dt)
            poses[:, t] = pose
        return poses[0] if single else poses

    def _coupling_poses(
        self,
        pose0: np.ndarray,
        wrenches: np.ndarray,
        live_poses: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if self.ell_c_source == "live":
            return np.asarray(live_poses, dtype=float)
        return self._implied_poses(pose0, wrenches, dt)

    def _rollout(
        self,
        robot_pos0: np.ndarray,
        pose0: np.ndarray,
        ref_poses: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """actions: (K, H, 2) velocities -> robot traj + realized wrenches."""
        dt = float(self.cfg["dt"])
        wrenches, positions, object_poses = self.planning_engine.rollout_batch(
            actions, ref_poses, robot_pos0, dt
        )
        implied = self._implied_poses(pose0, wrenches, dt)
        return {
            "wrenches": wrenches,
            "positions": positions,
            "object_poses": object_poses,
            "implied_poses": implied,
        }

    def _cost(
        self,
        actions: np.ndarray,
        info: dict[str, np.ndarray],
        z: np.ndarray,
        gamma_r: np.ndarray,
        ref_poses: np.ndarray,
        pose0: np.ndarray,
    ) -> np.ndarray:
        wrenches = info["wrenches"]
        positions = info["positions"]
        dt = float(self.cfg["dt"])
        coup_poses = self._coupling_poses(
            pose0, wrenches, info["object_poses"], dt
        )
        k, h, _ = actions.shape
        cost = float(self.cfg["r_r"]) * np.sum(actions**2, axis=(1, 2))
        w_c_pos = float(self.cfg.get("w_c_pos", 0.0))
        w_c_theta = float(self.cfg.get("w_c_theta", 0.0))
        w_g_pos = float(self.cfg.get("w_rob_goal_pos", 0.0))
        w_g_theta = float(self.cfg.get("w_rob_goal_theta", 0.0))
        for t in range(h):
            cost += robot_obstacle_cost(
                positions[:, t],
                self.obstacles,
                float(self.cfg["obstacle_margin"]),
                float(self.cfg["w_obstacle"]),
            )
            cost += 0.5 * self.consensus.rho * np.sum(
                (wrenches[:, t] - z[t] + gamma_r[t]) ** 2, axis=1
            )
            if w_c_pos > 0.0 or w_c_theta > 0.0:
                cost += goal_cost(
                    coup_poses[:, t], ref_poses[t], w_c_pos, w_c_theta
                )
            if w_g_pos > 0.0 or w_g_theta > 0.0:
                cost += goal_cost(
                    coup_poses[:, t], self.goal, w_g_pos, w_g_theta
                )
        return cost

    def _nominal_rollout(
        self,
        robot_pos0: np.ndarray,
        pose0: np.ndarray,
        ref_poses: np.ndarray,
        u_nom: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns w_rob, robot_path, object_poses, implied_poses."""
        dt = float(self.cfg["dt"])
        wrenches, path, obj = self.planning_engine.rollout_batch(
            np.asarray(u_nom, dtype=float)[None], ref_poses, robot_pos0, dt
        )
        implied = self._implied_poses(pose0, wrenches[0], dt)
        return wrenches[0], path[0], obj[0], implied

    def _cost_components(
        self,
        actions: np.ndarray,
        info: dict[str, np.ndarray],
        z: np.ndarray,
        gamma_r: np.ndarray,
        ref_poses: np.ndarray,
        pose0: np.ndarray,
    ) -> dict[str, float]:
        """Nominal cost breakdown (actions shape (H,2))."""
        wrenches = info["wrenches"]
        positions = info["positions"]
        object_poses = info.get("object_poses")
        if wrenches.ndim == 2:
            wrenches = wrenches[None]
            positions = positions[None]
            if object_poses is not None and object_poses.ndim == 2:
                object_poses = object_poses[None]
            actions_b = actions[None]
        else:
            actions_b = actions
        if object_poses is None:
            object_poses = np.zeros_like(wrenches)
        dt = float(self.cfg["dt"])
        coup = self._coupling_poses(pose0, wrenches, object_poses, dt)
        effort = float(self.cfg["r_r"]) * float(np.sum(actions_b[0] ** 2))
        obs = 0.0
        admm = 0.0
        coupling = 0.0
        rob_goal = 0.0
        w_c_pos = float(self.cfg.get("w_c_pos", 0.0))
        w_c_theta = float(self.cfg.get("w_c_theta", 0.0))
        w_g_pos = float(self.cfg.get("w_rob_goal_pos", 0.0))
        w_g_theta = float(self.cfg.get("w_rob_goal_theta", 0.0))
        for t in range(actions_b.shape[1]):
            obs += float(
                robot_obstacle_cost(
                    positions[0:1, t],
                    self.obstacles,
                    float(self.cfg["obstacle_margin"]),
                    float(self.cfg["w_obstacle"]),
                )[0]
            )
            admm += 0.5 * self.consensus.rho * float(
                np.sum((wrenches[0, t] - z[t] + gamma_r[t]) ** 2)
            )
            if w_c_pos > 0.0 or w_c_theta > 0.0:
                coupling += float(
                    goal_cost(coup[0:1, t], ref_poses[t], w_c_pos, w_c_theta)[0]
                )
            if w_g_pos > 0.0 or w_g_theta > 0.0:
                rob_goal += float(
                    goal_cost(coup[0:1, t], self.goal, w_g_pos, w_g_theta)[0]
                )
        return {
            "rob_effort_cost": effort,
            "rob_obstacle_cost": obs,
            "rob_admm_penalty": admm,
            "rob_coupling_cost": coupling,
            "rob_goal_cost": rob_goal,
        }

    def solve(
        self,
        robot_pos0: np.ndarray,
        pose0: np.ndarray,
        ref_poses: np.ndarray,
        u_nom: np.ndarray,
        z: np.ndarray,
        gamma_r: np.ndarray,
        sigma_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Returns u_nom, w_rob, robot_path, diagnostics."""
        pose0 = np.asarray(pose0, dtype=float).reshape(3)
        ref_poses = np.asarray(ref_poses, dtype=float)
        u_nom, _, rollout_info, samples = self.mppi.solve(
            u_nom,
            rollout_fn=lambda actions: self._rollout(
                robot_pos0, pose0, ref_poses, actions
            ),
            cost_fn=lambda actions, info: self._cost(
                actions, info, z, gamma_r, ref_poses, pose0
            ),
            project_fn=self._project_u,
            sigma_scale=sigma_scale,
        )
        u_nom = self._project_u(u_nom)
        w_rob, robot_path, live_obj, implied = self._nominal_rollout(
            robot_pos0, pose0, ref_poses, u_nom
        )

        positions = rollout_info["positions"]
        k = positions.shape[0]
        start = np.broadcast_to(robot_pos0.reshape(1, 1, 2), (k, 1, 2))
        fan = np.concatenate([start, positions], axis=1)
        nom_info = {
            "wrenches": w_rob,
            "positions": robot_path,
            "object_poses": live_obj,
            "implied_poses": implied,
        }
        costs = self._cost_components(
            u_nom, nom_info, z, gamma_r, ref_poses, pose0
        )
        diagnostics = {
            "robot_rollouts": fan,
            "implied_poses": implied,
            "live_object_poses": live_obj,
            **costs,
        }
        return u_nom, w_rob, robot_path, diagnostics
