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


class RobotSubproblem:
    def __init__(
        self,
        object_: QuasiStaticObject2D,
        obstacles: list[BaseSDF],
        cfg: dict[str, Any],
        consensus: WrenchConsensus,
        rng: np.random.Generator,
        planning_engine: PhysicsEngine2D,
    ) -> None:
        self.object_ = object_
        self.obstacles = obstacles
        self.cfg = cfg
        self.consensus = consensus
        self.rng = rng
        self.planning_engine = planning_engine
        self.horizon = int(cfg["horizon"])
        k_robot = int(cfg["k_robot"])
        self.robot_max_speed = float(
            cfg.get("robot_max_speed", cfg.get("seek_max_speed", 1.0))
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

    def _rollout(
        self,
        robot_pos0: np.ndarray,
        ref_poses: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """actions: (K, H, 2) velocities -> robot traj + realized wrenches."""
        dt = float(self.cfg["dt"])
        wrenches, positions = self.planning_engine.rollout_batch(
            actions, ref_poses, robot_pos0, dt
        )
        return {"wrenches": wrenches, "positions": positions}

    def _cost(
        self,
        actions: np.ndarray,
        info: dict[str, np.ndarray],
        z: np.ndarray,
        gamma_r: np.ndarray,
    ) -> np.ndarray:
        wrenches = info["wrenches"]
        positions = info["positions"]
        k, h, _ = actions.shape
        cost = float(self.cfg["r_r"]) * np.sum(actions**2, axis=(1, 2))
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
        return cost

    def _nominal_rollout(
        self, robot_pos0: np.ndarray, ref_poses: np.ndarray, u_nom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-simulate nominal u: returns w_rob (H,3), robot_path (H,2)."""
        dt = float(self.cfg["dt"])
        wrenches, path = self.planning_engine.rollout_batch(
            np.asarray(u_nom, dtype=float)[None], ref_poses, robot_pos0, dt
        )
        return wrenches[0], path[0]

    def _cost_components(
        self,
        actions: np.ndarray,
        info: dict[str, np.ndarray],
        z: np.ndarray,
        gamma_r: np.ndarray,
    ) -> dict[str, float]:
        """Nominal cost breakdown (actions shape (H,2))."""
        wrenches = info["wrenches"]
        positions = info["positions"]
        if wrenches.ndim == 2:
            wrenches = wrenches[None]
            positions = positions[None]
            actions_b = actions[None]
        else:
            actions_b = actions
        effort = float(self.cfg["r_r"]) * float(np.sum(actions_b[0] ** 2))
        obs = 0.0
        admm = 0.0
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
        return {
            "rob_effort_cost": effort,
            "rob_obstacle_cost": obs,
            "rob_admm_penalty": admm,
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
        del pose0
        u_nom, _, rollout_info, samples = self.mppi.solve(
            u_nom,
            rollout_fn=lambda actions: self._rollout(robot_pos0, ref_poses, actions),
            cost_fn=lambda actions, info: self._cost(actions, info, z, gamma_r),
            project_fn=self._project_u,
            sigma_scale=sigma_scale,
        )
        u_nom = self._project_u(u_nom)
        w_rob, robot_path = self._nominal_rollout(robot_pos0, ref_poses, u_nom)

        positions = rollout_info["positions"]
        k = positions.shape[0]
        start = np.broadcast_to(robot_pos0.reshape(1, 1, 2), (k, 1, 2))
        fan = np.concatenate([start, positions], axis=1)
        nom_info = {"wrenches": w_rob, "positions": robot_path}
        costs = self._cost_components(u_nom, nom_info, z, gamma_r)
        diagnostics = {"robot_rollouts": fan, **costs}
        return u_nom, w_rob, robot_path, diagnostics
