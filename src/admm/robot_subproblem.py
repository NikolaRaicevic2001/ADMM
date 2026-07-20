"""Robot-level subproblem: MPPI over velocity controls u^r in R^2."""

from __future__ import annotations

from typing import Any

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from contact.resolution import simulate_contact_step
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import robot_obstacle_cost
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
    ) -> None:
        self.object_ = object_
        self.obstacles = obstacles
        self.cfg = cfg
        self.consensus = consensus
        self.rng = rng
        self.horizon = int(cfg["horizon"])
        self.mppi = MPPIOptimizer(
            n_samples=int(cfg["k_robot"]),
            temperature=float(cfg["nu_r"]),
            sigma=np.asarray(cfg["sigma_robot"], dtype=float),
            sampler=GaussianSampler(rng),
        )

    def _sim_params(self) -> dict[str, Any]:
        return dict(
            f_max=float(self.cfg["f_max"]),
            mu_c=float(self.cfg["mu_c"]),
            obstacles=self.obstacles,
            n_substeps=int(self.cfg["n_contact_substeps"]),
            contact_step_margin=float(self.cfg["contact_step_margin"]),
            max_contact_step=float(self.cfg["max_contact_step"]),
            obstacle_margin=float(self.cfg["obstacle_margin"]),
            pushout_iters=int(self.cfg["object_pushout_iters"]),
        )

    def _rollout(
        self,
        robot_pos0: np.ndarray,
        ref_poses: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """actions: (K, H, 2) velocities -> robot traj + realized wrenches."""
        k, h, _ = actions.shape
        dt = float(self.cfg["dt"])
        params = self._sim_params()
        robot_pos = np.tile(robot_pos0, (k, 1))
        wrenches = np.zeros((k, h, 3))
        positions = np.zeros((k, h, 2))
        for t in range(h):
            pose_ref = np.tile(ref_poses[t], (k, 1))
            _, robot_pos, wrench = simulate_contact_step(
                self.object_,
                pose_ref,
                robot_pos,
                actions[:, t],
                dt,
                freeze_object=True,
                **params,
            )
            wrenches[:, t] = wrench
            positions[:, t] = robot_pos
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

    def _nominal_wrenches(
        self, robot_pos0: np.ndarray, ref_poses: np.ndarray, u_nom: np.ndarray
    ) -> np.ndarray:
        dt = float(self.cfg["dt"])
        params = self._sim_params()
        robot_pos = robot_pos0.copy()
        w_rob = np.zeros((self.horizon, 3))
        for t in range(self.horizon):
            _, robot_pos, wrench = simulate_contact_step(
                self.object_,
                ref_poses[t],
                robot_pos,
                u_nom[t],
                dt,
                freeze_object=True,
                **params,
            )
            w_rob[t] = np.asarray(wrench).reshape(3)
        return w_rob

    def solve(
        self,
        robot_pos0: np.ndarray,
        pose0: np.ndarray,
        ref_poses: np.ndarray,
        u_nom: np.ndarray,
        z: np.ndarray,
        gamma_r: np.ndarray,
        sigma_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns u_nom (H,2), w_rob (H,3). Uses x^{o,ref} inside contact sim."""
        del pose0  # kept for API symmetry with the ADMM solver
        u_nom, _, _ = self.mppi.solve(
            u_nom,
            rollout_fn=lambda actions: self._rollout(robot_pos0, ref_poses, actions),
            cost_fn=lambda actions, info: self._cost(actions, info, z, gamma_r),
            sigma_scale=sigma_scale,
        )
        w_rob = self._nominal_wrenches(robot_pos0, ref_poses, u_nom)
        return u_nom, w_rob
