"""Robot-level subproblem: velocity MPPI with contact wrench consensus."""

from __future__ import annotations

from typing import Any

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from contact.resolution import simulate_contact_step
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import robot_obstacle_cost
from geometry.base_sdf import BaseSDF
from utils.math_utils import softmax


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
        k = int(self.cfg["k_robot"])
        h = self.horizon
        dt = float(self.cfg["dt"])
        sigma = np.asarray(self.cfg["sigma_robot"], dtype=float) * sigma_scale
        eps = self.rng.standard_normal((k, h, 2)) * sigma
        u_k = u_nom[None] + eps

        params = self._sim_params()
        cost = np.zeros(k)
        robot_pos = np.tile(robot_pos0, (k, 1))
        for t in range(h):
            pose_ref = np.tile(ref_poses[t], (k, 1))
            _, robot_pos, wrench = simulate_contact_step(
                self.object_,
                pose_ref,
                robot_pos,
                u_k[:, t],
                dt,
                freeze_object=True,
                **params,
            )
            cost += float(self.cfg["r_r"]) * np.sum(u_k[:, t] ** 2, axis=1)
            cost += robot_obstacle_cost(
                robot_pos,
                self.obstacles,
                float(self.cfg["obstacle_margin"]),
                float(self.cfg["w_obstacle"]),
            )
            # wrench: (K, 3) -> treat as single-step batch for penalty
            cost += 0.5 * self.consensus.rho * np.sum(
                (wrench - z[t] + gamma_r[t]) ** 2, axis=1
            )

        weights = softmax(-cost / float(self.cfg["nu_r"]))
        u_nom = u_nom + np.einsum("k,kti->ti", weights, eps)

        robot_pos = robot_pos0.copy()
        w_rob = np.zeros((h, 3))
        for t in range(h):
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
        return u_nom, w_rob
