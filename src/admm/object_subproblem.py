"""Object-level subproblem: sample (p_c, f_n, f_t) → world CoM wrench → MPPI."""

from __future__ import annotations

from typing import Any

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from contact.wrench_map import contact_force_to_com_wrench
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import obstacle_cost
from geometry.base_sdf import BaseSDF
from utils.math_utils import goal_cost, rotate, softmax


def sample_contact_points(
    shape: BaseSDF,
    p_mean: np.ndarray,
    sigma_p: float,
    tau_n: float,
    rng: np.random.Generator,
    k: int,
    max_tries: int = 8,
) -> np.ndarray:
    """Rejection sample K body-frame contact points per horizon step. Returns (K, H, 2)."""
    h = len(p_mean)
    points = np.zeros((k, h, 2))
    for t in range(h):
        mean_pt = p_mean[t]
        _, mean_grad = shape.sdf_and_grad(mean_pt[None, :])
        n_mean = -mean_grad[0]
        cand = shape.project_to_boundary(mean_pt + sigma_p * rng.standard_normal((k, 2)))
        accepted = np.zeros(k, dtype=bool)
        for _ in range(max_tries):
            _, grad_c = shape.sdf_and_grad(cand)
            aligned = (-grad_c) @ n_mean >= tau_n
            accepted |= aligned
            if accepted.all():
                break
            redraw = ~accepted
            cand[redraw] = shape.project_to_boundary(
                mean_pt + sigma_p * rng.standard_normal((int(redraw.sum()), 2))
            )
        cand[~accepted] = mean_pt
        points[:, t] = cand
    return points


def actions_to_wrenches(
    object_: QuasiStaticObject2D,
    poses: np.ndarray,
    p_body: np.ndarray,
    f_n: np.ndarray,
    f_t: np.ndarray,
) -> np.ndarray:
    """Map (p_body, fn, ft) at each pose to world CoM wrench. poses (K,3) or (3,)."""
    single_pose = poses.ndim == 1
    if single_pose:
        poses = np.broadcast_to(poses, (p_body.shape[0], 3)).copy()
    n_world, t_world, _, _ = object_.geometry(p_body, poses[:, 2])
    f_world = f_n[:, None] * n_world + f_t[:, None] * t_world
    p_world = poses[:, :2] + rotate(poses[:, 2], p_body)
    return contact_force_to_com_wrench(poses, p_world, f_world)


class ObjectSubproblem:
    def __init__(
        self,
        object_: QuasiStaticObject2D,
        obstacles: list[BaseSDF],
        goal: np.ndarray,
        cfg: dict[str, Any],
        consensus: WrenchConsensus,
        rng: np.random.Generator,
    ) -> None:
        self.object_ = object_
        self.obstacles = obstacles
        self.goal = np.asarray(goal, dtype=float)
        self.cfg = cfg
        self.consensus = consensus
        self.rng = rng
        self.horizon = int(cfg["horizon"])
        self.mu_c = float(cfg["mu_c"])
        self.f_max = float(cfg["f_max"])

    def solve(
        self,
        pose0: np.ndarray,
        p_mean: np.ndarray,
        f_n_nom: np.ndarray,
        f_t_nom: np.ndarray,
        z: np.ndarray,
        gamma_o: np.ndarray,
        sigma_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns f_n_nom, f_t_nom, p_mean, w_obj (H,3), ref_poses (H,3)."""
        k = int(self.cfg["k_object"])
        h = self.horizon
        sigma_p = float(self.cfg["sigma_p"]) * sigma_scale
        sigma_fn = float(self.cfg["sigma_fn"]) * sigma_scale
        sigma_ft = float(self.cfg["sigma_ft"]) * sigma_scale
        dt = float(self.cfg["dt"])

        p_k = sample_contact_points(
            self.object_.shape,
            p_mean,
            sigma_p,
            float(self.cfg["tau_n"]),
            self.rng,
            k,
            int(self.cfg["max_rejection_tries"]),
        )
        eps_fn = self.rng.standard_normal((k, h)) * sigma_fn
        eps_ft = self.rng.standard_normal((k, h)) * sigma_ft
        f_n_k = np.clip(f_n_nom[None] + eps_fn, 0.0, self.f_max)
        f_t_k = f_t_nom[None] + eps_ft
        # Friction cone |ft| <= mu_c * fn
        f_t_k = np.clip(f_t_k, -self.mu_c * f_n_k, self.mu_c * f_n_k)

        poses = np.tile(pose0, (k, 1))
        traj = np.zeros((k, h, 3))
        wrenches = np.zeros((k, h, 3))
        for t in range(h):
            w = actions_to_wrenches(
                self.object_, poses, p_k[:, t], f_n_k[:, t], f_t_k[:, t]
            )
            wrenches[:, t] = w
            poses = self.object_.propagate(poses, w, dt)
            traj[:, t] = poses

        running = goal_cost(
            traj[:, :-1].reshape(-1, 3),
            self.goal,
            float(self.cfg["q_pos"]),
            float(self.cfg["q_theta"]),
        ).reshape(k, -1).sum(1)
        running += obstacle_cost(
            self.object_.shape,
            traj[:, :-1].reshape(-1, 3),
            self.obstacles,
            float(self.cfg["obstacle_margin"]),
            float(self.cfg["w_obstacle"]),
        ).reshape(k, -1).sum(1)
        terminal = goal_cost(
            traj[:, -1],
            self.goal,
            float(self.cfg["qf_pos"]),
            float(self.cfg["qf_theta"]),
        )
        effort = float(self.cfg["r_o"]) * np.sum(wrenches**2, axis=(1, 2))
        admm = self.consensus.penalty_cost_batch(wrenches, z, gamma_o)
        costs = running + terminal + effort + admm
        weights = softmax(-costs / float(self.cfg["nu_o"]))

        f_n_nom = np.clip(f_n_nom + np.einsum("k,kt->t", weights, eps_fn), 0.0, self.f_max)
        f_t_nom = f_t_nom + np.einsum("k,kt->t", weights, eps_ft)
        f_t_nom = np.clip(f_t_nom, -self.mu_c * f_n_nom, self.mu_c * f_n_nom)
        p_mean = self.object_.shape.project_to_boundary(
            np.einsum("k,kti->ti", weights, p_k)
        )

        # Nominal wrench + reference rollout
        pose = pose0.copy()
        w_obj = np.zeros((h, 3))
        ref_poses = np.zeros((h, 3))
        for t in range(h):
            w = actions_to_wrenches(
                self.object_,
                pose,
                p_mean[t][None],
                f_n_nom[t : t + 1],
                f_t_nom[t : t + 1],
            )[0]
            w_obj[t] = w
            pose = self.object_.propagate(pose, w, dt)
            ref_poses[t] = pose
        return f_n_nom, f_t_nom, p_mean, w_obj, ref_poses


class ContactPointEstimator:
    def __init__(
        self,
        object_: QuasiStaticObject2D,
        obstacles: list[BaseSDF],
        goal: np.ndarray,
        cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        self.object_ = object_
        self.obstacles = obstacles
        self.goal = goal
        self.cfg = cfg
        self.rng = rng

    def estimate(
        self, pose: np.ndarray, mean_prev: np.ndarray, f_n_nom: float, sigma_scale: float = 1.0
    ) -> np.ndarray:
        mean = self.object_.shape.project_to_boundary(mean_prev[None, :])[0]
        sigma = float(self.cfg["sigma_init_est"]) * sigma_scale
        tau = float(self.cfg["tau_p_est"]) * sigma_scale
        for _ in range(int(self.cfg["r_est"])):
            samples = mean + sigma * self.rng.standard_normal((int(self.cfg["n_p_est"]), 2))
            samples = self.object_.shape.project_to_boundary(samples)
            cost = self._score(pose, samples, f_n_nom)
            weights = softmax(-cost / tau)
            mean = self.object_.shape.project_to_boundary(
                (weights[:, None] * samples).sum(axis=0, keepdims=True)
            )[0]
            sigma = max(sigma * float(self.cfg["gamma_est"]), float(self.cfg["sigma_min_est"]))
        return mean

    def _score(self, pose: np.ndarray, samples: np.ndarray, f_n_nom: float) -> np.ndarray:
        n = len(samples)
        dt = float(self.cfg["dt"])
        zero_ft = np.zeros(n)
        w_o = actions_to_wrenches(
            self.object_, pose, samples, np.full(n, f_n_nom), zero_ft
        )
        traj_pose = np.tile(pose, (n, 1))
        cost = np.zeros(n)
        for _ in range(int(self.cfg["h_p_est"])):
            traj_pose = self.object_.propagate(traj_pose, w_o, dt)
            cost += goal_cost(
                traj_pose, self.goal, float(self.cfg["q_pos"]), float(self.cfg["q_theta"])
            )
            cost += obstacle_cost(
                self.object_.shape,
                traj_pose,
                self.obstacles,
                float(self.cfg["obstacle_margin"]),
                float(self.cfg["w_obstacle"]),
            )
        cost += goal_cost(
            traj_pose, self.goal, float(self.cfg["qf_pos"]), float(self.cfg["qf_theta"])
        )
        return cost
