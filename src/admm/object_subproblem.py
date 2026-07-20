"""Object-level subproblem: MPPI over contact actions (p_c, f_n, f_t).

Action vector per timestep is a_t = [p_x, p_y, f_n, f_t] in the object body frame.
Wrenches are derived via J_c^T f_c inside the rollout — never sampled directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from contact.wrench_map import contact_force_to_com_wrench
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import obstacle_cost
from geometry.base_sdf import BaseSDF
from mppi.mppi_core import MPPIOptimizer
from mppi.sampler import GaussianSampler
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
    """Map (p_body, fn, ft) at each pose to world CoM wrench."""
    single_pose = poses.ndim == 1
    if single_pose:
        poses = np.broadcast_to(poses, (p_body.shape[0], 3)).copy()
    n_world, t_world, _, _ = object_.geometry(p_body, poses[:, 2])
    f_world = f_n[:, None] * n_world + f_t[:, None] * t_world
    p_world = poses[:, :2] + rotate(poses[:, 2], p_body)
    return contact_force_to_com_wrench(poses, p_world, f_world)


def pack_actions(p_mean: np.ndarray, f_n: np.ndarray, f_t: np.ndarray) -> np.ndarray:
    """Stack body contact + forces into (H, 4) action trajectory."""
    return np.concatenate([p_mean, f_n[:, None], f_t[:, None]], axis=1)


def unpack_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (..., 4) -> p (..., 2), f_n (...,), f_t (...,)."""
    return actions[..., :2], actions[..., 2], actions[..., 3]


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
        self.mppi = MPPIOptimizer(
            n_samples=int(cfg["k_object"]),
            temperature=float(cfg["nu_o"]),
            sampler=GaussianSampler(rng),
        )

    def _sample_actions(
        self, nominal: np.ndarray, n_samples: int, sigma_scale: float
    ) -> np.ndarray:
        """Custom sampler: rejection on p_c, Gaussian on forces, friction-cone clip."""
        p_mean, f_n_nom, f_t_nom = unpack_actions(nominal)
        p_k = sample_contact_points(
            self.object_.shape,
            p_mean,
            float(self.cfg["sigma_p"]) * sigma_scale,
            float(self.cfg["tau_n"]),
            self.rng,
            n_samples,
            int(self.cfg["max_rejection_tries"]),
        )
        eps_fn = self.rng.standard_normal((n_samples, self.horizon)) * (
            float(self.cfg["sigma_fn"]) * sigma_scale
        )
        eps_ft = self.rng.standard_normal((n_samples, self.horizon)) * (
            float(self.cfg["sigma_ft"]) * sigma_scale
        )
        f_n_k = np.clip(f_n_nom[None] + eps_fn, 0.0, self.f_max)
        f_t_k = np.clip(
            f_t_nom[None] + eps_ft, -self.mu_c * f_n_k, self.mu_c * f_n_k
        )
        return np.concatenate([p_k, f_n_k[..., None], f_t_k[..., None]], axis=-1)

    def _project_actions(self, nominal: np.ndarray) -> np.ndarray:
        p, f_n, f_t = unpack_actions(nominal)
        p = self.object_.shape.project_to_boundary(p)
        f_n = np.clip(f_n, 0.0, self.f_max)
        f_t = np.clip(f_t, -self.mu_c * f_n, self.mu_c * f_n)
        return pack_actions(p, f_n, f_t)

    def _rollout(self, pose0: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        """actions: (K, H, 4) -> poses (K,H,3), wrenches (K,H,3)."""
        k, h, _ = actions.shape
        dt = float(self.cfg["dt"])
        p, f_n, f_t = unpack_actions(actions)
        poses = np.tile(pose0, (k, 1))
        traj = np.zeros((k, h, 3))
        wrenches = np.zeros((k, h, 3))
        for t in range(h):
            w = actions_to_wrenches(self.object_, poses, p[:, t], f_n[:, t], f_t[:, t])
            wrenches[:, t] = w
            poses = self.object_.propagate(poses, w, dt)
            traj[:, t] = poses
        return {"poses": traj, "wrenches": wrenches}

    def _cost(
        self,
        actions: np.ndarray,
        info: dict[str, np.ndarray],
        z: np.ndarray,
        gamma_o: np.ndarray,
    ) -> np.ndarray:
        traj = info["poses"]
        wrenches = info["wrenches"]
        k = traj.shape[0]
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
        return running + terminal + effort + admm

    def _nominal_rollout(
        self, pose0: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic rollout of a single (H,4) action seq -> w_obj, ref_poses."""
        h = self.horizon
        dt = float(self.cfg["dt"])
        p, f_n, f_t = unpack_actions(actions)
        pose = pose0.copy()
        w_obj = np.zeros((h, 3))
        ref_poses = np.zeros((h, 3))
        for t in range(h):
            w = actions_to_wrenches(
                self.object_, pose, p[t][None], f_n[t : t + 1], f_t[t : t + 1]
            )[0]
            w_obj[t] = w
            pose = self.object_.propagate(pose, w, dt)
            ref_poses[t] = pose
        return w_obj, ref_poses

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
        nominal = pack_actions(p_mean, f_n_nom, f_t_nom)

        new_nominal, _, _ = self.mppi.solve(
            nominal,
            rollout_fn=lambda actions: self._rollout(pose0, actions),
            cost_fn=lambda actions, info: self._cost(actions, info, z, gamma_o),
            sample_fn=self._sample_actions,
            project_fn=self._project_actions,
            sigma_scale=sigma_scale,
        )

        p_mean, f_n_nom, f_t_nom = unpack_actions(new_nominal)
        w_obj, ref_poses = self._nominal_rollout(pose0, new_nominal)
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
