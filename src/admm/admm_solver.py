"""Master ADMM coordination loop for wrench-consensus pushing MPC."""

from __future__ import annotations

from typing import Any

import numpy as np

from admm.consensus_spaces import WrenchConsensus
from admm.object_subproblem import ContactPointEstimator, ObjectSubproblem
from admm.robot_subproblem import RobotSubproblem
from contact.resolution import simulate_contact_step
from dynamics.object_2d import QuasiStaticObject2D
from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate, shift_horizon_zero_tail, wrap_angle


class ADMMSolver:
    def __init__(
        self,
        object_: QuasiStaticObject2D,
        robot_pos: np.ndarray,
        obstacles: list[BaseSDF],
        goal: np.ndarray,
        cfg: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.object_ = object_
        self.robot_pos = np.asarray(robot_pos, dtype=float).copy()
        self.obstacles = obstacles
        self.goal = np.asarray(goal, dtype=float)
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(int(cfg.get("random_seed", 0)))

        h = int(cfg["horizon"])
        self.consensus = WrenchConsensus(h, float(cfg["rho"]), float(cfg["max_dual"]))
        self.estimator = ContactPointEstimator(object_, obstacles, goal, cfg, self.rng)
        self.object_sp = ObjectSubproblem(
            object_, obstacles, goal, cfg, self.consensus, self.rng
        )
        self.robot_sp = RobotSubproblem(
            object_, obstacles, cfg, self.consensus, self.rng
        )

        q0 = object_.body_frame_point(self.robot_pos, object_.pose)
        p0 = object_.shape.project_to_boundary(q0[None, :])[0]
        self.p_mean = np.tile(p0, (h, 1))
        self.f_n_nom = np.zeros(h)
        self.f_t_nom = np.zeros(h)
        self.u_nom = np.zeros((h, 2))
        self.z = np.zeros((h, 3))
        self.gamma_o = np.zeros((h, 3))
        self.gamma_r = np.zeros((h, 3))

    def _sigma_scale(self, pose: np.ndarray) -> float:
        pos_err = np.linalg.norm(pose[:2] - self.goal[:2])
        theta_err = abs(wrap_angle(pose[2] - self.goal[2]))
        normalized = max(
            pos_err / float(self.cfg["goal_pos_tol"]),
            theta_err / float(self.cfg["goal_theta_tol"]),
        )
        return float(
            np.clip(
                normalized / float(self.cfg["sigma_anneal_band"]),
                float(self.cfg["min_sigma_scale"]),
                1.0,
            )
        )

    def control_step(self) -> tuple[np.ndarray, list[tuple[float, float]]]:
        pose0 = self.object_.pose.copy()
        self.f_n_nom = shift_horizon_zero_tail(self.f_n_nom)
        self.f_t_nom = shift_horizon_zero_tail(self.f_t_nom)
        self.u_nom = shift_horizon_zero_tail(self.u_nom)
        self.p_mean = shift_horizon_zero_tail(self.p_mean)
        # For p_mean last slot, re-seed from previous last projected point later
        self.z = shift_horizon_zero_tail(self.z)
        self.gamma_o = shift_horizon_zero_tail(self.gamma_o)
        self.gamma_r = shift_horizon_zero_tail(self.gamma_r)

        sigma_scale = self._sigma_scale(pose0)
        p0 = self.estimator.estimate(
            pose0, self.p_mean[0], float(self.f_n_nom[0]), sigma_scale
        )
        self.p_mean = np.tile(p0, (int(self.cfg["horizon"]), 1))

        # Seek toward contact if not touching
        p_world = pose0[:2] + rotate(pose0[2], p0)
        gap = p_world - self.robot_pos
        gap_norm = np.linalg.norm(gap)
        if gap_norm > float(self.cfg["contact_step_margin"]):
            speed = np.clip(
                gap_norm / float(self.cfg["dt"]),
                float(self.cfg["seek_min_speed"]),
                float(self.cfg["seek_max_speed"]),
            )
            self.u_nom = np.tile((gap / gap_norm) * speed, (int(self.cfg["horizon"]), 1))

        residuals: list[tuple[float, float]] = []
        for _ in range(int(self.cfg["n_admm"])):
            self.f_n_nom, self.f_t_nom, self.p_mean, w_obj, ref_poses = self.object_sp.solve(
                pose0,
                self.p_mean,
                self.f_n_nom,
                self.f_t_nom,
                self.z,
                self.gamma_o,
                sigma_scale,
            )
            self.u_nom, w_rob = self.robot_sp.solve(
                self.robot_pos,
                pose0,
                ref_poses,
                self.u_nom,
                self.z,
                self.gamma_r,
                sigma_scale,
            )

            z_new = self.consensus.z_update(w_obj, w_rob)
            gamma_o_new = self.consensus.dual_update(w_obj, z_new, self.gamma_o)
            gamma_r_new = self.consensus.dual_update(w_rob, z_new, self.gamma_r)

            primal = np.concatenate([w_obj - z_new, w_rob - z_new])
            dual = float(self.cfg["rho"]) * (z_new - self.z)
            residuals.append(
                (float(np.linalg.norm(primal)), float(np.linalg.norm(dual)))
            )
            self.z, self.gamma_o, self.gamma_r = z_new, gamma_o_new, gamma_r_new
            if (
                residuals[-1][0] <= float(self.cfg["eps_r"])
                and residuals[-1][1] <= float(self.cfg["eps_s"])
            ):
                break

        return self.u_nom[0].copy(), residuals

    def run(
        self, max_steps: int | None = None, verbose: bool = True
    ) -> dict[str, Any]:
        max_steps = max_steps or int(self.cfg["max_control_steps"])
        log: dict[str, Any] = {
            "object_pose": [self.object_.pose.copy()],
            "robot_pos": [self.robot_pos.copy()],
            "residuals": [],
        }
        reached = False
        dt = float(self.cfg["dt"])
        params = dict(
            f_max=float(self.cfg["f_max"]),
            mu_c=float(self.cfg["mu_c"]),
            obstacles=self.obstacles,
            n_substeps=int(self.cfg["n_contact_substeps"]),
            contact_step_margin=float(self.cfg["contact_step_margin"]),
            max_contact_step=float(self.cfg["max_contact_step"]),
            obstacle_margin=float(self.cfg["obstacle_margin"]),
            pushout_iters=int(self.cfg["object_pushout_iters"]),
            freeze_object=False,
        )

        for step in range(max_steps):
            u0, residuals = self.control_step()
            new_pose, new_robot, _ = simulate_contact_step(
                self.object_,
                self.object_.pose[None],
                self.robot_pos[None],
                u0[None],
                dt,
                **params,
            )
            self.object_.pose = np.asarray(new_pose).reshape(3)
            self.robot_pos = np.asarray(new_robot).reshape(2)

            log["object_pose"].append(self.object_.pose.copy())
            log["robot_pos"].append(self.robot_pos.copy())
            log["residuals"].extend(residuals)

            pos_err = np.linalg.norm(self.object_.pose[:2] - self.goal[:2])
            theta_err = abs(wrap_angle(self.object_.pose[2] - self.goal[2]))
            if verbose and step % 20 == 0:
                print(
                    f"step {step:4d}  pos_err={pos_err:.3f}  "
                    f"theta_err={theta_err:.3f}  admm_iters={len(residuals)}"
                )
            if (
                pos_err < float(self.cfg["goal_pos_tol"])
                and theta_err < float(self.cfg["goal_theta_tol"])
            ):
                reached = True
                if verbose:
                    print(f"goal reached at step {step}")
                break

        log["reached"] = reached
        log["object_pose"] = np.array(log["object_pose"])
        log["robot_pos"] = np.array(log["robot_pos"])
        return log
