"""Single-point contact resolution returning world CoM wrench."""

from __future__ import annotations

import numpy as np

from contact.wrench_map import contact_force_to_com_wrench
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.obstacles import push_object_out_of_obstacles, push_point_out_of_obstacles
from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate


def resolve_contact(
    object_: QuasiStaticObject2D,
    pose: np.ndarray,
    robot_pos_free: np.ndarray,
    robot_vel_cmd: np.ndarray,
    dt: float,
    f_max: float,
    mu_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form single-contact resolution.

    Returns f_n, f_t, contact_body, in_contact, wrench_com (..., 3).
    """
    single = pose.ndim == 1
    pose_b = np.atleast_2d(pose)
    robot_b = np.atleast_2d(robot_pos_free)
    vel_b = np.atleast_2d(robot_vel_cmd)

    q = object_.body_frame_point(robot_b, pose_b)
    d, grad = object_.shape.sdf_and_grad(q)
    in_contact = d <= 0.0
    n_body = -grad
    t_body = np.stack([-n_body[..., 1], n_body[..., 0]], axis=-1)
    n_world = rotate(pose_b[:, 2], n_body)
    t_world = rotate(pose_b[:, 2], t_body)

    penetration = np.clip(-d, 0.0, None)
    f_n = np.clip(penetration / (dt * object_.D[0]), 0.0, f_max)
    v_t = np.einsum("ij,ij->i", vel_b, t_world)
    sliding = np.abs(v_t) > 1e-4
    f_t = np.where(sliding, -mu_c * f_n * np.sign(v_t), 0.0)
    f_n = np.where(in_contact, f_n, 0.0)
    f_t = np.where(in_contact, f_t, 0.0)

    contact_body = q - d[:, None] * grad
    f_world = f_n[:, None] * n_world + f_t[:, None] * t_world
    p_c_world = pose_b[:, :2] + rotate(pose_b[:, 2], contact_body)
    wrench = contact_force_to_com_wrench(pose_b, p_c_world, f_world)
    wrench = np.where(in_contact[:, None], wrench, 0.0)

    if single:
        return f_n[0], f_t[0], contact_body[0], bool(in_contact[0]), wrench[0]
    return f_n, f_t, contact_body, in_contact, wrench


def _contact_substep(
    object_: QuasiStaticObject2D,
    pose: np.ndarray,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    dt: float,
    f_max: float,
    mu_c: float,
    obstacles: list[BaseSDF],
    contact_step_margin: float,
    max_contact_step: float,
    obstacle_margin: float,
    pushout_iters: int,
    freeze_object: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One collision-safe sub-step. Returns (new_pose, new_robot_pos, wrench_com)."""
    single = pose.ndim == 1
    pose_b = np.atleast_2d(pose)
    robot_b = np.atleast_2d(robot_pos)
    vel_b = np.atleast_2d(robot_vel)

    q_current = object_.body_frame_point(robot_b, pose_b)
    d_current = np.asarray(object_.shape.sdf(q_current), dtype=float).reshape(-1)
    disp = dt * vel_b
    disp_norm = np.linalg.norm(disp, axis=-1)
    safe_dist = np.where(
        d_current > 0.0, d_current + contact_step_margin, max_contact_step
    )
    for obs in obstacles:
        if np.all(
            np.linalg.norm(robot_b - obs.center, axis=-1)
            > obs.bounding_radius + max_contact_step
        ):
            continue
        d_obs = np.asarray(obs.sdf(robot_b), dtype=float).reshape(-1)
        obs_safe = np.where(d_obs > 0.0, d_obs + contact_step_margin, max_contact_step)
        safe_dist = np.minimum(safe_dist, obs_safe)
    scale = np.ones_like(disp_norm)
    mask = disp_norm > 1e-12
    scale[mask] = np.clip(safe_dist[mask] / disp_norm[mask], 0.0, 1.0)
    robot_free = robot_b + scale[:, None] * disp
    robot_free = push_point_out_of_obstacles(robot_free, obstacles)

    _, _, _, _, wrench = resolve_contact(
        object_, pose_b, robot_free, vel_b, dt, f_max, mu_c
    )

    if freeze_object:
        new_pose = pose_b.copy()
    else:
        new_pose = object_.propagate(pose_b, wrench, dt)
        new_pose = push_object_out_of_obstacles(
            object_.shape, new_pose, obstacles, obstacle_margin, pushout_iters
        )

    q_check = object_.body_frame_point(robot_free, new_pose)
    d_check, grad_check = object_.shape.sdf_and_grad(q_check)
    penetrating = d_check < 0.0
    q_proj = q_check - d_check[:, None] * grad_check
    corrected = new_pose[:, :2] + rotate(new_pose[:, 2], q_proj)
    new_robot_pos = np.where(penetrating[:, None], corrected, robot_free)
    new_robot_pos = push_point_out_of_obstacles(new_robot_pos, obstacles)

    if single:
        return new_pose[0], new_robot_pos[0], wrench[0]
    return new_pose, new_robot_pos, wrench


def simulate_contact_step(
    object_: QuasiStaticObject2D,
    pose: np.ndarray,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    dt: float,
    f_max: float,
    mu_c: float,
    obstacles: list[BaseSDF] | None = None,
    n_substeps: int = 4,
    contact_step_margin: float = 0.003,
    max_contact_step: float = 0.008,
    obstacle_margin: float = 0.015,
    pushout_iters: int = 4,
    freeze_object: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coupled control step with substeps.

    Returns (new_pose, new_robot_pos, mean_wrench_com) with wrench shape (..., 3).
    If freeze_object=True, object pose stays on the provided trajectory (robot MPPI
    with x^{o,ref}); wrench is still resolved against that pose.
    """
    obstacles = obstacles or []
    single = np.asarray(pose).ndim == 1
    pose_b = np.atleast_2d(pose)
    robot_b = np.atleast_2d(robot_pos)
    vel_b = np.atleast_2d(robot_vel)
    sub_dt = dt / n_substeps
    wrench_sum = np.zeros((pose_b.shape[0], 3))
    for _ in range(n_substeps):
        pose_b, robot_b, wrench = _contact_substep(
            object_,
            pose_b,
            robot_b,
            vel_b,
            sub_dt,
            f_max,
            mu_c,
            obstacles,
            contact_step_margin,
            max_contact_step,
            obstacle_margin,
            pushout_iters,
            freeze_object=freeze_object,
        )
        wrench_sum = wrench_sum + wrench
    wrench_mean = wrench_sum / n_substeps
    if single:
        return pose_b[0], robot_b[0], wrench_mean[0]
    return pose_b, robot_b, wrench_mean
