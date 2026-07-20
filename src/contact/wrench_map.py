"""World-frame CoM wrench map: w = Jc^T f."""

from __future__ import annotations

import numpy as np


def contact_force_to_com_wrench(
    pose_xytheta: np.ndarray,
    p_c_world: np.ndarray,
    f_world: np.ndarray,
) -> np.ndarray:
    """Map world contact force at p_c to world CoM wrench (fx, fy, tau_com).

    Parameters
    ----------
    pose_xytheta : (..., 3) object pose [px, py, theta]
    p_c_world : (..., 2) contact point in world frame
    f_world : (..., 2) contact force in world frame

    Returns
    -------
    wrench : (..., 3)
    """
    pose = np.asarray(pose_xytheta, dtype=float)
    p_c = np.asarray(p_c_world, dtype=float)
    f = np.asarray(f_world, dtype=float)

    single = pose.ndim == 1
    if single:
        pose = pose[None, :]
        p_c = np.atleast_2d(p_c)
        f = np.atleast_2d(f)

    r = p_c - pose[:, :2]
    tau = r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]
    w = np.concatenate([f, tau[:, None]], axis=-1)
    return w[0] if single else w


def body_contact_force_to_com_wrench(
    pose_xytheta: np.ndarray,
    p_body: np.ndarray,
    f_n: np.ndarray,
    f_t: np.ndarray,
    n_world: np.ndarray,
    t_world: np.ndarray,
) -> np.ndarray:
    """Build world force from (fn, ft) along n/t, map through Jc^T at body contact point."""
    from utils.math_utils import rotate

    pose = np.atleast_2d(np.asarray(pose_xytheta, dtype=float))
    p_body = np.atleast_2d(np.asarray(p_body, dtype=float))
    f_n = np.asarray(f_n, dtype=float).reshape(-1)
    f_t = np.asarray(f_t, dtype=float).reshape(-1)
    n_world = np.atleast_2d(n_world)
    t_world = np.atleast_2d(t_world)

    f_world = f_n[:, None] * n_world + f_t[:, None] * t_world
    p_world = pose[:, :2] + rotate(pose[:, 2], p_body)
    return contact_force_to_com_wrench(pose, p_world, f_world)
