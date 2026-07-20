"""Obstacle push-out and hinge costs."""

from __future__ import annotations

import numpy as np

from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate


def obstacle_cost(
    shape: BaseSDF,
    poses: np.ndarray,
    obstacles: list[BaseSDF],
    margin: float,
    weight: float,
) -> np.ndarray:
    """Hinge penalty on object boundary samples vs obstacle SDFs."""
    poses = np.atleast_2d(poses)
    if not hasattr(shape, "boundary_samples"):
        # Circles/boxes: sample a coarse ring around the bounding circle
        n = 16
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        samples = shape.bounding_radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        samples = shape.boundary_samples

    verts_world = poses[:, None, :2] + rotate(poses[:, 2, None], samples[None])
    flat = verts_world.reshape(-1, 2)
    cost = np.zeros(len(poses))
    for obs in obstacles:
        d = obs.sdf(flat).reshape(len(poses), -1)
        violation = np.clip(margin - d, 0.0, None)
        cost += weight * np.sum(violation**2, axis=1)
    return cost


def robot_obstacle_cost(
    robot_pos: np.ndarray,
    obstacles: list[BaseSDF],
    margin: float,
    weight: float,
) -> np.ndarray:
    robot_pos = np.atleast_2d(robot_pos)
    cost = np.zeros(len(robot_pos))
    for obs in obstacles:
        d = obs.sdf(robot_pos)
        violation = np.clip(margin - d, 0.0, None)
        cost += weight * violation**2
    return cost


def push_point_out_of_obstacles(
    points: np.ndarray, obstacles: list[BaseSDF]
) -> np.ndarray:
    points = np.asarray(points, dtype=float).copy()
    single = points.ndim == 1
    points = np.atleast_2d(points)
    for obs in obstacles:
        if np.all(np.linalg.norm(points - obs.center, axis=-1) > obs.bounding_radius):
            continue
        d, grad = obs.sdf_and_grad(points)
        inside = d < 0.0
        points = np.where(inside[..., None], points - d[..., None] * grad, points)
    return points[0] if single else points


def push_object_out_of_obstacles(
    shape: BaseSDF,
    pose: np.ndarray,
    obstacles: list[BaseSDF],
    margin: float,
    iterations: int = 4,
) -> np.ndarray:
    single = pose.ndim == 1
    pose = np.atleast_2d(pose).copy()
    if not hasattr(shape, "boundary_samples"):
        n = 16
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        samples = shape.bounding_radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        samples = shape.boundary_samples

    for _ in range(iterations):
        for obs in obstacles:
            reach = shape.bounding_radius + obs.bounding_radius + margin
            if np.all(np.linalg.norm(pose[:, :2] - obs.center, axis=-1) > reach):
                continue
            verts = pose[:, None, :2] + rotate(pose[:, None, 2], samples[None])
            k, v, _ = verts.shape
            d, grad = obs.sdf_and_grad(verts.reshape(-1, 2))
            d, grad = d.reshape(k, v), grad.reshape(k, v, 2)
            idx = np.argmin(d, axis=1)
            d_worst = d[np.arange(k), idx]
            grad_worst = grad[np.arange(k), idx]
            pose[:, :2] -= np.where(
                (d_worst < 0.0)[:, None], d_worst[:, None] * grad_worst, 0.0
            )
    return pose[0] if single else pose
