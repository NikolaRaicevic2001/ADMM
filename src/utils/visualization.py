"""Matplotlib trajectory visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon

from geometry.analytical_2d import BoxSDF, CircleSDF, PolygonSDF
from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate


def _obstacle_patch(obs: BaseSDF):
    if isinstance(obs, CircleSDF):
        return MplCircle(obs.center, obs.radius, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    if isinstance(obs, BoxSDF):
        corners = obs.center + rotate(
            obs.angle,
            np.array(
                [
                    [-obs.half_extents[0], -obs.half_extents[1]],
                    [obs.half_extents[0], -obs.half_extents[1]],
                    [obs.half_extents[0], obs.half_extents[1]],
                    [-obs.half_extents[0], obs.half_extents[1]],
                ]
            ),
        )
        return MplPolygon(corners, closed=True, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    if isinstance(obs, PolygonSDF):
        return MplPolygon(obs.vertices, closed=True, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    raise TypeError(f"unknown obstacle type {type(obs)}")


def plot_overview(
    log: dict[str, Any],
    shape: BaseSDF,
    obstacles: list[BaseSDF],
    goal: np.ndarray,
    save_path: Path,
    n_poses: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for obs in obstacles:
        ax.add_patch(_obstacle_patch(obs))

    idx = np.linspace(0, len(log["object_pose"]) - 1, n_poses).astype(int)
    verts0 = getattr(shape, "vertices", None)
    if verts0 is None:
        raise ValueError("overview plot requires polygonal object shape")
    for i, k in enumerate(idx):
        pose = log["object_pose"][k]
        verts = pose[:2] + rotate(pose[2], verts0)
        alpha = 0.15 + 0.65 * i / max(len(idx) - 1, 1)
        ax.add_patch(
            MplPolygon(verts, closed=True, fc="tab:blue", ec="tab:blue", alpha=alpha, zorder=3)
        )

    goal_verts = goal[:2] + rotate(goal[2], verts0)
    ax.add_patch(
        MplPolygon(
            goal_verts,
            closed=True,
            fill=False,
            ec="tab:green",
            lw=2,
            ls="--",
            zorder=4,
            label="goal",
        )
    )
    ax.plot(log["robot_pos"][:, 0], log["robot_pos"][:, 1], "-", color="tab:red", lw=1, alpha=0.6)
    ax.plot(*log["robot_pos"][0], "o", color="tab:red", ms=7, zorder=5, label="robot")
    ax.plot(*log["object_pose"][0, :2], "s", color="tab:blue", ms=6, zorder=5, label="object start")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    status = "reached" if log["reached"] else "not reached"
    ax.set_title(f"ADMM wrench-consensus pushing (goal {status})")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(log: dict[str, Any], save_path: Path, eps: float = 1.0) -> None:
    residuals = np.array(log["residuals"])
    if residuals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(residuals[:, 0], label=r"primal residual $\|r\|$", lw=1)
    ax.plot(residuals[:, 1], label=r"dual residual $\|s\|$", lw=1)
    ax.axhline(eps, color="k", lw=0.7, ls=":", label="tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("ADMM iteration (concatenated over control steps)")
    ax.set_ylabel("residual norm")
    ax.set_title("ADMM wrench consensus convergence")
    ax.legend(fontsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_animation(
    log: dict[str, Any],
    shape: BaseSDF,
    obstacles: list[BaseSDF],
    goal: np.ndarray,
    save_path: Path,
    stride: int = 2,
    fps: int = 15,
) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    verts0 = getattr(shape, "vertices")
    fig, ax = plt.subplots(figsize=(6, 6))
    for obs in obstacles:
        ax.add_patch(_obstacle_patch(obs))
    goal_verts = goal[:2] + rotate(goal[2], verts0)
    ax.add_patch(
        MplPolygon(goal_verts, closed=True, fill=False, ec="tab:green", lw=2, ls="--", zorder=4)
    )
    object_patch = MplPolygon(verts0, closed=True, fc="tab:blue", ec="k", alpha=0.85, zorder=3)
    ax.add_patch(object_patch)
    (robot_dot,) = ax.plot([], [], "o", color="tab:red", ms=8, zorder=5)
    (robot_trail,) = ax.plot([], [], "-", color="tab:red", lw=1, alpha=0.5, zorder=3)

    all_xy = np.concatenate([log["object_pose"][:, :2], log["robot_pos"]], axis=0)
    margin = 0.1
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)
    ax.set_aspect("equal")

    frames = list(range(0, len(log["object_pose"]), stride))

    def update(i: int):
        pose = log["object_pose"][i]
        object_patch.set_xy(pose[:2] + rotate(pose[2], verts0))
        robot_dot.set_data([log["robot_pos"][i, 0]], [log["robot_pos"][i, 1]])
        robot_trail.set_data(log["robot_pos"][: i + 1, 0], log["robot_pos"][: i + 1, 1])
        return object_patch, robot_dot, robot_trail

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
