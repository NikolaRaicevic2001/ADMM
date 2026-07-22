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

# Purely cosmetic: draws the dashed goal outline larger than the object's
# actual footprint so the arrived object visually reads as "inside" the goal marker.
GOAL_MARKER_SCALE = 1.2


def _box_corners(obs: BoxSDF) -> np.ndarray:
    return obs.center + rotate(
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


def _obstacle_patch(obs: BaseSDF):
    if isinstance(obs, CircleSDF):
        return MplCircle(obs.center, obs.radius, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    if isinstance(obs, BoxSDF):
        return MplPolygon(_box_corners(obs), closed=True, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    if isinstance(obs, PolygonSDF):
        return MplPolygon(obs.vertices, closed=True, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    raise TypeError(f"unknown obstacle type {type(obs)}")


def _obstacle_xy(obs: BaseSDF) -> np.ndarray:
    """Corner / extent samples used to size a static scene view."""
    if isinstance(obs, CircleSDF):
        c, r = obs.center, float(obs.radius)
        return np.array([[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]])
    if isinstance(obs, BoxSDF):
        return _box_corners(obs)
    if isinstance(obs, PolygonSDF):
        return obs.vertices
    raise TypeError(f"unknown obstacle type {type(obs)}")


def scene_axis_limits(
    shape: BaseSDF,
    obstacles: list[BaseSDF],
    goal: np.ndarray,
    object_pose: np.ndarray,
    robot_pos: np.ndarray,
    pad_frac: float = 0.28,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fixed view from environment geometry (start, goal, robot, obstacles).

    Grows the tight AABB by ``pad_frac`` in total (~28% wider/taller), split
    evenly on both sides, so every run of the same env shares xlim/ylim.
    """
    verts0 = getattr(shape, "vertices", None)
    if verts0 is None:
        r = float(shape.bounding_radius)
        local = np.array([[-r, -r], [r, -r], [r, r], [-r, r]], dtype=float)
    else:
        local = np.asarray(verts0, dtype=float)

    pts: list[np.ndarray] = [
        np.asarray(object_pose[:2], dtype=float) + rotate(float(object_pose[2]), local),
        np.asarray(goal[:2], dtype=float) + rotate(float(goal[2]), local * GOAL_MARKER_SCALE),
        np.asarray(robot_pos, dtype=float).reshape(1, 2),
    ]
    for obs in obstacles:
        pts.append(_obstacle_xy(obs))

    all_xy = np.concatenate(pts, axis=0)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    dx = max(float(xmax - xmin), 1e-3)
    dy = max(float(ymax - ymin), 1e-3)
    mx = 0.5 * pad_frac * dx
    my = 0.5 * pad_frac * dy
    return (float(xmin - mx), float(xmax + mx)), (float(ymin - my), float(ymax + my))


def _apply_scene_limits(
    ax,
    shape: BaseSDF,
    obstacles: list[BaseSDF],
    goal: np.ndarray,
    object_pose: np.ndarray,
    robot_pos: np.ndarray,
    pad_frac: float = 0.28,
) -> None:
    xlim, ylim = scene_axis_limits(
        shape, obstacles, goal, object_pose, robot_pos, pad_frac=pad_frac
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _draw_scene_base(ax, shape, obstacles, goal, log):
    for obs in obstacles:
        ax.add_patch(_obstacle_patch(obs))
    verts0 = getattr(shape, "vertices", None)
    if verts0 is None:
        raise ValueError("plot requires polygonal object shape")
    goal_verts = goal[:2] + rotate(goal[2], verts0 * GOAL_MARKER_SCALE)
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
    return verts0


def _overlay_plans(ax, log: dict[str, Any], n_plans: int = 6) -> None:
    """Draw a few object-MPPI vs robot-implied object horizons + robot path plans."""
    if "object_plan" not in log or len(log["object_plan"]) == 0:
        return
    n = len(log["object_plan"])
    idx = np.unique(np.linspace(0, n - 1, min(n_plans, n)).astype(int))
    for j, k in enumerate(idx):
        alpha = 0.25 + 0.55 * j / max(len(idx) - 1, 1)
        op = log["object_plan"][k]
        rp = log["robot_object_plan"][k]
        rpath = log["robot_plan_path"][k]
        label_o = "object MPPI plan" if j == len(idx) - 1 else None
        label_r = "robot-implied object plan" if j == len(idx) - 1 else None
        label_u = "robot MPPI path" if j == len(idx) - 1 else None
        ax.plot(
            op[:, 0],
            op[:, 1],
            "-",
            color="tab:cyan",
            lw=1.8,
            alpha=alpha,
            zorder=6,
            label=label_o,
        )
        ax.plot(
            rp[:, 0],
            rp[:, 1],
            "-",
            color="darkorange",
            lw=1.8,
            alpha=alpha,
            zorder=6,
            label=label_r,
        )
        ax.plot(
            rpath[:, 0],
            rpath[:, 1],
            "--",
            color="magenta",
            lw=1.2,
            alpha=alpha,
            zorder=6,
            label=label_u,
        )
        # Mark plan start (current state when plan was made)
        ax.plot(op[0, 0], op[0, 1], "o", color="tab:cyan", ms=3, alpha=alpha, zorder=7)
        ax.plot(rp[0, 0], rp[0, 1], "o", color="darkorange", ms=3, alpha=alpha, zorder=7)


def plot_overview(
    log: dict[str, Any],
    shape: BaseSDF,
    obstacles: list[BaseSDF],
    goal: np.ndarray,
    save_path: Path,
    n_poses: int = 8,
    view_pad_frac: float = 0.28,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    verts0 = _draw_scene_base(ax, shape, obstacles, goal, log)

    idx = np.linspace(0, len(log["object_pose"]) - 1, n_poses).astype(int)
    for i, k in enumerate(idx):
        pose = log["object_pose"][k]
        verts = pose[:2] + rotate(pose[2], verts0)
        alpha = 0.15 + 0.65 * i / max(len(idx) - 1, 1)
        ax.add_patch(
            MplPolygon(verts, closed=True, fc="tab:blue", ec="tab:blue", alpha=alpha, zorder=3)
        )

    ax.plot(
        log["robot_pos"][:, 0],
        log["robot_pos"][:, 1],
        "-",
        color="tab:red",
        lw=1,
        alpha=0.6,
        label="robot executed",
    )
    ax.plot(*log["robot_pos"][0], "o", color="tab:red", ms=7, zorder=5, label="robot start")
    ax.plot(
        *log["object_pose"][0, :2],
        "s",
        color="tab:blue",
        ms=6,
        zorder=5,
        label="object start",
    )
    _overlay_plans(ax, log)

    _apply_scene_limits(
        ax,
        shape,
        obstacles,
        goal,
        log["object_pose"][0],
        log["robot_pos"][0],
        pad_frac=view_pad_frac,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    status = "reached" if log["reached"] else "not reached"
    ax.set_title(f"ADMM pushing + plan overlays (goal {status})")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_plan_comparison(
    log: dict[str, Any],
    shape: BaseSDF,
    obstacles: list[BaseSDF],
    goal: np.ndarray,
    save_path: Path,
    n_panels: int = 4,
    view_pad_frac: float = 0.28,
) -> None:
    """Multi-panel view of object vs robot-implied object horizons at selected steps."""
    if "object_plan" not in log or len(log["object_plan"]) == 0:
        return
    n = len(log["object_plan"])
    idx = np.unique(np.linspace(0, n - 1, min(n_panels, n)).astype(int))
    fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 4.2), squeeze=False)
    verts0 = getattr(shape, "vertices")

    for ax, k in zip(axes[0], idx):
        for obs in obstacles:
            ax.add_patch(_obstacle_patch(obs))
        goal_verts = goal[:2] + rotate(goal[2], verts0 * GOAL_MARKER_SCALE)
        ax.add_patch(
            MplPolygon(goal_verts, closed=True, fill=False, ec="tab:green", lw=1.5, ls="--")
        )
        # Executed state at plan time (pose before this step's action)
        pose = log["object_pose"][k]
        ax.add_patch(
            MplPolygon(
                pose[:2] + rotate(pose[2], verts0),
                closed=True,
                fc="tab:blue",
                ec="k",
                alpha=0.5,
                zorder=3,
            )
        )
        ax.plot(*log["robot_pos"][k], "o", color="tab:red", ms=6, zorder=5)

        op = log["object_plan"][k]
        rp = log["robot_object_plan"][k]
        rpath = log["robot_plan_path"][k]
        ax.plot(op[:, 0], op[:, 1], "-", color="tab:cyan", lw=2, label="object MPPI")
        ax.plot(rp[:, 0], rp[:, 1], "-", color="darkorange", lw=2, label="robot→object")
        ax.plot(rpath[:, 0], rpath[:, 1], "--", color="magenta", lw=1.5, label="robot path")
        # Terminal markers
        ax.plot(op[-1, 0], op[-1, 1], "c*", ms=10)
        ax.plot(rp[-1, 0], rp[-1, 1], marker="*", color="darkorange", ms=10)

        gap = float(np.linalg.norm(op[-1, :2] - rp[-1, :2]))
        ax.set_title(f"step {k}  terminal Δxy={gap:.3f}m", fontsize=9)
        _apply_scene_limits(
            ax,
            shape,
            obstacles,
            goal,
            log["object_pose"][0],
            log["robot_pos"][0],
            pad_frac=view_pad_frac,
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Object MPPI plan vs object path implied by robot wrenches", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(log: dict[str, Any], save_path: Path, eps: float = 1.0) -> None:
    residuals = np.array(log["residuals"])
    if residuals.size == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=False)

    ax = axes[0]
    ax.plot(residuals[:, 0], label=r"primal residual $\|r\|$", lw=1)
    ax.plot(residuals[:, 1], label=r"dual residual $\|s\|$", lw=1)
    ax.axhline(eps, color="k", lw=0.7, ls=":", label="tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("ADMM iteration (concatenated over control steps)")
    ax.set_ylabel("residual norm")
    ax.set_title("ADMM wrench consensus convergence")
    ax.legend(fontsize=8)

    ax = axes[1]
    if "object_plan" in log and len(log["object_plan"]) > 0:
        gaps = np.linalg.norm(
            log["object_plan"][:, -1, :2] - log["robot_object_plan"][:, -1, :2], axis=1
        )
        ax.plot(gaps, color="purple", lw=1.2)
        ax.set_xlabel("control step")
        ax.set_ylabel("terminal plan Δxy [m]")
        ax.set_title("Object vs robot-implied object plan disagreement")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    fig.tight_layout()
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
    wrench_arrow_scale: float = 0.04,
    view_pad_frac: float = 0.28,
) -> None:
    """GIF with diagnostic overlays: contact cloud, wrench arrows, robot fan, HUD."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyArrowPatch

    verts0 = getattr(shape, "vertices")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for obs in obstacles:
        ax.add_patch(_obstacle_patch(obs))
    goal_verts = goal[:2] + rotate(goal[2], verts0 * GOAL_MARKER_SCALE)
    ax.add_patch(
        MplPolygon(goal_verts, closed=True, fill=False, ec="tab:green", lw=2, ls="--", zorder=4)
    )
    object_patch = MplPolygon(verts0, closed=True, fc="tab:blue", ec="k", alpha=0.75, zorder=3)
    ax.add_patch(object_patch)

    (robot_dot,) = ax.plot([], [], "o", color="tab:red", ms=9, zorder=8, label="robot")
    (robot_trail,) = ax.plot([], [], "-", color="tab:red", lw=1, alpha=0.45, zorder=3)
    (obj_plan_line,) = ax.plot(
        [], [], "-", color="tab:cyan", lw=2.0, alpha=0.85, zorder=6, label="object plan"
    )
    (rob_obj_plan_line,) = ax.plot(
        [], [], "-", color="darkorange", lw=2.0, alpha=0.85, zorder=6, label="robot→object"
    )
    (rob_plan_line,) = ax.plot(
        [], [], "--", color="magenta", lw=1.4, alpha=0.85, zorder=6, label="robot plan"
    )
    (contact_cloud,) = ax.plot(
        [], [], ".", color="deepskyblue", ms=3, alpha=0.35, zorder=7, label="pc samples"
    )
    (target_pc_dot,) = ax.plot([], [], "*", color="cyan", ms=14, zorder=9, label="target pc")

    telem0 = (log.get("telemetry") or [None])[0]
    n_fan = 0
    if telem0 is not None and "robot_rollouts" in telem0:
        n_fan = int(np.asarray(telem0["robot_rollouts"]).shape[0])
    fan_lines = [
        ax.plot([], [], "-", color="orange", lw=0.6, alpha=0.18, zorder=5)[0]
        for _ in range(n_fan)
    ]

    arrow_obj = FancyArrowPatch(
        (0, 0), (0, 0), arrowstyle="-|>", mutation_scale=12, color="cyan", lw=2, zorder=10
    )
    arrow_rob = FancyArrowPatch(
        (0, 0), (0, 0), arrowstyle="-|>", mutation_scale=12, color="darkorange", lw=2, zorder=10
    )
    ax.add_patch(arrow_obj)
    ax.add_patch(arrow_rob)

    hud = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9),
        zorder=20,
    )

    _apply_scene_limits(
        ax,
        shape,
        obstacles,
        goal,
        log["object_pose"][0],
        log["robot_pos"][0],
        pad_frac=view_pad_frac,
    )
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)

    n_pose = len(log["object_pose"])
    n_plan = len(log.get("object_plan", []))
    n_telem = len(log.get("telemetry") or [])
    frames = list(range(0, n_pose, stride))

    def _set_arrow(arrow, origin, wrench_xy):
        tip = origin + wrench_arrow_scale * wrench_xy
        arrow.set_positions(tuple(origin), tuple(tip))

    def update(i: int):
        pose = log["object_pose"][i]
        object_patch.set_xy(pose[:2] + rotate(pose[2], verts0))
        robot_dot.set_data([log["robot_pos"][i, 0]], [log["robot_pos"][i, 1]])
        robot_trail.set_data(log["robot_pos"][: i + 1, 0], log["robot_pos"][: i + 1, 1])

        pk = min(i, n_plan - 1) if n_plan > 0 else -1
        tk = min(i, n_telem - 1) if n_telem > 0 else -1

        if pk >= 0:
            op = log["object_plan"][pk]
            rp = log["robot_object_plan"][pk]
            rpath = log["robot_plan_path"][pk]
            obj_plan_line.set_data(op[:, 0], op[:, 1])
            rob_obj_plan_line.set_data(rp[:, 0], rp[:, 1])
            rob_plan_line.set_data(rpath[:, 0], rpath[:, 1])

        if tk >= 0:
            t = log["telemetry"][tk]
            pcs = np.asarray(t["contact_samples_pc"])
            contact_cloud.set_data(pcs[:, 0], pcs[:, 1]) if pcs.size else contact_cloud.set_data([], [])
            target = np.asarray(t["target_pc"])
            target_pc_dot.set_data([target[0]], [target[1]])
            com = np.asarray(t.get("object_com", pose[:2]))
            _set_arrow(arrow_obj, com, np.asarray(t["w_obj_world"])[:2])
            _set_arrow(arrow_rob, com, np.asarray(t["w_rob_world"])[:2])
            fan = np.asarray(t.get("robot_rollouts", np.zeros((0, 1, 2))))
            for li, line in enumerate(fan_lines):
                if li < fan.shape[0]:
                    line.set_data(fan[li, :, 0], fan[li, :, 1])
                    line.set_visible(True)
                else:
                    line.set_data([], [])
                    line.set_visible(False)
            sat = "YES" if t.get("dual_saturated") else "no"
            hud.set_text(
                f"MPC step {t.get('step', tk)} | ADMM iters {t.get('admm_iters', '?')}\n"
                f"||w_o-w_r||_0 = {t['primal_residual']:.3f}  "
                f"||g_o||={t['dual_norm_obj']:.2f}  ||g_r||={t['dual_norm_rob']:.2f}  sat={sat}\n"
                f"Obj task={t['obj_task_cost']:.2f}  Obj ADMM pen={t['obj_admm_penalty']:.2f}\n"
                f"Rob effort={t['rob_effort_cost']:.2f}  Rob ADMM pen={t['rob_admm_penalty']:.2f}"
            )
        else:
            contact_cloud.set_data([], [])
            target_pc_dot.set_data([], [])
            hud.set_text("")

        return (
            object_patch, robot_dot, robot_trail, obj_plan_line, rob_obj_plan_line,
            rob_plan_line, contact_cloud, target_pc_dot, hud, *fan_lines,
        )

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
