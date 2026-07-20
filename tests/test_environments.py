"""Feasibility checks for named push environments."""

from __future__ import annotations

import numpy as np

from utils.config import load_config
from utils.environments import (
    build_scenario,
    list_environments,
    min_obstacle_clearance,
)
from utils.math_utils import rotate


def test_start_and_goal_clear_of_obstacles() -> None:
    cfg = load_config()
    margin = float(cfg["obstacle_margin"])
    for name in list_environments():
        sc = build_scenario(cfg, name)
        d0 = min_obstacle_clearance(sc.shape, sc.object_.pose, sc.obstacles)
        dg = min_obstacle_clearance(sc.shape, sc.goal, sc.obstacles)
        assert d0 >= margin, f"{name}: start penetrates obstacles (clearance={d0:.4f})"
        assert dg >= margin, f"{name}: goal penetrates obstacles (clearance={dg:.4f})"


def test_robot_starts_outside_object() -> None:
    cfg = load_config()
    for name in list_environments():
        sc = build_scenario(cfg, name)
        q = sc.object_.body_frame_point(sc.robot_pos, sc.object_.pose)
        d = float(sc.shape.sdf(q[None])[0])
        assert d >= 0.0, f"{name}: robot starts inside object (sdf={d:.4f})"


def test_corridor_channel_fits_object() -> None:
    """Channel vertical gap must exceed T-shape height plus margin."""
    cfg = load_config()
    sc = build_scenario(cfg, "corridor")
    # Reconstruct wall faces from known layout: top bottom-face and bottom top-face
    top = sc.obstacles[0]
    bot = sc.obstacles[1]
    top_inner = top.center[1] - top.half_extents[1]
    bot_inner = bot.center[1] + bot.half_extents[1]
    gap = top_inner - bot_inner
    # T-shape body-frame y span
    ys = sc.shape.vertices[:, 1]
    height = float(ys.max() - ys.min())
    assert gap > height + float(cfg["obstacle_margin"]), (
        f"corridor gap {gap:.3f} too tight for object height {height:.3f}"
    )


def test_gate_slot_fits_object() -> None:
    cfg = load_config()
    sc = build_scenario(cfg, "gate")
    upper = sc.obstacles[0]
    lower = sc.obstacles[1]
    upper_inner = upper.center[1] - upper.half_extents[1]
    lower_inner = lower.center[1] + lower.half_extents[1]
    gap = upper_inner - lower_inner
    ys = sc.shape.vertices[:, 1]
    height = float(ys.max() - ys.min())
    assert gap > height + float(cfg["obstacle_margin"]), (
        f"gate slot {gap:.3f} too tight for object height {height:.3f}"
    )


def test_goal_com_not_inside_any_obstacle() -> None:
    cfg = load_config()
    for name in list_environments():
        sc = build_scenario(cfg, name)
        for i, obs in enumerate(sc.obstacles):
            d = float(obs.sdf(sc.goal[:2][None])[0])
            assert d > 0.0, f"{name}: goal CoM inside obstacle {i} (sdf={d:.4f})"
