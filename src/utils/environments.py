"""Named push environments: object start, robot start, goal, obstacles.

Each environment is designed so that:
  - the object at the start pose does not penetrate obstacles
  - the object at the goal pose does not penetrate obstacles (with margin)
  - a geometrically plausible passage exists between start and goal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dynamics.object_2d import QuasiStaticObject2D
from geometry.analytical_2d import BoxSDF, CircleSDF, PolygonSDF, t_shape_vertices
from geometry.base_sdf import BaseSDF
from utils.math_utils import rotate


@dataclass
class Scenario:
    name: str
    description: str
    shape: BaseSDF
    object_: QuasiStaticObject2D
    robot_pos: np.ndarray
    goal: np.ndarray
    obstacles: list[BaseSDF]


def _make_object(
    cfg: dict,
    pose: np.ndarray,
    obstacles: list[BaseSDF],
    shape: BaseSDF | None = None,
) -> tuple[BaseSDF, QuasiStaticObject2D]:
    shape = shape or PolygonSDF(t_shape_vertices())
    object_ = QuasiStaticObject2D(
        shape,
        pose=np.asarray(pose, dtype=float),
        mu=float(cfg["mu"]),
        mass=float(cfg["object_mass"]),
        gravity=float(cfg["gravity"]),
        limit_surface_c=float(cfg["limit_surface_c"]),
        limit_surface_r=float(cfg["limit_surface_r"]),
        obstacles=obstacles,
        obstacle_margin=float(cfg["obstacle_margin"]),
        pushout_iters=int(cfg["object_pushout_iters"]),
    )
    return shape, object_


def min_obstacle_clearance(
    shape: BaseSDF, pose: np.ndarray, obstacles: list[BaseSDF]
) -> float:
    """Minimum signed distance from object boundary samples to any obstacle."""
    samples = getattr(shape, "boundary_samples", None)
    if samples is None:
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        samples = shape.bounding_radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    verts = pose[:2] + rotate(pose[2], samples)
    return float(min(obs.sdf(verts).min() for obs in obstacles)) if obstacles else np.inf


def env_clutter(cfg: dict) -> Scenario:
    """Open tabletop with three separated obstacles; goal is clear of all of them.

    Layout (meters):
      start (0,0) ----> goal (0.50, 0.48, pi/4)
      Obstacles sit off the direct diagonal so the T-shape can pass with room.
    """
    obstacles: list[BaseSDF] = [
        CircleSDF(center_xy=np.array([0.08, 0.32]), radius=0.04),
        BoxSDF(
            center_xy=np.array([0.38, 0.10]),
            half_extents=np.array([0.04, 0.035]),
            angle=0.25,
        ),
        PolygonSDF(np.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])),
    ]
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.50, 0.48, np.pi / 4])
    shape, object_ = _make_object(cfg, start, obstacles)
    return Scenario(
        name="clutter",
        description="Push T-shape past separated circle/box/triangle clutter to a clear goal",
        shape=shape,
        object_=object_,
        robot_pos=np.array([-0.05, -0.06]),
        goal=goal,
        obstacles=obstacles,
    )


def env_corridor(cfg: dict) -> Scenario:
    """Horizontal corridor between two walls; goal is past the exit and clear.

    Channel centerline y≈0.15. Wall faces leave a ~0.24 m vertical gap (T-shape
    height ≈0.15 m), so the object fits without rotating. A side post sits north
    of the exit — not on the goal.
    """
    # Top wall: y in [0.27, 0.33], x in [0.02, 0.42]
    # Bottom wall: y in [-0.03, 0.03], x in [0.02, 0.42]
    # Channel: y in (0.03, 0.27) → gap 0.24 m
    obstacles: list[BaseSDF] = [
        BoxSDF(
            center_xy=np.array([0.22, 0.30]),
            half_extents=np.array([0.20, 0.03]),
            angle=0.0,
        ),
        BoxSDF(
            center_xy=np.array([0.22, 0.00]),
            half_extents=np.array([0.20, 0.03]),
            angle=0.0,
        ),
        # Side post after the corridor, north of the centerline (away from goal)
        CircleSDF(center_xy=np.array([0.52, 0.34]), radius=0.04),
    ]
    start = np.array([-0.10, 0.15, 0.0])
    goal = np.array([0.70, 0.15, 0.0])
    shape, object_ = _make_object(cfg, start, obstacles)
    return Scenario(
        name="corridor",
        description="Push through a narrow horizontal corridor to a clear exit goal",
        shape=shape,
        object_=object_,
        robot_pos=np.array([-0.18, 0.08]),
        goal=goal,
        obstacles=obstacles,
    )


def env_gate(cfg: dict) -> Scenario:
    """Vertical gate with a wide enough slot; goal beyond the gate, clear of props.

    Gate at x≈0.28. Slot y ∈ (0.02, 0.24) → 0.22 m (T height ≈0.15 m at θ≈0).
    Start and goal both centered on the slot so the object can pass upright.
    """
    obstacles: list[BaseSDF] = [
        # Upper pillar: y in [0.24, 0.44]
        BoxSDF(
            center_xy=np.array([0.28, 0.34]),
            half_extents=np.array([0.035, 0.10]),
            angle=0.0,
        ),
        # Lower pillar: y in [-0.18, 0.02]
        BoxSDF(
            center_xy=np.array([0.28, -0.08]),
            half_extents=np.array([0.035, 0.10]),
            angle=0.0,
        ),
        # Decorations clear of the post-gate goal region
        CircleSDF(center_xy=np.array([0.48, 0.38]), radius=0.04),
        PolygonSDF(np.array([[0.52, -0.18], [0.64, -0.18], [0.58, -0.06]])),
    ]
    start = np.array([-0.05, 0.13, 0.0])
    goal = np.array([0.62, 0.13, np.pi / 6])
    shape, object_ = _make_object(cfg, start, obstacles)
    return Scenario(
        name="gate",
        description="Pass through a vertical gate slot, then reach a lightly rotated goal",
        shape=shape,
        object_=object_,
        robot_pos=np.array([-0.12, 0.05]),
        goal=goal,
        obstacles=obstacles,
    )


ENVIRONMENTS: dict[str, Callable[[dict], Scenario]] = {
    "clutter": env_clutter,
    "corridor": env_corridor,
    "gate": env_gate,
}

DEFAULT_ENV = "clutter"


def list_environments() -> list[str]:
    return sorted(ENVIRONMENTS.keys())


def build_scenario(cfg: dict, env_name: str = DEFAULT_ENV) -> Scenario:
    key = env_name.lower().strip()
    if key not in ENVIRONMENTS:
        known = ", ".join(list_environments())
        raise ValueError(f"Unknown env '{env_name}'. Choose from: {known}")
    return ENVIRONMENTS[key](cfg)
