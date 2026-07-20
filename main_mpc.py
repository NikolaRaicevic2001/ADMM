"""Receding-horizon MPC entry point for wrench-consensus ADMM pushing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from admm.admm_solver import ADMMSolver
from dynamics.object_2d import QuasiStaticObject2D
from geometry.analytical_2d import BoxSDF, CircleSDF, PolygonSDF, t_shape_vertices
from utils.config import load_config
from utils.visualization import plot_overview, plot_residuals, save_animation

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def build_scenario(cfg: dict):
    shape = PolygonSDF(t_shape_vertices())
    object_ = QuasiStaticObject2D(
        shape,
        pose=np.array([0.0, 0.0, 0.0]),
        mu=float(cfg["mu"]),
        mass=float(cfg["object_mass"]),
        gravity=float(cfg["gravity"]),
        limit_surface_c=float(cfg["limit_surface_c"]),
        limit_surface_r=float(cfg["limit_surface_r"]),
        obstacles=[],  # filled below
        obstacle_margin=float(cfg["obstacle_margin"]),
        pushout_iters=int(cfg["object_pushout_iters"]),
    )
    robot_pos = np.array([-0.05, -0.06])
    goal = np.array([0.45, 0.5, np.pi / 4])
    obstacles = [
        CircleSDF(center_xy=np.array([0.06, 0.30]), radius=0.04),
        BoxSDF(
            center_xy=np.array([0.36, 0.16]),
            half_extents=np.array([0.04, 0.035]),
            angle=0.2,
        ),
        PolygonSDF(np.array([[0.08, 0.40], [0.18, 0.40], [0.13, 0.49]])),
    ]
    object_.obstacles = obstacles
    return shape, object_, robot_pos, goal, obstacles


def run(cfg_path: Path | None = None, max_steps: int | None = None):
    cfg = load_config(cfg_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["random_seed"]))
    shape, object_, robot_pos, goal, obstacles = build_scenario(cfg)

    solver = ADMMSolver(object_, robot_pos, obstacles, goal, cfg, rng)
    log = solver.run(max_steps=max_steps)

    print(f"final object pose: {log['object_pose'][-1]}")
    print(f"goal pose:          {goal}")
    print(f"goal reached:       {log['reached']}")

    overview_path = unique_path(RESULTS_DIR / "trajectory_overview.png")
    plot_overview(log, shape, obstacles, goal, overview_path)
    print(f"saved {overview_path}")

    residual_path = unique_path(RESULTS_DIR / "admm_residuals.png")
    plot_residuals(log, residual_path, eps=float(cfg["eps_r"]))
    print(f"saved {residual_path}")

    if cfg.get("save_animation", True):
        try:
            anim_path = unique_path(RESULTS_DIR / "pushing_animation.gif")
            save_animation(log, shape, obstacles, goal, anim_path)
            print(f"saved {anim_path}")
        except Exception as exc:
            print(f"skipped animation: {exc}")

    return log


if __name__ == "__main__":
    run()
