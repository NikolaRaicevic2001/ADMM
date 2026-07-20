"""Receding-horizon MPC entry point for wrench-consensus ADMM pushing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from admm.admm_solver import ADMMSolver
from utils.config import load_config
from utils.environments import DEFAULT_ENV, build_scenario, list_environments
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADMM-MPC planar pushing with world-frame CoM wrench consensus",
    )
    parser.add_argument(
        "--env",
        choices=list_environments(),
        default=DEFAULT_ENV,
        help=f"Push environment (default: {DEFAULT_ENV}). "
        f"Available: {', '.join(list_environments())}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: config/base_config.yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max control steps from config",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="Print available environments and exit",
    )
    return parser.parse_args(argv)


def run(
    cfg_path: Path | None = None,
    max_steps: int | None = None,
    env_name: str = DEFAULT_ENV,
):
    cfg = load_config(cfg_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["random_seed"]))
    scenario = build_scenario(cfg, env_name)

    print(f"env={scenario.name}: {scenario.description}")

    solver = ADMMSolver(
        scenario.object_,
        scenario.robot_pos,
        scenario.obstacles,
        scenario.goal,
        cfg,
        rng,
    )
    log = solver.run(max_steps=max_steps)

    print(f"final object pose: {log['object_pose'][-1]}")
    print(f"goal pose:          {scenario.goal}")
    print(f"goal reached:       {log['reached']}")

    tag = scenario.name
    overview_path = unique_path(RESULTS_DIR / f"trajectory_overview_{tag}.png")
    plot_overview(
        log, scenario.shape, scenario.obstacles, scenario.goal, overview_path
    )
    print(f"saved {overview_path}")

    residual_path = unique_path(RESULTS_DIR / f"admm_residuals_{tag}.png")
    plot_residuals(log, residual_path, eps=float(cfg["eps_r"]))
    print(f"saved {residual_path}")

    if cfg.get("save_animation", True):
        try:
            anim_path = unique_path(RESULTS_DIR / f"pushing_animation_{tag}.gif")
            save_animation(
                log, scenario.shape, scenario.obstacles, scenario.goal, anim_path
            )
            print(f"saved {anim_path}")
        except Exception as exc:
            print(f"skipped animation: {exc}")

    return log


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_envs:
        for name in list_environments():
            scenario = build_scenario(load_config(args.config), name)
            print(f"  {name:12s}  {scenario.description}")
        return
    run(cfg_path=args.config, max_steps=args.max_steps, env_name=args.env)


if __name__ == "__main__":
    main()
