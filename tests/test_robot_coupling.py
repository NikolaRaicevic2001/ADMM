"""Tests for implied ell_c coupling and frozen/coupled rollouts."""

from __future__ import annotations

import numpy as np

from admm.admm_solver import ADMMSolver
from admm.consensus_spaces import WrenchConsensus
from admm.robot_subproblem import RobotSubproblem
from dynamics.physics_engine import build_engine_pair
from utils.config import load_config
from utils.environments import build_scenario


def test_implied_poses_match_euler():
    cfg = load_config()
    cfg["physics_backend"] = "analytical"
    sc = build_scenario(cfg, "clutter")
    pair = build_engine_pair(cfg, sc.object_, sc.obstacles)
    cons = WrenchConsensus(int(cfg["horizon"]), float(cfg["rho"]), float(cfg["max_dual"]))
    rob = RobotSubproblem(
        sc.object_,
        sc.obstacles,
        cfg,
        cons,
        np.random.default_rng(0),
        pair.planning,
        goal=sc.goal,
    )
    pose0 = sc.object_.pose.copy()
    h = int(cfg["horizon"])
    w = np.zeros((h, 3))
    w[:, 0] = 1.0
    dt = float(cfg["dt"])
    implied = rob._implied_poses(pose0, w, dt)
    pose = pose0.copy()
    for t in range(h):
        pose = sc.object_.propagate(pose, w[t], dt)
        assert np.allclose(implied[t], pose)


def test_ell_c_lower_when_wrenches_match_object_plan():
    cfg = load_config()
    cfg["physics_backend"] = "analytical"
    cfg["w_c_pos"] = 20.0
    cfg["w_c_theta"] = 5.0
    cfg["w_rob_goal_pos"] = 0.0
    cfg["w_rob_goal_theta"] = 0.0
    cfg["ell_c_source"] = "implied"
    sc = build_scenario(cfg, "clutter")
    pair = build_engine_pair(cfg, sc.object_, sc.obstacles)
    cons = WrenchConsensus(int(cfg["horizon"]), float(cfg["rho"]), float(cfg["max_dual"]))
    rob = RobotSubproblem(
        sc.object_,
        sc.obstacles,
        cfg,
        cons,
        np.random.default_rng(0),
        pair.planning,
        goal=sc.goal,
    )
    pose0 = sc.object_.pose.copy()
    h = int(cfg["horizon"])
    # Object plan: constant +x wrench
    w_plan = np.zeros((h, 3))
    w_plan[:, 0] = 2.0
    ref = np.zeros((h, 3))
    pose = pose0.copy()
    for t in range(h):
        pose = sc.object_.propagate(pose, w_plan[t], float(cfg["dt"]))
        ref[t] = pose

    z = np.zeros((h, 3))
    gamma = np.zeros((h, 3))
    u = np.zeros((h, 2))
    good = {"wrenches": w_plan, "positions": np.zeros((h, 2)), "object_poses": ref}
    bad_w = w_plan.copy()
    bad_w[:, 0] = -2.0
    bad = {"wrenches": bad_w, "positions": np.zeros((h, 2)), "object_poses": ref}
    c_good = rob._cost_components(u, good, z, gamma, ref, pose0)
    c_bad = rob._cost_components(u, bad, z, gamma, ref, pose0)
    assert c_good["rob_coupling_cost"] < c_bad["rob_coupling_cost"]


def test_analytical_frozen_rollout_object_poses_are_refs():
    cfg = load_config()
    cfg["physics_backend"] = "analytical"
    cfg["robot_rollout_mode"] = "frozen"
    sc = build_scenario(cfg, "clutter")
    pair = build_engine_pair(cfg, sc.object_, sc.obstacles)
    h = 4
    ref = np.tile(sc.object_.pose, (h, 1))
    ref[:, 0] += np.linspace(0, 0.05, h)
    u = np.zeros((2, h, 2))
    w, p, obj = pair.planning.rollout_batch(u, ref, sc.robot_pos, float(cfg["dt"]))
    assert w.shape == (2, h, 3)
    assert p.shape == (2, h, 2)
    assert obj.shape == (2, h, 3)
    assert np.allclose(obj[0], ref)
    assert np.allclose(obj[1], ref)


def test_analytical_coupled_rollout_object_can_move():
    cfg = load_config()
    cfg["physics_backend"] = "analytical"
    cfg["robot_rollout_mode"] = "coupled"
    sc = build_scenario(cfg, "clutter")
    pair = build_engine_pair(cfg, sc.object_, sc.obstacles)
    h = 6
    ref = np.tile(sc.object_.pose, (h, 1))
    # Push from the right toward the object
    r0 = sc.object_.pose[:2] + np.array([0.12, 0.03])
    u = np.zeros((1, h, 2))
    u[0, :, 0] = -0.3
    w, p, obj = pair.planning.rollout_batch(u, ref, r0, float(cfg["dt"]))
    assert obj.shape == (1, h, 3)
    # Coupled object should not be forced to stay exactly on ref
    assert not np.allclose(obj[0], ref)


def test_admm_analytical_smoke_with_coupling():
    cfg = load_config()
    cfg["physics_backend"] = "analytical"
    cfg["robot_rollout_mode"] = "frozen"
    cfg["n_admm"] = 2
    cfg["horizon"] = 5
    cfg["k_object"] = 8
    cfg["k_robot"] = 8
    cfg["w_c_pos"] = 20.0
    cfg["w_c_theta"] = 5.0
    sc = build_scenario(cfg, "clutter")
    solver = ADMMSolver(sc.object_, sc.robot_pos, sc.obstacles, sc.goal, cfg)
    u0, residuals, plan = solver.control_step()
    assert np.isfinite(u0).all()
    telem = plan["telemetry"]
    assert "rob_coupling_cost" in telem
    assert np.isfinite(telem["rob_coupling_cost"])
