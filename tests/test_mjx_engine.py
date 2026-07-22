"""MJX physics backend verification matrix + ADMM smoke."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("jax")
mjx = pytest.importorskip("mujoco.mjx")

from admm.admm_solver import ADMMSolver
from dynamics.mjx_scene import build_execution_xml, build_planning_xml, t_shape_convex_boxes
from dynamics.physics_engine import build_engine_pair
from utils.config import load_config
from utils.environments import build_scenario


@pytest.fixture
def cfg_mjx():
    cfg = load_config()
    cfg["physics_backend"] = "mjx"
    return cfg


@pytest.fixture
def engines(cfg_mjx):
    sc = build_scenario(cfg_mjx, "clutter")
    return build_engine_pair(cfg_mjx, sc.object_, sc.obstacles), sc


def test_xml_loads_and_has_contact_sensors(cfg_mjx):
    import mujoco

    sc = build_scenario(cfg_mjx, "clutter")
    for builder in (build_planning_xml, build_execution_xml):
        m = mujoco.MjModel.from_xml_string(builder(cfg_mjx, sc.obstacles))
        names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)
        ]
        assert "c_cross" in names and "c_stem" in names
        mjx.put_model(m)


def test_t_shape_boxes_cover_outline():
    boxes = t_shape_convex_boxes()
    assert len(boxes) == 2
    # crossbar half-extents
    assert np.allclose(boxes[0][0][:2], [0.09, 0.015])


def test_execution_linear_friction_decay(engines, cfg_mjx):
    import jax.numpy as jnp

    pair, _ = engines
    ex = pair.execution
    ex.seed(np.zeros(3), np.array([0.5, 0.5]))
    qvel = np.array(ex._data.qvel)
    qvel[:] = 0.0
    qvel[ex._object_x] = 0.3
    ex._data = ex._data.replace(qvel=jnp.asarray(qvel))
    dt = float(cfg_mjx["dt"])
    mu = float(cfg_mjx["mu"])
    g = float(cfg_mjx["gravity"])
    pose0 = np.array(ex._data.qpos[ex._object_x : ex._object_x + 3])
    ex.step_execution(np.zeros(2), dt)
    pose1 = np.array(ex._data.qpos[ex._object_x : ex._object_x + 3])
    v1 = float(np.array(ex._data.qvel)[ex._object_x])
    expected = 0.3 - mu * g * dt
    assert abs(v1 - expected) < 0.02
    assert abs(pose1[1] - pose0[1]) < 1e-3
    assert abs(pose1[2] - pose0[2]) < 1e-3
    # no spring-back toward origin from frictionloss
    assert pose1[0] > pose0[0]


def test_execution_angular_friction_decay(engines, cfg_mjx):
    import jax.numpy as jnp

    pair, _ = engines
    ex = pair.execution
    ex.seed(np.zeros(3), np.array([0.5, 0.5]))
    qvel = np.array(ex._data.qvel)
    qvel[:] = 0.0
    qvel[ex._object_x + 2] = 1.0
    ex._data = ex._data.replace(qvel=jnp.asarray(qvel))
    dt = float(cfg_mjx["dt"])
    ex.step_execution(np.zeros(2), dt)
    w1 = float(np.array(ex._data.qvel)[ex._object_x + 2])
    assert w1 < 1.0
    assert abs(float(np.array(ex._data.qvel)[ex._object_x])) < 0.05
    assert abs(float(np.array(ex._data.qvel)[ex._object_x + 1])) < 0.05


def test_planning_contact_wrench_sign(engines, cfg_mjx):
    pair, sc = engines
    dt = float(cfg_mjx["dt"])
    ref = np.tile(sc.object_.pose.astype(float), (6, 1))
    r0 = sc.object_.pose[:2] + np.array([0.12, 0.03])
    u = np.zeros((1, 6, 2))
    u[0, :, 0] = -0.3
    wrenches, paths = pair.planning.rollout_batch(u, ref, r0, dt)
    assert wrenches.shape == (1, 6, 3)
    assert paths.shape == (1, 6, 2)
    assert np.isfinite(wrenches).all()
    # After contact establishes, force on object should push left (negative fx)
    late = wrenches[0, 2:, 0]
    assert np.mean(late) < -0.5


def test_rollout_batch_shapes(engines, cfg_mjx):
    pair, sc = engines
    k, h = 3, 4
    u = np.zeros((k, h, 2))
    ref = np.tile(sc.object_.pose, (h, 1))
    w, p = pair.planning.rollout_batch(u, ref, sc.robot_pos, float(cfg_mjx["dt"]))
    assert w.shape == (k, h, 3)
    assert p.shape == (k, h, 2)


def test_planning_rejects_execution_api(engines):
    pair, sc = engines
    with pytest.raises(RuntimeError):
        pair.planning.seed(sc.object_.pose, sc.robot_pos)
    with pytest.raises(RuntimeError):
        pair.execution.rollout_batch(
            np.zeros((1, 2, 2)), np.zeros((2, 3)), sc.robot_pos, 0.05
        )


def test_admm_control_step_analytical_smoke():
    cfg = load_config()
    cfg["physics_backend"] = "analytical"
    cfg["n_admm"] = 2
    cfg["horizon"] = 5
    cfg["k_object"] = 8
    cfg["k_robot"] = 8
    sc = build_scenario(cfg, "clutter")
    solver = ADMMSolver(sc.object_, sc.robot_pos, sc.obstacles, sc.goal, cfg)
    u0, residuals, plan = solver.control_step()
    assert np.isfinite(u0).all()
    assert "telemetry" in plan


def test_admm_control_step_mjx_smoke(cfg_mjx):
    cfg = dict(cfg_mjx)
    cfg["n_admm"] = 2
    cfg["horizon"] = 5
    cfg["k_object"] = 8
    cfg["k_robot"] = 8
    sc = build_scenario(cfg, "clutter")
    solver = ADMMSolver(sc.object_, sc.robot_pos, sc.obstacles, sc.goal, cfg)
    u0, residuals, plan = solver.control_step()
    assert np.isfinite(u0).all()
    assert "telemetry" in plan
