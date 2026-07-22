"""MuJoCo MJX physics backend with batched JIT planning rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynamics.mjx_scene import build_execution_xml, build_planning_xml
from dynamics.object_2d import QuasiStaticObject2D
from dynamics.physics_engine import EnginePair, PhysicsEngine2D
from geometry.base_sdf import BaseSDF

try:
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MJX backend requires mujoco, mujoco-mjx, and jax. "
        "Install with: pip install mujoco mujoco-mjx jax jaxlib"
    ) from exc


def _clip_speed(u: np.ndarray, vmax: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    speed = np.linalg.norm(u, axis=-1, keepdims=True)
    scale = np.ones_like(speed)
    mask = speed[..., 0] > vmax
    scale[mask] = vmax / np.maximum(speed[mask], 1e-12)
    return u * scale


class MjxPlanningEngine(PhysicsEngine2D):
    """Planning world: object welded to mocap; robot velocity-actuated."""

    def __init__(
        self,
        cfg: dict[str, Any],
        obstacles: list[BaseSDF],
    ) -> None:
        self.cfg = cfg
        self.vmax = float(cfg.get("robot_max_speed", 1.0))
        self.f_max = float(cfg["f_max"])
        self.mu_c = float(cfg["mu_c"])
        self.n_substeps = max(int(cfg.get("mjx_n_substeps", cfg.get("n_contact_substeps", 4))), 1)
        self.z = 0.05

        xml = build_planning_xml(cfg, obstacles)
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mjx_model = mjx.put_model(self.mj_model)
        self._object_body = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self._robot_body = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )
        # freejoint qpos: xyz + quat; robot slides at end
        self._robot_qadr = int(self.mj_model.jnt_qposadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_x")
        ])
        self._wrench_clip = self.f_max * (1.0 + self.mu_c)
        self._jitted_rollout = self._compile_rollout()

    def _compile_rollout(self):
        model = self.mjx_model
        n_sub = self.n_substeps
        robot_qadr = self._robot_qadr
        object_body = self._object_body
        robot_body = self._robot_body
        wrench_clip = self._wrench_clip
        z = self.z

        def single_rollout(u_seq, ref_poses, robot_pos0):
            # u_seq: (H, 2), ref_poses: (H, 3), robot_pos0: (2,)
            data0 = mjx.make_data(model)

            def init_state():
                qpos = data0.qpos
                # object freejoint from first ref pose
                pose0 = ref_poses[0]
                quat = jnp.array(
                    [jnp.cos(0.5 * pose0[2]), 0.0, 0.0, jnp.sin(0.5 * pose0[2])]
                )
                qpos = qpos.at[0:3].set(jnp.array([pose0[0], pose0[1], z]))
                qpos = qpos.at[3:7].set(quat)
                qpos = qpos.at[robot_qadr : robot_qadr + 2].set(robot_pos0)
                mocap_pos = data0.mocap_pos.at[0].set(
                    jnp.array([pose0[0], pose0[1], z])
                )
                mocap_quat = data0.mocap_quat.at[0].set(quat)
                return data0.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

            data = init_state()

            def horizon_step(data, inputs):
                u_t, pose_t = inputs
                quat = jnp.array(
                    [jnp.cos(0.5 * pose_t[2]), 0.0, 0.0, jnp.sin(0.5 * pose_t[2])]
                )
                mocap_pos = data.mocap_pos.at[0].set(
                    jnp.array([pose_t[0], pose_t[1], z])
                )
                mocap_quat = data.mocap_quat.at[0].set(quat)
                data = data.replace(
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                    ctrl=u_t,
                )

                def substep(d, _):
                    return mjx.step(model, d), None

                data, _ = jax.lax.scan(substep, data, None, length=n_sub)

                # Contact forces on robot (geom1); force on object is opposite.
                f_sum = data.sensordata[0:3] + data.sensordata[3:6]
                f_obj = -f_sum
                obj_xy = data.xpos[object_body, 0:2]
                rob_xy = data.xpos[robot_body, 0:2]
                r = rob_xy - obj_xy
                tau = r[0] * f_obj[1] - r[1] * f_obj[0]
                wrench = jnp.array([f_obj[0], f_obj[1], tau])
                wrench = jnp.clip(wrench, -wrench_clip, wrench_clip)
                return data, (wrench, rob_xy)

            _, (wrenches, paths) = jax.lax.scan(
                horizon_step, data, (u_seq, ref_poses)
            )
            return wrenches, paths

        batched = jax.vmap(single_rollout, in_axes=(0, None, None))
        return jax.jit(batched)

    def seed(self, object_pose: np.ndarray, robot_pos: np.ndarray) -> None:
        raise RuntimeError("seed is only valid on the execution engine")

    def step_execution(
        self, u_cmd: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("step_execution is only valid on the execution engine")

    def rollout_batch(
        self,
        u_seq: np.ndarray,
        ref_poses: np.ndarray,
        robot_pos0: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del dt  # physics timestep baked into MJCF from cfg["dt"] / n_substeps
        u_seq = _clip_speed(np.asarray(u_seq, dtype=float), self.vmax)
        ref_poses = np.asarray(ref_poses, dtype=float)
        robot_pos0 = np.asarray(robot_pos0, dtype=float).reshape(2)
        wrenches, paths = self._jitted_rollout(
            jnp.asarray(u_seq),
            jnp.asarray(ref_poses),
            jnp.asarray(robot_pos0),
        )
        return np.asarray(wrenches, dtype=float), np.asarray(paths, dtype=float)


class MjxExecutionEngine(PhysicsEngine2D):
    """Execution world: planar frictionloss object + velocity-actuated robot."""

    def __init__(
        self,
        cfg: dict[str, Any],
        obstacles: list[BaseSDF],
    ) -> None:
        self.cfg = cfg
        self.vmax = float(cfg.get("robot_max_speed", 1.0))
        self.n_substeps = max(int(cfg.get("mjx_n_substeps", cfg.get("n_contact_substeps", 4))), 1)
        self.z = 0.05

        xml = build_execution_xml(cfg, obstacles)
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mjx_model = mjx.put_model(self.mj_model)
        self._data = mjx.make_data(self.mjx_model)
        self._object_x = int(
            self.mj_model.jnt_qposadr[
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "object_x")
            ]
        )
        self._robot_x = int(
            self.mj_model.jnt_qposadr[
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_x")
            ]
        )
        self._jitted_step = self._compile_step()

    def _compile_step(self):
        model = self.mjx_model
        n_sub = self.n_substeps
        object_x = self._object_x
        robot_x = self._robot_x

        def step_fn(data, u_cmd):
            data = data.replace(ctrl=u_cmd)

            def substep(d, _):
                return mjx.step(model, d), None

            data, _ = jax.lax.scan(substep, data, None, length=n_sub)
            obj = data.qpos[object_x : object_x + 3]
            rob = data.qpos[robot_x : robot_x + 2]
            return data, obj, rob

        return jax.jit(step_fn)

    def seed(self, object_pose: np.ndarray, robot_pos: np.ndarray) -> None:
        pose = np.asarray(object_pose, dtype=float).reshape(3)
        rob = np.asarray(robot_pos, dtype=float).reshape(2)
        qpos = np.array(self._data.qpos)
        qvel = np.zeros_like(np.array(self._data.qvel))
        qpos[self._object_x : self._object_x + 3] = pose
        qpos[self._robot_x : self._robot_x + 2] = rob
        self._data = self._data.replace(
            qpos=jnp.asarray(qpos),
            qvel=jnp.asarray(qvel),
            ctrl=jnp.zeros(self.mj_model.nu),
        )

    def step_execution(
        self, u_cmd: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        del dt
        u = _clip_speed(np.asarray(u_cmd, dtype=float).reshape(2), self.vmax)
        self._data, obj, rob = self._jitted_step(self._data, jnp.asarray(u))
        return np.asarray(obj, dtype=float), np.asarray(rob, dtype=float)

    def rollout_batch(
        self,
        u_seq: np.ndarray,
        ref_poses: np.ndarray,
        robot_pos0: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("rollout_batch is only valid on the planning engine")


def build_mjx_engine_pair(
    cfg: dict[str, Any],
    object_: QuasiStaticObject2D,
    obstacles: list[BaseSDF],
) -> EnginePair:
    del object_  # geometry comes from canonical T in MJCF
    return EnginePair(
        planning=MjxPlanningEngine(cfg, obstacles),
        execution=MjxExecutionEngine(cfg, obstacles),
    )
