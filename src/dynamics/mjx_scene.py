"""MJCF scene builders for planar pushing (planning + execution worlds)."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.analytical_2d import BoxSDF, CircleSDF, PolygonSDF, t_shape_vertices
from geometry.base_sdf import BaseSDF


def t_shape_convex_boxes() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(half_size_xyz, pos_xyz), ...] for crossbar + stem in body frame."""
    # Matches analytical t_shape_vertices convex split; CoM near origin.
    crossbar_size = np.array([0.090, 0.015, 0.020])
    crossbar_pos = np.array([0.0, 0.030, 0.0])
    stem_size = np.array([0.015, 0.060, 0.020])
    stem_pos = np.array([0.0, -0.045, 0.0])
    return [(crossbar_size, crossbar_pos), (stem_size, stem_pos)]


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Return MuJoCo quaternion (w, x, y, z) for planar yaw about +z."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _fmt(xs: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in xs)


def _obstacle_geoms(obstacles: list[BaseSDF], z: float) -> str:
    chunks: list[str] = []
    for i, obs in enumerate(obstacles):
        if isinstance(obs, CircleSDF):
            c = np.asarray(obs.center, dtype=float)
            chunks.append(
                f'<geom name="obs_{i}" type="sphere" size="{float(obs.radius):.6g}"'
                f' pos="{c[0]:.6g} {c[1]:.6g} {z:.6g}" rgba="0.5 0.5 0.5 1"/>'
            )
        elif isinstance(obs, BoxSDF):
            c = np.asarray(obs.center_xy, dtype=float)
            he = np.asarray(obs.half_extents, dtype=float)
            ang = float(obs.angle)
            chunks.append(
                f'<geom name="obs_{i}" type="box" size="{he[0]:.6g} {he[1]:.6g} 0.04"'
                f' pos="{c[0]:.6g} {c[1]:.6g} {z:.6g}" euler="0 0 {ang:.6g}"'
                f' rgba="0.5 0.5 0.5 1"/>'
            )
        elif isinstance(obs, PolygonSDF):
            # Approximate with axis-aligned box from AABB (clutter triangles stay blocking).
            v = np.asarray(obs.vertices, dtype=float)
            lo = v.min(axis=0)
            hi = v.max(axis=0)
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            chunks.append(
                f'<geom name="obs_{i}" type="box" size="{half[0]:.6g} {half[1]:.6g} 0.04"'
                f' pos="{center[0]:.6g} {center[1]:.6g} {z:.6g}" rgba="0.5 0.5 0.5 1"/>'
            )
        else:
            c = np.asarray(obs.center, dtype=float)
            r = float(getattr(obs, "bounding_radius", 0.04))
            chunks.append(
                f'<geom name="obs_{i}" type="sphere" size="{r:.6g}"'
                f' pos="{c[0]:.6g} {c[1]:.6g} {z:.6g}" rgba="0.5 0.5 0.5 1"/>'
            )
    return "\n      ".join(chunks)


def _object_geoms(mass: float) -> str:
    boxes = t_shape_convex_boxes()
    m_each = mass / max(len(boxes), 1)
    parts = []
    for i, (size, pos) in enumerate(boxes):
        name = "obj_cross" if i == 0 else "obj_stem"
        parts.append(
            f'<geom name="{name}" type="box" size="{_fmt(size)}" pos="{_fmt(pos)}"'
            f' mass="{m_each:.6g}" rgba="0.2 0.45 0.85 1"/>'
        )
    return "\n        ".join(parts)


def _contact_sensors() -> str:
    # maxforce is JAX/MJX-compatible (netforce is not).
    return """
    <contact name="c_cross" geom1="robot_geom" geom2="obj_cross" data="force" reduce="maxforce"/>
    <contact name="c_stem" geom1="robot_geom" geom2="obj_stem" data="force" reduce="maxforce"/>
"""


def _option_block(cfg: dict[str, Any], dt_ctrl: float) -> str:
    n_sub = max(int(cfg.get("mjx_n_substeps", cfg.get("n_contact_substeps", 4))), 1)
    iters = int(cfg.get("mjx_solver_iterations", 20))
    timestep = float(dt_ctrl) / float(n_sub)
    return (
        f'<option timestep="{timestep:.8g}" gravity="0 0 -{float(cfg["gravity"]):.6g}" '
        f'iterations="{iters}" ls_iterations="{iters}" integrator="Euler"/>'
    )


def build_planning_xml(cfg: dict[str, Any], obstacles: list[BaseSDF]) -> str:
    """Dynamic object welded to mocap + velocity-actuated planar robot."""
    z = 0.05
    mass = float(cfg["object_mass"])
    r = float(cfg.get("robot_radius", 0.012))
    robot_mass = float(cfg.get("robot_mass", 1.0))
    mu_c = float(cfg["mu_c"])
    dt = float(cfg["dt"])
    vmax = float(cfg.get("robot_max_speed", 1.0))
    obs = _obstacle_geoms(obstacles, z)
    obj = _object_geoms(mass)
    return f"""
<mujoco model="admm_planning">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  {_option_block(cfg, dt)}
  <default>
    <geom contype="1" conaffinity="1" friction="{mu_c:.4g} 0.005 0.0001"
          solref="0.004 1" solimp="0.9 0.95 0.001" rgba="0.7 0.7 0.7 1"/>
    <joint damping="0.05" armature="0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 0" rgba="0.85 0.85 0.85 1"/>
    <body name="object_mocap" mocap="true" pos="0 0 {z}">
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 0 0 0.2"/>
    </body>
    <body name="object" pos="0 0 {z}">
      <freejoint name="object_free"/>
      {obj}
      <site name="object_com" pos="0 0 0" size="0.008"/>
    </body>
    <body name="robot" pos="0 0 {z}">
      <joint name="robot_x" type="slide" axis="1 0 0"/>
      <joint name="robot_y" type="slide" axis="0 1 0"/>
      <geom name="robot_geom" type="sphere" size="{r:.6g}" mass="{robot_mass:.6g}"
            rgba="0.9 0.25 0.2 1"/>
    </body>
    {obs}
  </worldbody>
  <equality>
    <weld name="object_weld" body1="object_mocap" body2="object"
          torquescale="1" solref="0.002 1" solimp="0.95 0.99 0.001"/>
  </equality>
  <actuator>
    <velocity name="robot_vx" joint="robot_x" kv="100" gear="1"
              ctrlrange="{-vmax:.4g} {vmax:.4g}" forcerange="-200 200"/>
    <velocity name="robot_vy" joint="robot_y" kv="100" gear="1"
              ctrlrange="{-vmax:.4g} {vmax:.4g}" forcerange="-200 200"/>
  </actuator>
  <sensor>
    {_contact_sensors()}
  </sensor>
</mujoco>
""".strip()


def build_execution_xml(cfg: dict[str, Any], obstacles: list[BaseSDF]) -> str:
    """Planar object with frictionloss tabletop + velocity-actuated robot."""
    z = 0.05
    mass = float(cfg["object_mass"])
    mu = float(cfg["mu"])
    g = float(cfg["gravity"])
    c = float(cfg["limit_surface_c"])
    r_ls = float(cfg["limit_surface_r"])
    f_lin = mu * mass * g
    f_ang = c * r_ls * mu * mass * g
    r = float(cfg.get("robot_radius", 0.012))
    robot_mass = float(cfg.get("robot_mass", 1.0))
    mu_c = float(cfg["mu_c"])
    dt = float(cfg["dt"])
    vmax = float(cfg.get("robot_max_speed", 1.0))
    obs = _obstacle_geoms(obstacles, z)
    obj = _object_geoms(mass)
    return f"""
<mujoco model="admm_execution">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  {_option_block(cfg, dt)}
  <default>
    <geom contype="1" conaffinity="1" friction="{mu_c:.4g} 0.005 0.0001"
          solref="0.004 1" solimp="0.9 0.95 0.001"/>
    <joint damping="0.0" armature="0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 0" rgba="0.85 0.85 0.85 1"/>
    <body name="object" pos="0 0 {z}">
      <joint name="object_x" type="slide" axis="1 0 0" frictionloss="{f_lin:.6g}"/>
      <joint name="object_y" type="slide" axis="0 1 0" frictionloss="{f_lin:.6g}"/>
      <joint name="object_yaw" type="hinge" axis="0 0 1" frictionloss="{f_ang:.6g}"/>
      {obj}
      <site name="object_com" pos="0 0 0" size="0.008"/>
    </body>
    <body name="robot" pos="0 0 {z}">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="0.05"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="0.05"/>
      <geom name="robot_geom" type="sphere" size="{r:.6g}" mass="{robot_mass:.6g}"
            rgba="0.9 0.25 0.2 1"/>
    </body>
    {obs}
  </worldbody>
  <actuator>
    <velocity name="robot_vx" joint="robot_x" kv="100" gear="1"
              ctrlrange="{-vmax:.4g} {vmax:.4g}" forcerange="-200 200"/>
    <velocity name="robot_vy" joint="robot_y" kv="100" gear="1"
              ctrlrange="{-vmax:.4g} {vmax:.4g}" forcerange="-200 200"/>
  </actuator>
  <sensor>
    {_contact_sensors()}
  </sensor>
</mujoco>
""".strip()


def build_coupled_planning_xml(cfg: dict[str, Any], obstacles: list[BaseSDF]) -> str:
    """Coupled planning: same plant as execution (dynamic planar object, no mocap weld)."""
    return build_execution_xml(cfg, obstacles).replace(
        'model="admm_execution"', 'model="admm_coupled_planning"'
    )


def assert_t_shape_matches_analytical() -> None:
    """Sanity: convex boxes cover analytical outline AABB."""
    verts = t_shape_vertices()
    assert float(verts[:, 0].min()) >= -0.090 - 1e-9
    assert float(verts[:, 0].max()) <= 0.090 + 1e-9
