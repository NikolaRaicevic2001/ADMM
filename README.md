# Robot–Object ADMM-MPC

Modular **2D decentralized ADMM-MPC** for contact-rich planar pushing of a rigid T-shaped object by a point robot. The two agents (object planner and robot planner) do **not** share a joint trajectory; they negotiate agreement on a common **world-frame CoM wrench** sequence

$$
w = \begin{bmatrix} f_x & f_y & \tau \end{bmatrix}^\top \in \mathbb{R}^3
$$

at the object center of mass. Geometry, dynamics, contact, MPPI, and ADMM are split into separate packages so the stack can later grow toward 3D simulators without rewriting the coordination layer.

**What runs at each MPC step**

1. Shift warm-started horizons (controls, consensus, duals).
2. Estimate a contact point on the object perimeter.
3. Optionally seed robot velocities toward that contact if the robot is far away.
4. Run up to `n_admm` ADMM iterations: **object MPPI → robot MPPI → average wrenches → dual update**.
5. Execute only the robot’s first velocity command through a coupled contact simulator (object + robot both move).

---

## Layout

```text
ADMM/
├── config/
│   └── base_config.yaml      # All hyperparameters (rho, MPPI, friction, costs, viz)
├── main_mpc.py               # CLI entry: build env → ADMMSolver.run → plots/GIF
├── requirements.txt
├── results/                  # Generated PNGs / GIFs (unique suffixes if files exist)
├── tests/                    # Unit + smoke tests (SDF, wrench, consensus, envs, MPPI)
└── src/
    ├── geometry/             # Analytical 2D signed-distance fields (SDFs)
    ├── dynamics/             # Quasi-static object + PhysicsEngine2D (analytical / MJX)
    ├── contact/              # Jcᵀ wrench map + analytical contact (legacy backend)
    ├── mppi/                 # Shared sample→rollout→cost→softmax→aggregate core
    ├── admm/                 # Wrench consensus + object/robot subproblems + solver
    └── utils/                # Config, math helpers, named environments, visualization
```

### Package responsibilities

| Path | Role |
|------|------|
| `main_mpc.py` | Loads YAML config, builds a named scenario (`clutter` / `corridor` / `gate`), runs `ADMMSolver`, writes overview / plan comparison / residual plots and an optional diagnostic GIF under `results/`. |
| `config/base_config.yaml` | Hyperparameters including `physics_backend: analytical \| mjx` (default analytical). |
| `src/geometry/` | Analytical 2D SDFs; T-shape from `t_shape_vertices()`. |
| `src/dynamics/object_2d.py` | `QuasiStaticObject2D` limit-surface model used by **object MPPI**. |
| `src/dynamics/physics_engine.py` | `PhysicsEngine2D` ABC + `EnginePair` factory: NumPy-only `seed` / `step_execution` / `rollout_batch`. |
| `src/dynamics/analytical_engine.py` | SDF contact adapter wrapping `simulate_contact_step`. |
| `src/dynamics/mjx_scene.py` | MJCF builders for planning (mocap weld) and execution (`frictionloss`) worlds. |
| `src/dynamics/mjx_engine.py` | MJX backend: JIT `vmap`×`scan` planning rollouts; NumPy boundary for ADMM. |
| `src/dynamics/robot_2d.py` | Reference kinematic robot model (ADMM path uses the physics engine). |
| `src/dynamics/obstacles.py` | Hinge obstacle costs and SDF push-out helpers. |
| `src/contact/wrench_map.py` | $w = J_c^\top f$ world CoM wrench map. |
| `src/contact/resolution.py` | Analytical contact used by the analytical physics backend. |
| `src/mppi/` | Shared MPPI core + Gaussian sampler. |
| `src/admm/` | Wrench consensus, object/robot subproblems, `ADMMSolver`. |
| `src/utils/` | Config, environments, visualization, math helpers. |
| `tests/` | SDF, wrench, consensus, envs, MPPI, MJX verification matrix. |

### Physics backends

- **Default:** `physics_backend: analytical` (fast SDF contact via `simulate_contact_step`).
- Set `physics_backend: mjx` for MuJoCo MJX batched robot contact + execution (GPU preferred; CPU works).
- **Object MPPI** always stays analytical ($x^+=x+\Delta t\,(D\odot w)$).
- **Algorithm ↔ simulator contract** (NumPy only; no JAX types leak upward):
  - `rollout_batch(u_seq, ref_poses, robot_pos0, dt) → wrenches, paths` for robot MPPI
  - `seed` + `step_execution(u, dt) → pose, robot` for the MPC plant
- **Dual engines (mjx):** Planning = dynamic object **welded to mocap** + velocity-actuated planar finger (JIT `vmap`/`scan`). Execution = planar object with joint `frictionloss` (tabletop) + velocity-actuated finger. Soft contacts via `solref`/`solimp` (no CPU SDF inside the XLA kernel).
- CoM wrench from fixed-size contact sensors (`reduce=maxforce`); force on object = −force on robot.
- Roadmap: MJX → NVIDIA Warp implementing the same `PhysicsEngine2D` ABC.

### Control-flow sketch

```text
main_mpc.py
  └─ ADMMSolver.run (receding horizon)
       └─ control_step
            ├─ shift warm starts (u, p_c, f_n, f_t, z, γ)
            ├─ ContactPointEstimator → p_c mean
            ├─ optional seek-to-contact velocity seed
            └─ for k = 1..n_admm:
                 ├─ ObjectSubproblem.solve  → w^o, x^{o,ref}   (analytical)
                 ├─ RobotSubproblem.solve   → w^r, u^r         (planning engine)
                 ├─ z ← ½(w^o + w^r)
                 ├─ γ^o, γ^r ← clip(γ + (w − z))
                 └─ early stop if ‖r‖ ≤ ε_r and ‖s‖ ≤ ε_s
            └─ execute u_0^r via execution engine (coupled)
```

### Named environments

| Env | Idea |
|-----|------|
| `clutter` (default) | Open tabletop; separated circle / box / triangle; diagonal goal. |
| `corridor` | Horizontal channel between two walls; goal past the exit; a side post sits north of the exit (not on the goal). |
| `gate` | Vertical gate with a slot wide enough for the upright T; lightly rotated goal beyond the gate. |

Scene axis limits for plots/GIFs are **static per environment** (AABB of start, goal, robot start, obstacles, expanded by `view_pad_frac`), so every run of the same env shares the same zoom.

### Mathematical Formulation

This subsection matches the **implemented** equations (not a generic textbook ADMM). Symbols follow the code in `src/admm/`, `src/contact/`, and `src/dynamics/`. Display math uses GitHub-flavored `$$ ... $$` blocks so formulas render cleanly on GitHub and in most Markdown viewers.

#### 1. Problem setting

We push a rigid planar object with a point robot among static obstacles toward a goal pose.

| Symbol | Meaning | Code |
|--------|---------|------|
| $x^o_t = [p_x,\,p_y,\,\theta]^\top$ | Object pose in $\mathrm{SE}(2)$ | `object_.pose` |
| $p^r_t\in\mathbb{R}^2$ | Robot position (point) | `robot_pos` |
| $w_t\in\mathbb{R}^3$ | World-frame wrench about object CoM | consensus variable |
| $u^r_t\in\mathbb{R}^2$ | Robot velocity command | `u_nom` |
| $H$ | Prediction horizon | `horizon` (default 15) |
| $\Delta t$ | Timestep | `dt` (default 0.05 s) |
| $x^\star$ | Goal pose | `goal` |

**Object MPPI action** (body frame), not the consensus variable:

$$
a_t = \begin{bmatrix} p_c^{\mathrm{body}} & f_n & f_t \end{bmatrix}^\top \in \mathbb{R}^4
$$

where $p_c^{\mathrm{body}}$ lies on the object boundary, $f_n\ge 0$ is the inward normal force, and $f_t$ is the tangential force.

**Robot MPPI action:** velocity sequence $U = (u^r_0,\ldots,u^r_{H-1})$.

#### 2. Quasi-static object dynamics (limit surface)

Support friction $\mu$, mass $m$, gravity $g$, and shape factors $(c,r)$ define a diagonal compliance

$$
D = \mathrm{diag}\!\left(
  \frac{1}{\mu m g},\;
  \frac{1}{\mu m g},\;
  \frac{1}{c\,r\,\mu m g}
\right).
$$

One Euler step (then optional obstacle push-out):

$$
x^+ = x + \Delta t\,(D \odot w),\qquad \theta \leftarrow \mathrm{wrap}_{(-\pi,\pi]}(\theta).
$$

Interpretation: under a quasi-static elliptic limit surface, twist is proportional to applied wrench. Support friction $\mu$ appears **only** here; robot–object Coulomb friction uses a separate coefficient $\mu_c$.

#### 3. Contact geometry and CoM wrench map

Let $d(q)$ be the object body-frame SDF at body point $q$, with $\nabla d$ pointing outward. The **inward** contact normal / tangent are

$$
n_{\mathrm{body}} = -\nabla d,\qquad
t_{\mathrm{body}} = \begin{bmatrix}-n_y\\ n_x\end{bmatrix},
$$

rotated into the world frame by $\theta$. Contact force in the world frame:

$$
f = f_n\,n_{\mathrm{world}} + f_t\,t_{\mathrm{world}}.
$$

With moment arm $r = p_c - p_{\mathrm{CoM}}$ (world), the CoM wrench is

$$
w = J_c^\top f =
\begin{bmatrix}
f_x \\
f_y \\
r_x f_y - r_y f_x
\end{bmatrix}.
$$

**Friction cone (object planning):** hard projection

$$
0 \le f_n \le f_{\max},\qquad |f_t| \le \mu_c f_n.
$$

**Friction (contact resolution / execution):** if the robot point penetrates the object SDF ($d\le 0$),

$$
f_n = \mathrm{clip}\!\left(\frac{\mathrm{penetration}}{\Delta t\, D_{xx}},\, 0,\, f_{\max}\right),\qquad
f_t =
\begin{cases}
-\mu_c f_n\,\mathrm{sign}(v\cdot t) & |v\cdot t| > 10^{-4}, \\
0 & \text{otherwise}.
\end{cases}
$$

#### 4. Decentralized ADMM on wrench sequences

Over the horizon, both agents propose wrench trajectories $W^o, W^r \in \mathbb{R}^{H\times 3}$. They agree on a consensus trajectory $Z$ with scaled duals $\Gamma^o,\Gamma^r$ (code stores $\gamma \approx \lambda/\rho$):

$$
\begin{aligned}
z &\leftarrow \tfrac12\bigl(w^o + w^r\bigr), \\
\gamma^+ &\leftarrow \mathrm{clip}\bigl(\gamma + (w - z),\; [-\gamma_{\max},\,\gamma_{\max}]\bigr).
\end{aligned}
$$

The ADMM **augmented penalty** each subproblem sees is

$$
\frac{\rho}{2}\,\|W - Z + \Gamma\|_F^2
= \frac{\rho}{2}\sum_{t=0}^{H-1}\|w_t - z_t + \gamma_t\|_2^2.
$$

**Residuals** used for early stopping (concatenated over the horizon):

$$
r = \begin{bmatrix} W^o - Z \\ W^r - Z \end{bmatrix},\qquad
s = \rho\,(Z^+ - Z),\qquad
\text{stop if }\|r\|_2\le\varepsilon_r\ \text{and}\ \|s\|_2\le\varepsilon_s.
$$

Defaults: $\rho=1$, $\gamma_{\max}=\texttt{max\_dual}=4$, $\varepsilon_r=\varepsilon_s=1$, at most `n_admm=6` iterations per control step.

**Horizon warm-start (MPC advance):** every sequence (controls, $Z$, duals) is shifted forward and the new terminal slot is zeroed:

$$
\mathrm{seq}_t \leftarrow \mathrm{seq}_{t+1},\qquad \mathrm{seq}_{H-1} \leftarrow 0.
$$

#### 5. Object subproblem (MPPI)

**Sampling.** $K_o$ action trajectories. Contact points are rejection-sampled on the perimeter near the current mean; forces are Gaussian about the nominal $(f_n,f_t)$ then projected into the friction cone / $[0,f_{\max}]$. Wrenches are **never** free variables — always $w^o_t = J_c^\top f_t$.

**Rollout.** For each sample, map $a_t\to w_t$ and integrate the quasi-static object dynamics for $H$ steps.

**Cost** (vectorized over samples; $Q$ / $Q_f$ are the running / terminal goal weights):

$$
\begin{aligned}
J^o
&= \sum_{t=0}^{H-2}\ell_{\mathrm{goal}}(x_t;\, q_{\mathrm{pos}}, q_\theta)
+ \ell_{\mathrm{goal}}(x_{H-1};\, q^f_{\mathrm{pos}}, q^f_\theta) \\
&\quad + \sum_{t=0}^{H-2}\ell_{\mathrm{obs}}(x_t)
+ r_o\|W^o\|_F^2
+ \frac{\rho}{2}\|W^o - Z + \Gamma^o\|_F^2,
\end{aligned}
$$

with

$$
\ell_{\mathrm{goal}}(x) = q_{\mathrm{pos}}\|p-p^\star\|_2^2 + q_\theta\bigl(\theta-\theta^\star\bigr)^2
$$

and $\ell_{\mathrm{obs}}$ a squared hinge on SDF clearance vs obstacles. MPPI softmin weights use temperature $\nu_o$:

$$
\omega_k \propto \exp\!\bigl(-(J_k - \min_j J_j)/\nu_o\bigr),\qquad
a\leftarrow \sum_k \omega_k\, a^{(k)}.
$$

The resulting nominal actions are projected back onto the boundary / friction cone, then re-rolled to produce $W^o$ and reference poses $x^{o,\mathrm{ref}}_{0:H-1}$.

**Contact-point estimator** (once per control step, before ADMM): CEM-like search over perimeter points scored by a short open-loop push with $f_t=0$, then the best point is tiled across the horizon as $p_{\mathrm{mean}}$.

#### 6. Robot subproblem (MPPI)

**Sampling.** $K_r$ velocity sequences with additive Gaussian noise of std $\Sigma^r=\texttt{sigma\_robot}$ (annealed by distance-to-goal).

**Rollout.** Against the **frozen** object reference poses from the object subproblem, each sample is integrated with `simulate_contact_step(..., freeze_object=True)`. The realized wrench at each step is the mean over `n_contact_substeps` contact substeps. There is **no** SE(2) goal cost on the robot — matching $Z$ through contact is the task.

**Cost:**

$$
J^r
= r_r\sum_{t=0}^{H-1}\|u^r_t\|_2^2
+ \sum_{t=0}^{H-1}\ell_{\mathrm{obs}}^{\mathrm{robot}}(p^r_t)
+ \frac{\rho}{2}\|W^r - Z + \Gamma^r\|_F^2.
$$

#### 7. Execution and seek-to-contact

After ADMM, only $u^r_0$ is applied with `freeze_object=False`: robot moves, contact wrench is resolved, and the object integrates under that wrench.

If the gap from the robot to the estimated world contact point exceeds `contact_step_margin`, the solver first seeds $U$ with a clipped seek speed toward that point (so MPPI starts from an approach, not from rest far away).

**Sigma annealing.** Exploration scale shrinks as the object nears the goal:

$$
\sigma_{\mathrm{scale}}
= \mathrm{clip}\!\left(
  \frac{\max\!\bigl(\|p-p^\star\|/\varepsilon_p,\; |\theta-\theta^\star|/\varepsilon_\theta\bigr)}
       {\texttt{sigma\_anneal\_band}},
  \; \texttt{min\_sigma\_scale},\; 1
\right).
$$

#### 8. Why wrench consensus (design intent)

- **Object** is free to invent any friction-feasible contact plan that moves $x^o$ toward $x^\star$, expressed as desired wrenches $W^o$.
- **Robot** must *physically realize* wrenches $W^r$ by moving a point finger under SDF contact; many $W^o$ are unreachable.
- Averaging to $Z$ and accumulating duals $\Gamma$ pushes both agents toward a **mutually feasible** wrench schedule without a centralized joint optimizer.
- Failure modes (local contact face, short robot horizon, heavy $r_r$, dual saturation) are diagnosed via telemetry: contact sample cloud, target $p_c$, $w^o$ vs $w^r$ arrows, robot rollout fan, and cost / residual HUD on the GIF.

#### 9. Key defaults (from `config/base_config.yaml`)

| Group | Defaults |
|-------|----------|
| Timing | $\Delta t=0.05$, $H=15$, `n_admm=6`, up to 500 control steps |
| Samples | $K_o=K_r=64$, $\nu_o=\nu_r=1$ |
| Physics | $\mu=0.4$, $\mu_c=0.5$, $m=2$, $g=9.81$, $c=1$, $r=0.06$, $f_{\max}=4$ |
| ADMM | $\rho=1$, $\gamma_{\max}=4$, $\varepsilon_r=\varepsilon_s=1$ |
| Costs | $q_{\mathrm{pos}}=40$, $q_\theta=10$, $q^f_{\mathrm{pos}}=150$, $q^f_\theta=45$, $r_o=0.01$, $r_r=0.05$, $w_{\mathrm{obs}}=6\cdot 10^4$ |
| Goal | position tol $0.06\,\mathrm{m}$, angle tol $0.08\,\mathrm{rad}$ |

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
# Optional GPU: install a CUDA-enabled jaxlib matching your driver
#   (see https://jax.readthedocs.io/en/latest/installation.html)
```

## Run

```bash
PYTHONPATH=src python main_mpc.py                 # default: clutter (analytical backend)
PYTHONPATH=src python main_mpc.py --env corridor
PYTHONPATH=src python main_mpc.py --env gate
PYTHONPATH=src python main_mpc.py --list-envs
PYTHONPATH=src python main_mpc.py --env clutter --max-steps 100
PYTHONPATH=src python main_mpc.py --config /path/to/custom.yaml
```

For MJX, set `physics_backend: mjx` in `config/base_config.yaml` (or a custom config). First call JIT-compiles the batched rollout (slow once); subsequent MPC steps reuse the compiled kernel.

Outputs land in `results/` (overview PNG, plan-comparison PNG, residual PNG, optional GIF). Filenames are tagged by environment and get a numeric suffix if a file already exists.

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

## Visualization and telemetry

Each control step appends a `telemetry` dict to the run log (`log["telemetry"]`), including:

- ADMM iteration count, first-step and full-horizon primal disagreement $\|w^o-w^r\|$
- Dual norms $\|\Gamma^o\|$, $\|\Gamma^r\|$ and a saturation flag vs $\gamma_{\max}$
- Object task / ADMM penalty and robot effort / ADMM penalty
- Contact sample cloud $p_c$, selected target $p_c$, first-step wrenches, subsampled robot rollout fan

The GIF overlays those spatial quantities plus a monospace HUD. Plan colors: **cyan** = object MPPI object path, **orange** = object path under robot wrenches, **magenta** = robot plan path, **red** = executed robot.
