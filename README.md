# Robot Manipulation and Control

This repository implements and compares several contact-implicit manipulation control methods, from a classical backstepping baseline to optimization-based contact dynamics and ADMM-MPC planners. All simulations are in 2D and written in pure Python.

---

## Repository Structure

```
Robot_Manipulation_Control/
│
├── ADMM/
│   ├── main.py
│   ├── main_interaction_concensus.py
│   └── results/
│
├── Contact_Dynamics/
│   ├── admm_qp_planar.py
│   ├── admm_sdf_planar.py
│   ├── contact_qp_planar.py
│   ├── contact_qp_vertical.py
│   ├── contact_quasistatic_planar.py
│   ├── contact_sdf_planar.py
│   └── results/
│
├── Legacy/
│   └── Backstepping/
│       ├── baskstepping.py
│       └── two_point_backstepping.gif
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Files

### `ADMM/`

**`main.py`** — ADMM-MPC simulation where an actuated particle `x` steers a passive particle `y` to a goal via a compact-support repulsive force; `x`-trajectory solved as a banded QP with a behind-y geometric projection. Exports a GIF to `results/`.

**`main_interaction_concensus.py`** — ADMM consensus variant where the `z`-update uses adjoint gradient descent through `y`'s dynamics; adds a smooth planning tail outside the hard force cutoff radius to prevent gradient vanishing and planner deadlock.

---

### `Contact_Dynamics/`

**`contact_qp_planar.py`** — Interactive top-down rigid-body sim (Anitescu 2006) of a box and circle driven by a WASD puck; 5-DOF QP resolves contact impulses with friction at walls, puck-body, and box-circle interfaces each timestep.

**`contact_qp_vertical.py`** — Side-view sim (Anitescu 2006) of a single box under gravity; ground and stick contacts resolved each step by a 3-DOF QP with a CLARABEL/SCS solver chain and free-flight fallback.

**`contact_quasistatic_planar.py`** — Inertia-free top-down sim (Pang et al., ICRA 2021) of a box pushed by a WASD puck; QP minimizes a regularization energy with puck velocity entering the constraint RHS directly, so the box stops instantly when the puck stops.

**`contact_sdf_planar.py`** — Top-down sim (Yang & Jin, RA-L 2024) of a circle pushed by a WASD puck using ContactSDF; contact resolved via log-sum-exp smoothing of the D-SDF halfspace set with a `Q⁻¹/²` projection — no QP solver needed.

**`admm_qp_planar.py`** — ADMM-MPC planner built on top of `contact_qp_planar.py`; plans puck inputs over a 24-step horizon using a 10-DOF Anitescu dynamics model, with support for targeting either the box or circle to a user-specified pose.

**`admm_sdf_planar.py`** — ADMM-MPC planner using ContactSDF as the planning model; finite-difference trajectory cost gradients with a speed-ball ADMM projection, running through three phases: ADMM planning → APF return → done.

---

### `Legacy/Backstepping/`

**`baskstepping.py`** — Two-point backstepping controller for `ṗ₁ = u`, `ṗ₂ = p₁ − p₂`; Lyapunov-based control law drives `p₂` to a goal via `p₁`, integrated with RK4 and exported as a GIF.

**`two_point_backstepping.gif`** — Animation output from `baskstepping.py`.

---

## Dependencies

```
numpy  matplotlib  pillow
```

`Contact_Dynamics/` scripts additionally require `pygame` and `cvxpy` (with CLARABEL).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pygame cvxpy     # for Contact_Dynamics scripts
```
