# Robot–Object ADMM-MPC

Modular 2D decentralized ADMM-MPC for contact pushing. Consensus is on **world-frame CoM wrenches** \(w = [f_x, f_y, \tau]^\top\). Geometry, dynamics, contact, MPPI, and ADMM are isolated so the stack can later grow to 3D simulators.

## Layout

```text
config/base_config.yaml   # Hyperparameters (rho, MPPI, friction, costs)
src/
  geometry/               # Analytical 2D SDFs (circle, box, polygon, capsule)
  dynamics/               # QuasiStaticObject2D, KinematicRobot2D
  contact/                # Jc^T CoM wrench map + SimulateContactStep
  mppi/                   # Shared MPPIOptimizer (sample→rollout→weight→aggregate)
  admm/                   # WrenchConsensus + object/robot adapters + ADMMSolver
  utils/                  # Config, math helpers, matplotlib viz
tests/
main_mpc.py               # Receding-horizon entry point
results/                  # Plots / GIF outputs
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
PYTHONPATH=src python main_mpc.py                 # default: clutter
PYTHONPATH=src python main_mpc.py --env corridor
PYTHONPATH=src python main_mpc.py --env gate
PYTHONPATH=src python main_mpc.py --list-envs
PYTHONPATH=src python main_mpc.py --env clutter --max-steps 100
```

| Env | Description |
|-----|-------------|
| `clutter` | Separated circle / box / triangle; goal clear of clutter |
| `corridor` | Narrow horizontal channel; goal past the exit (no blocking post) |
| `gate` | Wide vertical gate slot; lightly rotated goal beyond the gate |

Outputs under `results/` are tagged by env, e.g. `trajectory_overview_corridor.png`.
## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

## Consensus contract

All ADMM wrenches are expressed in the **world frame about the object CoM**. Object MPPI samples contact parameters \((p_c, f_n, f_t)\) (friction cone) and maps \(w^o = J_c^\top f_c\). Robot MPPI samples velocities; realized contact wrenches enter the same consensus space. Duals use box anti-windup \(\pm\gamma_{\max}\) and shift with \(\gamma_{T-1} \leftarrow 0\) on the MPC horizon advance.
