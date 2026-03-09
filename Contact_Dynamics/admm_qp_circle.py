import sys
import pathlib
import pygame
import numpy as np
import cvxpy as cp 
import cv2

# ══════════════════════════════════════════════════════════════════════════════
#  Load physics backend — run setup section only (stop before while-loop)
# ══════════════════════════════════════════════════════════════════════════════
_src = pathlib.Path(__file__).with_name("contact_qp_xy2.py").read_text()
_ns  = {"pygame": pygame, "np": np, "cp": cp, "sys": sys}
exec(_src[: _src.index("\nwhile True:")], _ns)

detect_contacts  = _ns["detect_contacts"]
physics_step     = _ns["step"]
get_box_vertices = _ns["get_box_vertices"]
rotation_matrix  = _ns["rotation_matrix"]

WIDTH, HEIGHT = _ns["WIDTH"],  _ns["HEIGHT"]
MARGIN        = _ns["MARGIN"]
STICK_R       = _ns["STICK_R"]; STICK_SPEED = _ns["STICK_SPEED"]
CIRC_R        = _ns["CIRC_R"];  size        = _ns["size"]
h             = _ns["h"]
I_box         = _ns["I_box"];   mass_box    = _ns["mass_box"]

# ══════════════════════════════════════════════════════════════════════════════
#  ★  USER CONFIG  ★
# ══════════════════════════════════════════════════════════════════════════════
MANIPULATED_OBJECT = "circle"                          # "circle"  or  "box"
TARGET_CIRCLE      = np.array([660.0, 130.0])          # (x, y)
TARGET_BOX         = np.array([630.0, 430.0, np.pi/3]) # (x, y, theta_rad)

# ══════════════════════════════════════════════════════════════════════════════
#  Simulation initial state
# ══════════════════════════════════════════════════════════════════════════════
q_box       = np.array([WIDTH/2 - 120.0, HEIGHT/2, 0.0])
v_box       = np.zeros(3)
q_circ      = np.array([WIDTH/2 + 130.0, HEIGHT/2])
v_circ      = np.zeros(2)
STICK_START = np.array([WIDTH/2 - 300.0, HEIGHT/2])
stick_pos   = STICK_START.copy()

# ══════════════════════════════════════════════════════════════════════════════
#  ADMM / MPC hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════
N_PLAN     = 24       # planning horizon (timesteps)
RHO        = 8.0      # ADMM penalty  ρ
N_ADMM     = 25       # ADMM inner iterations per control step

EPS_FD_XY  = 0.8     # FD perturbation: position states [px]
EPS_FD_TH  = 0.018   # FD perturbation: theta state     [rad]
EPS_FD_P   = 0.8     # FD perturbation: puck control    [px]
EPS_FD_V   = 0.3     # FD perturbation: velocity states [px/step]

Q_TERM     = 700.0    # terminal tracking weight  Q_f
Q_RUN      = 10.0     # running tracking weight   Q
Q_VEL_TERM = 80.0     # terminal velocity cost  (object at rest at goal)
Q_VEL_RUN  = 0.0      # running velocity cost = 0: planner pushes freely,
                       # only the terminal step enforces zero velocity
R_RATE     = 0.5      # puck smoothness weight    R_r
R_ABS      = 0.04     # Tikhonov regularisation on δu
R_GUIDE    = 20.0     # guidance: puck → push position

INFLUENCE_R = 52.0    # soft-contact influence zone beyond hard contact [px]
PUSH_ALPHA  = 0.55    # velocity impulse per step at full contact [px/step]
                       # (old value 2.3 was a position delta; now it's a velocity
                       #  impulse — steady-state v = alpha/(1-DAMP) ≈ 32 px/step)
TORQUE_GAIN = 0.038   # rotational response scale in planning model
PUSH_MARGIN = 8.0     # extra offset behind object for push position [px]
ROT_LAT_K   = 0.18    # lateral-offset gain for rotation induction

PHYS_STEPS  = 3       # physics substeps per ADMM control step

# Planning-model timestep and velocity damping (mirrors physics backend)
h_plan    = h * PHYS_STEPS
# damping ratio = B_LIN / mass = 2.0 for all bodies (from contact_qp_xy2.py)
DAMP_LIN  = max(0.0, 1.0 - 10.0 * h_plan)   # linear velocity decay per plan step
DAMP_ROT  = max(0.0, 1.0 - 15.0 * h_plan)   # angular velocity decay per plan step

# Max puck move per CONTROL step (generous – speed projection enforces it)
MAX_STEP = STICK_SPEED * h * PHYS_STEPS * 1.9

R_BOX_PLAN  = size * 0.5
SAFE_R_BOX  = STICK_R + R_BOX_PLAN + 28.0   # enlarged: puck must not touch non-target
SAFE_R_CIRC = STICK_R + CIRC_R     + 28.0

# Obstacle-avoidance penalty inside the QP (issue 1 formulation change)
W_OBS_PEN    = 6000.0   # weight on puck-obstacle repulsion term in QP cost
R_OBS_EXTRA  = 35.0     # extra radius beyond SAFE_R at which penalty activates

# Artificial Potential Field — return phase
APF_REP_GAIN  = 18000.0  # repulsive gain  (pushes away from each object)
APF_REP_RMAX  = 160.0    # repulsion fades beyond this distance [px]
APF_ATT_GAIN  = 0.35     # attractive gain toward STICK_START
APF_STEP      = 14.0     # max puck displacement per frame during return [px]
APF_SAFE_DIST = 130.0    # declare "done" when clear of BOTH objects by this much

REACH_XY    = 10.0    # [px]  goal reached – translation
REACH_THETA = 0.15    # [rad] goal reached – rotation (box only)
RETURN_DIST = 8.0     # [px]  puck home (kept for compatibility)

# ── Derived planning-space layout ─────────────────────────────────────────────
#   circle mode  NS=6 : [bx, by, cx, cy, vcx, vcy]
#   box    mode  NS=8 : [bx, by, btheta, cx, cy, vbx, vby, vb_omega]
#
# Velocity states are appended at the end so all position indices are unchanged.
if MANIPULATED_OBJECT == "circle":
    NS         = 6
    TARGET_DIM = 2
    TARGET_POS = TARGET_CIRCLE.copy()
    OBS_SAFE_R = SAFE_R_BOX
    # C_sel: selects circle position (indices 2,3)
    C_sel      = np.zeros((2, 6)); C_sel[0, 2] = 1.0; C_sel[1, 3] = 1.0
    # C_vel: selects circle velocity (indices 4,5)
    C_vel      = np.zeros((2, 6)); C_vel[0, 4] = 1.0; C_vel[1, 5] = 1.0
    W_cost     = np.eye(2)
    W_vel      = np.eye(2)
    THETA_WEIGHT = 0.0
else:
    NS         = 8
    TARGET_DIM = 3
    TARGET_POS = TARGET_BOX.copy()
    OBS_SAFE_R = SAFE_R_CIRC
    # C_sel: selects box pose (indices 0,1,2)
    C_sel      = np.zeros((3, 8))
    C_sel[0, 0] = 1.0; C_sel[1, 1] = 1.0; C_sel[2, 2] = 1.0
    # C_vel: selects box velocity (indices 5,6,7)
    C_vel      = np.zeros((3, 8))
    C_vel[0, 5] = 1.0; C_vel[1, 6] = 1.0; C_vel[2, 7] = 1.0
    THETA_WEIGHT = (size / 2.0) ** 2    # 1 rad ~ (size/2) px in cost
    W_cost       = np.diag([1.0, 1.0, THETA_WEIGHT])
    W_vel        = np.diag([1.0, 1.0, THETA_WEIGHT])

NU = 2   # control dim: puck (x, y)

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def angle_wrap(a):
    return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)


def get_planning_state():
    """
    NS=6 circle: [bx, by, cx, cy, vcx, vcy]
    NS=8 box:    [bx, by, btheta, cx, cy, vbx, vby, vb_omega]
    Velocities from the physics simulation seed the planner so it knows the
    object is moving and can plan to decelerate it.
    """
    if MANIPULATED_OBJECT == "circle":
        return np.array([q_box[0], q_box[1],
                         q_circ[0], q_circ[1],
                         v_circ[0], v_circ[1]])
    return np.array([q_box[0], q_box[1], q_box[2],
                     q_circ[0], q_circ[1],
                     v_box[0],  v_box[1],  v_box[2]])


def obs_center_from_s(s):
    """Centre of the NON-target obstacle from planning state s."""
    if MANIPULATED_OBJECT == "circle":
        return s[0:2].copy()   # box xy
    return s[3:5].copy()       # circle xy


def get_obs_from_state(s):
    """Same as obs_center_from_s — alias used inside build_qp/project."""
    return obs_center_from_s(s)


def compute_push_pos(s_k):
    """
    Ideal puck push position for planning state s_k.

    Placed behind the manipulated object along obj→target.
    For box mode, a lateral offset induces the needed rotation.
    """
    if MANIPULATED_OBJECT == "circle":
        obj_xy    = s_k[2:4]
        obj_theta = None
    else:
        obj_xy    = s_k[0:2]
        obj_theta = float(s_k[2])

    diff     = TARGET_POS[:2] - obj_xy
    d        = np.linalg.norm(diff)
    push_dir = diff / (d + 1e-9)

    contact_r = STICK_R + (CIRC_R if MANIPULATED_OBJECT == "circle" else R_BOX_PLAN)
    pos = obj_xy - push_dir * (contact_r + PUSH_MARGIN)

    if obj_theta is not None:
        da      = angle_wrap(TARGET_POS[2] - obj_theta)
        lateral = np.array([-push_dir[1], push_dir[0]])
        offset  = np.clip(da * size * ROT_LAT_K, -size * 0.3, size * 0.3)
        pos     = pos + lateral * offset

    pos[0] = np.clip(pos[0], MARGIN+STICK_R+2, WIDTH -MARGIN-STICK_R-2)
    pos[1] = np.clip(pos[1], MARGIN+STICK_R+2, HEIGHT-MARGIN-STICK_R-2)
    return pos


def target_error(s):
    """Weighted error of manipulated object vs target (angle-wrapped)."""
    e = C_sel @ s - TARGET_POS
    if MANIPULATED_OBJECT == "box":
        e[2] = angle_wrap(e[2])
    return e


def is_goal_reached():
    if MANIPULATED_OBJECT == "circle":
        return np.linalg.norm(q_circ - TARGET_POS) < REACH_XY
    xy_err = np.linalg.norm(q_box[:2] - TARGET_BOX[:2])
    th_err = abs(angle_wrap(q_box[2] - TARGET_BOX[2]))
    return xy_err < REACH_XY and th_err < REACH_THETA

# ══════════════════════════════════════════════════════════════════════════════
#  Simplified planning dynamics  f(s, p)
# ══════════════════════════════════════════════════════════════════════════════

def _wall_clamp(s):
    """Clamp position states only; velocity states are left untouched."""
    s = s.copy()
    # Box position (indices 0,1) — always present
    s[0] = np.clip(s[0], MARGIN+R_BOX_PLAN, WIDTH -MARGIN-R_BOX_PLAN)
    s[1] = np.clip(s[1], MARGIN+R_BOX_PLAN, HEIGHT-MARGIN-R_BOX_PLAN)
    if MANIPULATED_OBJECT == "circle":
        # circle position at indices 2,3
        s[2] = np.clip(s[2], MARGIN+CIRC_R, WIDTH -MARGIN-CIRC_R)
        s[3] = np.clip(s[3], MARGIN+CIRC_R, HEIGHT-MARGIN-CIRC_R)
        # indices 4,5 = circle velocity — no clamp
    else:
        # s[2] = theta — no clamp
        # circle position at indices 3,4
        s[3] = np.clip(s[3], MARGIN+CIRC_R, WIDTH -MARGIN-CIRC_R)
        s[4] = np.clip(s[4], MARGIN+CIRC_R, HEIGHT-MARGIN-CIRC_R)
        # indices 5,6,7 = box velocity — no clamp
    return s


def plan_step(s, puck):
    """
    One-step simplified planning dynamics with velocity states.

    Push force (soft-contact ramp) now adds to the manipulated object's
    velocity rather than its position directly.  Velocity then decays with
    physics-consistent damping and integrates into position.  This lets the
    planner reason about momentum and brake the object by moving to its front.

    The non-manipulated (obstacle) object keeps the old positional impulse
    model — we only need accurate velocity dynamics for the controlled object.
    """
    s    = s.copy()
    puck = np.asarray(puck, dtype=float)

    if MANIPULATED_OBJECT == "circle":
        # ── State unpack ─────────────────────────────────────────────────────
        q_b = s[0:2].copy()   # box position   (obstacle, positional model)
        q_c = s[2:4].copy()   # circle position (target)
        v_c = s[4:6].copy()   # circle velocity (target)

        # ── Puck → Box (obstacle, direct positional impulse) ─────────────────
        d_b    = q_b - puck
        dist_b = np.linalg.norm(d_b) + 1e-9
        gap_b  = dist_b - (STICK_R + R_BOX_PLAN)
        if gap_b < INFLUENCE_R:
            alpha_b = np.clip(1.0 - gap_b / INFLUENCE_R, 0.0, 1.0) * PUSH_ALPHA
            q_b    += alpha_b * (d_b / dist_b)

        # ── Puck → Circle (target, via velocity) ─────────────────────────────
        d_c    = q_c - puck
        dist_c = np.linalg.norm(d_c) + 1e-9
        gap_c  = dist_c - (STICK_R + CIRC_R)
        if gap_c < INFLUENCE_R:
            alpha_c = np.clip(1.0 - gap_c / INFLUENCE_R, 0.0, 1.0) * PUSH_ALPHA
            v_c    += alpha_c * (d_c / dist_c)   # impulse → velocity

        # ── Damping + position integration ───────────────────────────────────
        v_c *= DAMP_LIN
        q_c += v_c

        # ── State pack ───────────────────────────────────────────────────────
        s[0:2] = q_b
        s[2:4] = q_c
        s[4:6] = v_c

    else:  # MANIPULATED_OBJECT == "box"
        # ── State unpack ─────────────────────────────────────────────────────
        q_b   = s[0:2].copy()   # box position  (target)
        theta = float(s[2])     # box angle     (target)
        q_c   = s[3:5].copy()   # circle position (obstacle, positional model)
        v_b   = s[5:7].copy()   # box linear velocity (target)
        om_b  = float(s[7])     # box angular velocity (target)

        # ── Puck → Box (target, via velocity) ────────────────────────────────
        d_b    = q_b - puck
        dist_b = np.linalg.norm(d_b) + 1e-9
        gap_b  = dist_b - (STICK_R + R_BOX_PLAN)
        if gap_b < INFLUENCE_R:
            alpha_b    = np.clip(1.0 - gap_b / INFLUENCE_R, 0.0, 1.0) * PUSH_ALPHA
            n_b        = d_b / dist_b
            push_b     = alpha_b * n_b
            v_b       += push_b                           # linear impulse
            contact_pt = puck + n_b * STICK_R
            r_arm      = contact_pt - q_b
            torque_z   = r_arm[0]*push_b[1] - r_arm[1]*push_b[0]
            om_b      += torque_z / (I_box / mass_box) * TORQUE_GAIN

        # ── Damping + pose integration ────────────────────────────────────────
        v_b  *= DAMP_LIN
        om_b *= DAMP_ROT
        q_b  += v_b
        theta += om_b

        # ── Puck → Circle (obstacle, direct positional impulse) ───────────────
        d_c    = q_c - puck
        dist_c = np.linalg.norm(d_c) + 1e-9
        gap_c  = dist_c - (STICK_R + CIRC_R)
        if gap_c < INFLUENCE_R:
            alpha_c = np.clip(1.0 - gap_c / INFLUENCE_R, 0.0, 1.0) * PUSH_ALPHA
            q_c    += alpha_c * (d_c / dist_c)

        # ── State pack ───────────────────────────────────────────────────────
        s[0:2] = q_b
        s[2]   = theta
        s[3:5] = q_c
        s[5:7] = v_b
        s[7]   = om_b

    return _wall_clamp(s)


def rollout(s0, P):
    X    = np.empty((N_PLAN + 1, NS))
    X[0] = s0
    for k in range(N_PLAN):
        X[k + 1] = plan_step(X[k], P[k])
    return X

# ══════════════════════════════════════════════════════════════════════════════
#  Finite-difference Jacobians
# ══════════════════════════════════════════════════════════════════════════════

def fd_jacobians(s_bar, p_bar):
    eps_s = np.full(NS, EPS_FD_XY)
    if MANIPULATED_OBJECT == "circle":
        # indices 4,5 = circle velocity
        eps_s[4:6] = EPS_FD_V
    else:
        # index 2 = theta; indices 5,6,7 = box velocity
        eps_s[2]   = EPS_FD_TH
        eps_s[5:8] = EPS_FD_V

    A = np.empty((NS, NS))
    for i in range(NS):
        sp = s_bar.copy(); sp[i] += eps_s[i]
        sm = s_bar.copy(); sm[i] -= eps_s[i]
        A[:, i] = (plan_step(sp, p_bar) - plan_step(sm, p_bar)) / (2.0*eps_s[i])

    B = np.empty((NS, NU))
    for i in range(NU):
        pp = p_bar.copy(); pp[i] += EPS_FD_P
        pm = p_bar.copy(); pm[i] -= EPS_FD_P
        B[:, i] = (plan_step(s_bar, pp) - plan_step(s_bar, pm)) / (2.0*EPS_FD_P)

    return A, B

# ══════════════════════════════════════════════════════════════════════════════
#  Build condensed QP matrices
# ══════════════════════════════════════════════════════════════════════════════

def build_qp(s0, P_bar):
    """
    Linearise J around (s0, P_bar) and return H, g for:
        J ≈ ½ δu^T H δu + g^T δu       (δu = u - P̄_flat)

    Includes: tracking cost, smoothness, Tikhonov, guidance cost.
    """
    n = 2 * N_PLAN

    X_bar = rollout(s0, P_bar)
    As = []; Bs = []
    for k in range(N_PLAN):
        A_k, B_k = fd_jacobians(X_bar[k], P_bar[k])
        As.append(A_k); Bs.append(B_k)

    H = np.zeros((n, n))
    g = np.zeros(n)
    G = np.zeros((NS, n))   # condensed sensitivity  δs_{k+1} = G_{k+1} δu

    for k in range(N_PLAN):
        G                  = As[k] @ G
        G[:, 2*k:2*k+2]   += Bs[k]

        w     = Q_TERM if k == N_PLAN - 1 else Q_RUN
        CG    = C_sel @ G                          # (TD, 2N)
        WCG   = W_cost @ CG                        # (TD, 2N)

        e_bar = C_sel @ X_bar[k+1] - TARGET_POS
        if MANIPULATED_OBJECT == "box":
            e_bar[2] = angle_wrap(e_bar[2])

        # Tracking:  w ||ē + CG δu||²_W
        H += 2.0 * w * CG.T @ WCG
        g += 2.0 * w * CG.T @ (W_cost @ e_bar)

    # Smoothness on puck velocity
    D = np.zeros((2*(N_PLAN-1), n))
    for k in range(N_PLAN-1):
        D[2*k:2*k+2, 2*k:2*k+2]   = -np.eye(2)
        D[2*k:2*k+2, 2*k+2:2*k+4] =  np.eye(2)
    d_nom = D @ P_bar.flatten()
    H += 2.0 * R_RATE * D.T @ D
    g += 2.0 * R_RATE * D.T @ d_nom

    # Tikhonov
    H += 2.0 * R_ABS * np.eye(n)

    # Guidance cost: R_guide * ||u_k - p_push_k||²
    # In δu coords: R_guide * ||δu_k + (p̄_k - p_push_k)||²
    for k in range(N_PLAN):
        push_p = compute_push_pos(X_bar[k])
        d_g    = P_bar[k] - push_p
        H[2*k:2*k+2, 2*k:2*k+2] += 2.0 * R_GUIDE * np.eye(2)
        g[2*k:2*k+2]             += 2.0 * R_GUIDE * d_g

    # ── Obstacle-avoidance penalty ────────────────────────────────────────────
    R_pen = OBS_SAFE_R + R_OBS_EXTRA
    for k in range(N_PLAN):
        obs_k  = get_obs_from_state(X_bar[k])
        d_obs  = P_bar[k] - obs_k
        dist_k = np.linalg.norm(d_obs) + 1e-9
        if dist_k < R_pen:
            n_k   = d_obs / dist_k
            delta = R_pen - dist_k
            H[2*k:2*k+2, 2*k:2*k+2] += 2.0 * W_OBS_PEN * np.outer(n_k, n_k)
            g[2*k:2*k+2]             -= 2.0 * W_OBS_PEN * delta * n_k

    # ── Velocity cost — encourages object to arrive at rest ──────────────────
    # Rebuild sensitivity G from scratch for the velocity loop
    # (G was already fully accumulated above; we need to re-accumulate here
    #  because the loop above consumed it)
    G_v = np.zeros((NS, 2 * N_PLAN))
    for k in range(N_PLAN):
        G_v                   = As[k] @ G_v
        G_v[:, 2*k:2*k+2]   += Bs[k]

        # Weight ramps linearly from Q_VEL_RUN (k=0) to Q_VEL_TERM (k=N_PLAN-1)
        w_v  = Q_VEL_RUN + (Q_VEL_TERM - Q_VEL_RUN) * k / max(N_PLAN - 1, 1)
        CvGv = C_vel @ G_v                       # (vel_dim, 2N)
        WCvGv = W_vel @ CvGv
        v_bar = C_vel @ X_bar[k + 1]             # planned velocity (target = 0)
        H += 2.0 * w_v * CvGv.T @ WCvGv
        g += 2.0 * w_v * CvGv.T @ (W_vel @ v_bar)

    return H, g, X_bar

# ══════════════════════════════════════════════════════════════════════════════
#  ADMM Controller
# ══════════════════════════════════════════════════════════════════════════════

class ADMMController:
    """
    Receding-horizon ADMM puck controller.

    Consensus split:
        min_{u, z}  J_quad(u)  +  I_Z(z)   s.t.  u = z

    Scaled dual  y = λ/ρ.  Iterations:
        u ← argmin_u  J_quad(u)  + ρ/2 ||u − z + y||²
              solved as  (H + ρI) δu = ρ(z − y − P̄) − g
        z ← Π_Z(u + y)      [analytical per-step projection]
        y ← y + u − z
    """

    def __init__(self):
        # Warm-start: straight line from current puck to initial push position
        s0     = get_planning_state()
        push_p = compute_push_pos(s0)
        P_init = np.linspace(stick_pos, push_p, N_PLAN)
        self.P_bar = P_init.copy()
        self.z     = P_init.flatten().copy()
        self.y     = np.zeros(2 * N_PLAN)

    def plan(self, s0, cur_puck, obs_ctr):
        P_bar      = self.P_bar.copy()
        P_bar_flat = P_bar.flatten()

        H, g, X_bar = build_qp(s0, P_bar)
        M_lhs       = H + RHO * np.eye(2 * N_PLAN)

        z = self.z.copy()
        y = self.y.copy()

        for _ in range(N_ADMM):
            # u-update: linear solve
            rhs     = RHO * (z - y - P_bar_flat) - g
            delta_u = np.linalg.solve(M_lhs, rhs)
            u_abs   = P_bar_flat + delta_u

            # z-update: projection (pass X_bar for per-step obstacle positions)
            z = self._project(u_abs + y, cur_puck, X_bar)

            # dual update
            y = y + u_abs - z

        self.z = z.copy()
        self.y = y.copy()

        z_plan = z.reshape(N_PLAN, 2)
        # Receding-horizon warm-start shift
        self.P_bar[:-1] = z_plan[1:]
        self.P_bar[-1]  = z_plan[-1]

        next_p = z_plan[0].copy()
        next_p[0] = np.clip(next_p[0], MARGIN+STICK_R+1, WIDTH -MARGIN-STICK_R-1)
        next_p[1] = np.clip(next_p[1], MARGIN+STICK_R+1, HEIGHT-MARGIN-STICK_R-1)
        return next_p, z_plan, X_bar

    def _project(self, v_flat, cur_puck, X_bar):
        """
        Π_Z = Π_{obs} ∘ Π_{bounds} ∘ Π_{speed}   (alternating projections)

        Z_obs    : ||p_k − obs_k|| ≥ PROJ_R  per step, using rolled-out obstacle
                   position obs_k = get_obs_from_state(X_bar[k]).
                   PROJ_R = OBS_SAFE_R + R_OBS_EXTRA ensures the hard boundary
                   is at least as tight as the QP penalty activation radius.
        Z_bounds : clip to workspace
        Z_speed  : ||p_k − p_{k-1}|| ≤ MAX_STEP (3 forward sweeps)
        """
        PROJ_R = OBS_SAFE_R + R_OBS_EXTRA   # hard projection radius
        z = v_flat.reshape(N_PLAN, 2).copy()

        # Per-step obstacle avoidance + workspace clamp
        for k in range(N_PLAN):
            obs_k = get_obs_from_state(X_bar[k])   # non-target at step k
            d     = z[k] - obs_k
            dist  = np.linalg.norm(d)
            if dist < PROJ_R:
                z[k] = obs_k + PROJ_R * d / (dist + 1e-9)
            z[k, 0] = np.clip(z[k, 0], MARGIN+STICK_R+1, WIDTH -MARGIN-STICK_R-1)
            z[k, 1] = np.clip(z[k, 1], MARGIN+STICK_R+1, HEIGHT-MARGIN-STICK_R-1)

        # Speed constraint  (3 forward sweeps)
        for _ in range(3):
            prev = cur_puck.copy()
            for k in range(N_PLAN):
                diff = z[k] - prev
                d    = np.linalg.norm(diff)
                if d > MAX_STEP:
                    z[k] = prev + diff / d * MAX_STEP
                prev = z[k]

        return z.flatten()

# ══════════════════════════════════════════════════════════════════════════════
#  Artificial Potential Field — return phase
# ══════════════════════════════════════════════════════════════════════════════

def apf_return_step(puck):
    """
    One step of APF gradient descent.

    Potential:
        U = Σ_obs  ½ k_rep (1/d - 1/d_max)²   if d < d_max   [repulsive]
            + ½ k_att ||puck - start||²                        [attractive]

    Gradient is negated (descend potential) then clamped to APF_STEP.
    Both box and circle are always treated as obstacles.
    """
    grad = np.zeros(2)

    # ── Repulsive terms ──────────────────────────────────────────────────────
    obstacles = [
        (q_box[:2],  STICK_R + R_BOX_PLAN + 4.0),
        (q_circ,     STICK_R + CIRC_R     + 4.0),
    ]
    for (centre, r_min) in obstacles:
        diff = puck - centre
        d    = np.linalg.norm(diff) + 1e-9
        d_eff = max(d - r_min, 1e-3)   # distance from object surface
        if d_eff < APF_REP_RMAX:
            # dU/dp = k_rep (1/d_eff - 1/d_max) * (-1/d_eff²) * (dp/dd)
            #       = k_rep (1/d_eff - 1/APF_REP_RMAX) / d_eff² * n̂
            n_hat  = diff / d
            coeff  = APF_REP_GAIN * (1.0/d_eff - 1.0/APF_REP_RMAX) / (d_eff**2)
            grad  -= coeff * n_hat   # ∇U_rep = −coeff·n̂ (points toward obs); −(−coeff·n̂) pushes away

    # ── Attractive term ──────────────────────────────────────────────────────
    grad -= APF_ATT_GAIN * (STICK_START - puck)   # minus because grad of ½k||p-g||² = k(p-g)

    # ── Gradient step ────────────────────────────────────────────────────────
    mag = np.linalg.norm(grad) + 1e-9
    step = min(mag, APF_STEP) / mag * grad   # scale to at most APF_STEP
    new  = puck - step                        # descend potential

    new[0] = np.clip(new[0], MARGIN+STICK_R+2, WIDTH -MARGIN-STICK_R-2)
    new[1] = np.clip(new[1], MARGIN+STICK_R+2, HEIGHT-MARGIN-STICK_R-2)
    return new


def apf_is_settled():
    """True when puck is safely clear of BOTH objects."""
    d_box  = np.linalg.norm(stick_pos - q_box[:2])  - (STICK_R + R_BOX_PLAN)
    d_circ = np.linalg.norm(stick_pos - q_circ)     - (STICK_R + CIRC_R)
    return d_box > APF_SAFE_DIST and d_circ > APF_SAFE_DIST

# ══════════════════════════════════════════════════════════════════════════════
#  Target-pose visualization
# ══════════════════════════════════════════════════════════════════════════════

def draw_target_circle(surf, target_xy, color=(0, 210, 210), lw=2):
    """
    Render the circle target: exact-radius outline + faint filled interior +
    crosshair, so the circle object should land inside when goal is reached.
    """
    cx, cy = int(target_xy[0]), int(target_xy[1])
    r = int(CIRC_R)
    # Faint fill
    fill = pygame.Surface((2*r+2, 2*r+2), pygame.SRCALPHA)
    pygame.draw.circle(fill, (*color, 40), (r+1, r+1), r)
    surf.blit(fill, (cx - r - 1, cy - r - 1))
    # Outline
    pygame.draw.circle(surf, color, (cx, cy), r, lw)
    # Crosshair
    pygame.draw.line(surf, color, (cx - r - 8, cy), (cx + r + 8, cy), 1)
    pygame.draw.line(surf, color, (cx, cy - r - 8), (cx, cy + r + 8), 1)


def draw_target_box(surf, target_xyt, color=(255, 165, 0), lw=2):
    """
    Render the box target: rotated exact-size outline + faint filled interior +
    direction marker, so the box should land inside when goal is reached.
    """
    tx, ty, theta = float(target_xyt[0]), float(target_xyt[1]), float(target_xyt[2])
    R    = rotation_matrix(theta)
    half = size / 2.0
    corners = np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    world   = [(R @ c) + np.array([tx, ty]) for c in corners]
    pts_i   = [tuple(c.astype(int)) for c in world]
    # Faint fill
    fill = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(fill, (*color, 40), pts_i)
    surf.blit(fill, (0, 0))
    # Outline
    pygame.draw.lines(surf, color, True, pts_i, lw)
    # Direction marker
    tip = (np.array([tx, ty]) + R @ np.array([size*0.38, 0])).astype(int)
    pygame.draw.line(surf, color, (int(tx), int(ty)), tuple(tip), 2)

# ══════════════════════════════════════════════════════════════════════════════
#  Pygame init
# ══════════════════════════════════════════════════════════════════════════════
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    f"ADMM Manipulation  ▸  {MANIPULATED_OBJECT.upper()}")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 14)

# ── Video recording ───────────────────────────────────────────────────────────
results_dir = pathlib.Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)  # Create folder if it doesn't exist

_video_path = results_dir / "admm_recording_qp_circle.mp4"
_fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
_video      = cv2.VideoWriter(str(_video_path), _fourcc, 45, (WIDTH, HEIGHT))

ctrl          = ADMMController()
phase         = "manipulate"
last_z_plan   = None
last_contacts = 0

# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _video.release()
            pygame.quit(); sys.exit()

    s0    = get_planning_state()
    obs_c = obs_center_from_s(s0)

    # Phase transitions
    if phase == "manipulate" and is_goal_reached():
        phase = "return"
        print("[ADMM] Goal reached — APF return engaged.")
    elif phase == "return" and apf_is_settled():
        phase = "done"
        print("[APF]  Puck safely clear of both objects. Done.")

    # Control
    if phase == "manipulate":
        next_puck, last_z_plan, _ = ctrl.plan(s0, stick_pos, obs_c)
    elif phase == "return":
        next_puck   = apf_return_step(stick_pos)
        last_z_plan = None
    else:
        next_puck = stick_pos.copy()

    stick_pos = next_puck.copy()

    # Physics substeps
    for _ in range(PHYS_STEPS):
        contacts = detect_contacts(q_box, q_circ, stick_pos)
        q_box, v_box, q_circ, v_circ = physics_step(
            q_box, v_box, q_circ, v_circ, contacts)
    last_contacts = len(contacts)

    # ── Rendering ─────────────────────────────────────────────────────────────
    screen.fill((245, 235, 210))
    pygame.draw.rect(screen, (100, 70, 40),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    # Target outline (drawn below objects so they sit on top)
    if MANIPULATED_OBJECT == "circle":
        draw_target_circle(screen, TARGET_POS)
    else:
        draw_target_box(screen, TARGET_POS)

    # Box
    verts = get_box_vertices(q_box).astype(int).tolist()
    pygame.draw.polygon(screen, (210, 80, 60), verts)
    pygame.draw.polygon(screen, (110, 30, 20), verts, 2)
    R_  = rotation_matrix(q_box[2])
    tip = (q_box[:2] + R_ @ np.array([size*0.38, 0])).astype(int)
    pygame.draw.line(screen, (255,255,255),
                     tuple(q_box[:2].astype(int)), tuple(tip), 3)

    # Circle
    ci = q_circ.astype(int)
    pygame.draw.circle(screen, (60, 180, 100), ci, int(CIRC_R))
    pygame.draw.circle(screen, (20, 100,  50), ci, int(CIRC_R), 2)
    pygame.draw.circle(screen, (20, 100,  50), ci, 4)

    # Planned puck path (ADMM horizon)
    if last_z_plan is not None and len(last_z_plan) > 1:
        for k in range(N_PLAN - 1):
            pygame.draw.line(screen, (100, 130, 230),
                             tuple(last_z_plan[k].astype(int)),
                             tuple(last_z_plan[k+1].astype(int)), 1)
        pygame.draw.circle(screen, (80, 100, 200),
                           tuple(last_z_plan[-1].astype(int)), 3)

    # Puck home marker
    sp = STICK_START.astype(int)
    pygame.draw.circle(screen, (130, 140, 240), sp, int(STICK_R)+1, 2)

    # Puck
    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (50,  90, 210), sc, int(STICK_R))
    pygame.draw.circle(screen, (20,  40, 130), sc, int(STICK_R), 2)

    # HUD
    err       = target_error(s0)
    xy_err    = float(np.linalg.norm(err[:2]))
    th_str    = (f"  θ_err={float(np.degrees(err[2])):.1f}°"
                 if MANIPULATED_OBJECT == "box" else "")
    phase_lbl = {"manipulate": "MANIPULATING",
                 "return":     "RETURNING",
                 "done":       "DONE ✓"}[phase]
    lines = [
        f"ADMM | {MANIPULATED_OBJECT.upper()} | {phase_lbl}",
        f"N={N_PLAN}  rho={RHO}  iters={N_ADMM}  phys×{PHYS_STEPS}/step  Qv_t={Q_VEL_TERM}",
        f"box   ({q_box[0]:.0f},{q_box[1]:.0f})"
        f"  θ={float(np.degrees(q_box[2])):.1f}°"
        f"  spd={float(np.hypot(v_box[0],v_box[1])):.1f}",
        f"circ  ({q_circ[0]:.0f},{q_circ[1]:.0f})"
        f"  spd={float(np.hypot(v_circ[0],v_circ[1])):.1f}",
        f"target_err: xy={xy_err:.1f}px{th_str}",
        f"puck  ({stick_pos[0]:.0f},{stick_pos[1]:.0f})"
        f"  contacts={last_contacts}",
        "cyan/orange-outline = target  |  blue path = ADMM horizon  |  return = APF gradient descent",
    ]
    for i, ln in enumerate(lines):
        screen.blit(font.render(ln, True, (50, 30, 10)), (10, 10 + i*17))

    pygame.display.flip()

    # Capture frame for video
    _frame = pygame.surfarray.array3d(screen)          # (W, H, 3) RGB
    _frame = np.transpose(_frame, (1, 0, 2))           # → (H, W, 3)
    _video.write(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR))

    clock.tick(45)