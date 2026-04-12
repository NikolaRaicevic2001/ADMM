"""
ADMM-MPC planar manipulation using the Anitescu QP as the planning model.

Physics backend: contact_qp_planar.py (loaded via exec at startup).
Planning state: 10-DOF [x_box, y_box, theta_box, x_circ, y_circ,
                         vbx, vby, vb_omega, vcx, vcy] — identical layout
                to the real Anitescu QP, eliminating planning/physics mismatch.
ADMM split: min J(u) + I_Z(z)  s.t. u = z
  u-update: linear solve  (H + rho*I) du = rho(z - y - P_bar) - g
  z-update: analytical projection (obstacle avoidance + speed constraint)
  y-update: dual ascent  y <- y + u - z
"""

import sys
import pathlib
import pygame
import numpy as np
import cv2
import cvxpy as cp

# ── Load physics backend (exec up to but not including the while-loop) ─────────
_src = pathlib.Path(__file__).with_name("contact_qp_planar.py").read_text()
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

# Physics constants pulled from backend for use in the numpy QP planner
M_all      = _ns["M_all"]
Minv       = _ns["Minv"]
GAMMA_PHYS = _ns["GAMMA"]
B_LIN_BOX  = _ns["B_LIN_BOX"]
B_ROT_BOX  = _ns["B_ROT_BOX"]
B_LIN_CIRC = _ns["B_LIN_CIRC"]

# ── User config ───────────────────────────────────────────────────────────────
MANIPULATED_OBJECT = "circle"                          # "circle"  or  "box"
TARGET_CIRCLE      = np.array([260.0, 130.0])          # (x, y)
TARGET_BOX         = np.array([630.0, 430.0, np.pi/3]) # (x, y, theta_rad)

# ── Initial state ─────────────────────────────────────────────────────────────
q_box       = np.array([WIDTH/2 - 120.0, HEIGHT/2, 0.0])
v_box       = np.zeros(3)
q_circ      = np.array([WIDTH/2 + 130.0, HEIGHT/2])
v_circ      = np.zeros(2)
STICK_START = np.array([WIDTH/2 - 300.0, HEIGHT/2])
stick_pos   = STICK_START.copy()

# ── ADMM / MPC hyper-parameters ───────────────────────────────────────────────
N_PLAN     = 24       # planning horizon (timesteps)
RHO        = 8.0      # ADMM penalty  rho
N_ADMM     = 25       # ADMM inner iterations per control step

EPS_FD_XY  = 0.8      # FD perturbation: position states [px]
EPS_FD_TH  = 0.018    # FD perturbation: theta state     [rad]
EPS_FD_P   = 8.0      # FD perturbation: puck control    [px]
EPS_FD_V   = 0.3      # FD perturbation: velocity states [px/step]

Q_TERM     = 640.0    # terminal tracking weight  Q_f
Q_RUN      = 10.0     # running tracking weight   Q
Q_VEL_TERM = 537.5    # terminal velocity cost
Q_VEL_RUN  = 3.0      # running velocity cost
R_RATE     = 0.5      # puck smoothness weight    R_r
R_ABS      = 0.04     # Tikhonov regularisation on delta_u
R_GUIDE    = 8.0      # guidance: puck -> push position

# Negative margin keeps gap < 0 so Baumgarte always generates a contact force,
# preventing the B_k=0 "freeze" bug near the object surface.
PUSH_MARGIN = -2.0
ROT_LAT_K   = 0.18    # lateral-offset gain for rotation induction

PHYS_STEPS      = 3   # physics substeps per control step (real execution)
PHYS_STEPS_PLAN = 3   # must equal PHYS_STEPS to avoid horizon mismatch
N_DUAL_ITERS    = 30  # dual projected-gradient iterations inside planner

MAX_STEP = STICK_SPEED * h * PHYS_STEPS * 1.9

R_BOX_PLAN  = size * 0.5
SAFE_R_BOX  = STICK_R + R_BOX_PLAN + 28.0
SAFE_R_CIRC = STICK_R + CIRC_R     + 28.0

W_OBS_PEN   = 6000.0  # puck-obstacle repulsion weight in QP cost
R_OBS_EXTRA = 35.0    # penalty activates this many px beyond SAFE_R

# APF constants for the puck return phase
APF_REP_GAIN  = 18000.0
APF_REP_RMAX  = 160.0
APF_ATT_GAIN  = 0.35
APF_STEP      = 14.0
APF_SAFE_DIST = 130.0

REACH_XY    = 10.0    # [px]  goal reached – translation
REACH_THETA = 0.15    # [rad] goal reached – rotation (box only)
RETURN_DIST = 8.0     # [px]  puck home

# ── Planning state layout: NS=10, same for both modes ─────────────────────────
# index: 0      1      2          3       4        5    6    7         8    9
# field: x_box  y_box  theta_box  x_circ  y_circ   vbx  vby  vb_omega  vcx  vcy

NS = 10
NU = 2   # control dim: puck (x, y)

if MANIPULATED_OBJECT == "circle":
    TARGET_DIM = 2
    TARGET_POS = TARGET_CIRCLE.copy()
    OBS_SAFE_R = SAFE_R_BOX
    # C_sel: selects circle position (indices 3, 4)
    C_sel      = np.zeros((2, 10))
    C_sel[0, 3] = 1.0; C_sel[1, 4] = 1.0
    # C_vel: selects circle velocity (indices 8, 9)
    C_vel      = np.zeros((2, 10))
    C_vel[0, 8] = 1.0; C_vel[1, 9] = 1.0
    W_cost       = np.eye(2)
    W_vel        = np.eye(2)
    THETA_WEIGHT = 0.0
else:
    TARGET_DIM = 3
    TARGET_POS = TARGET_BOX.copy()
    OBS_SAFE_R = SAFE_R_CIRC
    # C_sel: selects box pose (indices 0, 1, 2)
    C_sel      = np.zeros((3, 10))
    C_sel[0, 0] = 1.0; C_sel[1, 1] = 1.0; C_sel[2, 2] = 1.0
    # C_vel: selects box velocity (indices 5, 6, 7)
    C_vel      = np.zeros((3, 10))
    C_vel[0, 5] = 1.0; C_vel[1, 6] = 1.0; C_vel[2, 7] = 1.0
    THETA_WEIGHT = (size / 2.0) ** 2
    W_cost       = np.diag([1.0, 1.0, THETA_WEIGHT])
    W_vel        = np.diag([1.0, 1.0, THETA_WEIGHT])

print(f"[admm_qp_planar] mode={MANIPULATED_OBJECT}  N={N_PLAN}  rho={RHO}  admm_iters={N_ADMM}")
print(f"  target={TARGET_POS}  PUSH_MARGIN={PUSH_MARGIN}px  PHYS_STEPS={PHYS_STEPS}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def angle_wrap(a):
    return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)


def get_planning_state():
    # Full 10-DOF state matching Anitescu QP DOF order exactly
    return np.array([q_box[0], q_box[1], q_box[2],
                     q_circ[0], q_circ[1],
                     v_box[0],  v_box[1],  v_box[2],
                     v_circ[0], v_circ[1]])


def obs_center_from_s(s):
    # Returns position of the NON-manipulated obstacle body
    if MANIPULATED_OBJECT == "circle":
        return s[0:2].copy()   # box xy  (indices 0, 1)
    return s[3:5].copy()       # circle xy (indices 3, 4)


def get_obs_from_state(s):
    return obs_center_from_s(s)


def compute_push_pos(s_k):
    # Ideal puck position: behind the object along obj->target line.
    # For box mode, a lateral offset induces the required torque.
    if MANIPULATED_OBJECT == "circle":
        obj_xy    = s_k[3:5]
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


# ── Planning dynamics: real Anitescu QP solved via dual projected gradient ─────

def _physics_qp_numpy(kbar, contacts):
    # No contacts: unconstrained minimum v* = -M^{-1} kbar
    if not contacts:
        return -Minv @ kbar

    rows, b_list = [], []
    for phi, Jn, Jt, mu_c in contacts:
        gap = GAMMA_PHYS * phi / h
        rows.append(Jn + mu_c * Jt);  b_list.append(-gap)
        rows.append(Jn - mu_c * Jt);  b_list.append(-gap)

    A = np.array(rows)
    b = np.array(b_list)

    # Dual matrices: H_d = A M^{-1} A^T, step alpha = 1/lambda_max(H_d)
    AMinv = A @ Minv
    H_d   = AMinv @ A.T
    c_d   = AMinv @ kbar

    ev_max = np.linalg.eigvalsh(H_d)[-1]
    if ev_max < 1e-12:
        return -Minv @ kbar
    alpha = 1.0 / ev_max

    # Projected gradient ascent on dual: lam = max(0, lam + alpha * grad)
    lam = np.zeros(len(b))
    for _ in range(N_DUAL_ITERS):
        lam = np.maximum(0.0, lam + alpha * (b + c_d - H_d @ lam))

    return Minv @ (A.T @ lam - kbar)


def plan_step(s, puck):
    # Run PHYS_STEPS_PLAN substeps of the real physics to propagate state s
    puck = np.asarray(puck, dtype=float)

    q_b = s[0:3].copy()
    q_c = s[3:5].copy()
    v_b = s[5:8].copy()
    v_c = s[8:10].copy()

    for _ in range(PHYS_STEPS_PLAN):
        v_all = np.concatenate([v_b, v_c])

        f_b   = np.array([-B_LIN_BOX  * v_b[0],
                           -B_LIN_BOX  * v_b[1],
                           -B_ROT_BOX  * v_b[2]])
        f_c   = np.array([-B_LIN_CIRC * v_c[0],
                           -B_LIN_CIRC * v_c[1]])
        f_all = np.concatenate([f_b, f_c])

        kbar     = -M_all @ v_all - h * f_all
        contacts = detect_contacts(q_b, q_c, puck)
        v_next   = _physics_qp_numpy(kbar, contacts)

        q_b = q_b + h * v_next[:3]
        q_c = q_c + h * v_next[3:]
        v_b = v_next[:3]
        v_c = v_next[3:]

    return np.array([q_b[0], q_b[1], q_b[2],
                     q_c[0], q_c[1],
                     v_b[0], v_b[1], v_b[2],
                     v_c[0], v_c[1]])


def rollout(s0, P):
    X    = np.empty((N_PLAN + 1, NS))
    X[0] = s0
    for k in range(N_PLAN):
        X[k + 1] = plan_step(X[k], P[k])
    return X


# ── Finite-difference Jacobians A_k = df/ds, B_k = df/dp ─────────────────────

def fd_jacobians(s_bar, p_bar):
    # Per-state perturbation sizes respect different units (px, rad, px/step)
    eps_s = np.full(NS, EPS_FD_XY)
    eps_s[2]    = EPS_FD_TH
    eps_s[5:8]  = EPS_FD_V
    eps_s[8:10] = EPS_FD_V

    A = np.empty((NS, NS))
    for i in range(NS):
        sp = s_bar.copy(); sp[i] += eps_s[i]
        sm = s_bar.copy(); sm[i] -= eps_s[i]
        A[:, i] = (plan_step(sp, p_bar) - plan_step(sm, p_bar)) / (2.0 * eps_s[i])

    B = np.empty((NS, NU))
    for i in range(NU):
        pp = p_bar.copy(); pp[i] += EPS_FD_P
        pm = p_bar.copy(); pm[i] -= EPS_FD_P
        B[:, i] = (plan_step(s_bar, pp) - plan_step(s_bar, pm)) / (2.0 * EPS_FD_P)

    return A, B


# ── Build condensed QP matrices H, g: J approx 0.5 du^T H du + g^T du ────────

def build_qp(s0, P_bar):
    n = 2 * N_PLAN

    X_bar = rollout(s0, P_bar)
    As = []; Bs = []
    for k in range(N_PLAN):
        A_k, B_k = fd_jacobians(X_bar[k], P_bar[k])
        As.append(A_k); Bs.append(B_k)

    H = np.zeros((n, n))
    g = np.zeros(n)
    G = np.zeros((NS, n))

    for k in range(N_PLAN):
        G                  = As[k] @ G
        G[:, 2*k:2*k+2]   += Bs[k]

        w     = Q_TERM if k == N_PLAN - 1 else Q_RUN
        CG    = C_sel @ G
        WCG   = W_cost @ CG

        e_bar = C_sel @ X_bar[k+1] - TARGET_POS
        if MANIPULATED_OBJECT == "box":
            e_bar[2] = angle_wrap(e_bar[2])

        H += 2.0 * w * CG.T @ WCG
        g += 2.0 * w * CG.T @ (W_cost @ e_bar)

    # Puck smoothness: penalise step-to-step velocity changes
    D = np.zeros((2*(N_PLAN-1), n))
    for k in range(N_PLAN-1):
        D[2*k:2*k+2, 2*k:2*k+2]   = -np.eye(2)
        D[2*k:2*k+2, 2*k+2:2*k+4] =  np.eye(2)
    d_nom = D @ P_bar.flatten()
    H += 2.0 * R_RATE * D.T @ D
    g += 2.0 * R_RATE * D.T @ d_nom

    # Tikhonov regularisation
    H += 2.0 * R_ABS * np.eye(n)

    # Guidance: pull puck toward push position behind target object
    for k in range(N_PLAN):
        push_p = compute_push_pos(X_bar[k])
        d_g    = P_bar[k] - push_p
        H[2*k:2*k+2, 2*k:2*k+2] += 2.0 * R_GUIDE * np.eye(2)
        g[2*k:2*k+2]             += 2.0 * R_GUIDE * d_g

    # Obstacle-avoidance penalty (soft, activates within R_OBS_EXTRA of SAFE_R)
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

    # Velocity cost: encourage object to arrive at rest (weight ramps to Q_VEL_TERM)
    G_v = np.zeros((NS, 2 * N_PLAN))
    for k in range(N_PLAN):
        G_v                   = As[k] @ G_v
        G_v[:, 2*k:2*k+2]   += Bs[k]

        w_v  = Q_VEL_RUN + (Q_VEL_TERM - Q_VEL_RUN) * k / max(N_PLAN - 1, 1)
        CvGv  = C_vel @ G_v
        WCvGv = W_vel @ CvGv
        v_bar = C_vel @ X_bar[k + 1]
        H += 2.0 * w_v * CvGv.T @ WCvGv
        g += 2.0 * w_v * CvGv.T @ (W_vel @ v_bar)

    return H, g, X_bar


# ── ADMM Controller ───────────────────────────────────────────────────────────

class ADMMController:
    def __init__(self):
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
            rhs     = RHO * (z - y - P_bar_flat) - g
            delta_u = np.linalg.solve(M_lhs, rhs)
            u_abs   = P_bar_flat + delta_u
            z = self._project(u_abs + y, cur_puck, X_bar)
            y = y + u_abs - z

        self.z = z.copy()
        self.y = y.copy()

        z_plan = z.reshape(N_PLAN, 2)
        # Receding-horizon warm-start: shift plan by one step
        self.P_bar[:-1] = z_plan[1:]
        self.P_bar[-1]  = z_plan[-1]

        next_p = z_plan[0].copy()
        next_p[0] = np.clip(next_p[0], MARGIN+STICK_R+1, WIDTH -MARGIN-STICK_R-1)
        next_p[1] = np.clip(next_p[1], MARGIN+STICK_R+1, HEIGHT-MARGIN-STICK_R-1)
        return next_p, z_plan, X_bar

    def _project(self, v_flat, cur_puck, X_bar):
        # Z_obs: keep puck outside obstacle; Z_speed: limit per-step displacement
        PROJ_R = OBS_SAFE_R + R_OBS_EXTRA
        z = v_flat.reshape(N_PLAN, 2).copy()

        for k in range(N_PLAN):
            obs_k = get_obs_from_state(X_bar[k])
            d     = z[k] - obs_k
            dist  = np.linalg.norm(d)
            if dist < PROJ_R:
                z[k] = obs_k + PROJ_R * d / (dist + 1e-9)
            z[k, 0] = np.clip(z[k, 0], MARGIN+STICK_R+1, WIDTH -MARGIN-STICK_R-1)
            z[k, 1] = np.clip(z[k, 1], MARGIN+STICK_R+1, HEIGHT-MARGIN-STICK_R-1)

        for _ in range(3):
            prev = cur_puck.copy()
            for k in range(N_PLAN):
                diff = z[k] - prev
                d    = np.linalg.norm(diff)
                if d > MAX_STEP:
                    z[k] = prev + diff / d * MAX_STEP
                prev = z[k]

        return z.flatten()


# ── APF return phase: repulsion from both bodies + attraction to home ──────────

def apf_return_step(puck):
    grad = np.zeros(2)
    obstacles = [
        (q_box[:2],  STICK_R + R_BOX_PLAN + 4.0),
        (q_circ,     STICK_R + CIRC_R     + 4.0),
    ]
    for (centre, r_min) in obstacles:
        diff  = puck - centre
        d     = np.linalg.norm(diff) + 1e-9
        d_eff = max(d - r_min, 1e-3)
        if d_eff < APF_REP_RMAX:
            n_hat  = diff / d
            coeff  = APF_REP_GAIN * (1.0/d_eff - 1.0/APF_REP_RMAX) / (d_eff**2)
            grad  -= coeff * n_hat
    grad -= APF_ATT_GAIN * (STICK_START - puck)
    mag  = np.linalg.norm(grad) + 1e-9
    step = min(mag, APF_STEP) / mag * grad
    new  = puck - step
    new[0] = np.clip(new[0], MARGIN+STICK_R+2, WIDTH -MARGIN-STICK_R-2)
    new[1] = np.clip(new[1], MARGIN+STICK_R+2, HEIGHT-MARGIN-STICK_R-2)
    return new


def apf_is_settled():
    d_box  = np.linalg.norm(stick_pos - q_box[:2]) - (STICK_R + R_BOX_PLAN)
    d_circ = np.linalg.norm(stick_pos - q_circ)    - (STICK_R + CIRC_R)
    return d_box > APF_SAFE_DIST and d_circ > APF_SAFE_DIST


# ── Target-pose visualisation ─────────────────────────────────────────────────

def draw_target_circle(surf, target_xy, color=(0, 210, 210), lw=2):
    cx, cy = int(target_xy[0]), int(target_xy[1])
    r = int(CIRC_R)
    fill = pygame.Surface((2*r+2, 2*r+2), pygame.SRCALPHA)
    pygame.draw.circle(fill, (*color, 40), (r+1, r+1), r)
    surf.blit(fill, (cx - r - 1, cy - r - 1))
    pygame.draw.circle(surf, color, (cx, cy), r, lw)
    pygame.draw.line(surf, color, (cx - r - 8, cy), (cx + r + 8, cy), 1)
    pygame.draw.line(surf, color, (cx, cy - r - 8), (cx, cy + r + 8), 1)


def draw_target_box(surf, target_xyt, color=(255, 165, 0), lw=2):
    tx, ty, theta = float(target_xyt[0]), float(target_xyt[1]), float(target_xyt[2])
    R    = rotation_matrix(theta)
    half = size / 2.0
    corners = np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    world   = [(R @ c) + np.array([tx, ty]) for c in corners]
    pts_i   = [tuple(c.astype(int)) for c in world]
    fill = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(fill, (*color, 40), pts_i)
    surf.blit(fill, (0, 0))
    pygame.draw.lines(surf, color, True, pts_i, lw)
    tip = (np.array([tx, ty]) + R @ np.array([size*0.38, 0])).astype(int)
    pygame.draw.line(surf, color, (int(tx), int(ty)), tuple(tip), 2)


# ── Pygame init + video writer ────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    f"ADMM-MPC (LCS dynamics)  ▸  {MANIPULATED_OBJECT.upper()}")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 14)

results_dir = pathlib.Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

_video_path = results_dir / "admm_recording_qp_final.mp4"
_fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
_video      = cv2.VideoWriter(str(_video_path), _fourcc, 45, (WIDTH, HEIGHT))

ctrl          = ADMMController()
phase         = "manipulate"
last_z_plan   = None
last_contacts = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _video.release()
            pygame.quit(); sys.exit()

    s0    = get_planning_state()
    obs_c = obs_center_from_s(s0)

    if phase == "manipulate" and is_goal_reached():
        phase = "return"
        print("[ADMM] Goal reached — APF return engaged.")
    elif phase == "return" and apf_is_settled():
        phase = "done"
        print("[APF]  Puck safely clear of both objects. Done.")

    if phase == "manipulate":
        next_puck, last_z_plan, _ = ctrl.plan(s0, stick_pos, obs_c)
    elif phase == "return":
        next_puck   = apf_return_step(stick_pos)
        last_z_plan = None
    else:
        next_puck = stick_pos.copy()

    stick_pos = next_puck.copy()

    for _ in range(PHYS_STEPS):
        contacts = detect_contacts(q_box, q_circ, stick_pos)
        q_box, v_box, q_circ, v_circ = physics_step(
            q_box, v_box, q_circ, v_circ, contacts)
    last_contacts = len(contacts)

    # ── Render ────────────────────────────────────────────────────────────────
    screen.fill((245, 235, 210))
    pygame.draw.rect(screen, (100, 70, 40),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    if MANIPULATED_OBJECT == "circle":
        draw_target_circle(screen, TARGET_POS)
    else:
        draw_target_box(screen, TARGET_POS)

    verts = get_box_vertices(q_box).astype(int).tolist()
    pygame.draw.polygon(screen, (210, 80, 60), verts)
    pygame.draw.polygon(screen, (110, 30, 20), verts, 2)
    R_  = rotation_matrix(q_box[2])
    tip = (q_box[:2] + R_ @ np.array([size*0.38, 0])).astype(int)
    pygame.draw.line(screen, (255,255,255),
                     tuple(q_box[:2].astype(int)), tuple(tip), 3)

    ci = q_circ.astype(int)
    pygame.draw.circle(screen, (60, 180, 100), ci, int(CIRC_R))
    pygame.draw.circle(screen, (20, 100,  50), ci, int(CIRC_R), 2)
    pygame.draw.circle(screen, (20, 100,  50), ci, 4)

    if last_z_plan is not None and len(last_z_plan) > 1:
        for k in range(N_PLAN - 1):
            pygame.draw.line(screen, (100, 130, 230),
                             tuple(last_z_plan[k].astype(int)),
                             tuple(last_z_plan[k+1].astype(int)), 1)
        pygame.draw.circle(screen, (80, 100, 200),
                           tuple(last_z_plan[-1].astype(int)), 3)

    sp = STICK_START.astype(int)
    pygame.draw.circle(screen, (130, 140, 240), sp, int(STICK_R)+1, 2)

    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (50,  90, 210), sc, int(STICK_R))
    pygame.draw.circle(screen, (20,  40, 130), sc, int(STICK_R), 2)

    err       = target_error(s0)
    xy_err    = float(np.linalg.norm(err[:2]))
    th_str    = (f"  θ_err={float(np.degrees(err[2])):.1f}°"
                 if MANIPULATED_OBJECT == "box" else "")
    phase_lbl = {"manipulate": "MANIPULATING",
                 "return":     "RETURNING",
                 "done":       "DONE ✓"}[phase]
    lines = [
        f"ADMM-LCS | {MANIPULATED_OBJECT.upper()} | {phase_lbl}",
        f"N={N_PLAN}  rho={RHO}  iters={N_ADMM}  phys×{PHYS_STEPS}/step  plan×{PHYS_STEPS_PLAN}",
        f"FD_P={EPS_FD_P}px  PUSH_MARGIN={PUSH_MARGIN}px  (real EPS_HAT={_ns['EPS_HAT']}px)",
        f"box   ({q_box[0]:.0f},{q_box[1]:.0f})"
        f"  θ={float(np.degrees(q_box[2])):.1f}°"
        f"  spd={float(np.hypot(v_box[0],v_box[1])):.1f}",
        f"circ  ({q_circ[0]:.0f},{q_circ[1]:.0f})"
        f"  spd={float(np.hypot(v_circ[0],v_circ[1])):.1f}",
        f"target_err: xy={xy_err:.1f}px{th_str}",
        f"puck  ({stick_pos[0]:.0f},{stick_pos[1]:.0f})"
        f"  contacts={last_contacts}",
        "Planning model: real Anitescu QP (dual proj-grad, no CVXPY)",
        "cyan/orange-outline = target  |  blue path = ADMM horizon",
    ]
    for i, ln in enumerate(lines):
        screen.blit(font.render(ln, True, (50, 30, 10)), (10, 10 + i*17))

    pygame.display.flip()

    _frame = pygame.surfarray.array3d(screen)
    _frame = np.transpose(_frame, (1, 0, 2))
    _video.write(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR))

    clock.tick(45)
