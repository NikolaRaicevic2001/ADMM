import pygame
import numpy as np
import sys
import cv2
import pathlib

# ═══════════════════════════════════════════════════════════════════════
#  PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 900, 600
MARGIN        = 40
h             = 1 / 240
mu_stick      = 0.3
mu_wall       = 0.1

CIRC_R        = 35.0
E_CIRC        = 6.0
E_diag        = np.array([E_CIRC, E_CIRC])
Q_diag        = E_diag / h**2
Q_half_diag   = np.sqrt(Q_diag)
Q_half_inv    = 1.0 / Q_half_diag

SIGMA_D       = 4.0
GAMMA         = 0.5
STICK_R       = 12.0
STICK_SPEED   = 500.0
EPS_HAT       = 8.0

# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO
# ═══════════════════════════════════════════════════════════════════════
Q_GOAL       = np.array([680.0, 310.0])
Q_CIRC_INIT  = np.array([WIDTH / 2.0, HEIGHT / 2.0])
PUCK_INIT    = np.array([200.0, 430.0])
PUCK_RETURN  = np.array([200.0, 430.0])

OBS_CENTERS  = [np.array([535.0, 165.0]),
                np.array([565.0, 415.0])]
OBS_RADII    = [40.0, 35.0]

# ═══════════════════════════════════════════════════════════════════════
#  ADMM / MPC HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
N_HORIZON    = 15
ADMM_ITERS   = 10
ALPHA_V      = 20.0
RHO          = 0.05

P_WEIGHT     = 10.0

# ── FIX 1: APPROACH_W fade ────────────────────────────────────────────
# APPROACH_W is full strength when dist_goal > APPROACH_FADE_START,
# then linearly decays to 0 at APPROACH_FADE_END.
# This stops the puck oscillating when the circle is almost at the goal.
APPROACH_W          = 0.75
APPROACH_FADE_START = 120.0   # px — begin fading here
APPROACH_FADE_END   =  30.0   # px — fully off here (just keep pushing)

# ── FIX 2: circle obstacle repulsion cost ────────────────────────────
# Same soft-penalty form as PUCK_OBS_W but applied to the planned circle
# trajectory so the optimizer routes the push path around obstacles.
PUCK_OBS_W      = 1000.0   # puck ↔ obstacle repulsion (existing)
CIRC_OBS_W      = 2000.0   # circle ↔ obstacle repulsion (NEW)
OBS_MARGIN      = 14.0     # puck clearance (px beyond obs_r + STICK_R)
CIRC_OBS_MARGIN = 10.0     # circle clearance (px beyond obs_r + CIRC_R)

S_WEIGHT     = 0.0
FD_EPS       = 10.0
NORM_THRESH  = 1e-4
GOAL_THRESH  = 10.0
RETURN_THRESH= 5.0
KP_RETURN    = 8.0

# ═══════════════════════════════════════════════════════════════════════
#  PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════════════════

def csdf_circle(x_query, center, radius):
    delta = x_query - center
    dist  = np.linalg.norm(delta)
    if dist < 1e-10:
        return -radius, np.array([1.0, 0.0])
    return dist - radius, delta / dist

def jac_circ(n):
    t  = np.array([-n[1], n[0]])
    Jn = np.array([n[0], n[1]])
    Jt = np.array([t[0], t[1]])
    return Jn, Jt

def detect_contacts(q_circ, stick_pos, stick_vel):
    contacts = []
    cx, cy   = q_circ

    for phi_val, n_vec in [
        (cx - CIRC_R - MARGIN,              np.array([ 1.,  0.])),
        ((WIDTH  - MARGIN) - (cx + CIRC_R), np.array([-1.,  0.])),
        (cy - CIRC_R - MARGIN,              np.array([ 0.,  1.])),
        ((HEIGHT - MARGIN) - (cy + CIRC_R), np.array([ 0., -1.])),
    ]:
        if phi_val <= EPS_HAT:
            Jn, Jt = jac_circ(n_vec)
            contacts.append((GAMMA * phi_val, Jn, Jt, mu_wall))

    for obs_c, obs_r in zip(OBS_CENTERS, OBS_RADII):
        phi_obs, n_obs = csdf_circle(q_circ, obs_c, CIRC_R + obs_r)
        if phi_obs <= EPS_HAT and np.linalg.norm(n_obs) > 1e-8:
            Jn, Jt = jac_circ(n_obs)
            contacts.append((GAMMA * phi_obs, Jn, Jt, 0.0))

    phi_pc, n_pc = csdf_circle(q_circ, stick_pos, STICK_R + CIRC_R)
    if phi_pc <= EPS_HAT and np.linalg.norm(n_pc) > 1e-8:
        b_offset = float(n_pc @ stick_vel)
        phi_eff  = GAMMA * phi_pc - b_offset * h
        Jn, Jt   = jac_circ(n_pc)
        contacts.append((phi_eff, Jn, Jt, mu_stick))

    return contacts

def dsdf_step(contacts):
    if not contacts:
        return np.zeros(2)
    z_query = np.zeros(2)
    ns, bs  = [], []
    for phi_eff, Jn, Jt, mu_c in contacts:
        for J_ij in [Jn + mu_c * Jt, Jn - mu_c * Jt]:
            Qinv_Jt = Q_half_inv * J_ij
            norm    = np.linalg.norm(Qinv_Jt)
            if norm < 1e-10:
                continue
            ns.append(-Qinv_Jt / norm)
            bs.append(-phi_eff / norm)
    if not ns:
        return np.zeros(2)
    ns = np.array(ns); bs = np.array(bs)
    scores    = ns @ z_query + bs
    all_terms = np.concatenate([[0.0], SIGMA_D * scores])
    max_t     = np.max(all_terms)
    lse       = max_t + np.log(np.sum(np.exp(all_terms - max_t)))
    d_sdf     = lse / SIGMA_D
    sw        = np.exp(all_terms - max_t); sw /= np.sum(sw)
    grad      = sw[1:] @ ns
    z_plus    = z_query - d_sdf * grad
    return Q_half_inv * z_plus / h

# ═══════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════

def push_dir(q):
    d = Q_GOAL - q
    return d / (np.linalg.norm(d) + 1e-8)

def push_pos(q):
    return q - push_dir(q) * (CIRC_R + STICK_R)

def clamp_puck(pos):
    return np.clip(pos,
                   [MARGIN + STICK_R,         MARGIN + STICK_R],
                   [WIDTH - MARGIN - STICK_R, HEIGHT - MARGIN - STICK_R])

# ═══════════════════════════════════════════════════════════════════════
#  ROLLOUT
# ═══════════════════════════════════════════════════════════════════════

def rollout(q0, u0, V):
    q, u  = q0.copy(), u0.copy()
    q_seq = [q.copy()]
    u_seq = [u.copy()]
    for vk in V:
        u = clamp_puck(u + h * vk)
        q = q + h * dsdf_step(detect_contacts(q, u, vk))
        q_seq.append(q.copy())
        u_seq.append(u.copy())
    return np.array(q_seq), np.array(u_seq)

# ═══════════════════════════════════════════════════════════════════════
#  COST
# ═══════════════════════════════════════════════════════════════════════

def traj_cost(q0, u0, V):
    """
    J  =  P_WEIGHT      · ‖q_N − goal‖²
        + APPROACH_W_eff· ‖u_N − push_pos(q0)‖²   (fades near goal)
        + CIRC_OBS_W    · Σ_k max(0, −circ_clearance_k)²   ← NEW
        + PUCK_OBS_W    · Σ_k max(0, −puck_clearance_k)²
        + S_WEIGHT      · Σ‖V[k]‖²
    """
    q_seq, u_seq = rollout(q0, u0, V)

    # Terminal circle-to-goal
    dqN = q_seq[-1] - Q_GOAL
    J   = P_WEIGHT * float(dqN @ dqN)

    # ── FIX 1: fade APPROACH_W as circle nears goal ───────────────────
    dist_goal = np.linalg.norm(q0 - Q_GOAL)
    if dist_goal >= APPROACH_FADE_START:
        aw_eff = APPROACH_W
    elif dist_goal <= APPROACH_FADE_END:
        aw_eff = 0.0
    else:
        t      = (dist_goal - APPROACH_FADE_END) / (APPROACH_FADE_START - APPROACH_FADE_END)
        aw_eff = APPROACH_W * t

    if aw_eff > 0.0:
        pp  = push_pos(q0)
        duN = u_seq[-1] - pp
        J  += aw_eff * float(duN @ duN)

    # ── FIX 2: circle obstacle repulsion (every planned step) ─────────
    if CIRC_OBS_W > 0.0:
        for qk in q_seq:
            for obs_c, obs_r in zip(OBS_CENTERS, OBS_RADII):
                clr = np.linalg.norm(qk - obs_c) - (CIRC_R + obs_r + CIRC_OBS_MARGIN)
                if clr < 0.0:
                    J += CIRC_OBS_W * clr * clr

    # Puck obstacle repulsion (every planned step, unchanged from v3)
    if PUCK_OBS_W > 0.0:
        for uk in u_seq:
            for obs_c, obs_r in zip(OBS_CENTERS, OBS_RADII):
                clr = np.linalg.norm(uk - obs_c) - (STICK_R + obs_r + OBS_MARGIN)
                if clr < 0.0:
                    J += PUCK_OBS_W * clr * clr

    if S_WEIGHT > 0.0:
        J += S_WEIGHT * float(np.sum(V * V))

    return J

# ═══════════════════════════════════════════════════════════════════════
#  FD GRADIENT
# ═══════════════════════════════════════════════════════════════════════

def grad_fd(q0, u0, V):
    g  = np.zeros_like(V)
    J0 = traj_cost(q0, u0, V)
    for k in range(len(V)):
        for d in range(2):
            Vp       = V.copy()
            Vp[k, d] += FD_EPS
            g[k, d]  = (traj_cost(q0, u0, Vp) - J0) / FD_EPS
    return g

# ═══════════════════════════════════════════════════════════════════════
#  SPEED-BALL PROJECTION
# ═══════════════════════════════════════════════════════════════════════

def proj_ball(V, max_speed):
    Z  = V.copy()
    nm = np.linalg.norm(V, axis=1)
    for k in range(len(V)):
        if nm[k] > max_speed:
            Z[k] = V[k] * (max_speed / nm[k])
    return Z

# ═══════════════════════════════════════════════════════════════════════
#  ADMM SOLVER
# ═══════════════════════════════════════════════════════════════════════

def admm_solve(q0, u0, V_init):
    V   = V_init.copy()
    Z   = proj_ball(V, STICK_SPEED)
    Lam = np.zeros_like(V)

    for _ in range(ADMM_ITERS):
        aug = grad_fd(q0, u0, V) + Lam + RHO * (V - Z)
        nms = np.linalg.norm(aug, axis=1)
        for k in range(len(V)):
            if nms[k] > NORM_THRESH:
                aug[k] /= nms[k]
            else:
                aug[k]  = 0.0
        V = V - ALPHA_V * aug
        Z = proj_ball(V + Lam / RHO, STICK_SPEED)
        Lam = Lam + RHO * (V - Z)

    return Z

# ═══════════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════════
q_circ     = Q_CIRC_INIT.copy()
stick_pos  = PUCK_INIT.copy()
stick_prev = PUCK_INIT.copy()
phase      = "admm"
V_prev     = None
V_last     = None
plan_q     = None
plan_u     = None

# ═══════════════════════════════════════════════════════════════════════
#  PYGAME INIT
# ═══════════════════════════════════════════════════════════════════════
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ADMM-MPC  |  ContactSDF v4  |  Obstacle-Aware")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 14)

# ── Video recording ───────────────────────────────────────────────────────────
results_dir = pathlib.Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)  # Create folder if it doesn't exist

_video_path = results_dir / "admm_recording_sdf.mp4"
_fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
_video      = cv2.VideoWriter(str(_video_path), _fourcc, 45, (WIDTH, HEIGHT))

_R        = int(CIRC_R)
_gs       = _R * 2 + 8
goal_surf = pygame.Surface((_gs, _gs), pygame.SRCALPHA)
pygame.draw.circle(goal_surf, (220, 55, 55,  70), (_R+4, _R+4), _R)
pygame.draw.circle(goal_surf, (220, 55, 55, 255), (_R+4, _R+4), _R, 4)

overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
dragging_obs = None

# ═══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _video.release()
            pygame.quit(); sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mp = np.array(event.pos, dtype=float)
            for i, (oc, orad) in enumerate(zip(OBS_CENTERS, OBS_RADII)):
                if np.linalg.norm(mp - oc) < orad + 8:
                    dragging_obs = i
                    break

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_obs is not None:
                V_prev = None
            dragging_obs = None

        elif event.type == pygame.MOUSEMOTION and dragging_obs is not None:
            OBS_CENTERS[dragging_obs] = np.array(event.pos, dtype=float)

    stick_prev = stick_pos.copy()

    # ── ADMM MPC ──────────────────────────────────────────────────────
    if phase == "admm":
        pdir = push_dir(q_circ)

        if V_prev is not None:
            V_warm = np.vstack([V_prev[1:], [pdir * STICK_SPEED]])
        else:
            V_warm = np.tile(pdir * STICK_SPEED, (N_HORIZON, 1)).astype(float)

        Z_opt  = admm_solve(q_circ, stick_pos, V_warm)
        V_prev = Z_opt.copy()

        plan_q, plan_u = rollout(q_circ, stick_pos, Z_opt)

        V_last    = Z_opt[0].copy()
        stick_pos = clamp_puck(stick_pos + h * V_last)

        if np.linalg.norm(q_circ - Q_GOAL) < GOAL_THRESH:
            phase  = "return"
            V_prev = None

    # ── Return phase ──────────────────────────────────────────────────
    elif phase == "return":
        err  = PUCK_RETURN - stick_pos
        dist = np.linalg.norm(err)
        if dist > 1e-8:
            spd       = min(STICK_SPEED, KP_RETURN * dist)
            stick_vel = err / dist * spd
        else:
            stick_vel = np.zeros(2)
        stick_pos = clamp_puck(stick_pos + h * stick_vel)
        if dist < RETURN_THRESH:
            phase = "done"

    # ── Real physics step ──────────────────────────────────────────────
    stick_vel_real = (stick_pos - stick_prev) / h
    contacts       = detect_contacts(q_circ, stick_pos, stick_vel_real)
    q_circ         = q_circ + h * dsdf_step(contacts)

    # ══════════════════════════════════════════════════════════════════
    #  RENDER
    # ══════════════════════════════════════════════════════════════════
    screen.fill((210, 230, 250))
    pygame.draw.rect(screen, (50, 80, 130),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    # ── Planned horizon ghosts ────────────────────────────────────────
    if plan_q is not None and phase == "admm":
        overlay.fill((0, 0, 0, 0))
        N = len(plan_q) - 1

        pts_q = [plan_q[i].astype(int).tolist() for i in range(N+1)]
        pts_u = [plan_u[i].astype(int).tolist() for i in range(N+1)]
        if N > 1:
            pygame.draw.lines(overlay, (60, 200, 100, 55), False, pts_q, 1)
            pygame.draw.lines(overlay, (90, 140, 240, 45), False, pts_u, 1)

        for i in range(1, N+1):
            t     = i / N
            alpha = int(210 * (1.0 - t))

            r_c = max(4, int(CIRC_R * (0.55 + 0.45 * (1.0 - t))))
            try:
                pygame.draw.circle(overlay, (55, 210, 110, alpha),
                                   plan_q[i].astype(int).tolist(), r_c, 2)
            except Exception:
                pass

            r_p = max(3, int(STICK_R * (0.55 + 0.45 * (1.0 - t))))
            try:
                pygame.draw.circle(overlay, (100, 150, 240, alpha),
                                   plan_u[i].astype(int).tolist(), r_p, 1)
            except Exception:
                pass

        screen.blit(overlay, (0, 0))

    # ── Goal ghost ────────────────────────────────────────────────────
    gx, gy = int(Q_GOAL[0]), int(Q_GOAL[1])
    screen.blit(goal_surf, (gx - _R - 4, gy - _R - 4))
    pygame.draw.line(screen, (220, 55, 55), (gx-14, gy), (gx+14, gy), 2)
    pygame.draw.line(screen, (220, 55, 55), (gx, gy-14), (gx, gy+14), 2)

    # ── Obstacles ─────────────────────────────────────────────────────
    for idx, (obs_c, obs_r) in enumerate(zip(OBS_CENTERS, OBS_RADII)):
        oc = (int(obs_c[0]), int(obs_c[1]))
        r  = int(obs_r)
        pygame.draw.circle(screen, (190, 75, 75), oc, r)
        rim_col = (255, 220, 100) if idx == dragging_obs else (110, 25, 25)
        pygame.draw.circle(screen, rim_col, oc, r, 3)
        # Circle clearance ring (CIRC_R + obs_r + CIRC_OBS_MARGIN)
        pygame.draw.circle(screen, (215, 110, 110),
                           oc, int(obs_r + CIRC_R + CIRC_OBS_MARGIN), 1)
        # Puck clearance ring (STICK_R + obs_r + OBS_MARGIN)
        pygame.draw.circle(screen, (160, 160, 230),
                           oc, int(obs_r + STICK_R + OBS_MARGIN), 1)
        lbl = font.render(f"obs{idx+1}", True, (255, 230, 200))
        screen.blit(lbl, (oc[0] - 14, oc[1] - 7))

    # ── Puck home marker ──────────────────────────────────────────────
    rx, ry = int(PUCK_RETURN[0]), int(PUCK_RETURN[1])
    pygame.draw.circle(screen, (100, 110, 200), (rx, ry), int(STICK_R), 2)

    # ── Push-point crosshair ──────────────────────────────────────────
    if phase == "admm":
        pp = push_pos(q_circ).astype(int)
        pygame.draw.line(screen, (180, 140, 60),
                         (pp[0]-11, pp[1]), (pp[0]+11, pp[1]), 2)
        pygame.draw.line(screen, (180, 140, 60),
                         (pp[0], pp[1]-11), (pp[0], pp[1]+11), 2)

    # ── Applied velocity arrow ────────────────────────────────────────
    if phase == "admm" and V_last is not None:
        vmag = np.linalg.norm(V_last)
        if vmag > 1.0:
            vdir = V_last / vmag
            sc   = stick_pos.astype(int)
            tip  = (sc + vdir * 30).astype(int)
            pygame.draw.line(screen, (255, 200, 50), tuple(sc), tuple(tip), 3)

    # ── Circle ────────────────────────────────────────────────────────
    ci = q_circ.astype(int)
    pygame.draw.circle(screen, (60, 180, 100), tuple(ci), _R)
    pygame.draw.circle(screen, (20, 100,  50), tuple(ci), _R, 2)
    pygame.draw.circle(screen, (20, 100,  50), tuple(ci), 4)

    # ── Puck ──────────────────────────────────────────────────────────
    puck_col = {"admm":   (50,  90, 210),
                "return": (190, 120,  40),
                "done":   (110, 110, 110)}.get(phase, (50, 90, 210))
    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, puck_col,      tuple(sc), int(STICK_R))
    pygame.draw.circle(screen, (20, 40, 130), tuple(sc), int(STICK_R), 2)

    # ── HUD ───────────────────────────────────────────────────────────
    dist_goal = np.linalg.norm(q_circ - Q_GOAL)
    sv_mag    = np.linalg.norm(stick_vel_real)
    vL_str = (f"({V_last[0]:.0f},{V_last[1]:.0f})"
              f"  |{np.linalg.norm(V_last):.0f}|"
              if V_last is not None else "—")

    # Compute effective APPROACH_W for HUD display
    if dist_goal >= APPROACH_FADE_START:
        aw_disp = APPROACH_W
    elif dist_goal <= APPROACH_FADE_END:
        aw_disp = 0.0
    else:
        aw_disp = APPROACH_W * (dist_goal - APPROACH_FADE_END) / (APPROACH_FADE_START - APPROACH_FADE_END)

    for i, txt in enumerate([
        "ADMM-MPC  |  ContactSDF v4  |  obstacle-aware",
        "Yang & Jin, IEEE RA-L 2024  |  FIX: circ obs cost + approach fade",
        f"Phase         : {phase}",
        f"Circle        : ({q_circ[0]:.0f},{q_circ[1]:.0f})"
        f"  Δgoal = {dist_goal:.1f} px  (thresh {GOAL_THRESH:.0f})",
        f"Puck          : ({stick_pos[0]:.0f},{stick_pos[1]:.0f})",
        f"stick_vel mag : {sv_mag:.0f} px/s  (max {STICK_SPEED:.0f})",
        f"V_opt[0]      : {vL_str} px/s",
        f"Contacts      : {len(contacts)}",
        f"N={N_HORIZON}  L={ADMM_ITERS}  α={ALPHA_V}"
        f"  ρ={RHO}  FD={FD_EPS}",
        f"APPROACH_W    : {aw_disp:.3f}  (fades {APPROACH_FADE_START:.0f}→{APPROACH_FADE_END:.0f} px)",
        f"CIRC_OBS_W    : {CIRC_OBS_W:.0f}  (NEW: circle path avoidance)",
        "Drag red circles to reposition obstacles",
    ]):
        screen.blit(font.render(txt, True, (20, 40, 90)), (10, 10 + i*17))

    # ── Legend ────────────────────────────────────────────────────────
    for i, (lbl, col) in enumerate([
        ("● circle",      (60,  180, 100)),
        ("◎ goal",        (220,  55,  55)),
        ("● puck",        puck_col),
        ("● obstacle",    (190,  75,  75)),
        ("○ home",        (100, 110, 200)),
        ("+ push pt",     (180, 140,  60)),
        ("→ V_opt",       (255, 200,  50)),
        ("○ plan-circle", (55,  210, 110)),
        ("○ plan-puck",   (100, 150, 240)),
        ("○ circ clear",  (215, 110, 110)),
        ("○ puck clear",  (160, 160, 230)),
    ]):
        screen.blit(font.render(lbl, True, col), (WIDTH - 130, 10 + i*17))

    pygame.display.flip()

    # Capture frame for video
    _frame = pygame.surfarray.array3d(screen)          # (W, H, 3) RGB
    _frame = np.transpose(_frame, (1, 0, 2))           # → (H, W, 3)
    _video.write(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR))

    clock.tick(30)