"""
Pang, T. et al. "A Convex Quasistatic Time-stepping Scheme for Rigid Multibody
Systems with Contact and Friction." ICRA 2021.

Top-down planar view. Single box driven by a kinematic puck (WASD). No inertia:
objective is min 0.5 v^T E v (regularisation), not a mass matrix — bodies stop
instantly when puck stops. Moving puck velocity enters the constraint RHS as b_offset.
"""

import pygame
import numpy as np
import cvxpy as cp
import sys

# ── Simulation parameters ─────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 600
MARGIN        = 40

h         = 1 / 720
mu_stick  = 0.3
mu_wall   = 0.1

size      = 80.0

# Regularisation matrix E (replaces mass matrix; controls compliance)
E_BOX_LIN = 8.0
E_BOX_ROT = 8.0
E_all     = np.diag([E_BOX_LIN, E_BOX_LIN, E_BOX_ROT])

EPS_HAT = 2.0
GAMMA   = 0.75

STICK_R     = 12.0
STICK_SPEED = 350.0

# ── Initial state ─────────────────────────────────────────────────────────────
q_box          = np.array([WIDTH / 2.0 - 120.0, HEIGHT / 2.0, 0.0])
stick_pos      = np.array([WIDTH / 2.0 - 300.0, HEIGHT / 2.0])
stick_pos_prev = stick_pos.copy()

# ── Pygame ────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quasistatic (Pang 2021) - top-down tabletop")
clock = pygame.time.Clock()
font  = pygame.font.SysFont("monospace", 14)

print(f"[contact_quasistatic_planar] h={h:.5f}  mu_stick={mu_stick}  gamma={GAMMA}  EPS_HAT={EPS_HAT}px")


# ── Geometry ──────────────────────────────────────────────────────────────────

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def get_box_vertices(q_box):
    R    = rotation_matrix(q_box[2])
    half = size / 2.0
    corners = np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    return (R @ corners.T).T + q_box[:2]

def closest_point_on_segment(p, a, b):
    ab    = b - a
    denom = ab @ ab
    if denom < 1e-12:
        return a.copy(), 0.0
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return a + t * ab, t


# ── 3-DOF Jacobians [vx_box, vy_box, omega_box] ───────────────────────────────

def jac_single_box(n, r):
    t  = np.array([-n[1], n[0]])
    Jn = np.array([n[0], n[1],  r[0]*n[1]-r[1]*n[0]])
    Jt = np.array([t[0], t[1],  r[0]*t[1]-r[1]*t[0]])
    return Jn, Jt


# ── Contact detection: walls×box (b_offset=0), puck×box (b_offset = n·v_puck) ─

def detect_contacts(q_box, stick_pos, stick_vel):
    contacts   = []
    verts      = get_box_vertices(q_box)
    edge_pairs = [(0,1),(1,2),(2,3),(3,0)]

    for vtx in verts:
        r = vtx - q_box[:2]
        for phi_val, n_vec in [
            (vtx[0] - MARGIN,               np.array([ 1.0,  0.0])),
            ((WIDTH -MARGIN) - vtx[0],      np.array([-1.0,  0.0])),
            (vtx[1] - MARGIN,               np.array([ 0.0,  1.0])),
            ((HEIGHT-MARGIN) - vtx[1],      np.array([ 0.0, -1.0])),
        ]:
            if phi_val <= EPS_HAT:
                Jn, Jt = jac_single_box(n_vec, r)
                contacts.append((phi_val, Jn, Jt, mu_wall, 0.0))

    for i, j in edge_pairs:
        closest, _ = closest_point_on_segment(stick_pos, verts[i], verts[j])
        delta = closest - stick_pos
        dist  = np.linalg.norm(delta)
        phi   = dist - STICK_R
        if phi <= EPS_HAT and dist > 1e-6:
            n           = delta / dist
            b_offset    = float(n @ stick_vel)
            Jn, Jt      = jac_single_box(n, closest - q_box[:2])
            contacts.append((phi, Jn, Jt, mu_stick, b_offset))

    return contacts


# ── Quasistatic QP: min 0.5 v^T E v  s.t. (Jn +/- mu Jt) v >= b_offset - gamma*phi/h

def step(q_box, contacts):
    if len(contacts) == 0:
        return q_box.copy()

    A_rows = []
    b_vals = []
    for phi, Jn, Jt, mu_c, b_offset in contacts:
        gap  = GAMMA * phi / h
        rhs  = b_offset - gap
        A_rows.append(Jn + mu_c * Jt);  b_vals.append(rhs)
        A_rows.append(Jn - mu_c * Jt);  b_vals.append(rhs)

    A_ineq = np.vstack(A_rows)
    b_ineq = np.array(b_vals)

    v_var = cp.Variable(3)
    prob  = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(v_var, E_all)),
        [A_ineq @ v_var >= b_ineq]
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        v_sol = v_var.value
    else:
        print(f"[WARN] QP status={prob.status}  returning zero velocity")
        v_sol = np.zeros(3)

    return q_box + h * v_sol


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    stick_pos_prev = stick_pos.copy()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]: stick_pos[0] -= STICK_SPEED * h
    if keys[pygame.K_d]: stick_pos[0] += STICK_SPEED * h
    if keys[pygame.K_w]: stick_pos[1] -= STICK_SPEED * h
    if keys[pygame.K_s]: stick_pos[1] += STICK_SPEED * h

    stick_pos[0] = np.clip(stick_pos[0], MARGIN+STICK_R, WIDTH -MARGIN-STICK_R)
    stick_pos[1] = np.clip(stick_pos[1], MARGIN+STICK_R, HEIGHT-MARGIN-STICK_R)

    # Finite-difference puck velocity for constraint RHS
    stick_vel = (stick_pos - stick_pos_prev) / h

    contacts = detect_contacts(q_box, stick_pos, stick_vel)
    q_box    = step(q_box, contacts)

    # ── Render ────────────────────────────────────────────────────────────────
    screen.fill((225, 245, 215))

    pygame.draw.rect(screen, (60, 110, 60),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    verts_draw = get_box_vertices(q_box).astype(int).tolist()
    pygame.draw.polygon(screen, (210, 80, 60), verts_draw)
    pygame.draw.polygon(screen, (110, 30, 20), verts_draw, 2)
    R   = rotation_matrix(q_box[2])
    tip = (q_box[:2] + R @ np.array([size*0.38, 0])).astype(int)
    pygame.draw.line(screen, (255,255,255), tuple(q_box[:2].astype(int)), tuple(tip), 3)

    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (50, 90, 210), sc, int(STICK_R))
    pygame.draw.circle(screen, (20, 40, 130), sc, int(STICK_R), 2)

    nc = len(contacts)
    hud = [
        "QUASISTATIC (Pang 2021) - top-down",
        "No inertia: bodies stop instantly when puck stops",
        f"h={h:.4f}  mu_stick={mu_stick}  gamma={GAMMA}",
        f"box  pos=({q_box[0]:.0f},{q_box[1]:.0f})  theta={float(np.degrees(q_box[2])):.1f}deg",
        f"active contacts: {nc}",
        "WASD = move puck",
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (20, 60, 20))
        screen.blit(surf, (10, 10 + i * 17))

    pygame.display.flip()
    clock.tick(120)
