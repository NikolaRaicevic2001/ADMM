"""
Quasistatic simulator inspired by:
  Pang, T. et al. "A Convex Quasistatic Time-stepping Scheme for Rigid Multibody
  Systems with Contact and Friction." ICRA 2021.

TOP-DOWN tabletop view. Two bodies (box + circle) driven by a kinematic puck.

NON-PENETRATION CONSTRAINT (correct form)
  Gap function:  phi = dist(body, puck) - STICK_R
  Gap rate:      dphi/dt = n^T * v_body_contact - n^T * v_puck
  Constraint:    dphi/dt + gamma*phi/h >= 0
  =>  Jn @ v_body  >=  n^T * v_puck  -  gamma*phi/h
                       ^^^^^^^^^^^^
                       This term was missing before -- caused puck to sink in.

For wall contacts v_wall = 0, so the term vanishes and nothing changes there.
"""

import pygame
import numpy as np
import cvxpy as cp
import sys

# =====================================================
# Simulation parameters
# =====================================================
WIDTH, HEIGHT = 900, 600
MARGIN        = 40

h           = 1 / 720
mu_stick    = 0.3
mu_wall     = 0.1
mu_bb       = 0.2

# Box
size    = 80.0
# Circle
CIRC_R  = 35.0

# Regularisation matrix E
E_BOX_LIN = 8.0
E_BOX_ROT = 8.0
E_CIRC    = 8.0
E_all = np.diag([E_BOX_LIN, E_BOX_LIN, E_BOX_ROT, E_CIRC, E_CIRC])

EPS_HAT = 2.0
GAMMA   = 0.75

# Puck
STICK_R     = 12.0
STICK_SPEED = 350.0

# =====================================================
# Initial state  (positions only -- no velocity state)
# =====================================================
q_box  = np.array([WIDTH / 2.0 - 120.0, HEIGHT / 2.0, 0.0])
q_circ = np.array([WIDTH / 2.0 + 130.0, HEIGHT / 2.0])

stick_pos      = np.array([WIDTH / 2.0 - 300.0, HEIGHT / 2.0])
stick_pos_prev = stick_pos.copy()   # track velocity of puck

# =====================================================
# Pygame
# =====================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quasistatic (Pang 2021) - top-down tabletop")
clock = pygame.time.Clock()
font  = pygame.font.SysFont("monospace", 14)


# =====================================================
# Geometry helpers
# =====================================================

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


# =====================================================
# Jacobian helpers
# DOF layout: [vx_box, vy_box, omega_box, vx_circ, vy_circ]
# =====================================================

def jac_single_box(n, r):
    t  = np.array([-n[1], n[0]])
    Jn = np.array([n[0], n[1],  r[0]*n[1]-r[1]*n[0],  0.0, 0.0])
    Jt = np.array([t[0], t[1],  r[0]*t[1]-r[1]*t[0],  0.0, 0.0])
    return Jn, Jt

def jac_single_circ(n):
    t  = np.array([-n[1], n[0]])
    Jn = np.array([0.0, 0.0, 0.0,  n[0], n[1]])
    Jt = np.array([0.0, 0.0, 0.0,  t[0], t[1]])
    return Jn, Jt

def jac_box_vs_circ(n, r_box):
    t      = np.array([-n[1], n[0]])
    Jn_box = np.array([n[0], n[1],  r_box[0]*n[1]-r_box[1]*n[0]])
    Jt_box = np.array([t[0], t[1],  r_box[0]*t[1]-r_box[1]*t[0]])
    Jn = np.concatenate([-Jn_box, [n[0], n[1]]])
    Jt = np.concatenate([-Jt_box, [t[0], t[1]]])
    return Jn, Jt


# =====================================================
# Contact detection
# Contact tuple: (phi, Jn, Jt, mu_c, b_offset)
#   b_offset = n^T @ v_obstacle  (puck velocity projected onto normal)
#            = 0 for walls and body-body contacts (obstacles are stationary)
# =====================================================

def detect_contacts(q_box, q_circ, stick_pos, stick_vel):
    """
    stick_vel : current puck velocity (2,), computed from finite difference.
    Returns list of (phi, Jn, Jt, mu_c, b_offset).
    """
    contacts  = []
    verts     = get_box_vertices(q_box)
    edge_pairs = [(0,1),(1,2),(2,3),(3,0)]

    # ---- Walls vs Box (per vertex, b_offset=0) -----------------------------
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

    # ---- Walls vs Circle (per face, b_offset=0) ----------------------------
    cx, cy = q_circ
    for phi_val, n_vec in [
        (cx - CIRC_R - MARGIN,              np.array([ 1.0,  0.0])),
        ((WIDTH -MARGIN) - (cx+CIRC_R),     np.array([-1.0,  0.0])),
        (cy - CIRC_R - MARGIN,              np.array([ 0.0,  1.0])),
        ((HEIGHT-MARGIN) - (cy+CIRC_R),     np.array([ 0.0, -1.0])),
    ]:
        if phi_val <= EPS_HAT:
            Jn, Jt = jac_single_circ(n_vec)
            contacts.append((phi_val, Jn, Jt, mu_wall, 0.0))

    # ---- Puck vs Box (per edge, b_offset = n @ stick_vel) ------------------
    # The puck is a moving obstacle so we must account for its velocity.
    # Correct constraint: Jn @ v_box >= n^T*v_puck - gamma*phi/h
    for i, j in edge_pairs:
        closest, _ = closest_point_on_segment(stick_pos, verts[i], verts[j])
        delta = closest - stick_pos
        dist  = np.linalg.norm(delta)
        phi   = dist - STICK_R
        if phi <= EPS_HAT and dist > 1e-6:
            n           = delta / dist
            b_offset    = float(n @ stick_vel)   # puck approach velocity
            Jn, Jt      = jac_single_box(n, closest - q_box[:2])
            contacts.append((phi, Jn, Jt, mu_stick, b_offset))

    # ---- Puck vs Circle (circle-circle, b_offset = n @ stick_vel) ----------
    delta = q_circ - stick_pos
    dist  = np.linalg.norm(delta)
    phi   = dist - STICK_R - CIRC_R
    if phi <= EPS_HAT and dist > 1e-6:
        n        = delta / dist
        b_offset = float(n @ stick_vel)          # puck approach velocity
        Jn, Jt   = jac_single_circ(n)
        contacts.append((phi, Jn, Jt, mu_stick, b_offset))

    # ---- Box vs Circle (per edge, b_offset=0) ------------------------------
    for i, j in edge_pairs:
        closest, _ = closest_point_on_segment(q_circ, verts[i], verts[j])
        delta = q_circ - closest
        dist  = np.linalg.norm(delta)
        phi   = dist - CIRC_R
        if phi <= EPS_HAT and dist > 1e-6:
            n = delta / dist
            Jn, Jt = jac_box_vs_circ(n, closest - q_box[:2])
            contacts.append((phi, Jn, Jt, mu_bb, 0.0))

    return contacts


# =====================================================
# One-step quasistatic QP
# =====================================================

def step(q_box, q_circ, contacts):
    """
    min  0.5 v^T E v
    s.t. (Jn +/- mu Jt) @ v  >=  b_offset - gamma*phi/h

    b_offset = n^T @ v_puck for puck contacts, 0 otherwise.
    Unconstrained minimum: v = 0 (bodies at rest).
    Constraints push v != 0 only when obstacles are in contact.
    """
    if len(contacts) == 0:
        return q_box.copy(), q_circ.copy()

    A_rows = []
    b_vals = []
    for phi, Jn, Jt, mu_c, b_offset in contacts:
        gap  = GAMMA * phi / h
        rhs  = b_offset - gap        # b_offset corrects for moving puck
        A_rows.append(Jn + mu_c * Jt);  b_vals.append(rhs)
        A_rows.append(Jn - mu_c * Jt);  b_vals.append(rhs)

    A_ineq = np.vstack(A_rows)
    b_ineq = np.array(b_vals)

    v_var = cp.Variable(5)
    prob  = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(v_var, E_all)),
        [A_ineq @ v_var >= b_ineq]
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        v_sol = v_var.value
    else:
        v_sol = np.zeros(5)

    return q_box + h * v_sol[:3], q_circ + h * v_sol[3:]


# =====================================================
# Main loop
# =====================================================
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

    # Finite-difference puck velocity (used in contact RHS)
    stick_vel = (stick_pos - stick_pos_prev) / h

    contacts      = detect_contacts(q_box, q_circ, stick_pos, stick_vel)
    q_box, q_circ = step(q_box, q_circ, contacts)

    # --- Rendering ----------------------------------------------------------
    screen.fill((225, 245, 215))

    pygame.draw.rect(screen, (60, 110, 60),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    # Box
    verts_draw = get_box_vertices(q_box).astype(int).tolist()
    pygame.draw.polygon(screen, (210, 80, 60), verts_draw)
    pygame.draw.polygon(screen, (110, 30, 20), verts_draw, 2)
    R   = rotation_matrix(q_box[2])
    tip = (q_box[:2] + R @ np.array([size*0.38, 0])).astype(int)
    pygame.draw.line(screen, (255,255,255), tuple(q_box[:2].astype(int)), tuple(tip), 3)

    # Circle
    ci = q_circ.astype(int)
    pygame.draw.circle(screen, (60, 180, 100), ci, int(CIRC_R))
    pygame.draw.circle(screen, (20, 100,  50), ci, int(CIRC_R), 2)
    pygame.draw.circle(screen, (20, 100,  50), ci, 4)

    # Puck
    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (50, 90, 210), sc, int(STICK_R))
    pygame.draw.circle(screen, (20, 40, 130), sc, int(STICK_R), 2)

    # HUD
    nc = len(contacts)
    hud = [
        "QUASISTATIC (Pang 2021) - top-down",
        "No inertia: bodies stop instantly when puck stops",
        f"h={h:.4f}  mu_stick={mu_stick}  mu_bb={mu_bb}  gamma={GAMMA}",
        f"box  pos=({q_box[0]:.0f},{q_box[1]:.0f})  theta={float(np.degrees(q_box[2])):.1f}deg",
        f"circ pos=({q_circ[0]:.0f},{q_circ[1]:.0f})",
        f"active contacts: {nc}",
        "WASD = move puck",
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (20, 60, 20))
        screen.blit(surf, (10, 10 + i * 17))

    pygame.display.flip()
    clock.tick(120)
