"""
Exact implementation of:
  Anitescu, M. "Optimization-based simulation of nonsmooth rigid multibody dynamics."
  Math. Program. 105, 113-143 (2006).  [ANL Preprint P1161]

TOP-DOWN (tabletop) VIEW  -  two bodies: a square box and a circular disk.
Both are driven by a kinematic puck (WASD).

MULTI-BODY QP
  Combined generalised velocity: v_all = [vx_box, vy_box, omega_box, vx_circ, vy_circ]
  Combined mass matrix: M_all = diag(m_box, m_box, I_box, m_circ, m_circ)

  For each contact the Jacobian row spans the full 5-DOF state:
    kinematic obstacle -> single body:  row has zeros for the other body
    box <-> circle:                     row has -J_box | +J_circ

  min_{v+}  0.5 (v+)^T M_all v+  +  kbar^T v+
  kbar = -M_all v^l - h f^l

  subject to, for each active contact j:
    (Jn^j +/- mu Jt^j) @ v+  +  gamma * phi^j / h  >= 0
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

h           = 1 / 120
mu_stick    = 0.3     # puck-body friction
mu_wall     = 0.1     # wall-body friction
mu_bb       = 0.2     # box-circle friction

# --- Box ---
mass_box = 2.0 * 2.0
size     = 80.0
I_box    = (1.0 / 6.0) * mass_box * size ** 2
B_LIN_BOX = mass_box * 2.0
B_ROT_BOX = I_box    * 2.0

# --- Circle / disk ---
mass_circ  = 1.5
CIRC_R     = 35.0
# disk moment of inertia: 0.5 m r^2 (not used in DOF but kept for reference)
B_LIN_CIRC = mass_circ * 2.0

# Combined mass matrix  (5 DOF: box x,y,theta  + circle x,y)
M_all = np.diag([mass_box, mass_box, I_box, mass_circ, mass_circ])
Minv  = np.linalg.inv(M_all)

EPS_HAT = 2.0
GAMMA   = 0.2

# Puck
STICK_R     = 12.0
STICK_SPEED = 350.0

# =====================================================
# Initial state
# =====================================================
# Box
q_box = np.array([WIDTH / 2.0 - 120.0, HEIGHT / 2.0, 0.0])
v_box = np.zeros(3)

# Circle
q_circ = np.array([WIDTH / 2.0 + 130.0, HEIGHT / 2.0])
v_circ = np.zeros(2)

# Puck
stick_pos = np.array([WIDTH / 2.0 - 300.0, HEIGHT / 2.0])

# =====================================================
# Pygame
# =====================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Anitescu 2006 - top-down, two bodies")
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
    corners = np.array([[-half, -half], [half, -half],
                         [half,  half], [-half,  half]])
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
# =====================================================
# Full system DOF layout:
#   indices 0,1,2  -> box  (vx, vy, omega)
#   indices 3,4    -> circle (vx, vy)
#
# Jn/Jt are built as 5-vectors with zeros in the unused slots.

def jac_single_box(n, r):
    """Jacobian rows for a contact on the BOX only (circle slots = 0)."""
    t  = np.array([-n[1], n[0]])
    Jn = np.array([n[0], n[1],  r[0]*n[1] - r[1]*n[0],  0.0, 0.0])
    Jt = np.array([t[0], t[1],  r[0]*t[1] - r[1]*t[0],  0.0, 0.0])
    return Jn, Jt


def jac_single_circ(n):
    """Jacobian rows for a contact on the CIRCLE only (box slots = 0).
    Circle has no rotation DOF, so contact velocity = CM velocity."""
    t  = np.array([-n[1], n[0]])
    Jn = np.array([0.0, 0.0, 0.0,  n[0], n[1]])
    Jt = np.array([0.0, 0.0, 0.0,  t[0], t[1]])
    return Jn, Jt


def jac_box_vs_circ(n, r_box):
    """
    Jacobian for box-circle contact.
    n points from box surface TOWARD circle (outward from box).
    Non-penetration:  n^T v_circ_contact  -  n^T v_box_contact  >= -phi/h
    => Jn_all = [-Jn_box | +Jn_circ]
    """
    t = np.array([-n[1], n[0]])
    # box part (negated because box is the obstacle from circle's perspective)
    Jn_box_part = np.array([n[0], n[1],  r_box[0]*n[1] - r_box[1]*n[0]])
    Jt_box_part = np.array([t[0], t[1],  r_box[0]*t[1] - r_box[1]*t[0]])
    Jn = np.concatenate([-Jn_box_part, [n[0], n[1]]])
    Jt = np.concatenate([-Jt_box_part, [t[0], t[1]]])
    return Jn, Jt


# =====================================================
# Contact detection
# =====================================================

def detect_contacts(q_box, q_circ, stick_pos):
    """
    Returns list of (phi, Jn, Jt, mu_c).
    Jacobians are already 5-vectors over the full combined state.
    """
    contacts = []
    verts = get_box_vertices(q_box)

    # ---- Walls vs Box (per vertex) -----------------------------------------
    for vtx in verts:
        r = vtx - q_box[:2]

        phi = vtx[0] - MARGIN
        if phi <= EPS_HAT:
            Jn, Jt = jac_single_box(np.array([1.0, 0.0]), r)
            contacts.append((phi, Jn, Jt, mu_wall))

        phi = (WIDTH - MARGIN) - vtx[0]
        if phi <= EPS_HAT:
            Jn, Jt = jac_single_box(np.array([-1.0, 0.0]), r)
            contacts.append((phi, Jn, Jt, mu_wall))

        phi = vtx[1] - MARGIN
        if phi <= EPS_HAT:
            Jn, Jt = jac_single_box(np.array([0.0, 1.0]), r)
            contacts.append((phi, Jn, Jt, mu_wall))

        phi = (HEIGHT - MARGIN) - vtx[1]
        if phi <= EPS_HAT:
            Jn, Jt = jac_single_box(np.array([0.0, -1.0]), r)
            contacts.append((phi, Jn, Jt, mu_wall))

    # ---- Walls vs Circle (per face) ----------------------------------------
    cx, cy = q_circ

    phi = cx - CIRC_R - MARGIN
    if phi <= EPS_HAT:
        Jn, Jt = jac_single_circ(np.array([1.0, 0.0]))
        contacts.append((phi, Jn, Jt, mu_wall))

    phi = (WIDTH - MARGIN) - (cx + CIRC_R)
    if phi <= EPS_HAT:
        Jn, Jt = jac_single_circ(np.array([-1.0, 0.0]))
        contacts.append((phi, Jn, Jt, mu_wall))

    phi = cy - CIRC_R - MARGIN
    if phi <= EPS_HAT:
        Jn, Jt = jac_single_circ(np.array([0.0, 1.0]))
        contacts.append((phi, Jn, Jt, mu_wall))

    phi = (HEIGHT - MARGIN) - (cy + CIRC_R)
    if phi <= EPS_HAT:
        Jn, Jt = jac_single_circ(np.array([0.0, -1.0]))
        contacts.append((phi, Jn, Jt, mu_wall))

    # ---- Puck vs Box (per edge, closest point) -----------------------------
    edge_pairs = [(0,1), (1,2), (2,3), (3,0)]
    for i, j in edge_pairs:
        closest, _ = closest_point_on_segment(stick_pos, verts[i], verts[j])
        delta = closest - stick_pos
        dist  = np.linalg.norm(delta)
        phi   = dist - STICK_R
        if phi <= EPS_HAT and dist > 1e-6:
            n   = delta / dist
            r   = closest - q_box[:2]
            Jn, Jt = jac_single_box(n, r)
            contacts.append((phi, Jn, Jt, mu_stick))

    # ---- Puck vs Circle (circle-circle) ------------------------------------
    delta = q_circ - stick_pos
    dist  = np.linalg.norm(delta)
    phi   = dist - STICK_R - CIRC_R
    if phi <= EPS_HAT and dist > 1e-6:
        n = delta / dist       # outward: puck_centre -> circle_centre
        Jn, Jt = jac_single_circ(n)
        contacts.append((phi, Jn, Jt, mu_stick))

    # ---- Box vs Circle -----------------------------------------------------
    # Closest point on each box edge to the circle centre.
    for i, j in edge_pairs:
        closest, _ = closest_point_on_segment(q_circ, verts[i], verts[j])
        delta = q_circ - closest
        dist  = np.linalg.norm(delta)
        phi   = dist - CIRC_R
        if phi <= EPS_HAT and dist > 1e-6:
            n = delta / dist   # outward from box edge -> circle centre
            r = closest - q_box[:2]
            Jn, Jt = jac_box_vs_circ(n, r)
            contacts.append((phi, Jn, Jt, mu_bb))

    return contacts


# =====================================================
# One-step multi-body QP
# =====================================================

def step(q_box, v_box, q_circ, v_circ, contacts):
    # Stack combined velocity and forces
    v_all = np.concatenate([v_box, v_circ])

    f_box  = np.array([-B_LIN_BOX  * v_box[0],
                       -B_LIN_BOX  * v_box[1],
                       -B_ROT_BOX  * v_box[2]])
    f_circ = np.array([-B_LIN_CIRC * v_circ[0],
                       -B_LIN_CIRC * v_circ[1]])
    f_all  = np.concatenate([f_box, f_circ])

    kbar = -M_all @ v_all - h * f_all

    if len(contacts) == 0:
        v_next = v_all + h * (Minv @ f_all)
    else:
        A_rows = []
        b_vals = []
        for phi, Jn, Jt, mu_c in contacts:
            gap = GAMMA * phi / h
            A_rows.append(Jn + mu_c * Jt)
            b_vals.append(-gap)
            A_rows.append(Jn - mu_c * Jt)
            b_vals.append(-gap)

        A_ineq = np.vstack(A_rows)
        b_ineq = np.array(b_vals)

        v_var = cp.Variable(5)
        prob  = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(v_var, M_all) + kbar @ v_var),
            [A_ineq @ v_var >= b_ineq]
        )
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate"):
            v_next = v_var.value
        else:
            v_next = v_all + h * (Minv @ f_all)

    v_box_next  = v_next[:3]
    v_circ_next = v_next[3:]

    q_box_next  = q_box  + h * v_box_next
    q_circ_next = q_circ + h * v_circ_next

    return q_box_next, v_box_next, q_circ_next, v_circ_next


# =====================================================
# Main loop
# =====================================================
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]: stick_pos[0] -= STICK_SPEED * h
    if keys[pygame.K_d]: stick_pos[0] += STICK_SPEED * h
    if keys[pygame.K_w]: stick_pos[1] -= STICK_SPEED * h
    if keys[pygame.K_s]: stick_pos[1] += STICK_SPEED * h

    stick_pos[0] = np.clip(stick_pos[0], MARGIN + STICK_R, WIDTH  - MARGIN - STICK_R)
    stick_pos[1] = np.clip(stick_pos[1], MARGIN + STICK_R, HEIGHT - MARGIN - STICK_R)

    contacts = detect_contacts(q_box, q_circ, stick_pos)
    q_box, v_box, q_circ, v_circ = step(q_box, v_box, q_circ, v_circ, contacts)

    # --- Rendering ----------------------------------------------------------
    screen.fill((245, 235, 210))

    # Table border
    pygame.draw.rect(screen, (100, 70, 40),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    # Box
    verts_draw = get_box_vertices(q_box).astype(int).tolist()
    pygame.draw.polygon(screen, (210, 80, 60), verts_draw)
    pygame.draw.polygon(screen, (110, 30, 20), verts_draw, 2)
    # direction marker
    R   = rotation_matrix(q_box[2])
    tip = (q_box[:2] + R @ np.array([size * 0.38, 0])).astype(int)
    pygame.draw.line(screen, (255,255,255), tuple(q_box[:2].astype(int)), tuple(tip), 3)

    # Circle
    ci = q_circ.astype(int)
    pygame.draw.circle(screen, (60, 180, 100), ci, int(CIRC_R))
    pygame.draw.circle(screen, (20, 100,  50), ci, int(CIRC_R), 2)
    # small dot at centre so you can see it's moved
    pygame.draw.circle(screen, (20, 100,  50), ci, 4)

    # Puck
    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (50, 90, 210), sc, int(STICK_R))
    pygame.draw.circle(screen, (20, 40, 130), sc, int(STICK_R), 2)

    # HUD
    nc = len(contacts)
    hud = [
        "Anitescu 2006 - top-down  |  red=box  green=circle  blue=puck",
        f"h={h:.4f}  mu_stick={mu_stick}  mu_bb={mu_bb}  gamma={GAMMA}",
        f"box  pos=({q_box[0]:.0f},{q_box[1]:.0f})  theta={float(np.degrees(q_box[2])):.1f}deg"
        f"  spd={float(np.hypot(v_box[0],v_box[1])):.1f}",
        f"circ pos=({q_circ[0]:.0f},{q_circ[1]:.0f})"
        f"  spd={float(np.hypot(v_circ[0],v_circ[1])):.1f}",
        f"active contacts: {nc}",
        "WASD = move puck",
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (50, 30, 10))
        screen.blit(surf, (10, 10 + i * 17))

    pygame.display.flip()
    clock.tick(120)
