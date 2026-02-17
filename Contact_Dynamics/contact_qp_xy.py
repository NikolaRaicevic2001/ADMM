"""
Exact implementation of:
  Anitescu, M. "Optimization-based simulation of nonsmooth rigid multibody dynamics."
  Math. Program. 105, 113-143 (2006).  [ANL Preprint P1161]

TOP-DOWN (tabletop) VIEW  -  x/y plane, no gravity in-plane.
  - Gravity acts INTO the screen (perpendicular), so f_ext = 0 in QP.
  - Table sliding friction is viscous damping: f_drag = -b*v
  - Boundaries: 4 walls at screen edges.
  - Stick: circular puck (radius STICK_R) moved with WASD.

CONTACT DETECTION
  Walls : checked per vertex  (vertex is the deepest penetrating feature)
  Puck  : checked per EDGE using closest point on segment to puck centre.
          This catches face-centre contacts that vertex-only detection misses.

KEY EQUATIONS (paper)
  q^{l+1} = q^l + h v^{l+1}

  min_{v+}  0.5 (v+)^T M v+  +  kbar^T v+
  kbar = -M v^l - h f^l

  subject to, for each active contact j, both tangent directions:
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
MARGIN        = 40            # wall inset from screen edge (pixels)

h       = 1 / 120             # time step (s)
mu      = 0.3                 # stick-box Coulomb friction coefficient
mu_wall = 0.1                 # wall-box Coulomb friction (walls are smooth)
mass    = 2.0
size    = 80.0                # box side length (pixels)
I       = (1.0 / 6.0) * mass * size ** 2

# Viscous table damping (simulates kinetic friction with table surface)
B_LIN = mass * 0.8
B_ROT = I    * 0.8

# Generalised mass matrix  M = diag(m, m, I)
M    = np.diag([mass, mass, I])
Minv = np.linalg.inv(M)

# Contact activation sentinel (paper section 2.1: include phi <= eps_hat)
EPS_HAT = 2.0

# Stabilization gain (paper eq. 2.12)
GAMMA = 0.2

# Stick (circular puck)
STICK_R     = 12.0            # puck radius (pixels)  -- smaller than before
STICK_SPEED = 350.0           # keyboard speed (pixels/s)

# =====================================================
# Initial state
# =====================================================
q = np.array([WIDTH / 2.0, HEIGHT / 2.0, 0.0])   # [x, y, theta]
v = np.zeros(3)                                    # [vx, vy, omega]

stick_pos = np.array([WIDTH / 2.0 - 200.0, HEIGHT / 2.0])

# =====================================================
# Pygame setup
# =====================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Anitescu 2006 - top-down tabletop")
clock = pygame.time.Clock()
font  = pygame.font.SysFont("monospace", 15)


# =====================================================
# Geometry helpers
# =====================================================

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def get_vertices(q):
    """Four corners of the square in world coordinates."""
    R    = rotation_matrix(q[2])
    half = size / 2.0
    local_corners = np.array([[-half, -half],
                               [ half, -half],
                               [ half,  half],
                               [-half,  half]])
    return (R @ local_corners.T).T + q[:2]


# =====================================================
# Generalised Jacobian rows  (paper section 2.1)
# =====================================================

def make_jacobians(n, r):
    """
    n : unit outward normal at contact (2,)
    r : lever arm from COM to contact point (2,)
    Jn = [n_x, n_y,  r_x*n_y - r_y*n_x]
    Jt = [t_x, t_y,  r_x*t_y - r_y*t_x]   where t = 90 deg CCW of n
    """
    t  = np.array([-n[1], n[0]])
    Jn = np.array([n[0], n[1],  r[0]*n[1] - r[1]*n[0]])
    Jt = np.array([t[0], t[1],  r[0]*t[1] - r[1]*t[0]])
    return Jn, Jt


# =====================================================
# Contact detection  (paper section 2.1, active set A)
# =====================================================

def closest_point_on_segment(p, a, b):
    """Closest point on segment [a,b] to point p, and parameter t in [0,1]."""
    ab    = b - a
    denom = ab @ ab
    if denom < 1e-12:
        return a.copy(), 0.0
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return a + t * ab, t


def detect_contacts(q, stick_pos):
    """
    Returns list of (phi, n, r, mu_c).
      phi  : signed gap (< 0 means penetrating)
      n    : outward unit normal (away from obstacle, into free space)
      r    : lever arm from box COM to contact point
      mu_c : friction coefficient for this contact pair
    """
    contacts = []
    verts = get_vertices(q)   # shape (4,2), corners in order

    # -- Four walls: checked per vertex ----------------------------------
    # Vertices are the deepest penetrating feature for flat walls.
    for vtx in verts:
        # Left wall  (x = MARGIN), outward normal = [+1, 0]
        phi = vtx[0] - MARGIN
        if phi <= EPS_HAT:
            contacts.append((phi, np.array([1.0, 0.0]), vtx - q[:2], mu_wall))

        # Right wall  (x = WIDTH-MARGIN), outward normal = [-1, 0]
        phi = (WIDTH - MARGIN) - vtx[0]
        if phi <= EPS_HAT:
            contacts.append((phi, np.array([-1.0, 0.0]), vtx - q[:2], mu_wall))

        # Top wall  (y = MARGIN), outward normal = [0, +1]
        phi = vtx[1] - MARGIN
        if phi <= EPS_HAT:
            contacts.append((phi, np.array([0.0, 1.0]), vtx - q[:2], mu_wall))

        # Bottom wall  (y = HEIGHT-MARGIN), outward normal = [0, -1]
        phi = (HEIGHT - MARGIN) - vtx[1]
        if phi <= EPS_HAT:
            contacts.append((phi, np.array([0.0, -1.0]), vtx - q[:2], mu_wall))

    # -- Circular puck: checked per box edge -----------------------------
    # For each edge we find the closest point on that edge to the puck
    # centre. This correctly handles contact anywhere on a face:
    #   - face centre: closest point is the foot of the perpendicular,
    #                  normal is perpendicular to the face (correct push)
    #   - near corner: closest point is the corner itself (same as before)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edge_pairs:
        closest, _ = closest_point_on_segment(stick_pos, verts[i], verts[j])
        delta = closest - stick_pos
        dist  = np.linalg.norm(delta)
        phi   = dist - STICK_R
        if phi <= EPS_HAT and dist > 1e-6:
            n = delta / dist         # outward normal: puck_centre -> edge point
            r = closest - q[:2]     # lever arm: box_COM -> contact point
            contacts.append((phi, n, r, mu))

    return contacts


# =====================================================
# One-step QP   (Theorem 2.1 / section 2.3, Anitescu 2006)
# =====================================================

def step(q, v, contacts, f):
    """
    f : generalised external force [Fx, Fy, torque] (includes damping).
    Returns (q_next, v_next).
    """
    kbar = -M @ v - h * f

    if len(contacts) == 0:
        v_next = v + h * (Minv @ f)
        return q + h * v_next, v_next

    A_rows = []
    b_vals = []

    for phi, n, r, mu_c in contacts:
        Jn, Jt = make_jacobians(n, r)
        gap_correction = GAMMA * phi / h

        # +tangent: (Jn + mu*Jt) @ v+ + gamma*phi/h >= 0
        A_rows.append(Jn + mu_c * Jt)
        b_vals.append(-gap_correction)

        # -tangent: (Jn - mu*Jt) @ v+ + gamma*phi/h >= 0
        A_rows.append(Jn - mu_c * Jt)
        b_vals.append(-gap_correction)

    A_ineq = np.vstack(A_rows)
    b_ineq = np.array(b_vals)

    v_var = cp.Variable(3)
    objective = cp.Minimize(0.5 * cp.quad_form(v_var, M) + kbar @ v_var)
    prob = cp.Problem(objective, [A_ineq @ v_var >= b_ineq])
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        v_next = v_var.value
    else:
        v_next = v + h * (Minv @ f)

    return q + h * v_next, v_next


# =====================================================
# Main loop
# =====================================================
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # -- Stick control (WASD) ------------------------------------------------
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]: stick_pos[0] -= STICK_SPEED * h
    if keys[pygame.K_d]: stick_pos[0] += STICK_SPEED * h
    if keys[pygame.K_w]: stick_pos[1] -= STICK_SPEED * h
    if keys[pygame.K_s]: stick_pos[1] += STICK_SPEED * h

    stick_pos[0] = np.clip(stick_pos[0], MARGIN + STICK_R, WIDTH  - MARGIN - STICK_R)
    stick_pos[1] = np.clip(stick_pos[1], MARGIN + STICK_R, HEIGHT - MARGIN - STICK_R)

    # -- External force: viscous damping only, no in-plane gravity -----------
    f_ext = np.array([-B_LIN * v[0],
                      -B_LIN * v[1],
                      -B_ROT * v[2]])

    # -- Contacts and QP step ------------------------------------------------
    contacts = detect_contacts(q, stick_pos)
    q, v     = step(q, v, contacts, f_ext)

    # -- Rendering -----------------------------------------------------------
    screen.fill((245, 235, 210))

    # Table border
    wall_rect = pygame.Rect(MARGIN, MARGIN, WIDTH - 2*MARGIN, HEIGHT - 2*MARGIN)
    pygame.draw.rect(screen, (100, 70, 40), wall_rect, 6)

    # Box
    verts_draw = get_vertices(q).astype(int).tolist()
    pygame.draw.polygon(screen, (220, 80, 60), verts_draw)
    pygame.draw.polygon(screen, (120, 30, 20), verts_draw, 2)

    # Direction marker on box
    centre = q[:2].astype(int)
    R      = rotation_matrix(q[2])
    tip    = (q[:2] + R @ np.array([size * 0.38, 0])).astype(int)
    pygame.draw.line(screen, (255, 255, 255), tuple(centre), tuple(tip), 3)

    # Stick puck
    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (40, 80, 200), sc, int(STICK_R))
    pygame.draw.circle(screen, (20, 40, 120), sc, int(STICK_R), 2)

    # HUD
    nc        = len(contacts)
    speed_mag = float(np.hypot(v[0], v[1]))
    hud = [
        "Anitescu 2006 - top-down tabletop",
        f"h={h:.4f}  mu_stick={mu}  mu_wall={mu_wall}  gamma={GAMMA}",
        f"box pos=({q[0]:.1f}, {q[1]:.1f})  theta={float(np.degrees(q[2])):.1f} deg",
        f"speed={speed_mag:.1f}  omega={float(v[2]):.2f}",
        f"active contacts: {nc}",
        "",
        "WASD = move puck",
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (50, 30, 10))
        screen.blit(surf, (10, 10 + i * 18))

    pygame.display.flip()
    clock.tick(120)
