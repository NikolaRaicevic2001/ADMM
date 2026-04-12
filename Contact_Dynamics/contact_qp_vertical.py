"""
Anitescu, M. "Optimization-based simulation of nonsmooth rigid multibody dynamics."
Math. Program. 105, 113-143 (2006).  [ANL Preprint P1161]

Single rigid box under gravity, side-view. Contacts: ground + vertical stick (WASD).
QP per step: min 0.5 v+^T M v+ + kbar^T v+  s.t. (Jn +/- mu Jt) v+ + gamma*phi/h >= 0
"""

import pygame
import numpy as np
import cvxpy as cp
import sys

# ── Simulation parameters ─────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 600

h    = 1 / 360
g    = 900
mu   = 0.8
mass = 20.0
size = 60.0
I    = (1.0 / 6.0) * mass * size ** 2

M    = np.diag([mass, mass, I])
Minv = np.linalg.inv(M)

EPS_HAT = 2.0
GAMMA   = 0.2

# ── Initial state ─────────────────────────────────────────────────────────────
q = np.array([WIDTH / 2.0, HEIGHT / 3.0, 0.0])
v = np.zeros(3)

# ── Pygame ────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Anitescu 2006 – exact QP rigid body")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 15)

stick_x = WIDTH / 2.0 - 200
stick_y = HEIGHT / 2.0
speed   = 300.0

print(f"[contact_qp_vertical] h={h:.5f}  mu={mu}  gamma={GAMMA}  EPS_HAT={EPS_HAT}px")


# ── Geometry ──────────────────────────────────────────────────────────────────

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def get_vertices(q):
    R    = rotation_matrix(q[2])
    half = size / 2.0
    local_corners = np.array([[-half, -half],
                               [ half, -half],
                               [ half,  half],
                               [-half,  half]])
    return (R @ local_corners.T).T + q[:2]


# ── Generalised Jacobians: Jn=[nx, ny, rx*ny - ry*nx], Jt=[tx, ty, rx*ty - ry*tx] ──

def make_jacobians(n, r):
    t  = np.array([-n[1], n[0]])
    Jn = np.array([n[0], n[1],  r[0]*n[1] - r[1]*n[0]])
    Jt = np.array([t[0], t[1],  r[0]*t[1] - r[1]*t[0]])
    return Jn, Jt


# ── Contact detection: ground + stick wall, active when phi <= EPS_HAT ────────

def detect_contacts(q):
    contacts = []
    verts    = get_vertices(q)
    ground_y = HEIGHT - 50

    for vtx in verts:
        phi = ground_y - vtx[1]
        if phi <= EPS_HAT:
            n = np.array([0.0, -1.0])
            r = vtx - q[:2]
            contacts.append((phi, n, r))

        side      = np.sign(q[0] - stick_x)
        phi_stick = side * (vtx[0] - stick_x)
        if phi_stick <= EPS_HAT and abs(vtx[1] - stick_y) < 120:
            n = np.array([side, 0.0])
            r = vtx - q[:2]
            contacts.append((phi_stick, n, r))

    return contacts


# ── One-step QP: kbar = -Mv - hf; free-flight if no contacts ─────────────────

def step(q, v, contacts, f):
    kbar = -M @ v - h * f

    if len(contacts) == 0:
        v_next = v + h * (Minv @ f)
        q_next = q + h * v_next
        return q_next, v_next

    A_rows = []
    b_vals = []

    for phi, n, r in contacts:
        Jn, Jt = make_jacobians(n, r)
        gap_correction = GAMMA * phi / h
        A_rows.append( Jn + mu * Jt )
        b_vals.append( -gap_correction )
        A_rows.append( Jn - mu * Jt )
        b_vals.append( -gap_correction )

    A_ineq = np.vstack(A_rows)
    b_ineq = np.array(b_vals)

    v_var = cp.Variable(3)
    objective   = cp.Minimize(0.5 * cp.quad_form(v_var, M) + kbar @ v_var)
    constraints = [A_ineq @ v_var >= b_ineq]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        v_next = v_var.value
    else:
        print(f"[WARN] QP status={prob.status}  falling back to free-flight")
        v_next = v + h * (Minv @ f)

    q_next = q + h * v_next
    return q_next, v_next


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]: stick_x -= speed * h
    if keys[pygame.K_d]: stick_x += speed * h
    if keys[pygame.K_w]: stick_y -= speed * h
    if keys[pygame.K_s]: stick_y += speed * h

    f_ext    = np.array([0.0, mass * g, 0.0])
    contacts = detect_contacts(q)
    q, v     = step(q, v, contacts, f_ext)

    # ── Render ────────────────────────────────────────────────────────────────
    screen.fill((240, 240, 240))

    ground_y = HEIGHT - 50
    pygame.draw.line(screen, (0, 0, 0), (0, ground_y), (WIDTH, ground_y), 4)

    pygame.draw.line(screen, (30, 30, 200),
                     (int(stick_x), int(stick_y - 110)),
                     (int(stick_x), int(stick_y + 110)), 8)

    verts_draw = get_vertices(q).astype(int).tolist()
    pygame.draw.polygon(screen, (200, 60, 60), verts_draw)
    pygame.draw.polygon(screen, (100, 20, 20), verts_draw, 2)

    nc  = len(contacts)
    hud = [
        f"Anitescu 2006  |  h={h:.4f}  mu={mu}  gamma={GAMMA}",
        f"pos=({q[0]:.1f}, {q[1]:.1f})  theta={np.degrees(q[2]):.1f}°",
        f"vel=({v[0]:.1f}, {v[1]:.1f})  omega={v[2]:.2f}",
        f"active contacts: {nc}   (sentinel ε̂={EPS_HAT}px)",
        "",
        "WASD = move stick",
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (40, 40, 40))
        screen.blit(surf, (10, 10 + i * 18))

    pygame.display.flip()
    clock.tick(120)
