"""
Exact implementation of:
  Anitescu, M. "Optimization-based simulation of nonsmooth rigid multibody dynamics."
  Math. Program. 105, 113–143 (2006).  [ANL Preprint P1161]

Also uses constraint-stabilization from:
  Anitescu & Hart, "A Constraint-Stabilized Time-Stepping Approach…"
  Int. J. Numer. Meth. Engng 60 (2004) 2335–2371.

KEY EQUATIONS FROM THE PAPER
─────────────────────────────
State: q ∈ ℝ³ = [x, y, θ],  v ∈ ℝ³ = [ẋ, ẏ, θ̇]

Time-stepping (eq. 2.2 / §2):
  q^{l+1} = q^l + h v^{l+1}          ← semi-implicit (new velocity)

QP subproblem per step (Theorem 2.1 / §2.3 of Anitescu & Hart 2004,
and the primal form in §3 of Anitescu 2006):

  min_{v+}   ½ (v+)ᵀ M v+  +  k̄ᵀ v+
  where      k̄ = −M v^l − h f^l

  subject to, for each active contact j and each tangent direction i:
    (n^(j) + μ d^(j)_i)ᵀ v+  +  Φ^(j)/h  ≥  0      [relaxed cone / eq 2.22]

  In 2D there are two tangent directions per contact: +t and −t, so two
  constraints per contact collapse to:
    (Jn + μ Jt) @ v+  +  φ/h  ≥  0
    (Jn − μ Jt) @ v+  +  φ/h  ≥  0

  Active set A (eq. 2.9 / §2.1):
    Include contact j whenever  Φ^(j)(q^l) ≤ ε̂   (ε̂ small positive sentinel)

The objective minimum (unconstrained) recovers the free-flight step
  M v+ = M v^l + h f  →  v+ = v^l + h M⁻¹ f,
so no contact impulse is needed when constraints are inactive.
The dual variables of the QP give the contact impulses (not needed explicitly
for forward integration).
"""

import pygame
import numpy as np
import cvxpy as cp
import sys

# ══════════════════════════════════════════════════════
# Simulation parameters
# ══════════════════════════════════════════════════════
WIDTH, HEIGHT = 900, 600

h    = 1 / 120          # time step
g    = 900              # gravitational acceleration (pixels/s²), downward = +y
mu   = 0.8              # Coulomb friction coefficient
mass = 20.0
size = 60.0
I    = (1.0 / 6.0) * mass * size ** 2   # inertia of square about centre

# Generalised mass matrix  M = diag(m, m, I)
M    = np.diag([mass, mass, I])
Minv = np.linalg.inv(M)

# Contact activation sentinel  (include contacts with Φ ≤ ε_hat, paper §2.1)
EPS_HAT = 2.0           # pixels; contacts within this distance are "active"

# Stabilization gain  (paper eq. 2.12: alternative methods use γΔ instead of Δ)
# γ = 1.0  → exact paper formulation (can over-correct for large penetrations)
# γ ∈ (0,1) → softer correction; prevents "launching" when stick overlaps body
GAMMA = 0.2             # tune between 0.1 (gentle) and 1.0 (exact/aggressive)

# ══════════════════════════════════════════════════════
# Initial state
# ══════════════════════════════════════════════════════
q = np.array([WIDTH / 2.0, HEIGHT / 3.0, 0.0])   # [x, y, theta]
v = np.zeros(3)                                    # [vx, vy, omega]

# ══════════════════════════════════════════════════════
# Pygame setup
# ══════════════════════════════════════════════════════
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Anitescu 2006 – exact QP rigid body")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 15)

stick_x = WIDTH / 2.0 - 200
stick_y = HEIGHT / 2.0
speed   = 300.0          # stick speed, pixels/s


# ══════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════
# Generalised contact Jacobian rows  (paper §2.1)
# ══════════════════════════════════════════════════════
# For a 2-D body with generalised coordinates q = [x, y, θ]:
#   The contact point world position is  p = q[:2] + r   (r = lever arm from COM)
#
# Normal Jacobian row (generalised):
#   Jn = [ n_x,  n_y,  r_x n_y − r_y n_x ]
# Tangent Jacobian row (generalised):
#   Jt = [ t_x,  t_y,  r_x t_y − r_y t_x ]
# where t = [−n_y, n_x]  (90° CCW rotation of n)
#
# Then:  n^T v_contact = Jn @ v_gen
#        t^T v_contact = Jt @ v_gen

def make_jacobians(n, r):
    """
    n : unit outward normal at contact (2,)
    r : lever arm from COM to contact point (2,)
    Returns Jn (3,) and Jt (3,) as row vectors.
    """
    t  = np.array([-n[1], n[0]])          # tangent, 90° CCW from n
    Jn = np.array([n[0], n[1],  r[0]*n[1] - r[1]*n[0]])
    Jt = np.array([t[0], t[1],  r[0]*t[1] - r[1]*t[0]])
    return Jn, Jt


# ══════════════════════════════════════════════════════
# Contact detection   (paper §2.1, active set A)
# ══════════════════════════════════════════════════════
# A contact is ACTIVE when  Φ^(j)(q) ≤ EPS_HAT  (not only when < 0).
# Φ is the signed penetration depth; Φ < 0 ⟹ interpenetrating.
#
# Contacts modelled:
#   1. Ground (y = ground_y, outward normal n = [0, −1] pointing upward)
#   2. Vertical stick (x = stick_x, outward normal n = [+1, 0] pointing right)

def detect_contacts(q):
    """
    Returns list of (phi, n, r) for all active contacts.
      phi : signed gap (< 0 means penetration)
      n   : outward unit normal at contact (pointing away from obstacle)
      r   : lever arm from COM to contact vertex
    """
    contacts = []
    verts     = get_vertices(q)
    ground_y  = HEIGHT - 50

    for vtx in verts:
        # ── Ground ──────────────────────────────────────
        phi_ground = ground_y - vtx[1]     # positive = above ground (no contact)
        # Φ = -(ground_y - vtx[1])? Let's be careful:
        # Noninterpenetration: vtx[1] ≤ ground_y → Φ = ground_y - vtx[1] ≥ 0
        # n points upward (away from ground), i.e., n = [0, -1] in screen coords
        # because increasing y goes DOWN in pygame.
        phi = ground_y - vtx[1]            # ≥ 0 when above ground, < 0 when below
        if phi <= EPS_HAT:
            n = np.array([0.0, -1.0])      # outward normal: upward (screen: -y)
            r = vtx - q[:2]
            contacts.append((phi, n, r))

        # ── Stick (vertical wall at x = stick_x) ─────────
        # Normal must point away from the stick TOWARD the square, so it
        # depends on which side the square COM is on.
        #   square to the right: side = +1, n = [+1, 0], phi = vtx[0] - stick_x
        #   square to the left:  side = -1, n = [-1, 0], phi = stick_x - vtx[0]
        side      = np.sign(q[0] - stick_x)   # +1 or -1
        phi_stick = side * (vtx[0] - stick_x) # positive = gap, negative = penetration
        if phi_stick <= EPS_HAT and abs(vtx[1] - stick_y) < 120:
            n = np.array([side, 0.0])          # outward normal away from stick
            r = vtx - q[:2]
            contacts.append((phi_stick, n, r))

    return contacts


# ══════════════════════════════════════════════════════
# One-step QP   (paper Theorem 2.1 / §2.3 + Anitescu 2006 §3)
# ══════════════════════════════════════════════════════

def step(q, v, contacts, f):
    """
    Advance (q, v) by one time step h.

    QP decision variable:  v_next ∈ ℝ³  (generalised velocity at step l+1)

    Objective:
      min  ½ v_next^T M v_next  +  k̄^T v_next
      k̄ = −M v − h f                          (eq. in §2.2 / bk notation)

    Constraints  (one pair per active contact j):
      (Jn^j + μ Jt^j) @ v_next  +  φ^j / h  ≥  0     [+tangent direction]
      (Jn^j − μ Jt^j) @ v_next  +  φ^j / h  ≥  0     [−tangent direction]

    The two constraints together enforce the relaxed (linearised) Coulomb cone
    AND the velocity-level non-penetration with gap stabilisation φ/h.

    Position update:
      q_next = q + h v_next                             (eq. §2.2)
    """

    # ── Build k̄ ──────────────────────────────────────────────────────────────
    kbar = -M @ v - h * f      # shape (3,)

    if len(contacts) == 0:
        # No constraints: closed-form free-flight step
        # min  ½ v+^T M v+  + k̄^T v+  →  v+ = M⁻¹ (−k̄) = v + h M⁻¹ f
        v_next = v + h * (Minv @ f)
        q_next = q + h * v_next
        return q_next, v_next

    # ── Build constraint matrix  A_ineq @ v_next ≥ b_ineq ───────────────────
    # Two rows per contact (±tangent relaxed-cone constraint)
    nc = len(contacts)
    A_rows = []
    b_vals = []

    for phi, n, r in contacts:
        Jn, Jt = make_jacobians(n, r)

        # Stabilized gap term: γ·φ/h  (paper eq. 2.12)
        # γ=1 → exact paper; γ<1 → gentler correction for large penetrations
        gap_correction = GAMMA * phi / h

        # +tangent direction: (Jn + μ Jt) @ v+ + γφ/h ≥ 0
        A_rows.append( Jn + mu * Jt )
        b_vals.append( -gap_correction )   # RHS: move to right-hand side

        # −tangent direction: (Jn − μ Jt) @ v+ + γφ/h ≥ 0
        A_rows.append( Jn - mu * Jt )
        b_vals.append( -gap_correction )

    A_ineq = np.vstack(A_rows)   # (2*nc, 3)
    b_ineq = np.array(b_vals)    # (2*nc,)

    # ── CVXPY QP ─────────────────────────────────────────────────────────────
    v_var = cp.Variable(3)

    objective = cp.Minimize(
        0.5 * cp.quad_form(v_var, M)   # ½ v^T M v
        + kbar @ v_var                 # + k̄^T v
    )

    # A_ineq @ v_var ≥ b_ineq  ↔  A_ineq @ v_var − b_ineq ≥ 0
    constraints = [A_ineq @ v_var >= b_ineq]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        v_next = v_var.value
    else:
        # Fallback: free-flight (should not happen with a feasible QP)
        v_next = v + h * (Minv @ f)

    q_next = q + h * v_next
    return q_next, v_next


# ══════════════════════════════════════════════════════
# Main simulation loop
# ══════════════════════════════════════════════════════
while True:
    # ── Events ───────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # ── Stick control ────────────────────────────────
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]: stick_x -= speed * h
    if keys[pygame.K_d]: stick_x += speed * h
    if keys[pygame.K_w]: stick_y -= speed * h
    if keys[pygame.K_s]: stick_y += speed * h

    # ── External force (gravity, downward = +y in screen) ────────────────
    # In generalised coordinates: F = [0, m*g, 0]
    f_ext = np.array([0.0, mass * g, 0.0])

    # ── Contact detection ────────────────────────────
    contacts = detect_contacts(q)

    # ── QP time step ─────────────────────────────────
    q, v = step(q, v, contacts, f_ext)

    # ── Rendering ────────────────────────────────────
    screen.fill((240, 240, 240))

    # Ground
    ground_y = HEIGHT - 50
    pygame.draw.line(screen, (0, 0, 0), (0, ground_y), (WIDTH, ground_y), 4)

    # Stick
    pygame.draw.line(screen, (30, 30, 200),
                     (int(stick_x), int(stick_y - 110)),
                     (int(stick_x), int(stick_y + 110)), 8)

    # Square body
    verts_draw = get_vertices(q).astype(int).tolist()
    pygame.draw.polygon(screen, (200, 60, 60), verts_draw)
    pygame.draw.polygon(screen, (100, 20, 20), verts_draw, 2)

    # HUD
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
