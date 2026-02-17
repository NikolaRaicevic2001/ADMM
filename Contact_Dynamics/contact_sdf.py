"""
ContactSDF (Yang & Jin, IEEE RA-L 2024)
  https://arxiv.org/pdf/2408.09612

TOP-DOWN tabletop demo: circular disk only, driven by a puck (WASD).
"""

import pygame
import numpy as np
import sys

WIDTH, HEIGHT = 900, 600
MARGIN        = 40

h           = 1 / 120
mu_stick    = 0.3
mu_wall     = 0.1

CIRC_R  = 35.0

E_CIRC  = 6.0
E_diag  = np.array([E_CIRC, E_CIRC])

Q_diag       = E_diag / h**2
Q_half_diag  = np.sqrt(Q_diag)
Q_half_inv   = 1.0 / Q_half_diag

SIGMA_D = 4.0
GAMMA   = 0.5

STICK_R     = 12.0
STICK_SPEED = 350.0
EPS_HAT     = 8.0

# =====================================================
# Initial state
# =====================================================
q_circ = np.array([WIDTH / 2.0, HEIGHT / 2.0])

stick_pos      = np.array([WIDTH / 2.0 - 200.0, HEIGHT / 2.0])
stick_pos_prev = stick_pos.copy()

# =====================================================
# Pygame
# =====================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ContactSDF - circle only")
clock = pygame.time.Clock()
font  = pygame.font.SysFont("monospace", 14)


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

    # ---- Walls vs Circle ----------------------------------------------------
    cx, cy = q_circ
    for phi_val, n_vec in [
        (cx - CIRC_R - MARGIN,             np.array([ 1.,  0.])),
        ((WIDTH -MARGIN) - (cx+CIRC_R),    np.array([-1.,  0.])),
        (cy - CIRC_R - MARGIN,             np.array([ 0.,  1.])),
        ((HEIGHT-MARGIN) - (cy+CIRC_R),    np.array([ 0., -1.])),
    ]:
        if phi_val <= EPS_HAT:
            Jn, Jt  = jac_circ(n_vec)
            phi_eff = GAMMA * phi_val
            contacts.append((phi_eff, Jn, Jt, mu_wall))

    # ---- Puck vs Circle -----------------------------------------------------
    phi_pc, n_pc = csdf_circle(q_circ, stick_pos, STICK_R + CIRC_R)
    if phi_pc <= EPS_HAT and np.linalg.norm(n_pc) > 1e-8:
        b_offset = float(n_pc @ stick_vel)
        phi_eff  = GAMMA * phi_pc - b_offset * h
        Jn, Jt   = jac_circ(n_pc)
        contacts.append((phi_eff, Jn, Jt, mu_stick))

    return contacts


def dsdf_step(contacts):
    z_query = np.zeros(2)

    if not contacts:
        return np.zeros(2)

    ns, bs = [], []
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

    ns = np.array(ns)
    bs = np.array(bs)

    scores    = ns @ z_query + bs
    all_terms = np.concatenate([[0.0], SIGMA_D * scores])
    max_t     = np.max(all_terms)
    lse       = max_t + np.log(np.sum(np.exp(all_terms - max_t)))
    d_sdf     = lse / SIGMA_D

    sw   = np.exp(all_terms - max_t)
    sw  /= np.sum(sw)
    grad = sw[1:] @ ns

    z_plus = z_query - d_sdf * grad
    return Q_half_inv * z_plus / h


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

    stick_vel = (stick_pos - stick_pos_prev) / h

    contacts = detect_contacts(q_circ, stick_pos, stick_vel)
    v_next   = dsdf_step(contacts)

    q_circ = q_circ + h * v_next

    # ---- Rendering ----------------------------------------------------------
    screen.fill((210, 230, 250))

    pygame.draw.rect(screen, (50, 80, 130),
                     pygame.Rect(MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 6)

    # Circle
    ci = q_circ.astype(int)
    pygame.draw.circle(screen, (60, 180, 100), ci, int(CIRC_R))
    pygame.draw.circle(screen, (20, 100,  50), ci, int(CIRC_R), 2)
    pygame.draw.circle(screen, (20, 100,  50), ci, 4)

    # Puck
    sc = stick_pos.astype(int)
    pygame.draw.circle(screen, (50, 90, 210), sc, int(STICK_R))
    pygame.draw.circle(screen, (20, 40, 130), sc, int(STICK_R), 2)

    nc = len(contacts)
    hud = [
        "ContactSDF (Yang & Jin, RA-L 2024) - circle only",
        "D-SDF gradient projection (no QP solver)",
        f"sigma_d={SIGMA_D}  gamma={GAMMA}",
        f"circ pos=({q_circ[0]:.0f},{q_circ[1]:.0f})",
        f"active contacts: {nc}",
        "WASD = move puck",
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (20, 40, 90))
        screen.blit(surf, (10, 10 + i * 17))

    pygame.display.flip()
    clock.tick(120)
