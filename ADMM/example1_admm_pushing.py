import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle
from dataclasses import dataclass
from pathlib import Path

np.seterr(divide="ignore", invalid="ignore")

# ---------------------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------------------
RANDOM_SEED = 0
SAVE_ANIMATION = True
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------------------
# Simulation / MPPI / ADMM tuning parameters
# ---------------------------------------------------------------------------------------
DT = 0.05                          # s, shared robot + object timestep (Delta t)
HORIZON = 15                       # H^c, ADMM shared coordination horizon
N_ADMM = 6                         # N_ADMM, ADMM iterations per control step
K_OBJECT = 64                      # K_o, object-level rollouts
K_ROBOT = 64                       # K_r, robot-level rollouts
MAX_CONTROL_STEPS = 500

MU = 0.4                           # object-support Coulomb friction
MU_C = 0.5                         # robot-object Coulomb friction (Sec. friction_cone)
OBJECT_MASS = 2.0                  # kg (effective, tuned for cm-scale motion per step)
GRAVITY = 9.81
LIMIT_SURFACE_C = 1.0
LIMIT_SURFACE_R = 0.06             # m, characteristic length

F_MAX = 4.0                        # N, normal force cap (f_max)
RHO = 1.0                          # ADMM penalty parameter
MAX_DUAL_NORM = 1.0 * F_MAX         # N, cap on accumulated scaled-dual magnitude (anti-windup)
EPS_R, EPS_S = 1.0, 1.0            # ADMM primal / dual convergence tolerances

MPPI_LAMBDA = 1.0                  # MPPI temperature
SIGMA_O = 0.7                      # N, object-level normal-force perturbation std (sigma_o)
SIGMA_P = 0.012                    # m, contact-point perturbation std (sigma_p)
TAU_N = 0.7                        # normal-alignment rejection threshold (tau_n), Eq. reject_criterion
MAX_REJECTION_TRIES = 8            # redraw cap for contact-point rejection sampling (implementation detail)
SIGMA_ROBOT = np.array([0.12, 0.12])   # (vx, vy) perturbation std, m/s (Sigma_r)

Q_POS, Q_THETA = 40.0, 10.0            # running goal-cost weights
QF_POS, QF_THETA = 150.0, 45.0         # terminal goal-cost weights
Q_TRACK_POS, Q_TRACK_THETA = 250.0, 8.0    # robot-level tracking-cost weights (W)
W_OBSTACLE = 6.0e4                     # obstacle hinge-cost weight
OBSTACLE_MARGIN = 0.015                # m
CONTACT_STEP_MARGIN = 0.003            # m, allowed overshoot past the surface in one step
MAX_CONTACT_STEP = 0.008               # m, hard per-substep cap while already at/past the surface
N_CONTACT_SUBSTEPS = 4                 # sub-steps per control step, bounds sweep from either body
OBJECT_PUSHOUT_ITERS = 4               # Gauss-Seidel passes to resolve object-obstacle overlap
SEEK_MIN_SPEED, SEEK_MAX_SPEED = 0.4, 1.0  # m/s, re-seeded approach speed bounds

N_P_EST = 48        # N_p, estimator samples per round
R_EST = 4            # R_est, estimator annealing rounds
H_P_EST = 5           # H_p, estimator lookahead horizon
SIGMA_INIT_EST, SIGMA_MIN_EST = 0.006, 0.004    # m
GAMMA_EST = 0.6                                  # annealing rate
TAU_P_EST = 0.35                                 # estimator softmax temperature

GOAL_POS_TOL, GOAL_THETA_TOL = 0.06, 0.08   # m, rad: how close counts as "reached"
SIGMA_ANNEAL_BAND = 4.5                     # start shrinking exploration within this many tolerances
MIN_SIGMA_SCALE = 0.2                       # exploration never shrinks below this fraction


# ---------------------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------------------
def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotate(theta, v):
    """Rotate 2D vector(s) v (..., 2) by angle(s) theta, broadcasting over leading dims."""
    c, s = np.cos(theta), np.sin(theta)
    vx, vy = v[..., 0], v[..., 1]
    return np.stack([c * vx - s * vy, s * vx + c * vy], axis=-1)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def clip_dual(y, max_norm=MAX_DUAL_NORM):
    """Cap each timestep's scalar scaled-dual to a maximum magnitude.

    Classical ADMM convergence assumes convex sub-problems solved to exact minimizers; both
    sub-problems here are non-convex sampling-based MPPI solves, so nothing guarantees the
    primal residual actually shrinks. If it does not, y_o and y_r (each a running sum of that
    residual, warm-started and never reset across hundreds of control steps) grow without bound,
    and the "target" each side chases, z - y, becomes dominated by accumulated bias rather than
    the task. Capping the dual magnitude bounds how much accumulated bias can ever enter the
    target, without changing the update's form. y_o, y_r are scalars per timestep now (Sec.
    aug_lagrangian), so the cap is a plain elementwise clip rather than a vector-norm rescale."""
    return np.clip(y, -max_norm, max_norm)


def goal_cost(poses, goal, w_pos, w_theta):
    """(x^o - g)^T Q_o (x^o - g), Q_o = diag(w_pos, w_pos, w_theta)."""
    poses = np.atleast_2d(poses)
    diff_pos = poses[:, :2] - goal[:2]
    diff_theta = wrap_angle(poses[:, 2] - goal[2])
    return w_pos * np.einsum("ij,ij->i", diff_pos, diff_pos) + w_theta * diff_theta ** 2


# ---------------------------------------------------------------------------------------
# Object shape: exact signed-distance function of a simple (possibly non-convex) polygon
# ---------------------------------------------------------------------------------------
class PolygonShape:
    """Body-frame SDF d_B and gradient of a closed polygon, via nearest-edge distance and a
    winding-number inside/outside test. Exact almost everywhere (eikonal property holds)."""

    def __init__(self, vertices, boundary_samples_per_edge=4):
        self.vertices = np.asarray(vertices, float)
        self.boundary_samples = self._sample_boundary(boundary_samples_per_edge)
        self.edge_normals = self._edge_normals()
        self.center = self.vertices.mean(axis=0)
        self.bounding_radius = float(np.max(np.linalg.norm(self.vertices - self.center, axis=1)))

    def _sample_boundary(self, n):
        v = self.vertices
        pts = [v[i] + (v[(i + 1) % len(v)] - v[i]) * k / n for i in range(len(v)) for k in range(n)]
        return np.array(pts)

    def _edge_normals(self):
        """Fixed outward unit normal of every edge, oriented once via the polygon's signed
        area. Computed once at construction rather than on every sdf_and_grad call."""
        v = self.vertices
        edge_vecs = np.roll(v, -1, axis=0) - v
        signed_area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
        orientation = 1.0 if signed_area > 0 else -1.0
        edge_len = np.linalg.norm(edge_vecs, axis=1, keepdims=True)
        return orientation * np.stack([edge_vecs[:, 1], -edge_vecs[:, 0]], axis=1) / edge_len

    def sdf_and_grad(self, points):
        """The gradient at a point whose nearest feature is an edge interior is exactly that
        edge's fixed outward normal (correct even exactly on the boundary, where distance to
        the nearest point is zero and a naive "direction to nearest point" is undefined). Only
        at an actual vertex is the direction ambiguous; there, average the two adjacent edges."""
        points = np.atleast_2d(points)
        v = self.vertices
        n = len(v)
        edge_normals = self.edge_normals

        best_dist2 = np.full(len(points), np.inf)
        nearest = np.zeros_like(points)
        nearest_edge_normal = np.zeros_like(points)
        nearest_is_interior = np.zeros(len(points), dtype=bool)
        winding = np.zeros(len(points))

        for i in range(n):
            a, b = v[i], v[(i + 1) % n]
            ab = b - a
            raw_t = ((points - a) @ ab) / (ab @ ab)
            t = np.clip(raw_t, 0.0, 1.0)
            proj = a + t[:, None] * ab
            diff = points - proj
            dist2 = np.einsum("ij,ij->i", diff, diff)
            closer = dist2 < best_dist2
            best_dist2 = np.where(closer, dist2, best_dist2)
            nearest = np.where(closer[:, None], proj, nearest)
            nearest_edge_normal = np.where(closer[:, None], edge_normals[i], nearest_edge_normal)
            nearest_is_interior = np.where(closer, (raw_t > 0.0) & (raw_t < 1.0), nearest_is_interior)

            upward = (a[1] <= points[:, 1]) & (b[1] > points[:, 1])
            downward = (a[1] > points[:, 1]) & (b[1] <= points[:, 1])
            is_left = (b[0] - a[0]) * (points[:, 1] - a[1]) - (points[:, 0] - a[0]) * (b[1] - a[1])
            winding += np.where(upward & (is_left > 0), 1, 0)
            winding += np.where(downward & (is_left < 0), -1, 0)

        sign = np.where(winding != 0, -1.0, 1.0)
        dist = np.sqrt(best_dist2)

        diff = points - nearest
        diff_norm = np.linalg.norm(diff, axis=1, keepdims=True)
        vertex_dir = sign[:, None] * diff / np.clip(diff_norm, 1e-9, None)
        grad = np.where(nearest_is_interior[:, None], nearest_edge_normal, vertex_dir)

        # a point sitting exactly on a vertex has no well-defined nearest-point direction either;
        # use the average of the two edges meeting there instead.
        still_degenerate = (~nearest_is_interior) & (diff_norm[:, 0] < 1e-7)
        if np.any(still_degenerate):
            vertex_idx = np.argmin(
                np.linalg.norm(points[still_degenerate][:, None, :] - v[None, :, :], axis=2), axis=1)
            prev_edge = (vertex_idx - 1) % n
            averaged = edge_normals[prev_edge] + edge_normals[vertex_idx]
            averaged /= np.clip(np.linalg.norm(averaged, axis=1, keepdims=True), 1e-9, None)
            grad[still_degenerate] = averaged

        return sign * dist, grad

    def sdf(self, points):
        return self.sdf_and_grad(points)[0]

    def project_to_boundary(self, points):
        """Pi_dB(p) = p - d_B(p) grad d_B(p)  [Eq. ex1_boundary_projection]."""
        d, grad = self.sdf_and_grad(points)
        return points - d[:, None] * grad


def t_shape_vertices():
    """Capital-T outline, body frame, origin at the shape's centroid."""
    return np.array([
        [-0.090, 0.045], [0.090, 0.045], [0.090, 0.015], [0.015, 0.015],
        [0.015, -0.105], [-0.015, -0.105], [-0.015, 0.015], [-0.090, 0.015],
    ])


# ---------------------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------------------
@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float

    @property
    def bounding_radius(self):
        return self.radius

    def sdf(self, points):
        return np.linalg.norm(points - self.center, axis=-1) - self.radius

    def sdf_and_grad(self, points):
        diff = points - self.center
        dist = np.linalg.norm(diff, axis=-1)
        grad = diff / np.clip(dist, 1e-9, None)[..., None]
        return dist - self.radius, grad


@dataclass
class BoxObstacle:
    center: np.ndarray
    half_extents: np.ndarray
    angle: float = 0.0

    @property
    def bounding_radius(self):
        return float(np.linalg.norm(self.half_extents))

    def sdf(self, points):
        local = rotate(-self.angle, points - self.center)
        q = np.abs(local) - self.half_extents
        outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
        inside = np.clip(np.max(q, axis=-1), None, 0.0)
        return outside + inside

    def sdf_and_grad(self, points):
        local = rotate(-self.angle, points - self.center)
        q = np.abs(local) - self.half_extents
        outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
        inside = np.clip(np.max(q, axis=-1), None, 0.0)

        is_outside = np.any(q > 0.0, axis=-1)
        clipped = np.clip(q, 0.0, None)
        grad_outside = np.sign(local) * clipped / np.clip(np.linalg.norm(clipped, axis=-1, keepdims=True), 1e-9, None)
        onehot = np.zeros_like(q)
        onehot[np.arange(len(q)), np.argmax(q, axis=-1)] = 1.0
        grad_inside = np.sign(local) * onehot
        grad_local = np.where(is_outside[..., None], grad_outside, grad_inside)
        return outside + inside, rotate(self.angle, grad_local)


def obstacle_cost(shape, poses, obstacles):
    """Hinge penalty on the object's sampled boundary points against every obstacle SDF. Stands
    in for the formal object-environment contact-force penalty of Eq. obj_cost's second term
    (w_f^o(max(lambda^2_t - f_0, 0))^2): that term requires resolving an actual object-obstacle
    contact-force complementarity, which this 2D prototype does not model; a position-based hinge
    on SDF violation is a cheaper proxy that still keeps rollouts away from obstacles."""
    poses = np.atleast_2d(poses)
    verts_world = poses[:, None, :2] + rotate(poses[:, 2, None], shape.boundary_samples[None])
    flat = verts_world.reshape(-1, 2)
    cost = np.zeros(len(poses))
    for obs in obstacles:
        d = obs.sdf(flat).reshape(len(poses), -1)
        violation = np.clip(OBSTACLE_MARGIN - d, 0.0, None)
        cost += W_OBSTACLE * np.sum(violation ** 2, axis=1)
    return cost


def push_point_out_of_obstacles(points, obstacles):
    """Project any points inside a (static) obstacle back onto its boundary. Cheap centroid
    broad phase first, same reasoning as push_object_out_of_obstacles: this runs every
    sub-step of every MPPI rollout, so skipping the full SDF query when clearly far from an
    obstacle matters far more than it looks like it should."""
    for obs in obstacles:
        if np.all(np.linalg.norm(points - obs.center, axis=-1) > obs.bounding_radius):
            continue
        d, grad = obs.sdf_and_grad(points)
        inside = d < 0.0
        points = np.where(inside[..., None], points - d[..., None] * grad, points)
    return points


def push_object_out_of_obstacles(shape, pose, obstacles, iterations=OBJECT_PUSHOUT_ITERS):
    """Simple positional non-penetration correction: if any sampled boundary point ends up
    inside a (static) obstacle after the quasi-static update, translate the object out along
    that point's SDF gradient by the penetration depth. A basic kinematic fix, not a force
    response with reaction torque -- real hardware or IsaacGym deployment resolves object to
    obstacle contact in the physics engine; this exists only so the 2D prototype does not let
    obstacles be ignored entirely.

    One pass per obstacle only resolves whichever single boundary point is worst at that
    instant; a fast quasi-static step can leave several sample points penetrating at once
    (confirmed directly: a single pass left up to 3cm of residual penetration against a 4cm
    circle after one large push), and pushing out the worst one can put a different point
    into violation. Repeating a few passes, recomputing the worst point each time, is a basic
    Gauss-Seidel-style projection that converges for simple convex obstacle shapes.

    Cheap centroid-distance broad phase first: this runs on every propagate call across every
    MPPI rollout, so without it, evaluating the full per-boundary-point SDF of a polygon
    obstacle (its own nearest-edge search over every edge) even when nowhere near it dominated
    the whole control step's cost -- confirmed by profiling, roughly 4x slower overall than
    skipping obviously-safe obstacles."""
    single = pose.ndim == 1
    pose = np.atleast_2d(pose).copy()
    for _ in range(iterations):
        for obs in obstacles:
            reach = shape.bounding_radius + obs.bounding_radius + OBSTACLE_MARGIN
            if np.all(np.linalg.norm(pose[:, :2] - obs.center, axis=-1) > reach):
                continue
            verts = pose[:, None, :2] + rotate(pose[:, None, 2], shape.boundary_samples[None])
            k, v, _ = verts.shape
            d, grad = obs.sdf_and_grad(verts.reshape(-1, 2))
            d, grad = d.reshape(k, v), grad.reshape(k, v, 2)
            idx = np.argmin(d, axis=1)
            d_worst = d[np.arange(k), idx]
            grad_worst = grad[np.arange(k), idx]
            pose[:, :2] -= np.where((d_worst < 0.0)[:, None], d_worst[:, None] * grad_worst, 0.0)
    return pose[0] if single else pose


# ---------------------------------------------------------------------------------------
# Object dynamics (quasi-static, SE(2))
# ---------------------------------------------------------------------------------------
class QuasiStaticObject:
    """SE(2) object with quasi-static wrench-driven dynamics [Eq. qs_dynamics]."""

    def __init__(self, shape, pose):
        d_trans = 1.0 / (MU * OBJECT_MASS * GRAVITY)
        d_rot = 1.0 / (LIMIT_SURFACE_C * LIMIT_SURFACE_R * MU * OBJECT_MASS * GRAVITY)
        self.D = np.array([d_trans, d_trans, d_rot])
        self.shape = shape
        self.pose = np.asarray(pose, float)

    def body_frame_point(self, world_point, pose):
        """q_t = R^T(theta)(x_r - p_o), the robot position queried in the object's body frame
        [Sec. example1]."""
        return rotate(-pose[..., 2], world_point - pose[..., :2])

    def geometry(self, body_point, theta):
        """World-frame inward normal/tangent and body-frame moment arms at body_point (M,2)
        [Eqs. ex1_normal_tangent, ex1_boundary_projection]."""
        _, grad = self.shape.sdf_and_grad(body_point)
        n_body = -grad
        t_body = np.stack([-n_body[..., 1], n_body[..., 0]], axis=-1)
        gamma_n = body_point[..., 0] * n_body[..., 1] - body_point[..., 1] * n_body[..., 0]
        gamma_t = body_point[..., 0] * t_body[..., 1] - body_point[..., 1] * t_body[..., 0]
        return rotate(theta, n_body), rotate(theta, t_body), gamma_n, gamma_t

    def wrench(self, body_point, theta, f_n, f_t):
        """w^o(p, f_n, f_t) = J_c(p)^T f_c  [Eq. wrench_prop], general 2-DOF contact force.
        Object-level rollouts always pass f_t=0, since u^o_t never proposes a tangential
        component [Eq. contact_force]; the robot side (Sec. example1_gamma_r) passes the real
        Coulomb-resolved f_t to propagate its own simulated object trajectory."""
        n_world, t_world, gamma_n, gamma_t = self.geometry(body_point, theta)
        force = f_n[..., None] * n_world + f_t[..., None] * t_world
        torque = f_n * gamma_n + f_t * gamma_t
        return np.concatenate([force, torque[..., None]], axis=-1)

    def propagate(self, pose, w_o, dt=DT):
        """x_o <- x_o + dt D w_o  [Eq. qs_dynamics]."""
        new_pose = pose + dt * self.D * w_o
        new_pose[..., 2] = wrap_angle(new_pose[..., 2])
        return new_pose

    def world_vertices(self, pose=None):
        pose = self.pose if pose is None else pose
        return pose[:2] + rotate(pose[2], self.shape.vertices)


def rollout_object(object_, pose0, p_seq, f_n_seq, dt=DT, obstacles=()):
    """Batched quasi-static rollout under a per-rollout, per-timestep contact point sequence
    [Eqs. contact_force, wrench_prop, qs_dynamics]. p_seq: (K, H, 2) body-frame contact points
    (already accepted by rejection sampling, or the deterministic nominal broadcast with K=1);
    f_n_seq: (K, H) normal-force magnitudes. Object-level contact force is pure normal
    (f_t = 0 throughout), per Eq. contact_force. Returns poses (K, H, 3)."""
    k, h, _ = p_seq.shape
    pose = np.tile(pose0, (k, 1))
    poses = np.zeros((k, h, 3))
    zero_ft = np.zeros(k)
    for t in range(h):
        w_o = object_.wrench(p_seq[:, t], pose[:, 2], f_n_seq[:, t], zero_ft)
        pose = object_.propagate(pose, w_o, dt)
        pose = push_object_out_of_obstacles(object_.shape, pose, obstacles)
        poses[:, t] = pose
    return poses


def sample_contact_points(shape, p_mean, sigma_p, tau_n, rng, k, max_tries=MAX_REJECTION_TRIES):
    """Joint rejection sampling of K candidate contact points around each timestep's running
    mean [Sec. object_mppi_loop item 1, Eq. reject_criterion]: draw, project onto the boundary,
    accept only if normal-aligned with the mean, else redraw. p_mean: (H, 2) body-frame points,
    one running mean per timestep. Returns (K, H, 2) accepted body-frame points.

    The alignment test compares BODY-FRAME normals rather than rotating both into world frame at
    each rollout's own current orientation: a 2D rotation preserves dot products, so
    n_hat_mean . n_hat_candidate is unchanged whether compared before or after rotating both by
    the same angle. Comparing in body frame sidesteps the question of *which* rollout's evolving
    orientation to rotate by (each of the K rollouts has diverged onto its own trajectory by the
    time step t is reached), without changing what the test means.

    A hard cap on redraw attempts is a practical necessity for a vectorized, fixed-shape batch:
    without one, a mean sitting exactly at a sharp vertex between two near-perpendicular edges
    could reject indefinitely for an unlucky draw. Anything still unaccepted after max_tries
    falls back to the mean point itself, which is always accepted trivially against itself."""
    h = len(p_mean)
    points = np.zeros((k, h, 2))
    for t in range(h):
        mean_pt = p_mean[t]
        _, mean_grad = shape.sdf_and_grad(mean_pt[None, :])
        n_mean = -mean_grad[0]

        cand = shape.project_to_boundary(mean_pt + sigma_p * rng.standard_normal((k, 2)))
        accepted = np.zeros(k, dtype=bool)
        for _ in range(max_tries):
            _, grad_c = shape.sdf_and_grad(cand)
            aligned = (-grad_c) @ n_mean >= tau_n
            accepted |= aligned
            if accepted.all():
                break
            redraw = ~accepted
            cand[redraw] = shape.project_to_boundary(
                mean_pt + sigma_p * rng.standard_normal((int(redraw.sum()), 2)))
        cand[~accepted] = mean_pt
        points[:, t] = cand
    return points


# ---------------------------------------------------------------------------------------
# Robot-side contact resolution: single-point complementarity + Coulomb friction
# ---------------------------------------------------------------------------------------
def resolve_contact(object_, pose, robot_pos_free, robot_vel_cmd, dt=DT):
    """Closed-form single-contact resolution [Eqs. ex1_normal_complementarity, ex1_gamma_r].
    Batched over the leading dimension. Returns f_n, f_t, contact point (body frame), and an
    in-contact mask. A_r extracts only f_n for the ADMM layer (Sec. coupled_opt); the robot's
    own simulated object trajectory still needs both to propagate via object_.wrench."""
    q = object_.body_frame_point(robot_pos_free, pose)
    d, grad = object_.shape.sdf_and_grad(q)
    in_contact = d <= 0.0
    n_body = -grad
    t_body = np.stack([-n_body[..., 1], n_body[..., 0]], axis=-1)
    t_world = rotate(pose[..., 2], t_body)

    penetration = np.clip(-d, 0.0, None)
    f_n = np.clip(penetration / (dt * object_.D[0]), 0.0, F_MAX)
    v_t = np.einsum("...i,...i->...", robot_vel_cmd, t_world)
    sliding = np.abs(v_t) > 1e-4
    f_t = np.where(sliding, -MU_C * f_n * np.sign(v_t), 0.0)
    f_n = np.where(in_contact, f_n, 0.0)
    f_t = np.where(in_contact, f_t, 0.0)

    contact_body = q - d[..., None] * grad
    return f_n, f_t, contact_body, in_contact


def _contact_substep(object_, pose, robot_pos, robot_vel, dt, obstacles=()):
    """One collision-safe sub-step: robot moves kinematically, the contact resolves f_n/f_t,
    the object responds via its quasi-static dynamics, and the robot is kept out of the object.

    Obstacles are static (they never move or exert a reaction), so they need only a basic,
    strict non-penetration constraint, not a force response: the robot's allowed step is also
    capped by its clearance to every obstacle (same reasoning as the object cap below), and the
    object is translated out of any obstacle its boundary sample points end up inside after the
    quasi-static update. Real IsaacGym or hardware deployment resolves this in the physics
    engine automatically; this exists only so the 2D prototype does not let obstacles be ignored.

    The commanded displacement is capped to the robot's current clearance d_current plus a
    small fixed contact margin before anything else happens. This is not a heuristic: a signed
    distance is by definition a lower bound on distance to the surface in every direction, so
    moving no further than that (plus a margin far smaller than the object's thinnest feature)
    can never tunnel through, no matter how fast the commanded velocity is. Checking only the
    end-of-step position, as the correction below does, cannot catch a jump that lands clean on
    the far side of a thin part. Capping to a shrinking fraction of d_current instead of a fixed
    margin would work too, but asymptotically prevents ever actually reaching d <= 0, so contact
    would never register.

    Once the robot is already at or past the surface (d_current <= 0), d_current + margin is no
    longer a meaningful bound: it stops shrinking the allowed step as penetration deepens, so a
    large commanded velocity while already touching would sail straight through with no cap at
    all -- exactly the regime where a thin feature is most at risk, not least. There, the step is
    instead capped to a small fixed MAX_CONTACT_STEP, comfortably under the object's thinnest
    feature, so the bound never disappears regardless of sign or magnitude of d_current.

    This guard only bounds the ROBOT's own motion, not the OBJECT's: a large wrench can rotate
    or translate the object enough in one step that some other part of its boundary sweeps past
    the robot, which the guard above never sees and the end-of-step check below can miss for the
    same reason a fast robot could tunnel through a stationary object. simulate_contact_step
    calls this at a shrunk sub-step size so that sweep stays small on both sides at once."""
    q_current = object_.body_frame_point(robot_pos, pose)
    d_current = object_.shape.sdf(q_current)
    disp = dt * robot_vel
    disp_norm = np.linalg.norm(disp, axis=-1)
    safe_dist = np.where(d_current > 0.0, d_current + CONTACT_STEP_MARGIN, MAX_CONTACT_STEP)
    for obs in obstacles:
        if np.all(np.linalg.norm(robot_pos - obs.center, axis=-1) > obs.bounding_radius + MAX_CONTACT_STEP):
            continue
        d_obs = obs.sdf(robot_pos)
        obs_safe = np.where(d_obs > 0.0, d_obs + CONTACT_STEP_MARGIN, MAX_CONTACT_STEP)
        safe_dist = np.minimum(safe_dist, obs_safe)
    scale = np.clip(np.divide(safe_dist, disp_norm, out=np.ones_like(disp_norm), where=disp_norm > 1e-12), 0.0, 1.0)
    robot_free = robot_pos + scale[..., None] * disp
    robot_free = push_point_out_of_obstacles(robot_free, obstacles)

    f_n, f_t, contact_body, _ = resolve_contact(object_, pose, robot_free, robot_vel, dt)
    w_o = object_.wrench(contact_body, pose[..., 2], f_n, f_t)
    new_pose = object_.propagate(pose, w_o, dt)
    new_pose = push_object_out_of_obstacles(object_.shape, new_pose, obstacles)

    q_check = object_.body_frame_point(robot_free, new_pose)
    d_check, grad_check = object_.shape.sdf_and_grad(q_check)
    penetrating = d_check < 0.0
    q_proj = q_check - d_check[..., None] * grad_check
    corrected = new_pose[..., :2] + rotate(new_pose[..., 2], q_proj)
    new_robot_pos = np.where(penetrating[..., None], corrected, robot_free)
    new_robot_pos = push_point_out_of_obstacles(new_robot_pos, obstacles)
    return new_pose, new_robot_pos, f_n


def simulate_contact_step(object_, pose, robot_pos, robot_vel, dt=DT, n_substeps=N_CONTACT_SUBSTEPS, obstacles=()):
    """One coupled control-step, split into several smaller collision-safe sub-steps so that
    neither body's motion within the step is large enough to sweep past the other undetected.
    Returns the sub-step-averaged realized normal-force magnitude (A_r's per-timestep reading,
    Sec. coupled_opt), alongside the updated pose and robot position."""
    sub_dt = dt / n_substeps
    f_n_sum = np.zeros(robot_pos.shape[:-1])
    for _ in range(n_substeps):
        pose, robot_pos, f_n = _contact_substep(object_, pose, robot_pos, robot_vel, sub_dt, obstacles)
        f_n_sum = f_n_sum + f_n
    return pose, robot_pos, f_n_sum / n_substeps


# ---------------------------------------------------------------------------------------
# Step 0: contact point estimator (samples directly in R^2, importance-weighted mean update)
# ---------------------------------------------------------------------------------------
class ContactPointEstimator:
    def __init__(self, object_, obstacles, goal, rng):
        self.object_, self.obstacles, self.goal, self.rng = object_, obstacles, goal, rng

    def estimate(self, pose, mean_prev, f_n_nom, sigma_scale=1.0):
        """Cross-entropy search over body-frame points on the boundary [Sec. contact_estimator].
        Runs once per control step to seed the per-timestep mean sequence (Algorithm Step 0);
        not rejection-sampled itself -- the per-round update list of Sec. resampling has no
        accept/reject test, only a plain importance-weighted mean followed by resampling around
        the new mean.

        sigma_scale (from ADMMPushingController, shrinking as the object nears the goal) also
        narrows the search here and sharpens the softmax: without it, the estimator keeps
        re-exploring at full width near the goal and can wander onto a worse contact strategy
        after already finding a good one, which shows up as the object drifting back away from
        the goal after getting close, not just failing to fully converge."""
        mean = self.object_.shape.project_to_boundary(mean_prev[None, :])[0]
        sigma = SIGMA_INIT_EST * sigma_scale
        tau = TAU_P_EST * sigma_scale
        for _ in range(R_EST):
            samples = mean + sigma * self.rng.standard_normal((N_P_EST, 2))
            samples = self.object_.shape.project_to_boundary(samples)
            cost = self._score(pose, samples, f_n_nom)
            weights = softmax(-cost / tau)
            mean = self.object_.shape.project_to_boundary(
                (weights[:, None] * samples).sum(axis=0, keepdims=True))[0]
            sigma = max(sigma * GAMMA_EST, SIGMA_MIN_EST)
        return mean

    def _score(self, pose, samples, f_n_nom):
        """Forward-rollout cost per candidate [Sec. resampling, items 1-3]. No ADMM penalty
        term: f_n_nom is held fixed across every candidate in a round, so the object-side
        consensus term would be a constant offset, identical for every candidate."""
        n = len(samples)
        w_o = self.object_.wrench(samples, np.full(n, pose[2]), np.full(n, f_n_nom), np.zeros(n))
        traj_pose = np.tile(pose, (n, 1))
        cost = np.zeros(n)
        for _ in range(H_P_EST):
            traj_pose = self.object_.propagate(traj_pose, w_o)
            traj_pose = push_object_out_of_obstacles(self.object_.shape, traj_pose, self.obstacles)
            cost += goal_cost(traj_pose, self.goal, Q_POS, Q_THETA)
            cost += obstacle_cost(self.object_.shape, traj_pose, self.obstacles)
        cost += goal_cost(traj_pose, self.goal, QF_POS, QF_THETA)
        return cost


# ---------------------------------------------------------------------------------------
# Step 1: object-level MPPI (contact point and normal force jointly rejection-sampled)
# ---------------------------------------------------------------------------------------
class ObjectLevelMPPI:
    def __init__(self, object_, obstacles, goal, rng):
        self.object_, self.obstacles, self.goal, self.rng = object_, obstacles, goal, rng

    def solve(self, pose0, p_mean, f_n_nom, z_seq, y_o_seq, sigma_scale=1.0):
        """One object update [Eq. obj_update]: joint rejection-sampled (p^t, f_n^t) MPPI with
        the scalar ADMM penalty [Sec. object_mppi_loop, Algorithm Step 1]. p_mean, f_n_nom are
        the per-timestep running means (H, 2) and (H,); returns their importance-weighted
        updates plus the deterministic reference rollout at the new means."""
        sigma_o = SIGMA_O * sigma_scale
        sigma_p = SIGMA_P * sigma_scale

        p_k = sample_contact_points(self.object_.shape, p_mean, sigma_p, TAU_N, self.rng, K_OBJECT)
        eps_fn = self.rng.standard_normal((K_OBJECT, HORIZON)) * sigma_o
        f_n_k = np.clip(f_n_nom[None] + eps_fn, 0.0, F_MAX)

        poses = rollout_object(self.object_, pose0, p_k, f_n_k, obstacles=self.obstacles)
        running = goal_cost(poses[:, :-1].reshape(-1, 3), self.goal, Q_POS, Q_THETA).reshape(K_OBJECT, -1).sum(1)
        running += obstacle_cost(self.object_.shape, poses[:, :-1].reshape(-1, 3), self.obstacles).reshape(K_OBJECT, -1).sum(1)
        terminal = goal_cost(poses[:, -1], self.goal, QF_POS, QF_THETA)
        admm_diff = f_n_k - z_seq[None] + y_o_seq[None]
        admm = 0.5 * RHO * (admm_diff ** 2).sum(1)
        weights = softmax(-(running + terminal + admm) / MPPI_LAMBDA)

        # Importance-weighted mean update [Eqs. nominal_update, point_nominal_update]: safe to
        # average the accepted contact points directly (not resample) because rejection
        # sampling already guaranteed every p_k[:, t] is normal-consistent with p_mean[t].
        f_n_nom = np.clip(f_n_nom + np.einsum("k,kt->t", weights, eps_fn), 0.0, F_MAX)
        p_mean = self.object_.shape.project_to_boundary(np.einsum("k,kti->ti", weights, p_k))

        ref_poses = rollout_object(
            self.object_, pose0, p_mean[None], f_n_nom[None], obstacles=self.obstacles)[0]
        return f_n_nom, p_mean, ref_poses


# ---------------------------------------------------------------------------------------
# Step 2: robot-level MPPI (standard point-mass planning)
# ---------------------------------------------------------------------------------------
class RobotLevelMPPI:
    def __init__(self, object_, obstacles, rng):
        self.object_, self.obstacles, self.rng = object_, obstacles, rng

    def solve(self, robot_pos0, pose0, ref_poses, u_nom, z_seq, y_r_seq, sigma_scale=1.0):
        """One robot update [Eq. robot_update], MPPI over 2D velocity with the scalar ADMM
        penalty on A_r, the realized normal-force magnitude (Sec. coupled_opt). The full
        (f_n, f_t) contact force still propagates the robot's own simulated object trajectory
        internally; only f_n is extracted into the penalty."""
        eps = self.rng.standard_normal((K_ROBOT, HORIZON, 2)) * (SIGMA_ROBOT * sigma_scale)
        u_k = u_nom[None] + eps
        pose = np.tile(pose0, (K_ROBOT, 1))
        robot_pos = np.tile(robot_pos0, (K_ROBOT, 1))
        cost = np.zeros(K_ROBOT)
        for t in range(HORIZON):
            pose, robot_pos, f_n = simulate_contact_step(
                self.object_, pose, robot_pos, u_k[:, t], obstacles=self.obstacles)
            diff = pose - ref_poses[t]
            diff[:, 2] = wrap_angle(diff[:, 2])
            cost += Q_TRACK_POS * np.sum(diff[:, :2] ** 2, axis=1) + Q_TRACK_THETA * diff[:, 2] ** 2
            admm_diff = f_n - z_seq[t] + y_r_seq[t]
            cost += 0.5 * RHO * admm_diff ** 2
        weights = softmax(-cost / MPPI_LAMBDA)
        u_nom = u_nom + np.einsum("k,kti->ti", weights, eps)

        pose, robot_pos = pose0[None], robot_pos0[None]
        f_n_seq = np.zeros(HORIZON)
        for t in range(HORIZON):
            pose, robot_pos, f_n = simulate_contact_step(
                self.object_, pose, robot_pos, u_nom[t][None], obstacles=self.obstacles)
            f_n_seq[t] = f_n[0]
        return u_nom, f_n_seq


# ---------------------------------------------------------------------------------------
# ADMM coordination layer (Steps 0-5, Algorithm admm_mppi) + closed-loop execution
# ---------------------------------------------------------------------------------------
class ADMMPushingController:
    def __init__(self, object_, robot_pos, obstacles, goal, rng):
        self.object_ = object_
        self.robot_pos = np.asarray(robot_pos, float)
        self.obstacles = obstacles
        self.goal = np.asarray(goal, float)
        self.rng = rng
        self.estimator = ContactPointEstimator(object_, obstacles, goal, rng)
        self.object_mppi = ObjectLevelMPPI(object_, obstacles, goal, rng)
        self.robot_mppi = RobotLevelMPPI(object_, obstacles, rng)

        q0 = self.object_.body_frame_point(self.robot_pos, self.object_.pose)
        p0 = self.object_.shape.project_to_boundary(q0[None, :])[0]
        self.p_mean = np.tile(p0, (HORIZON, 1))
        self.f_n_nom = np.zeros(HORIZON)
        self.u_nom = np.zeros((HORIZON, 2))
        self.z = np.zeros(HORIZON)
        self.y_o = np.zeros(HORIZON)
        self.y_r = np.zeros(HORIZON)

    @staticmethod
    def _shift(seq):
        seq = np.roll(seq, -1, axis=0)
        seq[-1] = seq[-2]
        return seq

    def control_step(self):
        pose0 = self.object_.pose.copy()
        self.f_n_nom, self.u_nom = self._shift(self.f_n_nom), self._shift(self.u_nom)
        self.p_mean, self.z = self._shift(self.p_mean), self._shift(self.z)
        self.y_o, self.y_r = self._shift(self.y_o), self._shift(self.y_r)

        # Shrink MPPI exploration noise as the object nears the goal. A fixed-variance sampler
        # never converges tighter than its own noise floor: near the goal, a perturbation as
        # large as the one that was useful for a 60cm approach mostly just re-disturbs an
        # already-good pose, so the object visibly jitters around the target instead of
        # settling into it. Scale is 1 far from the goal and anneals linearly down to
        # MIN_SIGMA_SCALE (never fully off, so it can still correct residual error) once within
        # SIGMA_ANNEAL_BAND tolerances on whichever of position or orientation is furthest off.
        pos_err0 = np.linalg.norm(pose0[:2] - self.goal[:2])
        theta_err0 = abs(wrap_angle(pose0[2] - self.goal[2]))
        normalized_err = max(pos_err0 / GOAL_POS_TOL, theta_err0 / GOAL_THETA_TOL)
        sigma_scale = np.clip(normalized_err / SIGMA_ANNEAL_BAND, MIN_SIGMA_SCALE, 1.0)

        # Step 0 -- contact point estimation [Algorithm Step 0]: runs once per control step,
        # its single output broadcasts to seed every one of the H^c per-timestep means below.
        p0 = self.estimator.estimate(pose0, self.p_mean[0], self.f_n_nom[0], sigma_scale)
        self.p_mean = np.tile(p0, (HORIZON, 1))

        # re-seed the robot's nominal toward the target contact point whenever it isn't
        # already touching: MPPI exploring from a zero-velocity nominal almost never randomly
        # drifts into contact within one horizon, so pure noise alone cannot bootstrap it.
        # The speed is floored, not just gap/horizon: dividing a small remaining gap by the
        # whole horizon gives a vanishingly small command that the exploration noise (std
        # SIGMA_ROBOT) swamps completely, so the robot would hover near the target and never
        # actually close the last few millimeters. The SDF clearance guard in
        # simulate_contact_step is what keeps a large commanded speed safe, so there is no
        # need to be gentle here.
        p_world = pose0[:2] + rotate(pose0[2], p0)
        gap = p_world - self.robot_pos
        gap_norm = np.linalg.norm(gap)
        if gap_norm > CONTACT_STEP_MARGIN:
            speed = np.clip(gap_norm / DT, SEEK_MIN_SPEED, SEEK_MAX_SPEED)
            self.u_nom = np.tile((gap / gap_norm) * speed, (HORIZON, 1))

        residuals = []
        for _ in range(N_ADMM):
            self.f_n_nom, self.p_mean, ref_poses = self.object_mppi.solve(
                pose0, self.p_mean, self.f_n_nom, self.z, self.y_o, sigma_scale)
            self.u_nom, f_n_r_seq = self.robot_mppi.solve(
                self.robot_pos, pose0, ref_poses, self.u_nom, self.z, self.y_r, sigma_scale)

            # Step 3 -- consensus update [Eq. z_update]: average the two sides' bias-corrected
            # normal-force estimates and clip to non-negative; no cone projection needed now
            # that tangential force is out of consensus (Sec. friction_cone).
            z_new = np.maximum(0.0, 0.5 * (self.f_n_nom + self.y_o + f_n_r_seq + self.y_r))
            # Step 4 -- dual update [Eq. dual_update]
            y_o_new = clip_dual(self.y_o + self.f_n_nom - z_new)
            y_r_new = clip_dual(self.y_r + f_n_r_seq - z_new)

            # Step 5 -- convergence check
            primal = np.concatenate([self.f_n_nom - z_new, f_n_r_seq - z_new])
            dual = RHO * (z_new - self.z)
            residuals.append((np.linalg.norm(primal), np.linalg.norm(dual)))
            self.z, self.y_o, self.y_r = z_new, y_o_new, y_r_new
            if residuals[-1][0] <= EPS_R and residuals[-1][1] <= EPS_S:
                break

        return self.u_nom[0].copy(), residuals

    def run(self, max_steps=MAX_CONTROL_STEPS, verbose=True):
        log = {"object_pose": [self.object_.pose.copy()], "robot_pos": [self.robot_pos.copy()], "residuals": []}
        reached = False
        for step in range(max_steps):
            u0, residuals = self.control_step()
            new_pose, new_robot_pos, _ = simulate_contact_step(
                self.object_, self.object_.pose[None], self.robot_pos[None], u0[None], obstacles=self.obstacles)
            self.object_.pose, self.robot_pos = new_pose[0], new_robot_pos[0]

            log["object_pose"].append(self.object_.pose.copy())
            log["robot_pos"].append(self.robot_pos.copy())
            log["residuals"].extend(residuals)

            pos_err = np.linalg.norm(self.object_.pose[:2] - self.goal[:2])
            theta_err = abs(wrap_angle(self.object_.pose[2] - self.goal[2]))
            if verbose and step % 20 == 0:
                print(f"step {step:4d}  pos_err={pos_err:.3f}  theta_err={theta_err:.3f}  admm_iters={len(residuals)}")
            if pos_err < GOAL_POS_TOL and theta_err < GOAL_THETA_TOL:
                reached = True
                if verbose:
                    print(f"goal reached at step {step}")
                break
        log["reached"] = reached
        log["object_pose"] = np.array(log["object_pose"])
        log["robot_pos"] = np.array(log["robot_pos"])
        return log


# ---------------------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------------------
def _obstacle_patch(obs):
    if isinstance(obs, CircleObstacle):
        return MplCircle(obs.center, obs.radius, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    if isinstance(obs, BoxObstacle):
        corners = obs.center + rotate(obs.angle, np.array([
            [-obs.half_extents[0], -obs.half_extents[1]], [obs.half_extents[0], -obs.half_extents[1]],
            [obs.half_extents[0], obs.half_extents[1]], [-obs.half_extents[0], obs.half_extents[1]],
        ]))
        return MplPolygon(corners, closed=True, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    if isinstance(obs, PolygonShape):
        return MplPolygon(obs.vertices, closed=True, fc="tab:gray", ec="k", alpha=0.7, zorder=2)
    raise TypeError(f"unknown obstacle type {type(obs)}")


def plot_overview(log, shape, obstacles, goal, save_path, n_poses=8):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for obs in obstacles:
        ax.add_patch(_obstacle_patch(obs))

    idx = np.linspace(0, len(log["object_pose"]) - 1, n_poses).astype(int)
    for i, k in enumerate(idx):
        pose = log["object_pose"][k]
        verts = pose[:2] + rotate(pose[2], shape.vertices)
        alpha = 0.15 + 0.65 * i / (len(idx) - 1)
        ax.add_patch(MplPolygon(verts, closed=True, fc="tab:blue", ec="tab:blue", alpha=alpha, zorder=3))

    goal_verts = goal[:2] + rotate(goal[2], shape.vertices)
    ax.add_patch(MplPolygon(goal_verts, closed=True, fill=False, ec="tab:green", lw=2, ls="--", zorder=4, label="goal"))

    ax.plot(log["robot_pos"][:, 0], log["robot_pos"][:, 1], "-", color="tab:red", lw=1, alpha=0.6, zorder=3)
    ax.plot(*log["robot_pos"][0], "o", color="tab:red", ms=7, zorder=5, label="robot")
    ax.plot(*log["object_pose"][0, :2], "s", color="tab:blue", ms=6, zorder=5, label="object start")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    status = "reached" if log["reached"] else "not reached"
    ax.set_title(f"ADMM-OI-MPPI planar pushing, Example 1 (goal {status})")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(log, save_path):
    residuals = np.array(log["residuals"])
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(residuals[:, 0], label="primal residual $\\|r\\|$", lw=1)
    ax.plot(residuals[:, 1], label="dual residual $\\|s\\|$", lw=1)
    ax.axhline(EPS_R, color="k", lw=0.7, ls=":", label="tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("ADMM iteration (concatenated over control steps)")
    ax.set_ylabel("residual norm")
    ax.set_title("ADMM consensus convergence")
    ax.legend(fontsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_animation(log, shape, obstacles, goal, save_path, stride=2, fps=15):
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    for obs in obstacles:
        ax.add_patch(_obstacle_patch(obs))
    goal_verts = goal[:2] + rotate(goal[2], shape.vertices)
    ax.add_patch(MplPolygon(goal_verts, closed=True, fill=False, ec="tab:green", lw=2, ls="--", zorder=4))
    object_patch = MplPolygon(shape.vertices, closed=True, fc="tab:blue", ec="k", alpha=0.85, zorder=3)
    ax.add_patch(object_patch)
    (robot_dot,) = ax.plot([], [], "o", color="tab:red", ms=8, zorder=5)
    (robot_trail,) = ax.plot([], [], "-", color="tab:red", lw=1, alpha=0.5, zorder=3)

    all_xy = np.concatenate([log["object_pose"][:, :2], log["robot_pos"]], axis=0)
    margin = 0.1
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)
    ax.set_aspect("equal")

    frames = list(range(0, len(log["object_pose"]), stride))

    def update(i):
        pose = log["object_pose"][i]
        object_patch.set_xy(pose[:2] + rotate(pose[2], shape.vertices))
        robot_dot.set_data([log["robot_pos"][i, 0]], [log["robot_pos"][i, 1]])
        robot_trail.set_data(log["robot_pos"][:i + 1, 0], log["robot_pos"][:i + 1, 1])
        return object_patch, robot_dot, robot_trail

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def build_scenario():
    shape = PolygonShape(t_shape_vertices())
    object_ = QuasiStaticObject(shape, pose=np.array([0.0, 0.0, 0.0]))
    robot_pos = np.array([-0.05, -0.06])
    goal = np.array([0.45, 0.5, np.pi / 4])
    obstacles = [
        CircleObstacle(center=np.array([0.06, 0.30]), radius=0.04),
        BoxObstacle(center=np.array([0.36, 0.16]), half_extents=np.array([0.04, 0.035]), angle=0.2),
        PolygonShape(np.array([[0.08, 0.40], [0.18, 0.40], [0.13, 0.49]])),
    ]
    return shape, object_, robot_pos, goal, obstacles


def run(save_dir=RESULTS_DIR, max_steps=MAX_CONTROL_STEPS):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    shape, object_, robot_pos, goal, obstacles = build_scenario()

    controller = ADMMPushingController(object_, robot_pos, obstacles, goal, rng)
    log = controller.run(max_steps=max_steps)

    print(f"final object pose: {log['object_pose'][-1]}")
    print(f"goal pose:          {goal}")
    print(f"goal reached:       {log['reached']}")

    overview_path = unique_path(save_dir / "trajectory_overview.png")
    plot_overview(log, shape, obstacles, goal, overview_path)
    print(f"saved {overview_path}")

    residual_path = unique_path(save_dir / "admm_residuals.png")
    plot_residuals(log, residual_path)
    print(f"saved {residual_path}")

    if SAVE_ANIMATION:
        try:
            anim_path = unique_path(save_dir / "pushing_animation.gif")
            save_animation(log, shape, obstacles, goal, anim_path)
            print(f"saved {anim_path}")
        except Exception as exc:  # optional dependency (Pillow) may be missing
            print(f"skipped animation: {exc}")

    return log


if __name__ == "__main__":
    run()
