import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from llampc.rollout.rk6 import _rk4_step, rollout
import llampc.rollout.dynamic as dynamics

# State / control conventions: x = [px, py, ψ, vx, vy, ω], u = [δ, d]
PX_, PY_, PSI_, VX_, VY_, R_ = 0, 1, 2, 3, 4, 5
DELTA_, D_                    = 0, 1
NX = 6

# Default actuator limits
DELTA_MAX = 0.40
D_MIN, D_MAX = 0.0, 0.5

_rollout_traj = rollout(_rk4_step, dynamics.diffequation)

# ── Obstacle CBF: h_i(x) = ‖p − p_i‖² − (r_i + r_car)² ──────────────────
def _h_obstacle(x, p_obs, r_obs, r_car):
    dx, dy = x[PX_] - p_obs[0], x[PY_] - p_obs[1]
    return dx*dx + dy*dy - (r_obs + r_car)**2

def _h_min(x, obstacles, r_car):
    """h(x) = min_i h_i(x)  over all obstacles.  Positive ⇔ safe."""
    if not obstacles:
        return np.inf
    return min(_h_obstacle(x, p, r, r_car) for p, r in obstacles)

# ── Predictive HOCBF condition (trajectory-min over an N-step rollout) ──
def _cbf_psi(x, u, dt, lla_params, known_params, obstacles, r_car, alpha, N,
             return_states=False):
    u_dyn = np.array([u[D_], u[DELTA_]])  # [δ, d] -> [pwm, steer] for diffequation
    xs_rolled = np.asarray(_rollout_traj(lla_params, known_params, x, u_dyn, dt, N))
    xs = np.vstack([x, xs_rolled])  
    h_now = _h_min(x, obstacles, r_car)
    decay_base = 1.0 - alpha * dt
    psis = np.empty(N)
    for i in range(1, N + 1):
        h_i = _h_min(xs[i], obstacles, r_car)
        decay = decay_base ** i
        psis[i - 1] = h_i - decay * h_now
    psi = float(psis.min())
    if return_states:
        return psi, xs
    return psi

# ── The QP-solver function ──────────────────────────────────────────────
def cbf_qp_pacejka(x, u_nom, lla_params, known_params, obstacles,
                    r_car=0.04,
                    dt=0.01, alpha=2.5, N=10,
                    delta_max=DELTA_MAX, d_min=D_MIN, d_max=D_MAX,
                    w_delta=1.0, w_d=1/0.35,
                    eps_fd=(0.05, 1e-3), policy=None):

    x = np.asarray(x, dtype=np.float64)
    u_nom = np.asarray(u_nom, dtype=np.float64)

    psi0, rollout_xs = _cbf_psi(x, u_nom, dt, lla_params, known_params,
                                obstacles, r_car, alpha, N, return_states=True)
    if psi0 >= 0:
        return u_nom.copy(), dict(active=False, psi=psi0,
                                  grad_psi=np.zeros(2), rollout=rollout_xs)

    grad = np.zeros(2)
    for i in range(2):
        du = np.zeros(2); du[i] = eps_fd[i]
        psi_p = _cbf_psi(x, u_nom + du, dt, lla_params, known_params,
                          obstacles, r_car, alpha, N)
        psi_m = _cbf_psi(x, u_nom - du, dt, lla_params, known_params,
                          obstacles, r_car, alpha, N)
        grad[i] = (psi_p - psi_m) / (2.0 * eps_fd[i])

    Wi = np.array([1.0 / w_delta, 1.0 / w_d])
    den = float(grad @ (Wi * grad))
    if den < 1e-10:
        return u_nom.copy(), dict(active=False, psi=psi0, grad_psi=grad,
                                  rollout=rollout_xs)
    delta_u = (-psi0 / den) * (Wi * grad)
    u_safe = u_nom + delta_u

    if psi0 < 0:
        brake_psi_scale = 0.3
        bf = min(1.0, -psi0 / brake_psi_scale)
        u_safe[D_] = (1.0 - bf) * float(u_nom[D_]) + bf * d_min

    u_safe[DELTA_] = float(np.clip(u_safe[DELTA_], -delta_max, delta_max))
    u_safe[D_]     = float(np.clip(u_safe[D_],     d_min,     d_max))
    return u_safe, dict(active=True, psi=psi0, grad_psi=grad,
                        rollout=rollout_xs)
