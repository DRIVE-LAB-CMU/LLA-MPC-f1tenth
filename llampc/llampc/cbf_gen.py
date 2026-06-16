import casadi as ca
from acados_template import AcadosModel,  AcadosOcp, AcadosOcpSolver
import numpy as np
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from scipy.linalg import block_diag

from llampc.params import F110


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
    return min(_h_obstacle(x, p, r, r_car) for p, r in obstacles)

# ── Predictive HOCBF condition (trajectory-min over an N-step rollout) ──
def _cbf_psi(x, u, dt, lla_params, known_params, obstacles, r_car, alpha, N):
    """ψ(x, u; θ) = min_{i ∈ 1..N} [ h(x_i(u)) − (1 − α·dt)^i · h(x_0) ].

    Discrete exponential-CBF form. The required margin decays geometrically
    as (1 − α·dt)^i, which stays strictly positive and monotone for
    α·dt < 1, so far-horizon nodes still carry a real constraint and the
    factor can never go negative (which would invert the constraint and
    permit collisions downrange).

    The minimum runs over i = 1..N only. Node 0 is the current state x,
    where h_i = h_now and the old (1 − α·i·dt) form gave exactly
    h_now − 1·h_now = 0, pinning ψ ≤ 0 on every call and making the
    'already safe' guard unfirable. x_0 is also not a future the control
    can steer toward, so it does not belong in the predictive min.
    """
    xs_rolled = np.asarray(_rollout_traj(lla_params, known_params, x, u, dt, N))
    xs = np.vstack([x, xs_rolled])  # shape (N+1, 6), xs[0]=x, xs[1..N]=rollout
    h_now = _h_min(x, obstacles, r_car)
    decay_base = 1.0 - alpha * dt
    psis = np.empty(N)
    for i in range(1, N + 1):
        h_i = _h_min(xs[i], obstacles, r_car)
        decay = decay_base ** i
        psis[i - 1] = h_i - decay * h_now
    return float(psis.min())


# ── The QP-solver function ──────────────────────────────────────────────
def cbf_qp_pacejka(x, u_nom, lla_params, known_params, obstacles,
                    r_car=0.04,
                    dt=0.01, alpha=2.5, N=10,
                    delta_max=DELTA_MAX, d_min=D_MIN, d_max=D_MAX,
                    w_delta=1.0, w_d=1/0.35,
                    eps_fd=(0.05, 1e-3)):
    """Predictive HOCBF + weighted-2-D-QP obstacle-avoidance filter for
    the Pacejka race-car.

    Parameters
    ----------
    x         : (6,) current state [px, py, ψ, vx, vy, ω]
    u_nom     : (2,) nominal control [δ_nom, d_nom] from LQR / pure-pursuit
    theta     : (6,) Pacejka params [B_f, C_f, D_f, B_r, C_r, D_r] —
                supply the *estimator's* output (LLA bank cell, DOB+θ_dry, ...)
    obstacles : list of (p_obs ∈ ℝ², r_obs ∈ ℝ) tuples
    r_car     : car radius (added to each obstacle's r)
    dt        : integration step
    alpha     : HOCBF discount (continuous-time α). Keep α·dt < 1 so the
                discrete decay (1 − α·dt)^i stays positive.
    N         : predictive horizon (number of RK4 steps in the rollout)
    delta_max, d_min, d_max : actuator limits
    w_delta, w_d : QP cost weights; larger w_d ⇒ prefer steering over braking
    eps_fd    : central-difference perturbation for ∂ψ/∂u. The steering
                step must be large enough to cross the kink in the
                min-over-rollout (a sub-0.1° probe reads ≈0 at the obstacle
                bearing and the avoidance steer collapses to zero), so it
                defaults to ~3° on δ.

    Returns
    -------
    u_safe    : (2,) safe control after the QP correction
    info      : dict {
                  'active'   : True iff the filter modified u_nom,
                  'psi'      : current ψ value at u_nom,
                  'grad_psi' : (2,) finite-difference gradient ∇_u ψ,
                }
    """
    x = np.asarray(x, dtype=np.float64)
    u_nom = np.asarray(u_nom, dtype=np.float64)

    # Evaluate ψ at u_nom; if already safe, return as-is.
    psi0 = _cbf_psi(x, u_nom, dt, lla_params, known_params, obstacles, r_car, alpha, N)
    if psi0 >= 0:
        return u_nom.copy(), dict(active=False, psi=psi0,
                                    grad_psi=np.zeros(2))

    # Central-difference gradient ∇_u ψ at u_nom. Central (vs one-sided)
    # so the gradient can see whichever steering direction reduces the
    # predicted penetration even when ψ has a kink at the obstacle bearing.
    grad = np.zeros(2)
    for i in range(2):
        du = np.zeros(2); du[i] = eps_fd[i]
        psi_p = _cbf_psi(x, u_nom + du, dt, lla_params, known_params,
                          obstacles, r_car, alpha, N)
        psi_m = _cbf_psi(x, u_nom - du, dt, lla_params, known_params,
                          obstacles, r_car, alpha, N)
        grad[i] = (psi_p - psi_m) / (2.0 * eps_fd[i])

    # Weighted QP: min ½‖u − u_nom‖²_W  s.t.  ψ_0 + ∇ψ·(u − u_nom) ≥ 0
    # with W = diag(w_delta, w_d).  Analytical half-space projection:
    #   u_safe = u_nom − (ψ_0 / ‖∇ψ‖²_{W⁻¹}) · (W⁻¹ ∇ψ)
    Wi = np.array([1.0 / w_delta, 1.0 / w_d])
    den = float(grad @ (Wi * grad))
    if den < 1e-10:
        return u_nom.copy(), dict(active=False, psi=psi0, grad_psi=grad)
    delta_u = (-psi0 / den) * (Wi * grad)
    u_safe = u_nom + delta_u

    # While unsafe (ψ < 0), the filter may brake but must never command MORE
    # throttle than nominal: a positive d-correction here just drives the car
    # into the obstacle faster. Clamp the throttle channel to <= nominal
    # before the box clip.
    if psi0 < 0:
        u_safe[D_] = min(u_safe[D_], float(u_nom[D_]))

    u_safe[DELTA_] = float(np.clip(u_safe[DELTA_], -delta_max, delta_max))
    u_safe[D_]     = float(np.clip(u_safe[D_],     d_min,     d_max))
    return u_safe, dict(active=True, psi=psi0, grad_psi=grad)