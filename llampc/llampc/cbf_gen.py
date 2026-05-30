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
D_MIN, D_MAX = -1.0, 1.0

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
    """ψ(x, u; θ) = min_{i ∈ 0..N} [ h(x_i(u)) − (1 − α·i·dt)·h(x_0) ].
    Trajectory-min keeps the filter reactive to obstacles the endpoint-
    only check would skip past at speed."""
    xs_rolled = np.asarray(_rollout_traj(lla_params, known_params, x, u, dt, N))
    xs = np.vstack([x, xs_rolled])  # shape (N+1, 6), xs[0]=x, xs[1..N]=rollout
    h_now = _h_min(x, obstacles, r_car)
    psis = np.empty(N + 1)
    for i in range(N + 1):
        h_i = _h_min(xs[i], obstacles, r_car)
        psis[i] = h_i - (1.0 - alpha * i * dt) * h_now
    return float(psis.min())


# ── The QP-solver function ──────────────────────────────────────────────
def cbf_qp_pacejka(x, u_nom, lla_params, known_params, obstacles, 
                    r_car=0.04,
                    dt=0.01, alpha=2.5, N=10,
                    delta_max=DELTA_MAX, d_min=D_MIN, d_max=D_MAX,
                    w_delta=1.0, w_d=1/0.35,
                    eps_fd=(1e-3, 1e-3)):
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
    alpha     : HOCBF discount (continuous-time α)
    N         : predictive horizon (number of RK4 steps in the rollout)
    delta_max, d_min, d_max : actuator limits
    w_delta, w_d : QP cost weights; larger w_d ⇒ prefer steering over braking
    eps_fd    : finite-difference perturbation for ∂ψ/∂u

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

    # Finite-difference gradient ∇_u ψ at u_nom.
    grad = np.zeros(2)
    for i in range(2):
        du = np.zeros(2); du[i] = eps_fd[i]
        psi_p = _cbf_psi(x, u_nom + du, dt, lla_params, known_params,
                          obstacles, r_car, alpha, N)
        grad[i] = (psi_p - psi0) / eps_fd[i]

    # Weighted QP: min ½‖u − u_nom‖²_W  s.t.  ψ_0 + ∇ψ·(u − u_nom) ≥ 0
    # with W = diag(w_delta, w_d).  Analytical half-space projection:
    #   u_safe = u_nom − (ψ_0 / ‖∇ψ‖²_{W⁻¹}) · (W⁻¹ ∇ψ)
    Wi = np.array([1.0 / w_delta, 1.0 / w_d])
    den = float(grad @ (Wi * grad))
    if den < 1e-10:
        return u_nom.copy(), dict(active=False, psi=psi0, grad_psi=grad)
    delta_u = (-psi0 / den) * (Wi * grad)
    u_safe = u_nom + delta_u
    u_safe[DELTA_] = float(np.clip(u_safe[DELTA_], -delta_max, delta_max))
    u_safe[D_]     = float(np.clip(u_safe[D_],     d_min,     d_max))
    return u_safe, dict(active=True, psi=psi0, grad_psi=grad)