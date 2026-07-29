from __future__ import annotations

import numpy as np
from numba import njit

from .utils import steering_constraint, cur_constraint


@njit(cache=True)
def vehicle_dynamics_fiala(x: np.ndarray, u_init: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fiala/brush-tire single-track model with wheel-speed, load-transfer,
    and motor-current states. Mirrors the acados `export_model` (exact branch).

        Args:
            x (numpy.ndarray (10,)): state vector
                x0: x position (global)
                x1: y position (global)
                x2: yaw angle (phi)
                x3: longitudinal velocity (vx)
                x4: lateral velocity (vy)
                x5: yaw rate (omega)
                x6: rear wheel speed (omega_w)
                x7: load-transfer state (dFz)
                x8: steer angle (delta)
            u_init (numpy.ndarray (2,)): control input vector
                u0: motor current
                u1: steering angle velocity of front wheels (ddelta/dt)
            params (numpy.ndarray (21,)):
                0  mass
                1  Iz
                2  lf
                3  lr
                4  rw       rear wheel radius
                5  Iw       rear driveline rotational inertia
                6  Im       motor rotor inertia
                7  pole_pairs
                8  gear_ratio
                9  lam      motor flux linkage
                10 h        cg height
                11 Cf       front cornering stiffness
                12 Cr       rear cornering stiffness
                13 muf      front friction coefficient
                14 mur      rear friction coefficient
                15 Cro      rolling resistance coefficient
                16 c        dFz relaxation rate [1/s]
                17 s_min    min steer angle
                18 s_max    max steer angle
                19 sv_min   min steer velocity
                20 sv_max   max steer velocity

        Returns:
            f (numpy.ndarray (10,)): state derivatives
    """
    g = 9.81

    # states
    PHI     = x[2]
    VX      = x[3]
    VY      = x[4]
    OMEGA   = x[5]
    OMEGA_W = x[6]
    DFZ     = x[7]
    DELTA   = x[8]

    # params
    mass        = params[0]
    Iz          = params[1]
    lf          = params[2]
    lr          = params[3]
    rw          = params[4]
    Iw          = params[5]
    Im          = params[6]
    pole_pairs  = params[7]
    gear_ratio  = params[8]
    lam         = params[9]
    h           = params[10]
    Cf          = params[11]
    Cr          = params[12]
    muf         = params[13]
    mur         = params[14]
    Cro         = params[15]
    c           = params[16]
    s_min       = params[17]
    s_max       = params[18]
    sv_min      = params[19]
    sv_max      = params[20]
    cur_min     = params[21]
    cur_max     = params[22]

    L = lf + lr

    # controls (constrain steer rate against current steer angle/limits, like the ST model does)
    u = np.array([
        cur_constraint(u_init[0], cur_min, cur_max),
        steering_constraint(DELTA, u_init[1], s_min, s_max, sv_min, sv_max),
    ])
    CURRENT = u[0]
    D_DELTA = u[1]

    # low-speed regularization so kappa/alpha stay finite near VX = 0
    eps = 0.1
    vx_dyn = np.sqrt(VX**2 + eps)

    # --- load transfer: dFz is a state, not recomputed algebraically ---
    Ffz = mass * g * lr / L - DFZ
    Frz = mass * g * lf / L + DFZ
    Ffmax = muf * Ffz
    Frmax = mur * Frz  # coupling lives in sigma_r, no separate friction-circle derate

    # --- slip angles ---
    alphaf = DELTA - np.arctan2(OMEGA * lf + VY, vx_dyn)
    alphar = np.arctan2(OMEGA * lr - VY, vx_dyn)

    # --- rear longitudinal slip ratio from wheel speed ---
    kappa = (rw * OMEGA_W - VX) / vx_dyn

    # --- combined slips (front assumed free-rolling, shares kappa formula) ---
    sigma_f = np.sqrt(np.tan(alphaf)**2 + kappa**2 + 1e-2)
    sigma_r = np.sqrt(np.tan(alphar)**2 + kappa**2 + 1e-2)

    sigmaf_max = 3 * Ffmax / Cf
    sigmar_max = 3 * Frmax / Cr

    Csf = Cf * sigma_f
    Ff_total = Csf - Csf**2 / (3 * Ffmax) + Csf**3 / (27 * Ffmax**2)
    if sigma_f >= sigmaf_max:
        Ff_total = Ffmax

    Csr = Cr * sigma_r
    Fr_total = Csr - Csr**2 / (3 * Frmax) + Csr**3 / (27 * Frmax**2)
    if sigma_r >= sigmar_max:
        Fr_total = Frmax

    # --- front: lateral + longitudinal share Ff_total via sigma_f ---
    Ffy = Ff_total * np.tan(alphaf) / sigma_f
    Ffx = Ff_total * kappa / sigma_f

    # --- rear: lateral + longitudinal share Fr_total via sigma_r ---
    v_eps = 0.5
    F_roll = Cro * Frz * np.tanh(VX / v_eps)
    Frx = Fr_total * kappa / sigma_r - F_roll
    Fry = Fr_total * np.tan(alphar) / sigma_r

    # --- drive torque on rear wheel from motor current ---
    tau_drive = gear_ratio * 1.5 * pole_pairs * lam * CURRENT

    # --- realized longitudinal chassis force drives load-transfer relaxation ---
    Fx_chassis = Frx + Ffx * np.cos(DELTA) - Ffy * np.sin(DELTA)

    f = np.array([
        VX * np.cos(PHI) - VY * np.sin(PHI),                                     # dx0
        VX * np.sin(PHI) + VY * np.cos(PHI),                                     # dx1
        OMEGA,                                                                    # dx2
        (Frx + Ffx * np.cos(DELTA) - Ffy * np.sin(DELTA)) / mass + VY * OMEGA,    # dx3
        (Fry + Ffy * np.cos(DELTA) + Ffx * np.sin(DELTA)) / mass - VX * OMEGA,    # dx4
        (lf * (Ffy * np.cos(DELTA) + Ffx * np.sin(DELTA)) - lr * Fry) / Iz,       # dx5
        (tau_drive - Frx * rw - Ffx * rw) / (Iw + Im),                            # dx6
        -c * (DFZ - (h / L) * Fx_chassis),                                        # dx7                                                                   # dx8
        D_DELTA,                                                                  # dx9
    ])

    return f


@njit(cache=True)
def get_standardized_state_fiala(x: np.ndarray) -> np.ndarray:
    """[X, Y, steering_angle, speed, yaw, yaw_rate, v_y]"""
    return np.array([x[0], x[1], x[9], x[3], x[2], x[5], x[4]])