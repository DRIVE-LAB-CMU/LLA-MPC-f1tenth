""" Runge Kutta sixth order integration.
    Uses same syntax as scipy.integrate.odeint
    `fun` should be of the form fun(t, y, control)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from numba import njit, float64, boolean, int64
from numba.experimental import jitclass

@njit
def integrate_batch(integrator, dynamics, diffeq, x_t, u_t, dt):
    """Batched version of _integrate"""
    odesol = integrator(
        dynamics,
        diffeq,
        x_t,
        u_t,
        dt
        
        )
    return odesol

@njit
def odeintRK6(dynamics, diffequation, y0, t, control):
    gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    y_next = np.empty((len(t)-1, len(y0)))

    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        k1 = h * diffequation(dynamics, t[i], y0, control)
        k2 = h * diffequation(dynamics, t[i]+h/4, y0+k1/4, control)
        k3 = h * diffequation(dynamics, t[i]+3/8*h, y0+3/32*k1+9/32*k2, control)
        k4 = h * diffequation(dynamics, t[i]+12/13*h, y0+1932/2197*k1-7200/2197*k2+7296/2197*k3, control)
        k5 = h * diffequation(dynamics, t[i]+h, y0+439/216*k1-8*k2+3680/513*k3-845/4104*k4, control)
        k6 = h * diffequation(dynamics, t[i]+h/2, y0-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5, control)
        K = np.asarray([k1, k2, k3, k4, k5, k6])
        # y_next[i,:] = y0 + gamma@K
        # y0 = y0 + gamma@K

        y_next[i] = y0 +np.sum(gamma[:, None] * K, axis=0)
        y0 = y_next[i]
        
    return y_next

# @njit
# def odeintRK6_batch(dynamics_batch, diffequation, y0, t, control):

#     n_models = y0.shape[0]
#     gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
#     y_next = np.empty((len(t)-1, n_models, y0_batch.shape[1]))

#     for i in range(len(t)-1):
#         h = t[i+1]-t[i]
#         k1 = h * diffequation(dynamics_batch, t[i], y0, control_batch)
#         k2 = h * diffequation(dynamics_batch, t[i]+h/4, y0_batch+k1/4, control_batch)
#         k3 = h * diffequation(dynamics_batch, t[i]+3/8*h, y0_batch+3/32*k1+9/32*k2, control_batch)
#         k4 = h * diffequation(dynamics_batch, t[i]+12/13*h, y0_batch+1932/2197*k1-7200/2197*k2+7296/2197*k3, control_batch)
#         k5 = h * diffequation(dynamics_batch ,t[i]+h, y0_batch+439/216*k1-8*k2+3680/513*k3-845/4104*k4, control_batch)
#         k6 = h * diffequation(dynamics_batch, t[i]+h/2, y0_batch-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5, control_batch)
#         K = np.empty((6, y0_batch.shape[0], y0_batch.shape[1]))
#         K[0] = k1
#         K[1] = k2
#         K[2] = k3
#         K[3] = k4
#         K[4] = k5
#         K[5] = k6
#         # y_next[i] = y0_batch + np.tensordot(gamma, K, axes=1)
#         # y0_batch = y0_batch + np.tensordot(gamma, K, axes=1)

#         y_next[i] = y0_batch +np.sum(gamma[:, None, None] * K, axis=0)
#         y0_batch = y_next[i]

#     return y_next

@njit
def odeintRK4_batch(dynamics_batch, diffequation, x_t, u_t, h):
    """Batched version of RK4"""

    # RK4 steps
    k1 = h * diffequation(dynamics_batch, 0, x_t, u_t)
    k2 = h * diffequation(dynamics_batch, h / 2, x_t + k1 / 2, u_t)
    k3 = h * diffequation(dynamics_batch, h / 2, x_t + k2 / 2, u_t)
    k4 = h * diffequation(dynamics_batch, h, x_t + k3, u_t)
        
    y_next = x_t + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # print(y_next)

    return y_next

@njit
def odeintEuler_batch(dynamics_batch, diffequation, y0_batch, t, control_batch):
    n_models = y0_batch.shape[0]
    y_next = np.empty((len(t)-1, n_models, y0_batch.shape[1]))

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]

        # Single Euler step
        k = h * diffequation(dynamics_batch, t[i], y0_batch, control_batch)

        # Update
        y_next[i] = y0_batch + k
        y0_batch = y0_batch + k

    return y_next
