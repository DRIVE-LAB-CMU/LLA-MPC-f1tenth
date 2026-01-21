""" Runge Kutta sixth order integration.
    Uses same syntax as scipy.integrate.odeint
    `fun` should be of the form fun(t, y, control)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
# from numba import njit, float64, boolean, int64
# from numba.experimental import jitclass
import jax
import jax.numpy as jnp
from functools import partial



@partial(jax.jit, static_argnums=(0, 1))
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

def rk4Factory(bank_params, diffequation, h):
    """Returns a function that integrates with fixed bank_params."""
    
    def odeintRK4_batch(known_params, x0, u):

        
        def step(b_p, x_t):
            return diffequation(b_p,  known_params,x_t, u)

        def rk4(b_p):
            k1 = h * step(b_p, x0)
            k2 = h * step(b_p, x0 + k1 / 2)
            k3 = h * step(b_p, x0 + k2 / 2)
            k4 = h * step(b_p, x0 + k3)
            return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return jax.vmap(rk4)(bank_params)
    
    return jax.jit(odeintRK4_batch)

def eulerFactory(bank_params, diffequation, h):
    """Returns a function that performs one Euler integration step with fixed bank_params."""    
    def odeintEuler_batch(known_params, x0, u):
        def step(b_p, x_t):
            return diffequation(b_p, known_params, x_t, u)

        def euler(b_p):
            return x0 + h * step(b_p, x0)

        return jax.vmap(euler)(bank_params)

    return jax.jit(odeintEuler_batch)

def rk6Factory(bank_params, diffequation, h):
    """Returns a function that performs one RK6 integration step with fixed bank_params."""

    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    def odeintRK6_batch(known_params, x0, u):
        def step(b_p, x_t):
            return diffequation(b_p, known_params, x_t, u)

        def rk6(b_p):
            k1 = h * step(b_p, x0)
            k2 = h * step(b_p, x0 + k1 * (1/4))
            k3 = h * step(b_p, x0 + k1 * (3/32) + k2 * (9/32))
            k4 = h * step(b_p, x0 + k1 * (1932/2197) - k2 * (7200/2197) + k3 * (7296/2197))
            k5 = h * step(b_p, x0 + k1 * (439/216) - k2 * 8 + k3 * (3680/513) - k4 * (845/4104))
            k6 = h * step(b_p, x0 - k1 * (8/27) + k2 * 2 - k3 * (3544/2565) + k4 * (1859/4104) - k5 * (11/40))

            K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
            return x0 + jnp.sum(gamma[:, None] * K, axis=0)

        return jax.vmap(rk6)(bank_params)

    return jax.jit(odeintRK6_batch)
