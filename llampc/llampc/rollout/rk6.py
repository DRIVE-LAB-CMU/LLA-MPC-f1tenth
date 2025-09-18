""" Runge Kutta sixth order integration.
    Uses same syntax as scipy.integrate.odeint
    `fun` should be of the form fun(t, y, control)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

def odeintRK6(diffeq, y0, t, control, state_add):
    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

    hs = jnp.diff(t)
    dts = t[:-1]

    def step(carry, i):
        y = carry 
        h, dt = i

        k1 = h * diffeq(dt, y, control, state_add)
        k2 = h * diffeq(dt + h / 4, y + k1 / 4, control, state_add)
        k3 = h * diffeq(dt + 3/8 * h, y + 3/32 * k1 + 9/32 * k2, control, state_add)
        k4 = h * diffeq(dt + 12/13 * h, y + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3, control, state_add)
        k5 = h * diffeq(dt + h, y + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4, control, state_add)
        k6 = h * diffeq(dt + h / 2, y - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5, control, state_add)

        K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
        y_carry = y + jnp.sum(gamma[:, None] * K, axis=0)

        return y_carry, y_carry

    _, y_next = jax.lax.scan(step, y0, (hs, dts))

    return y_next

odeintRK6 = jax.jit(odeintRK6, static_argnums=(0,))
    
def odeintRK6_batch(diffeq, y0_batch, t, control_batch, state_add):
    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

    hs = jnp.diff(t)
    dts = t[:-1]  

    def step(carry, i):
        y_batch = carry
        h, dt = i

        k1 = h * diffeq(dt, y_batch, control_batch, state_add)
        k2 = h * diffeq(dt + h / 4, y_batch + k1 / 4, control_batch, state_add)
        k3 = h * diffeq(dt + 3/8 * h, y_batch + 3/32 * k1 + 9/32 * k2, control_batch, state_add)
        k4 = h * diffeq(dt + 12/13 * h, y_batch + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3, control_batch, state_add)
        k5 = h * diffeq(dt + h, y_batch + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4, control_batch, state_add)
        k6 = h * diffeq(dt + h / 2, y_batch - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5, control_batch, state_add)

        K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
        y_carry = y_batch + jnp.sum(gamma[:, None, None] * K, axis=0)

        return y_carry, y_carry

    # Run scan over time steps
    _, y_next = jax.lax.scan(step, y0_batch, (hs, dts))

    return y_next

odeintRK6_batch = jax.jit(odeintRK6_batch, static_argnums=(0,))

def odeintRK4_batch(diffeq, y0_batch, t, control_batch, state_add):
    hs = jnp.diff(t)
    dts = t[:-1]

    def step(carry, i):
        y_batch = carry
        h, dt = i

        k1 = h * diffeq(dt, y_batch, control_batch, state_add)
        k2 = h * diffeq(dt + h / 2, y_batch + k1 / 2, control_batch, state_add)
        k3 = h * diffeq(dt + h / 2, y_batch + k2 / 2, control_batch, state_add)
        k4 = h * diffeq(dt + h, y_batch + k3, control_batch, state_add)

        y_carry = y_batch + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return y_carry, y_carry

    _, y_next = jax.lax.scan(step, y0_batch, (hs, dts))

    return y_next
# odeintRK4_batch = jax.jit(odeintRK4_batch, static_argnums=(0,))



def odeintEuler_batch(diffeq, y0_batch, t, control_batch, state_add):
    hs = jnp.diff(t)
    dts = t[:-1]

    def step(carry, i):
        y_batch = carry
        h, dt = i

        k = h * diffeq(dt, y_batch, control_batch, state_add)
        y_carry = y_batch + k

        return y_carry, y_carry

    _, y_next = jax.lax.scan(step, y0_batch, (hs, dts))

    return y_next

odeintEuler_batch = jax.jit(odeintEuler_batch, static_argnums=(0,))