""" Runge Kutta sixth order integration.
    Uses same syntax as scipy.integrate.odeint
    `fun` should be of the form fun(t, y, u)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
# from numba import njit, float64, boolean, int64
# from numba.experimental import jitclass
import jax
import jax.numpy as jnp
from functools import partial


def _rk4_step(diffequation, h, lla_params, known_params, x, u):
    k1 = h * diffequation(lla_params, known_params, x,        u)
    k2 = h * diffequation(lla_params, known_params, x + k1/2, u)
    k3 = h * diffequation(lla_params, known_params, x + k2/2, u)
    k4 = h * diffequation(lla_params, known_params, x + k3,   u)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6


def _rk6_step(diffequation, h, lla_params, known_params, x, u):
    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    k1 = h * diffequation(lla_params, known_params, x, u)
    k2 = h * diffequation(lla_params, known_params, x + k1*(1/4), u)
    k3 = h * diffequation(lla_params, known_params, x + k1*(3/32)      + k2*(9/32), u)
    k4 = h * diffequation(lla_params, known_params, x + k1*(1932/2197) - k2*(7200/2197) + k3*(7296/2197), u)
    k5 = h * diffequation(lla_params, known_params, x + k1*(439/216)   - k2*8           + k3*(3680/513)  - k4*(845/4104), u)
    k6 = h * diffequation(lla_params, known_params, x - k1*(8/27)      + k2*2           - k3*(3544/2565) + k4*(1859/4104) - k5*(11/40), u)
    K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
    return x + jnp.sum(gamma[:, None] * K, axis=0)


def _euler_step(diffequation, h, lla_params, known_params, x, u):
    return x + h * diffequation(lla_params, known_params, x, u)


def rollout(integrator_step, diffequation):
    """
    Returns a JIT'd N-step trajectory rollout using the given integrator step.
    integrator_step: one of _rk4_step, _rk6_step, _euler_step
    Returns (N, nx) trajectory.
    """
    def _rollout(lla_params, known_params, x0, u, h, N):
        def step(x, _):
            x_next = integrator_step(diffequation, h, lla_params, known_params, x, u)
            return x_next, x_next
        _, x_traj = jax.lax.scan(step, x0, None, length=N)
        return x_traj

    return jax.jit(_rollout, static_argnames=('N',))

def rk4Factory(bank_params, diffequation, h):
    """
    Returns a function that performs one RK4 integration step with fixed bank_params.
    Matches the logic used in rk6Factory for consistency.
    """
    
    def odeintRK4_batch(known_params, x0, u):
        def step(b_p, x_t):
            return diffequation(b_p, known_params, x_t, u)

        # 1. Move state into the arguments of the inner function
        def rk4(b_p, x_curr):
            k1 = h * step(b_p, x_curr)
            k2 = h * step(b_p, x_curr + k1 / 2)
            k3 = h * step(b_p, x_curr + k2 / 2)
            k4 = h * step(b_p, x_curr + k3)
            
            result = x_curr + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            return result.astype(jnp.float32)

        # 2. Use JAX vmap to handle mapping based on the input rank of x0
        if x0.ndim == 1:
            # Single state (Initialization): map params (0), broadcast state (None)
            # This is likely where your (1000, 6) vs (6, 6) error was coming from
            return jax.vmap(rk4, in_axes=(0, None))(bank_params, x0)
        else:
            # Batched state (Runtime): map params (0), map states (0)
            return jax.vmap(rk4, in_axes=(0, 0))(bank_params, x0)

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

# def rk6Factory(bank_params, diffequation, h):
#     """Returns a function that performs one RK6 integration step with fixed bank_params."""

#     gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
#     def odeintRK6_batch(known_params, x0, u):
#         def step(b_p, x_t):
#             return diffequation(b_p, known_params, x_t, u)

#         def rk6(b_p):
#             k1 = h * step(b_p, x0)
#             k2 = h * step(b_p, x0 + k1 * (1/4))
#             k3 = h * step(b_p, x0 + k1 * (3/32) + k2 * (9/32))
#             k4 = h * step(b_p, x0 + k1 * (1932/2197) - k2 * (7200/2197) + k3 * (7296/2197))
#             k5 = h * step(b_p, x0 + k1 * (439/216) - k2 * 8 + k3 * (3680/513) - k4 * (845/4104))
#             k6 = h * step(b_p, x0 - k1 * (8/27) + k2 * 2 - k3 * (3544/2565) + k4 * (1859/4104) - k5 * (11/40))

#             K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
#             return x0 + jnp.sum(gamma[:, None] * K, axis=0)

#         return jax.vmap(rk6)(bank_params)

#     return jax.jit(odeintRK6_batch)

def rk6Factory(bank_params, diffequation, h):
    """Returns a function that performs one RK6 integration step with fixed bank_params."""

    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    def odeintRK6_batch(known_params, x0, u):
        def step(b_p, x_t):
            return diffequation(b_p, known_params, x_t, u)

        # 1. We move the state into the arguments of the inner function
        def rk6(b_p, x_curr):
            k1 = h * step(b_p, x_curr)
            k2 = h * step(b_p, x_curr + k1 * (1/4))
            k3 = h * step(b_p, x_curr + k1 * (3/32) + k2 * (9/32))
            k4 = h * step(b_p, x_curr + k1 * (1932/2197) - k2 * (7200/2197) + k3 * (7296/2197))
            k5 = h * step(b_p, x_curr + k1 * (439/216) - k2 * 8 + k3 * (3680/513) - k4 * (845/4104))
            k6 = h * step(b_p, x_curr - k1 * (8/27) + k2 * 2 - k3 * (3544/2565) + k4 * (1859/4104) - k5 * (11/40))

            K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
            return x_curr + jnp.sum(gamma[:, None] * K, axis=0)

        # 2. Let JAX JIT compile the correct mapping behavior based on the input rank
        if x0.ndim == 1:
            # Single state (One-Step): map params (0), broadcast state (None)
            return jax.vmap(rk6, in_axes=(0, None))(bank_params, x0)
        else:
            # Batched state (Open-Loop): map params (0), map states (0)
            return jax.vmap(rk6, in_axes=(0, 0))(bank_params, x0)

    return jax.jit(odeintRK6_batch)

def eulerMultiStepFactory(bank_params, diffequation, h):
    """Returns a batched N-step Euler rollout function."""
    def odeintEuler_multistep(known_params, x0, U_seq):
        def single_model_scan(b_p):
            def step_fn(x_current, u_current):
                x_next = x_current + h * diffequation(b_p, known_params, x_current, u_current)
                return x_next, x_next  # (carry, output) for lax.scan
            
            # lax.scan automatically loops over the sequence of controls (U_seq)
            final_state, _ = jax.lax.scan(step_fn, x0, U_seq)
            return final_state

        return jax.vmap(single_model_scan)(bank_params)

    return jax.jit(odeintEuler_multistep)


def rk4MultiStepFactory(bank_params, diffequation, h):
    """Returns a batched N-step RK4 rollout function."""
    def odeintRK4_multistep(known_params, x0, U_seq):
        def single_model_scan(b_p):
            def step_fn(x_current, u_current):
                k1 = h * diffequation(b_p, known_params, x_current, u_current)
                k2 = h * diffequation(b_p, known_params, x_current + k1 / 2, u_current)
                k3 = h * diffequation(b_p, known_params, x_current + k2 / 2, u_current)
                k4 = h * diffequation(b_p, known_params, x_current + k3, u_current)
                x_next = x_current + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return x_next, x_next
            
            final_state, _ = jax.lax.scan(step_fn, x0, U_seq)
            return final_state

        return jax.vmap(single_model_scan)(bank_params)

    return jax.jit(odeintRK4_multistep)


def rk6MultiStepFactory(bank_params, diffequation, h):
    """Returns a batched N-step RK6 rollout function."""
    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    def odeintRK6_multistep(known_params, x0, U_seq):
        def single_model_scan(b_p):
            def step_fn(x_current, u_current):
                k1 = h * diffequation(b_p, known_params, x_current, u_current)
                k2 = h * diffequation(b_p, known_params, x_current + k1 * (1/4), u_current)
                k3 = h * diffequation(b_p, known_params, x_current + k1 * (3/32) + k2 * (9/32), u_current)
                k4 = h * diffequation(b_p, known_params, x_current + k1 * (1932/2197) - k2 * (7200/2197) + k3 * (7296/2197), u_current)
                k5 = h * diffequation(b_p, known_params, x_current + k1 * (439/216) - k2 * 8 + k3 * (3680/513) - k4 * (845/4104), u_current)
                k6 = h * diffequation(b_p, known_params, x_current - k1 * (8/27) + k2 * 2 - k3 * (3544/2565) + k4 * (1859/4104) - k5 * (11/40), u_current)

                K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
                x_next = x_current + jnp.sum(gamma[:, None] * K, axis=0)
                return x_next, x_next
            
            final_state, _ = jax.lax.scan(step_fn, x0, U_seq)
            return final_state

        return jax.vmap(single_model_scan)(bank_params)

    return jax.jit(odeintRK6_multistep)



def rk6FactoryMultiOrigin(bank_params, diffequation, h):
    """Returns a function that performs one RK6 integration step with fixed bank_params."""

    gamma = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    def odeintRK6_batch(known_params, x0, u):
        def step(b_p, x_t):
            return diffequation(b_p, known_params, x_t, u)

        # 1. We move the state into the arguments of the inner function
        def rk6(b_p, x_curr):
            k1 = h * step(b_p, x_curr)
            k2 = h * step(b_p, x_curr + k1 * (1/4))
            k3 = h * step(b_p, x_curr + k1 * (3/32) + k2 * (9/32))
            k4 = h * step(b_p, x_curr + k1 * (1932/2197) - k2 * (7200/2197) + k3 * (7296/2197))
            k5 = h * step(b_p, x_curr + k1 * (439/216) - k2 * 8 + k3 * (3680/513) - k4 * (845/4104))
            k6 = h * step(b_p, x_curr - k1 * (8/27) + k2 * 2 - k3 * (3544/2565) + k4 * (1859/4104) - k5 * (11/40))

            K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)
            return x_curr + jnp.sum(gamma[:, None] * K, axis=0)

        # 2. Let JAX JIT compile the correct mapping behavior based on the input rank
        if x0.ndim == 1:
            # Single state (One-Step): map params (0), broadcast state (None)
            return jax.vmap(rk6, in_axes=(0, None))(bank_params, x0)
        else:
            # Batched state (Open-Loop): map params (0), map states (0)
            return jax.vmap(rk6, in_axes=(0, 0))(bank_params, x0)

    return jax.jit(odeintRK6_batch)