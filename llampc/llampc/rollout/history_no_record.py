"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
# from numba import njit, float64, boolean, int64, prange
# from numba.experimental import jitclass

import jax
import jax.numpy as jnp
from functools import partial
from collections import deque
import time

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
# gpu = jax.devices("cpu")[0]

@jax.jit
def get_lookback_error(last_predicted_states, x_t, cost_weights):
    cost = jnp.sum(jnp.square(x_t[None, :] - last_predicted_states) * cost_weights[None, :], axis = 1)
    

    return cost

@jax.jit
def _step_bank(last_predicted_states,  running_cost, x_t, cost_weights):
    diff = x_t[None, :] - last_predicted_states
    cost = jnp.sum(jnp.square(diff) * cost_weights[None, :], axis = 1)
    running_cost = running_cost.at[:].add(cost)

    best_model = jnp.argmin(running_cost)


    return diff, cost,  running_cost, best_model

@jax.jit
def find_best_model(running_cost):
    return jnp.argmin(running_cost)


class LBHistory:

    def __init__(self, num_models, dt, cost_weights, state_size, integrator_factory, dynamics_bank, diffeq,
                 buffer_size = None, control_size = 2):
        self.num_models = num_models

        self.last_predicted_states = jax.device_put(jnp.zeros((self.num_models, state_size), dtype='float32'), device = gpu)

        self.running_cost = jax.device_put(jnp.zeros(self.num_models, dtype='float32'), device = gpu)
        
        self.dt = dt
        self.cost_weights = jax.device_put(jnp.array(cost_weights, dtype='float32'), device = gpu)
        self.state_size = state_size
        
        self.dynamics_bank = dynamics_bank
        self.integrator = integrator_factory(
            jax.device_put(dynamics_bank.param_bank, device = gpu),
            diffeq,
            dt
        )
        self.current_best_model = 0

        # control buffer to account for actuation delay 
        # note that this does not account for actuation speed, which is accounted for via 
        # the nmpc problem setup itself via max acceleratin and steering angle
        self.control_size = 2
        self.buffer_size = np.zeros(control_size) if buffer_size is None else buffer_size
        self.buffer = [deque() for _ in range(control_size)]

    def predict_states(self, x_t, u_t):
        """Batched version of _integrate"""
    
        t0 = time.perf_counter_ns()

        print(type(x_t))
        # buffered_u_t = u_t

        buffered_u_t = np.zeros_like(u_t)
        for i in range(self.control_size):
            self.buffer[i].append(u_t[i])
            if(len(self.buffer[i]) > self.buffer_size[i]):
                buffered_u_t[i] = self.buffer[i].popleft()
        
        t1 = time.perf_counter_ns()
        known_params = self.dynamics_bank.get_known_params()
        
        t2 = time.perf_counter_ns()
        self.last_predicted_states = self.integrator(
            known_params,
            x_t,
            buffered_u_t)
        
        
    def update_lookback_error(self, x_t):
        t0 = time.perf_counter_ns()
        gpu_x = jax.device_put(jnp.array(x_t, dtype = 'float32'), device = gpu)
        t1 = time.perf_counter_ns()
        diff, cost, self.running_cost, self.current_best_model = _step_bank(
            self.last_predicted_states,
            self.running_cost,
            gpu_x,
            self.cost_weights
        )

        return diff, cost
    
    def predict_multi_step(self, x_t, U_seq):
        """
        Rolls out the simulation N steps into the future.
        U_seq should be an array of shape (N, control_size).
        """
        # Note: If you want to buffer U_seq for actuation delay, 
        # you will need to pre-process U_seq before passing it to the integrator.
        
        gpu_x = jax.device_put(jnp.array(x_t, dtype='float32'), device=gpu)
        gpu_U = jax.device_put(jnp.array(U_seq, dtype='float32'), device=gpu)
        
        known_params = self.dynamics_bank.get_known_params()
        
        # The multi-step factory naturally expects (known_params, x0, U_seq)
        self.last_predicted_states = self.integrator(
            known_params,
            gpu_x,
            gpu_U
        )

    def update_multi_step_error(self, true_future_state):
        """
        Grades all 200,000 models against reality at the end of the N-step horizon.
        """
        gpu_x_true = jax.device_put(jnp.array(true_future_state, dtype='float32'), device=gpu)
        
        # We can perfectly reuse your existing _step_bank function!
        diff, cost, self.running_cost, self.current_best_model = _step_bank(
            self.last_predicted_states,
            self.running_cost,
            gpu_x_true,
            self.cost_weights
        )
        return diff, cost
    

    def reset(self):
        # Preserve device placement by using the same pattern as initialization
        self.last_predicted_states = jax.device_put(
            jnp.zeros((self.num_models, self.state_size), dtype='float32'), 
            device=gpu
        )
        self.current_best_model = 0
        self.running_cost = np.zeros(self.num_models, dtype='float32')

    def get_best_model(self):
        return self.current_best_model