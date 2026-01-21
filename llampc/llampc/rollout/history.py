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

@jax.jit
def get_lookback_error(last_predicted_states, x_t, cost_weights, queue_index):
    cost = jnp.sum(jnp.square(x_t[None, :] - last_predicted_states) * cost_weights[None, :], axis = 1)

    return cost

@jax.jit
def find_best_model(running_cost):
    return jnp.argmin(running_cost)


class LBHistory:

    def __init__(self, num_models, history_length, dt, cost_weights, state_size, integrator_factory, dynamics_bank, diffeq,
                 buffer_size = None, control_size = 2):
        self.num_models = num_models
        self.history_length = history_length

        self.last_predicted_states = jnp.zeros((self.num_models, state_size), dtype='float16')

        self.running_cost = np.zeros(self.num_models, dtype='float16')
        self.cost_history = np.zeros((self.num_models, self.history_length), dtype='float16')
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = jnp.array(cost_weights)
        self.state_size = state_size
        
        self.dynamics_bank = dynamics_bank
        self.integrator = integrator_factory(
            dynamics_bank.param_bank,
            diffeq,
            dt
        )
        

        # control buffer to account for actuation delay 
        # note that this does not account for actuation speed, which is accounted for via 
        # the nmpc problem setup itself via max acceleratin and steering angle
        self.control_size = 2
        self.buffer_size = np.zeros(control_size) if buffer_size is None else buffer_size
        self.buffer = [deque() for _ in range(control_size)]

    def predict_states(self, x_t, u_t):
        """Batched version of _integrate"""
    
        t0 = time.perf_counter_ns()
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
        
        t3 = time.perf_counter_ns()
        
        jax.block_until_ready(self.last_predicted_states)
        t4 = time.perf_counter_ns()
        
        print(f"Buffer: {(t1-t0)*1e-6:.3f}ms, "
          f"GetParams: {(t2-t1)*1e-6:.3f}ms, "
          f"Integrator: {(t3-t2)*1e-6:.3f}ms, "
          f"Sync: {(t4-t3)*1e-6:.3f}ms")
        
    def update_lookback_error(self, x_t):
        t0 = time.perf_counter_ns()
        
        self.running_cost -= self.cost_history[:, self.queue_index]
        
        t1 = time.perf_counter_ns()
        error_result = get_lookback_error(
            self.last_predicted_states,
            x_t, 
            self.cost_weights,
            self.queue_index
        )
        jax.block_until_ready(error_result)
        t2 = time.perf_counter_ns()
        cost = np.array(error_result)
        t3 = time.perf_counter_ns()
        self.cost_history[:, self.queue_index] = cost
        self.running_cost += cost
        self.queue_index = (self.queue_index + 1) % self.history_length

        t4 = time.perf_counter_ns()
        print(f"Prep cost: {(t1-t0)*1e-6:.3f}ms, "
          f"Jit call: {(t2-t1)*1e-6:.3f}ms, "
          f"CPU Transfer: {(t3-t2)*1e-6:.3f}ms, "
          f"Update: {(t4-t3)*1e-6:.3f}ms")



        return cost
    

    def reset(self):
        self.last_predicted_states = jnp.zeros((self.num_models, self.state_size))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0

    def get_best_model(self):
        return find_best_model(self.running_cost)

           
    
    
    
    
