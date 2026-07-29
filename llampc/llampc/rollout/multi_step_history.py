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
#gpu = jax.devices("gpu")[0]
gpu = jax.devices("cpu")[0]

@jax.jit
def start_traj_eval(last_predicted_states, gamma_count, x_t, index):
    last_predicted_states = last_predicted_states.at[index].set(x_t)  
    gamma_count = gamma_count.at[index].set(1.0) 

    return last_predicted_states, gamma_count

ANGLE_IDX = 2          # state index wrapped to (-pi, pi] before squaring


@jax.jit
def _slot_errors(last_predicted_states, x_t, cost_weights):
    """Weighted squared error of every (slot, model) prediction.

    last_predicted_states : (S, M, D)
    x_t                   : (D,)
    returns               : (S, M)
    """
    error = x_t[None, None, :] - last_predicted_states
    error = error.at[:, :, ANGLE_IDX].set(
        (error[:, :, ANGLE_IDX] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

    return jnp.sum(jnp.square(error) * cost_weights[None, None, :], axis=-1)


@jax.jit
def get_lookback_error(last_predicted_states, x_t, cost_weights, gamma_count):
    """Collapse the slot axis into one cost per model. returns (M,)"""
    per_slot = _slot_errors(last_predicted_states, x_t, cost_weights)

    return jnp.sum(per_slot * gamma_count[:, None], axis=0)


@jax.jit
def _step_bank(last_predicted_states, cost_history, running_cost, queue_index,
               x_t, cost_weights, gamma_count):
    cost = get_lookback_error(last_predicted_states, x_t, cost_weights, gamma_count)

    running_cost = running_cost.at[:].add(-cost_history[:, queue_index])
    cost_history = cost_history.at[:, queue_index].set(cost)
    running_cost = running_cost.at[:].add(cost)
    best_model = jnp.argmin(running_cost)

    return cost, cost_history, running_cost, best_model

@jax.jit
def find_best_model(running_cost):
    return jnp.argmin(running_cost)


class LBHistoryMultiStep:

    def __init__(self, num_models, history_length, dt, 
                 cost_weights, state_size, integrator_factory, dynamics_bank, diffeq,
                 buffer_size = None, control_size = 2, 
                 reset_interval = 1, skip_interval = 1, gamma = 1):
        self.num_models = num_models
        self.history_length = history_length

        self.reset_interval = max(reset_interval, 1)
        self.gamma = gamma
        self.skip_interval = max(skip_interval, 1)
        self.integration_slots = (self.reset_interval // self.skip_interval) + (self.reset_interval % self.skip_interval != 0)

        self.integration_count = -1 * np.ones(self.integration_slots, dtype=int)
        self.gamma_count = jax.device_put(
            jnp.zeros(self.integration_slots, dtype='float32'), 
            device = gpu
        )
        self.skip_counter = 0
        self.slot_queue = deque()

        self.last_predicted_states = jax.device_put(
            jnp.zeros((self.integration_slots, self.num_models, state_size), dtype='float32'), 
            device = gpu)

        for i in range(self.integration_slots):
            self.slot_queue.append(i)

        self.running_cost = jax.device_put(
            jnp.zeros(self.num_models, dtype='float32'), device = gpu)
        self.cost_history = jax.device_put(
            jnp.zeros((self.num_models, self.history_length), dtype='float32'), device = gpu)
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = jax.device_put(
            jnp.array(cost_weights, dtype='float32'), device = gpu)
        self.state_size = state_size
        
        self.dynamics_bank = dynamics_bank
        self.integrator = integrator_factory(
            jax.device_put(dynamics_bank.param_bank, device = gpu),
            diffeq,
            dt
        )
        self.slot_integrator = jax.jit(jax.vmap(self.integrator, in_axes=(None, 0, None)))
        self.current_best_model = 0

        # control buffer to account for actuation delay 
        # note that this does not account for actuation speed, which is accounted for via 
        # the nmpc problem setup itself via max acceleratin and steering angle
        self.control_size = control_size
        self.buffer_size = np.zeros(control_size, dtype = int) if buffer_size is None else buffer_size
        self.buffer = [deque() for _ in range(control_size)]


    def predict_states(self, x_t, u_t, reset=False):
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
        for i in range(self.integration_slots):
            if(self.integration_count[i] != -1):
                self.integration_count[i] += 1
                if(self.integration_count[i] >= self.reset_interval):
                    self.gamma_count = self.gamma_count.at[i].set(0)
                    self.integration_count[i] = -1
                    self.slot_queue.append(i)

        self.gamma_count  = self.gamma_count * self.gamma
        if(self.skip_counter == 0):
            index = self.slot_queue.popleft()
            self.integration_count[index] = 0
            self.last_predicted_states, self.gamma_count = start_traj_eval(
                self.last_predicted_states, self.gamma_count, x_t, index)

        self.last_predicted_states = self.slot_integrator(
            known_params,
            self.last_predicted_states,
            buffered_u_t)
        
        self.skip_counter = (self.skip_counter + 1) % self.skip_interval 

        t3 = time.perf_counter_ns()
        
        jax.block_until_ready(self.last_predicted_states)
        t4 = time.perf_counter_ns()
        
        # print(f"Buffer: {(t1-t0)*1e-6:.3f}ms, "
        #   f"GetParams: {(t2-t1)*1e-6:.3f}ms, "
        #   f"Integrator: {(t3-t2)*1e-6:.3f}ms, "
        #   f"Sync: {(t4-t3)*1e-6:.3f}ms")
        
    def update_lookback_error(self, x_t):
        t0 = time.perf_counter_ns()
        gpu_x = jax.device_put(jnp.array(x_t, dtype = 'float32'), device = gpu)
        t1 = time.perf_counter_ns()

        cost, self.cost_history, self.running_cost, self.current_best_model = _step_bank(
            self.last_predicted_states,
            self.cost_history,
            self.running_cost,
            self.queue_index,
            gpu_x,
            self.cost_weights,
            self.gamma_count
        )
        # print(f"BEST MODEL: self.current_best_model")
        # Advance queue_index
        self.queue_index = (self.queue_index + 1) % self.history_length
        t2 = time.perf_counter_ns()

        # print(cost)
        
        # print(f"Prep cost: {(t1-t0)*1e-6:.3f}ms, "
        #     f"Jit call: {(t2-t1)*1e-6:.3f}ms, ")
            # f"In-place update: {(t2-t1)*1e-6:.3f}ms, "
            # f"CPU Transfer: {(t3-t2)*1e-6:.3f}ms")

        # print(f"Prep cost: {(t1-t0)*1e-6:.3f}ms, ")
            # f"Jit call: {(t1-t0)*1e-6:.3f}ms, "
            # f"In-place update: {(t2-t1)*1e-6:.3f}ms, "
            # f"CPU Transfer: {(t3-t2)*1e-6:.3f}ms")
        return cost
    

    def reset(self):
        self.last_predicted_states = jax.device_put(
            jnp.zeros((self.integration_slots, self.num_models, self.state_size),
                    dtype='float32'), device=gpu)
        self.gamma_count = jax.device_put(
            jnp.zeros(self.integration_slots, dtype='float32'), device=gpu)
        self.integration_count[:] = -1
        self.slot_queue = deque(range(self.integration_slots))
        self.skip_counter = 0

        self.current_best_model = 0
        self.running_cost = jax.device_put(
            jnp.zeros(self.num_models, dtype='float32'), device=gpu)
        self.cost_history = jax.device_put(
            jnp.zeros((self.num_models, self.history_length), dtype='float32'), device=gpu)
        self.queue_index = 0

        self.buffer = [deque() for _ in range(self.control_size)]

    def get_best_model(self):
        return self.current_best_model
