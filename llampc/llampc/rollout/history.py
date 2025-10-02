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

# def _get_batch(num_models, x_t, u_t):
#     x_t_batch = np.empty((num_models, x_t.shape[0]), dtype=np.float64)
#     u_t_batch = np.empty((num_models, u_t.shape[0]), dtype=np.float64)
    
#     for i in range(num_models):
#         x_t_batch[i, :] = x_t
#         u_t_batch[i, :] = u_t

#     return x_t_batch, u_t_batch

# spec = [
#     ('num_models', int64),          
#     ('history_length', int64),       
#     ('last_predicted_states', float64[:, :]), 
#     ('running_cost', float64[:]),         
#     ('cost_history', float64[:, :]),       
#     ('queue_index', int64),
#     ('dt', float64),                          
#     ('cost_weights', float64[:]),  
#     ('state_size', int64),                    
# ]
# @jitclass(spec)

@jax.jit
def get_lookback_error(last_predicted_states, x_t, cost_weights, queue_index):
    # running_cost = running_cost - cost_history[:, queue_index]
    # cost = jnp.sum(jnp.square(x_t[None, :] - last_predicted_states) * cost_weights[None, :], axis = 1)
    # cost_history = cost_history.at[:, queue_index].set(cost)
    # running_cost = running_cost + cost
    cost = jnp.sum(jnp.square(x_t[None, :] - last_predicted_states) * cost_weights[None, :], axis = 1)

    return cost

@jax.jit
def find_best_model(running_cost):
    return jnp.argmin(running_cost)



# def cost_update(x_t, last_predicted_states, cost_history, running_cost, cost_weights, queue_index):
#     running_cost = running_cost - cost_history[:, queue_index]

#     diff = x_t[:, None] - last_predicted_states
#     cost = jnp.sum(jnp.square(diff) * cost_weights[:, None], axis=0)
#     cost_history = cost_history.at[:, queue_index].set(cost)
#     running_cost = running_cost + cost

#     return running_cost, cost_history
# cost_update = jit(cost_update)
# @jitclass(spec)
class LBHistory:

    def __init__(self, num_models, history_length, dt, cost_weights, state_size, integrator_factory, dynamics_bank, diffeq):
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
            diffeq
        )

    def predict_states(self, x_t, u_t):
        """Batched version of _integrate"""
        self.last_predicted_states = self.integrator(
            self.dynamics_bank.get_known_params(),
            x_t,
            u_t,
            self.dt)
        
    def update_lookback_error(self, x_t):
        self.running_cost -= self.cost_history[:, self.queue_index]
        cost = np.array(get_lookback_error(
            self.last_predicted_states,
            x_t, 
            self.cost_weights,
            self.queue_index
        ))
        self.cost_history[:, self.queue_index] = cost
        self.running_cost += cost
        self.queue_index = (self.queue_index + 1) % self.history_length
    

    def reset(self):
        self.last_predicted_states = jnp.zeros((self.num_models, self.state_size))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0

    def get_best_model(self):
        return find_best_model(self.running_cost)

           
    
    
    
    
