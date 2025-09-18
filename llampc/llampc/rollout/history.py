"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from llampc.rollout import integrate_batch
from numba import njit, float64, boolean, int64, prange
from numba.experimental import jitclass

@njit(parallel = True)
def _get_batch(num_models, x_t, u_t):
    x_t_batch = np.empty((num_models, x_t.shape[0]), dtype=np.float64)
    u_t_batch = np.empty((num_models, u_t.shape[0]), dtype=np.float64)
    
    for i in prange(num_models):
        x_t_batch[i, :] = x_t
        u_t_batch[i, :] = u_t

    return x_t_batch, u_t_batch

@njit
def predict_states(history, integrator, dynamics_bank, diffeq, x_t, u_t):
    x_batch, u_batch = _get_batch(history.num_models, x_t, u_t)
    history.last_predicted_states = integrate_batch(
        integrator, dynamics_bank, diffeq, x_batch, u_batch, 0, history.dt).T

@njit
def update_lookback_error(history, x_t):
    history.running_cost -= history.cost_history[:, history.queue_index]

    cost = np.sum(np.square(x_t[:, None] - history.last_predicted_states) * history.cost_weights[:, None], axis = 0)
    history.cost_history[:, history.queue_index] = cost
    history.running_cost += cost
    history.queue_index = (history.queue_index + 1) % history.history_length


spec = [
    ('num_models', int64),          
    ('history_length', int64),       
    ('last_predicted_states', float64[:, :]), 
    ('running_cost', float64[:]),         
    ('cost_history', float64[:, :]),       
    ('queue_index', int64),
    ('dt', float64),                          
    ('cost_weights', float64[:]),  
    ('state_size', int64),                    
]
@jitclass(spec)
class LBHistory:

    def __init__(self, num_models, history_length, dt, cost_weights, state_size):
        self.num_models = num_models
        self.history_length = history_length
        self.last_predicted_states = np.zeros((state_size, self.num_models))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = cost_weights
        self.state_size = state_size
    
    def reset(self):
        self.last_predicted_states = np.zeros((self.state_size, self.num_models))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0

    def get_best_model(history):
        return np.argmin(history.running_cost)

           
    
    
    
    