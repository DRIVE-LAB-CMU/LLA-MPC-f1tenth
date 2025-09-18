"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from llampc.rollout import odeintRK4_batch

class LBHistory:

    def __init__(self, num_models, history_length, dt, cost_weights,  dynamics, integrator= odeintRK4_batch,state_size = 6):
        self.num_models = num_models
        self.history_length = history_length
        self.last_predicted_states = np.zeros((state_size, self.num_models))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = cost_weights
        self.integrator = integrator # should be njit integrator function
        self.dynamics = dynamics # should be dynamics jitclass instance
        
    def predict_states(self, x_t, u_t):
        x_batch, u_batch = self._get_batch(x_t, u_t)
        self.last_predicted_states = self._integrate_batch(x_batch, u_batch, 0, self.dt).T
    
    def update_lookback_error(self, x_t):
        self.running_cost -= self.cost_history[:, self.queue_index]

        # print(x_t[:, None].shape)
        # print(self.last_predicted_states.shape)
        cost = np.sum(np.square(x_t[:, None] - self.last_predicted_states) * self.cost_weights[:, None], axis = 0)
        self.cost_history[:, self.queue_index] = cost
        self.running_cost += cost
        self.queue_index = (self.queue_index + 1) % self.history_length
        
    def get_best_model(self):
        return np.argmin(self.running_cost)
    
    def _get_batch(self, x_t, u_t):
        x_t_batch = np.tile(x_t.reshape(1, -1), (self.num_models, 1))
        u_t_batch = np.tile(u_t.reshape(1, -1), (self.num_models, 1))
        return x_t_batch, u_t_batch
    
    def _integrate_batch(self, x_t_batch, u_t_batch, t_start, t_end):
        """Batched version of _integrate"""
        odesol = self.integrator(
            dynamics = self.dynamics,
            y0_batch=x_t_batch,
            t=np.array([t_start, t_end]),
            control_batch = u_t_batch
            )
        return odesol[-1]