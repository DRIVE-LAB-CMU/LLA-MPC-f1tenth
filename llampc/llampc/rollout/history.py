"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import jax
from jax import jit
import jax.numpy as jnp



bank_argmin = jax.jit(jnp.argmin)
# @jitclass(spec)
class LBHistory:

    def __init__(self, num_models, history_length, dt, cost_weights, dynamics, integrator, state_size):
        self.num_models = num_models
        self.history_length = history_length
        self.last_predicted_states = jnp.zeros((state_size, self.num_models), dtype=jnp.float64)
        self.running_cost = jnp.zeros(self.num_models, dtype=jnp.float64)
        self.cost_history = jnp.zeros((self.num_models, self.history_length), dtype=jnp.float64)
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = cost_weights
        self.dynamics = dynamics
        self.integrator = integrator
        self.state_size = state_size

    def predict_states(self, x_t, u_t):
        x_batch, u_batch = self._get_batch(x_t, u_t)
        self.last_predicted_states = self._integrate_batch(x_batch, u_batch, 0, self.dt).T

    def update_lookback_error(self, x_t):
        self.running_cost, self.cost_history= self._cost_update(
            x_t, self.last_predicted_states, self.cost_history,
            self.running_cost,self.queue_index)
        
        self.queue_index = (self.queue_index + 1) % self.history_length

    def reset(self):
        self.last_predicted_states = jnp.zeros((self.state_size, self.num_models), dtype=jnp.float64)
        self.running_cost = jnp.zeros(self.num_models, dtype=jnp.float64)
        self.cost_history = jnp.zeros((self.num_models, self.history_length), dtype=jnp.float64)
        self.queue_index = 0

    def get_best_model(self):
        return bank_argmin(self.running_cost)
    
    @jit(static_argnums=0)
    def _cost_update(self, x_t, last_predicted_states, cost_history, running_cost, queue_index):
        running_cost = running_cost - cost_history[:, queue_index]

        diff = x_t[:, None] - last_predicted_states
        cost = jnp.sum(jnp.square(diff) * self.cost_weights[:, None], axis=0)
        cost_history = cost_history.at[:, queue_index].set(cost)
        running_cost = running_cost + cost

        return running_cost, cost_history

    @jit(static_argnums=0)
    def _get_batch(self, x_t, u_t):
        x_t_batch = jnp.broadcast_to(x_t[None, :], (self.num_models, x_t.shape[0]))
        u_t_batch = jnp.broadcast_to(u_t[None, :], (self.num_models, u_t.shape[0]))
        return x_t_batch, u_t_batch
    
    def _integrate_batch(self, x_t_batch, u_t_batch, t_start, t_end):
        """Batched version of _integrate"""
        odesol = self.integrator(
            self.dynamics.diffequation,
            x_t_batch,
            jnp.array([t_start, t_end]),
            u_t_batch,
            self.dynamics.get_state_add()
            )
        return odesol[-1]

