"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from llampc.rollout.rk6 import odeintRK6, odeintRK6_batch, odeintRK4_batch, odeintEuler_batch
import matplotlib.pyplot as plt


class ModelBank:

    def __init__(self, num_models, history_length, dt, cost_weights, state_size = 6, sim = False):
        self.num_models = num_models
        self.history_length = history_length
        self.last_predicted_states = np.zeros((state_size, self.num_models))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = cost_weights
        self.sim = sim
    
    def get_batch(self, x_t, u_t):
        x_t_batch = np.tile(x_t.reshape(1, -1), (self.num_models, 1))
        u_t_batch = np.tile(u_t.reshape(1, -1), (self.num_models, 1))
        return x_t_batch, u_t_batch

    def predict_states(self, x_t, u_t):
        x_batch, u_batch = self.get_batch(x_t, u_t)
        integrator = odeintRK4_batch #if not self.sim else odeintEuler_batch
        self.last_predicted_states = self.integrate_batch(x_batch, u_batch, 0, self.dt, integrator).T #num_models x state_size states

        # print(f"PREDICT {self.last_predicted_states.shape}")
    
    def update_lookback_error(self, x_t):
        #queue index begins pointing to the outdated index
        # print("RUNNING COST PER MODEL:")
        # print(self.running_cost)

        # print("PREDICTED STATES:")
        # print(self.last_predicted_states)

        # print("COST HISTORY")
        # print(self.cost_history)

        
        self.running_cost -= self.cost_history[:, self.queue_index]

        # print(x_t[:, None].shape)
        # print(self.last_predicted_states.shape)
        cost = np.sum(np.square(x_t[:, None] - self.last_predicted_states) * self.cost_weights[:, None], axis = 0)
        self.cost_history[:, self.queue_index] = cost
        self.running_cost += cost
        self.queue_index = (self.queue_index + 1) % self.history_length
        
    def get_best_model(self):
        return np.argmin(self.running_cost)

    def integrate_batch(self, x_t_batch, u_t_batch, t_start, t_end, integrator = odeintRK4_batch):
        """Batched version of _integrate"""
        fun = self._diffequation
        odesol = integrator(
            fun=fun,
            y0_batch=x_t_batch,
            t=[t_start, t_end],
            args_batch=(u_t_batch,))
        return odesol[-1]

    # def _integrate(self, x_t, u_t, t_start, t_end):
    # 	"""	integrates using an RK6 ODE solver
    # 		returns x_t at t_end = ∫ dxdt dt given x_t at t_start
    # 		x_t is either a 4x1 vector: [x, y, psi, v]^T
    # 				   or a 6x1 vector: [x, y, psi, dxdt, dydt, dpsidt]^T
    # 	"""
    # 	fun=self._diffequation
    # 	odesol = odeintRK6(
    # 		fun=fun, 
    # 		y0=x_t, 
    # 		t=[t_start, t_end], 
    # 		args=(u_t,))
    # 	return odesol[-1,:]


    # def plot_results(self, t, x, dxdt, u, friction_circle=False):
    # 	"""	plot states and inputs
    # 	"""
    # 	# plot position
    # 	plt.figure()
    # 	plt.plot(x[0,:], x[1,:])
    # 	plt.xlabel('x [m]')
    # 	plt.ylabel('y [m]')
    # 	plt.axis('equal')
    # 	plt.grid(True)

    # 	plt.figure()
    # 	plt.plot(t, x[0,:], label='x')
    # 	plt.plot(t, x[1,:], label='y')
    # 	plt.xlabel('time [s]')
    # 	plt.ylabel('position [m]')
    # 	plt.grid(True)
    # 	plt.legend()

    # 	# plot velocity
    # 	if dxdt is not None:
    # 		plt.figure()
    # 		plt.plot(t, dxdt[0,:], label='speed x')
    # 		plt.plot(t, dxdt[1,:], label='speed y')
    # 		if not friction_circle:
    # 			plt.plot(t, dxdt[2,:], label='yaw rate')
    # 		plt.plot(t, np.sqrt(dxdt[0,:]**2+dxdt[1,:]**2), '--', label='speed abs')
    # 		plt.xlabel('time [s]')
    # 		plt.ylabel('velocity [m/s]')
    # 		plt.grid(True)
    # 		plt.legend()

    # 		plt.figure()
    # 		plt.plot(dxdt[0,:], dxdt[1,:])
    # 		plt.xlabel('speed x [m/s]')
    # 		plt.ylabel('speed y [m/s]')
    # 		plt.axis('equal')
    # 		plt.grid(True)

    # 	# plot inertial heading
    # 	plt.figure()
    # 	if friction_circle:
    # 		plt.plot(t, np.arctan2(dxdt[1,:],dxdt[0,:]))
    # 	else:
    # 		plt.plot(t, x[2,:])
    # 	plt.ylabel('yaw (heading) [rad]')
    # 	plt.xlabel('time [s]')
    # 	plt.grid(True)

    # 	# plot inputs
    # 	plt.figure()
    # 	if friction_circle:
    # 		plt.plot(t[1:], u[0,:], label='force x')
    # 		plt.plot(t[1:], u[1,:], label='force y')
    # 	else:
    # 		plt.plot(t[1:], u[0,:], label='acceleration')
    # 		plt.plot(t[1:], u[1,:], label='steering')
    # 	plt.ylabel('inputs')
    # 	plt.grid(True)
    # 	plt.legend()
    # 	plt.show()