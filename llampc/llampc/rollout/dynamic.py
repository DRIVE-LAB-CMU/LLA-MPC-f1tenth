# integrate vehicle dynamics by 1 step
from llampc.rollout import ModelBank
import numpy as np


class DynamicBank(ModelBank):
	def __init__(self, lf, lr, mass, Iz, mean_dict, variation_dict, num_models, history_length, dt, cost_weights):
		#initializes a bank of models ready for vectorized calculation
		self.lf = lf
		self.lr = lr
		self.mass = mass
		self.Iz = Iz
		super().__init__(num_models, history_length, dt, cost_weights)

		for key in variation_dict.keys():
			setattr(
				self,
				key,
				mean_dict[key] * ones *
				(np.random.uniform(-variation_dict[key], variation_dict[key], self.num_models) + 1)
			)

	def get_model_params(self, index):
		return {
			'Bf': self.Bf[index], 
            'Br': self.Br[index], 
            'Cf': self.Cf[index], 
            'Cr': self.Cr[index], 
            'Df': self.Df[index], 
            'Dr': self.Dr[index], 
            'Cro':self.Cro[index], 
            'Cd': self.Cd[index], 
            'Ce': self.Ce[index], 
            'Cm': self.Cm[index], 
		}

	def get_model_params_arr(self, index):
		return np.array([
			self.Bf[index],
			self.Br[index],
			self.Cf[index],
			self.Cr[index],
			self.Df[index],
			self.Dr[index],
			self.Cro[index],
			self.Cd[index],
			self.Ce[index],
			self.Cm[index]
		])


    def _diffequation(self, t, x_batch, u_batch):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
			u is a 2x1 vector: [acc/pwm, steer]^T
		"""
		steer = u_batch[:, 1]
		psi = x_batch[:, 2]
		vx = x_batch[:, 3]
		vy = x_batch[:, 4]
		omega = x_batch[:, 5]

		Ffy, Frx, Fry = self.calc_forces(x_batch, u_batch)

		dxdt = np.zeros((self.num_models, 6))
		dxdt[:, 0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[:, 1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[:, 2] = omega
		dxdt[:, 3] = 1/self.mass * (Frx - Ffy*np.sin(steer)) + vy*omega
		dxdt[:, 4] = 1/self.mass * (Fry + Ffy*np.cos(steer)) - vx*omega
		dxdt[:, 5] = 1/self.Iz * (Ffy*self.lf*np.cos(steer) - Fry*self.lr)

		return dxdt

	def calc_forces(self, x_batch, u_batch, return_slip=False):
		acc = u_batch[:, 0]
		steer = u_batch[:, 1]
		psi = x_batch[:, 2]
		vx = x_batch[:, 3]
		vy = x_batch[:, 4]
		omega = x_batch[:, 5]

		if self.approx:
			# rolling friction and drag are ignored
			
			Frx = self.mass*acc

			# See Vehicle Dynamics and Control (Rajamani)
			alphaf = steer - (self.lf*omega + vy)/vx
			alphar = (self.lr*omega - vy)/vx
			Ffy = 2 * self.Cf * alphaf
			Fry = 2 * self.Cr * alphar

		else:
			Frx = self.mass * (acc * self.Ce - self.Cm * vx ) - self.Cr0 - self.Cr2 * (vx ** 2)

			alphaf = steer - np.arctan2((self.lf*omega + vy), abs(vx))
			alphar = np.arctan2((self.lr*omega - vy), abs(vx))
			Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
			Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))
		if return_slip:
			return Ffy, Frx, Fry, alphaf, alphar
		else:
			return Ffy, Frx, Fry # each of these should end up being num_models long

	# def _diffequation_batch(self, t, x_batch, u_batch):
	# 	"""Batched version of _diffequation using calc_forces_batch"""
	# 	psi = x_batch[:, 2]
	# 	vx = x_batch[:, 3]
	# 	vy = x_batch[:, 4]
	# 	omega = x_batch[:, 5]

	# 	# Get forces for all models at once
	# 	Ffy, Frx, Fry = self.calc_forces_batch(x_batch, u_batch)

	# 	return np.stack([
	# 		vx * np.cos(psi) - vy * np.sin(psi),
	# 		vx * np.sin(psi) + vy * np.cos(psi),
	# 		omega,
	# 		1 / self.mass * (Frx - Ffy * np.sin(u_batch[:, 1])) + vy * omega,
	# 		1 / self.mass * (Fry + Ffy * np.cos(u_batch[:, 1])) - vx * omega,
	# 		1 / self.Iz * (Ffy * self.lf * np.cos(u_batch[:, 1]) - Fry * self.lr)
	# 	], axis=1)

	# def calc_forces_batch(self, x_batch, u_batch, return_slip=False):
	# 	"""Batched version of calc_forces"""
	# 	steer = u_batch[:, 1]
	# 	psi = x_batch[:, 2]
	# 	vx = x_batch[:, 3]
	# 	vy = x_batch[:, 4]
	# 	omega = x_batch[:, 5]
	# 	acc = 

	# 	if self.approx:
	# 		# rolling friction and drag are ignored
	# 		acc = u_batch[:, 0]
	# 		Frx = self.mass * acc

	# 		# See Vehicle Dynamics and Control (Rajamani)
	# 		alphaf = steer - (self.lf * omega + vy) / vx
	# 		alphar = (self.lr * omega - vy) / vx
	# 		Ffy = 2 * self.Cf * alphaf
	# 		Fry = 2 * self.Cr * alphar

	# 	else:
	# 		if self.input_acc:
	# 			# rolling friction and drag are ignored
	# 			acc = u_batch[:, 0]
	# 			Frx = self.mass * acc
	# 		else:
	# 			# rolling friction and drag are modeled
	# 			pwm = u_batch[:, 0]
	# 			Frx = (self.Cm1 - self.Cm2 * vx) * pwm - self.Cr0 - self.Cr2 * (vx ** 2)

	# 		alphaf = steer - np.arctan2((self.lf * omega + vy), np.abs(vx))
	# 		alphar = np.arctan2((self.lr * omega - vy), np.abs(vx))
	# 		Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
	# 		Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))

	# 	if return_slip:
	# 		return Ffy, Frx, Fry, alphaf, alphar
	# 	else:
	# 		return Ffy, Frx, Fry

