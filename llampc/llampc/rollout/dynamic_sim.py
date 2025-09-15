# integrate vehicle dynamics by 1 step
from llampc.rollout import ModelBank
import numpy as np


class DynamicSimBank(ModelBank):
    def __init__(self, lf, lr, mass, I,  h, mean_dict, variation_dict, num_models, history_length, dt, cost_weights):
        #initializes a bank of models ready for vectorized calculation
        self.lf = lf
        self.lr = lr
        self.m = mass
        self.I = I
        self.h = h
        self.approx = False
        super().__init__(num_models, history_length, dt, cost_weights, state_size = 7, sim = True)

        for key in variation_dict.keys():
            setattr(
                self,
                key,
                mean_dict[key] * 
                (np.random.uniform(-variation_dict[key], variation_dict[key], self.num_models) + 1)
            )

    def get_model_params(self, index):
        return {
            'C_Sf': self.C_Sf[index], 
            'C_Sr': self.C_Sr[index],
            'mu': self.C_Sr[index],
        }

    def get_model_params_arr(self, index):
        return np.array([
            self.C_Sf[index],
            self.C_Sr[index],
            self.mu[index]
        ])


    def _diffequation(self, t, x_batch, u_batch, rates = None):
        """	write dynamics as first order ODE: dxdt = f(x(t))
            x is a 6x1 vector: [x, y, psi, vx, slip, omega, steer]^T
            u is a 2x1 vector: [acc/pwm, steer_rate]^T
            rates is a 
        """
        acc = u_batch[:, 0]
        steer_rate = u_batch[:, 1]

        psi = x_batch[:, 2]
        vx = x_batch[:, 3]
        slip =  x_batch[:, 4]
        omega = x_batch[:, 5]
        steer = x_batch[:, 6]

        base = (self.lr + self.lf)
        dxdt = np.zeros((self.num_models, 7))

        if abs(vx[0]) < 0.5:
            
            dxdt[:, 0] = vx*np.cos(psi) #x
            dxdt[:, 1] = vx*np.sin(psi) #y
            dxdt[:, 2] = acc / base * np.tan(steer) # psi
            dxdt[:, 3] = acc #vx
            dxdt[:, 4] = 0 #slip
            dxdt[:, 5] = (acc/base) * np.tan(steer) + vx / (base * np.square(np.cos(steer))) * steer_rate
            dxdt[:, 6] = steer_rate # steer
        else:
            g = 9.81

            
            # one = -self.mu*self.m/(vx*self.I*(self.lr+self.lf)) *(self.lf**2*self.C_Sf*(g*self.lr-acc*self.h) + self.lr**2*self.C_Sr*(g*self.lf + acc*self.h))*omega
            # two = self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(g*self.lf + acc*self.h) - self.lf*self.C_Sf*(g*self.lr - acc*self.h))*slip
            # three = (self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(g*self.lr - acc*self.h)*steer)
            dxdt[:, 0] = vx * np.cos(psi + slip) #x
            dxdt[:, 1] = vx * np.sin(psi + slip) #y
            dxdt[:, 2] = omega # psi
            dxdt[:, 3] = acc #vx
            dxdt[:, 4] = (self.mu/(vx**2*(self.lr+self.lf))*(self.C_Sr*(g*self.lf + acc*self.h)*self.lr - self.C_Sf*(g*self.lr - acc*self.h)*self.lf)-1)*omega \
                -self.mu/(vx*(self.lr+self.lf))*(self.C_Sr*(g*self.lf + acc*self.h) + self.C_Sf*(g*self.lr-acc*self.h))*slip \
                +self.mu/(vx*(self.lr+self.lf))*(self.C_Sf*(g*self.lr-acc*self.h))*steer #slip
            dxdt[:, 5] = -self.mu*self.m/(vx*self.I*(self.lr+self.lf)) *(self.lf**2*self.C_Sf*(g*self.lr-acc*self.h) + self.lr**2*self.C_Sr*(g*self.lf + acc*self.h))*omega \
                +self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(g*self.lf + acc*self.h) - self.lf*self.C_Sf*(g*self.lr - acc*self.h))*slip \
                +self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(g*self.lr - acc*self.h)*steer #omega
            dxdt[:, 6] = steer_rate # steer
        

        return dxdt

    # def calc_forces(self, x_batch, u_batch, return_slip=False):
    #     acc = u_batch[:, 0]
    #     steer = u_batch[:, 1]
    #     psi = x_batch[:, 2]
    #     vx = x_batch[:, 3]
    #     vy = x_batch[:, 4]
    #     omega = x_batch[:, 5]

    #     if self.approx:
    #         # rolling friction and drag are ignored
            
    #         Frx = self.mass*acc

    #         # See Vehicle Dynamics and Control (Rajamani)
    #         alphaf = steer - (self.lf*omega + vy)/vx
    #         alphar = (self.lr*omega - vy)/vx
    #         Ffy = 2 * self.Cf * alphaf
    #         Fry = 2 * self.Cr * alphar

    #     else:
    #         Frx = self.mass * (acc * self.Ce - self.Cm * vx ) - self.Cro - self.Cd * (vx ** 2)

    #         alphaf = np.where(abs(vx) < 1e-4, 0,  steer - np.arctan2((self.lf*omega + vy), abs(vx)))
    #         alphar = np.where(abs(vx) < 1e-4, 0, np.arctan2((self.lr*omega - vy), abs(vx)))
    #         Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
    #         Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))
    #     if return_slip:
    #         return Ffy, Frx, Fry, alphaf, alphar
    #     else:
    #         return Ffy, Frx, Fry # each of these should end up being num_models long

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

