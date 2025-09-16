# integrate vehicle dynamics by 1 step
from llampc.rollout import ModelBank
import numpy as np


class DynamicSimBank(ModelBank):
    def __init__(self, lf, lr, mass, I,  h, params_car, mean_dict, variation_dict, num_models, history_length, dt, cost_weights, ground_truth = True):
        #initializes a bank of models ready for vectorized calculation
        self.lf = lf
        self.lr = lr
        self.m = mass
        self.I = I
        self.h = h
        self.params_car = params_car
        self.approx = False
        super().__init__(num_models, history_length, dt, cost_weights, state_size = 7, sim = True)

        for key in variation_dict.keys():
            setattr(
                self,
                key,
                mean_dict[key] * 
                (np.random.uniform(-variation_dict[key], variation_dict[key], self.num_models) + 1)
            )

        if ground_truth:
            for key in variation_dict.keys():
                param_array = getattr(self, key)
                param_array[0] = mean_dict[key]


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
    


    def accl_constraints(self, vel, accl_batch, v_switch, a_max, v_min, v_max):
        """
        Batched acceleration constraints.

        Args:
            vel (np.ndarray): (N,) current velocities
            accl_batch (np.ndarray): (N,) unconstrained desired accelerations
            v_switch (float): switching velocity
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            np.ndarray: (N,) constrained accelerations
        """
        pos_limit = np.where(vel > v_switch, a_max * v_switch / vel, a_max)

        # Start with unconstrained
        constrained_accl = np.copy(accl_batch)

        # Conditions
        stop_condition = ((vel <= v_min) & (accl_batch <= 0)) | ((vel >= v_max) & (accl_batch >= 0))
        max_decel = accl_batch <= -a_max
        max_accel = accl_batch >= pos_limit

        constrained_accl[stop_condition] = 0.0
        constrained_accl[max_decel] = -a_max
        constrained_accl[max_accel] = pos_limit[max_accel]

        return constrained_accl


    
    def steering_constraint(self, steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
        """
        Batched steering velocity constraints.

        Args:
            steering_angle (np.ndarray): (N,) current steering angles
            steering_velocity (np.ndarray): (N,) desired steering velocities
            s_min (float): min steering angle
            s_max (float): max steering angle
            sv_min (float): min steering velocity
            sv_max (float): max steering velocity

        Returns:
            np.ndarray: (N,) constrained steering velocities
        """
        constrained_sv = np.copy(steering_velocity)

        # Conditions
        stop_condition = ((steering_angle <= s_min) & (steering_velocity <= 0)) | \
                        ((steering_angle >= s_max) & (steering_velocity >= 0))
        too_negative = steering_velocity <= sv_min
        too_positive = steering_velocity >= sv_max

        constrained_sv[stop_condition] = 0.0
        constrained_sv[too_negative] = sv_min
        constrained_sv[too_positive] = sv_max

        return constrained_sv



    
    
    def _diffequation(self, t, x_batch, u_batch, rates = None):
        """	write dynamics as first order ODE: dxdt = f(x(t))
            x is a 6x1 vector: [x, y, psi, vx, slip, omega, steer]^T
            u is a 2x1 vector: [acc/pwm, steer_rate]^T
            rates is a 
        """

        psi = x_batch[:, 2]
        vx = x_batch[:, 3]
        slip =  x_batch[:, 4]
        omega = x_batch[:, 5]
        steer = x_batch[:, 6]
        
        
        acc = u_batch[:, 0]
        steer_rate = u_batch[:, 1]

        acc= self.accl_constraints(
            vel=vx,
            accl_batch=acc,
            v_switch=self.params_car["v_switch"],
            a_max=self.params_car["a_max"],
            v_min=self.params_car["v_min"],
            v_max=self.params_car["v_max"]
        )

        steer_rate = self.steering_constraint(
            steering_angle=steer,
            steering_velocity=steer_rate,
            s_min=self.params_car["s_min"],
            s_max=self.params_car["s_max"],
            sv_min=self.params_car["sv_min"],
            sv_max=self.params_car["sv_max"]
        )

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

