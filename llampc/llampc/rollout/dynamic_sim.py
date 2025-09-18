# integrate vehicle dynamics by 1 step
import numpy as np

import jax
from jax import jit, lax
import jax.numpy as jnp



class DynamicSimBank():
    def __init__(self, 
                 lf, lr, 
                 mass, I, 
                 h, v_switch,
                 a_max, v_min, v_max,
                 s_min, s_max, 
                 sv_min, sv_max,
                 param_dict
                 ):
        # non-varying parameters
        self.lf = lf
        self.lr = lr
        self.m = mass
        self.I = I
        self.h = h
        self.v_switch = v_switch
        self.a_max = a_max
        self.v_min = v_min
        self.v_max = v_max
        self.s_min = s_min
        self.s_max = s_max
        self.sv_min = sv_min
        self.sv_max = sv_max

        #varying parameters
        
        # varying parameters
        for key, value in param_dict.items():
            setattr(self, key, value)

        # self.accl_constraints = jit(self._accl_constraints, static_argnums=(0,))
        # self.steering_constraint = jit(self._steering_constraint, static_argnums=(0,))
        # self.diffequation = jit(self.diffequation, static_argnums=(0,))

    def get_model_params_arr(self, index):
        return np.array([
            self.C_Sf[index],
            self.C_Sr[index],
            self.mu[index]
        ])
    
    def _accl_constraints(self, vel, accl_batch):
        """
        Batched acceleration constraints.

        Args:
            vel (jnp.ndarray): (N,) current velocities
            accl_batch (jnp.ndarray): (N,) unconstrained desired accelerations
            v_switch (float): switching velocity
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            jnp.ndarray: (N,) constrained accelerations
        """
        pos_limit = jnp.where(vel > self.v_switch, self.a_max * self.v_switch / vel, self.a_max)

        # Conditions
        stop_condition = ((vel <= self.v_min) & (accl_batch <= 0)) | ((vel >= self.v_max) & (accl_batch >= 0))
        max_decel = accl_batch <= -self.a_max
        max_accel = accl_batch >= pos_limit

        # Start with unconstrained
        constrained_accl = accl_batch

        constrained_accl = jnp.where(stop_condition, 0.0, constrained_accl)
        constrained_accl = jnp.where(max_decel, -self.a_max, constrained_accl)
        constrained_accl = jnp.where(max_accel, pos_limit, constrained_accl)

        return constrained_accl


    def _steering_constraint(self, steering_angle, steering_velocity):
        """
        Batched steering velocity constraints.

        Args:
            steering_angle (jnp.ndarray): (N,) current steering angles
            steering_velocity (jnp.ndarray): (N,) desired steering velocities
            s_min (float): min steering angle
            s_max (float): max steering angle
            sv_min (float): min steering velocity
            sv_max (float): max steering velocity

        Returns:
            jnp.ndarray: (N,) constrained steering velocities
        """

        # Conditions
        stop_condition = ((steering_angle <= self.s_min) & (steering_velocity <= 0)) | \
                        ((steering_angle >= self.s_max) & (steering_velocity >= 0))
        too_negative = steering_velocity <= self.sv_min
        too_positive = steering_velocity >= self.sv_max

        constrained_sv = steering_velocity
        constrained_sv = jnp.where(stop_condition, 0.0, constrained_sv)
        constrained_sv = jnp.where(too_negative, self.sv_min, constrained_sv)
        constrained_sv = jnp.where(too_positive, self.sv_max, constrained_sv)

        return constrained_sv

    def get_state_add(self):
        return None

    def diffequation(self, t, x_batch, u_batch, state_add):
        """	write dynamics as first order ODE: dxdt = f(x(t))
            x is a 6x1 vector: [x, y, psi, vx, slip, omega, steer]^T
            u is a 2x1 vector: [acc/pwm, steer_rate]^T
        """

        psi = x_batch[:, 2]
        vx = x_batch[:, 3]
        slip =  x_batch[:, 4]
        omega = x_batch[:, 5]
        steer = x_batch[:, 6]
        
        
        acc = u_batch[:, 0]
        steer_rate = u_batch[:, 1]

        acc= self._accl_constraints(
            vel=vx,
            accl_batch=acc
        )

        steer_rate = self._steering_constraint(
            steering_angle=steer,
            steering_velocity=steer_rate
        )

        base = (self.lr + self.lf)

        def low_speed(_):
            dxdt = jnp.stack([
                vx * jnp.cos(psi),  # x
                vx * jnp.sin(psi),  # y
                acc / base * jnp.tan(steer),  # psi
                acc, # vx
                jnp.zeros_like(vx), # slip = 0
                (acc / base) * jnp.tan(steer) + vx / (base * jnp.square(jnp.cos(steer))) * steer_rate,  # omega
                steer_rate  # steer
            ], axis=1)
            return dxdt

        def high_speed(_):
            g = 9.81
            base = self.lr + self.lf
            dxdt = jnp.stack([
                vx * jnp.cos(psi + slip),   # x
                vx * jnp.sin(psi + slip),   # y
                omega,                      # psi
                acc,                        # vx
                (self.mu/(vx**2 * base) * (self.C_Sr*(g*self.lf + acc*self.h)*self.lr - self.C_Sf*(g*self.lr - acc*self.h)*self.lf) - 1) * omega \
                - self.mu/(vx * base) * (self.C_Sr*(g*self.lf + acc*self.h) + self.C_Sf*(g*self.lr - acc*self.h)) * slip \
                + self.mu/(vx * base) * (self.C_Sf*(g*self.lr - acc*self.h)) * steer,   # slip
                -self.mu * self.m / (vx * self.I * base) * (self.lf**2 * self.C_Sf * (g * self.lr - acc * self.h) + self.lr**2 * self.C_Sr * (g * self.lf + acc * self.h)) * omega \
                + self.mu * self.m / (self.I * base) * (self.lr * self.C_Sr * (g * self.lf + acc * self.h) - self.lf * self.C_Sf * (g * self.lr - acc * self.h)) * slip \
                + self.mu * self.m / (self.I * base) * self.lf * self.C_Sf * (g * self.lr - acc * self.h) * steer,   # omega
                steer_rate                  # steer
            ], axis=1)
            return dxdt


        dxdt = lax.cond(jnp.abs(vx[0]) < 0.5, low_speed, high_speed, operand=None)

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

    #         alphaf = jnp.where(abs(vx) < 1e-4, 0,  steer - jnp.arctan2((self.lf*omega + vy), abs(vx)))
    #         alphar = jnp.where(abs(vx) < 1e-4, 0, jnp.arctan2((self.lr*omega - vy), abs(vx)))
    #         Ffy = self.Df * jnp.sin(self.Cf * jnp.arctan(self.Bf * alphaf))
    #         Fry = self.Dr * jnp.sin(self.Cr * jnp.arctan(self.Br * alphar))
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

    # 	return jnp.stack([
    # 		vx * jnp.cos(psi) - vy * jnp.sin(psi),
    # 		vx * jnp.sin(psi) + vy * jnp.cos(psi),
    # 		omega,
    # 		1 / self.mass * (Frx - Ffy * jnp.sin(u_batch[:, 1])) + vy * omega,
    # 		1 / self.mass * (Fry + Ffy * jnp.cos(u_batch[:, 1])) - vx * omega,
    # 		1 / self.Iz * (Ffy * self.lf * jnp.cos(u_batch[:, 1]) - Fry * self.lr)
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

    # 		alphaf = steer - jnp.arctan2((self.lf * omega + vy), jnp.abs(vx))
    # 		alphar = jnp.arctan2((self.lr * omega - vy), jnp.abs(vx))
    # 		Ffy = self.Df * jnp.sin(self.Cf * jnp.arctan(self.Bf * alphaf))
    # 		Fry = self.Dr * jnp.sin(self.Cr * jnp.arctan(self.Br * alphar))

    # 	if return_slip:
    # 		return Ffy, Frx, Fry, alphaf, alphar
    # 	else:
    # 		return Ffy, Frx, Fry

