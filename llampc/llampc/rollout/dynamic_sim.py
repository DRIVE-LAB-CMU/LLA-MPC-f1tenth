# integrate vehicle dynamics by 1 step

from numba import njit, float64, boolean, int64
from numba.experimental import jitclass
import jax
import jax.numpy as jnp
@jax.jit
def _accl_constraints(bank_params, known_params, vel, accl):
    lf, lr, m, I, h, v_switch, a_max, v_min, v_max, s_min, s_max, sv_min, sv_max = known_params
    C_Sf, C_Sr, mu = bank_params

    pos_limit = jnp.where(vel > v_switch, a_max * v_switch / vel, a_max)

    def stop_case(): return 0.0
    def decel_case(): return -a_max
    def accel_case(): return pos_limit
    def nominal(): return accl

    # Ordered condition checks
    return jax.lax.cond(
        ((vel <= v_min) & (accl <= 0)) | ((vel >= v_max) & (accl >= 0)),
        stop_case,
        lambda: jax.lax.cond(
            accl <= -a_max,
            decel_case,
            lambda: jax.lax.cond(
                accl >= pos_limit,
                accel_case,
                nominal
            )
        )
    )


@jax.jit
def _steering_constraint(bank_params, known_params, steering_angle, steering_velocity):
    lf, lr, m, I, h, v_switch, a_max, v_min, v_max, s_min, s_max, sv_min, sv_max = known_params
    C_Sf, C_Sr, mu = bank_params

    def stop_case(): return 0.0
    def too_negative_case(): return sv_min
    def too_positive_case(): return sv_max
    def nominal(): return steering_velocity

    return jax.lax.cond(
        ((steering_angle <= s_min) & (steering_velocity <= 0)) |
        ((steering_angle >= s_max) & (steering_velocity >= 0)),
        stop_case,
        lambda: jax.lax.cond(
            steering_velocity <= sv_min,
            too_negative_case,
            lambda: jax.lax.cond(
                steering_velocity >= sv_max,
                too_positive_case,
                nominal
            )
        )
    )

@jax.jit
def diffequation(bank_params, known_params, x_batch, u_batch):
    """	write dynamics as first order ODE: dxdt = f(x(t))
        x is a 6x1 vector: [x, y, psi, vx, slip, omega, steer]^T
        u is a 2x1 vector: [acc/pwm, steer_rate]^T
    """
    g = 9.81
    psi = x_batch[ 2]
    vx = x_batch[ 3]
    slip =  x_batch[ 4]
    omega = x_batch[ 5]
    steer = x_batch[ 6]
    
    
    acc = u_batch[0]
    steer_rate = u_batch[1]

    lf, lr, m, I, h, v_switch, a_max, v_min, v_max, s_min, s_max, sv_min, sv_max = known_params
    C_Sf, C_Sr, mu = bank_params

    acc= _accl_constraints(
        bank_params,
        known_params,
        vel=vx,
        accl=acc
    )

    steer_rate = _steering_constraint(
        bank_params, 
        known_params,
        steering_angle=steer,
        steering_velocity=steer_rate
    )

    base = (lr + lf)

    def low_speed(_):
        return jnp.stack([
            vx * jnp.cos(psi),  # x
            vx * jnp.sin(psi),  # y
            acc / base * jnp.tan(steer),  # psi
            acc,  # vx
            0.0,  # slip = 0
            (acc / base) * jnp.tan(steer) + vx / (base * jnp.square(jnp.cos(steer))) * steer_rate,  # omega
            steer_rate  # steer
        ])

    def high_speed(_):
        slip_term = (
            mu / (vx**2 * base) *
            (C_Sr * (g * lf + acc * h) * lr -
             C_Sf * (g * lr - acc * h) * lf)
            - 1
        ) * omega \
        - mu / (vx * base) * (
            C_Sr * (g * lf + acc * h) +
            C_Sf * (g * lr - acc * h)
        ) * slip \
        + mu / (vx * base) * (
            C_Sf * (g * lr - acc * h)
        ) * steer

        omega_term = -mu * m / (vx * I * base) * (
            lf**2 * C_Sf * (g * lr - acc * h) +
            lr**2 * C_Sr * (g * lf + acc * h)
        ) * omega \
        + mu * m / (I * base) * (
            lr * C_Sr * (g * lf + acc * h) -
            lf * C_Sf * (g * lr - acc * h)
        ) * slip \
        + mu * m / (I * base) * lf * C_Sf * (
            g * lr - acc * h
        ) * steer

        return jnp.stack([
            vx * jnp.cos(psi + slip),  # x
            vx * jnp.sin(psi + slip),  # y
            omega,  # psi
            acc,  # vx
            slip_term,  # slip
            omega_term,  # omega
            steer_rate  # steer
        ])

    return jax.lax.cond(jnp.abs(vx) < 0.5, low_speed, high_speed, operand=None)


# spec = [
#     # Non-varying scalar parameters
#     ('lf', float64),
#     ('lr', float64),
#     ('m', float64),
#     ('I', float64),
#     ('h', float64),
#     ('v_switch', float64),
#     ('a_max', float64),
#     ('v_min', float64),
#     ('v_max', float64),
#     ('s_min', float64),
#     ('s_max', float64),
#     ('sv_min', float64),
#     ('sv_max', float64),

#     # Varying parameters (arrays)
#     ('C_Sf', float64[:]),
#     ('C_Sr', float64[:]),
#     ('mu', float64[:]),

#     ('num_models', int64),
# ]
# @jitclass(spec)
class DynamicSimBank():
    def __init__(self, 
                 lf, lr, 
                 mass, I, 
                 h, v_switch,
                 a_max, v_min, v_max,
                 s_min, s_max, 
                 sv_min, sv_max,
                 C_Sf,
                 C_Sr, mu,
                 num_models
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
        self.C_Sf = C_Sf
        self.C_Sr = C_Sr
        self.mu = mu

        self.num_models = num_models

        self.param_bank = jnp.stack([
            self.C_Sf, self.C_Sr, self.mu
        ], axis=1)

        # super().__init__(num_mod`     els, history_length, dt, cost_weights, state_size = 7, sim = True)

    def get_model_params_arr(self, index):
        return jnp.array([
            self.C_Sf[index],
            self.C_Sr[index],
            self.mu[index]
        ])
    
    def get_known_params(self):
        return jnp.array([
            self.lf,
            self.lr,
            self.m,
            self.I,
            self.h,
            self.v_switch,
            self.a_max,
            self.v_min,
            self.v_max,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
        ])


    
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

