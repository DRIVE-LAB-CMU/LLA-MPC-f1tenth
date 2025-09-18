# integrate vehicle dynamics by 1 step
import numpy as np

from numba import njit, float64, boolean, int64
from numba.experimental import jitclass

@njit
def _accl_constraints(dynamic_bank, vel, accl_batch):
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
    pos_limit = np.where(vel > dynamic_bank.v_switch, dynamic_bank.a_max * dynamic_bank.v_switch / vel, dynamic_bank.a_max)

    # Start with unconstrained
    constrained_accl = np.copy(accl_batch)

    # Conditions
    stop_condition = ((vel <= dynamic_bank.v_min) & (accl_batch <= 0)) | ((vel >= dynamic_bank.v_max) & (accl_batch >= 0))
    max_decel = accl_batch <= -dynamic_bank.a_max
    max_accel = accl_batch >= pos_limit

    constrained_accl[stop_condition] = 0.0
    constrained_accl[max_decel] = -dynamic_bank.a_max
    constrained_accl[max_accel] = pos_limit[max_accel]

    return constrained_accl


@njit
def _steering_constraint(dynamic_bank, steering_angle, steering_velocity):
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
    stop_condition = ((steering_angle <= dynamic_bank.s_min) & (steering_velocity <= 0)) | \
                    ((steering_angle >= dynamic_bank.s_max) & (steering_velocity >= 0))
    too_negative = steering_velocity <= dynamic_bank.sv_min
    too_positive = steering_velocity >= dynamic_bank.sv_max

    constrained_sv[stop_condition] = 0.0
    constrained_sv[too_negative] = dynamic_bank.sv_min
    constrained_sv[too_positive] = dynamic_bank.sv_max

    return constrained_sv

@njit
def diffequation(dynamic_bank, t, x_batch, u_batch):
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

    acc= _accl_constraints(
        dynamic_bank,
        vel=vx,
        accl_batch=acc
    )

    steer_rate = _steering_constraint(
        dynamic_bank, 
        steering_angle=steer,
        steering_velocity=steer_rate
    )

    base = (dynamic_bank.lr + dynamic_bank.lf)
    dxdt = np.empty((vx.shape[0], 7))

    if np.abs(vx[0]) < 0.5:

        dxdt[:, 0] = vx * np.cos(psi)  # x
        dxdt[:, 1] = vx * np.sin(psi)  # y
        dxdt[:, 2] = acc / base * np.tan(steer)  # psi
        dxdt[:, 3] = acc  # vx
        dxdt[:, 4] = 0  # slip = 0
        dxdt[:, 5] = (acc / base) * np.tan(steer) + vx / (base * np.square(np.cos(steer))) * steer_rate  # omega
        dxdt[:, 6] = steer_rate  # steer
    else:
        g = 9.81
        
        # one = -dynamic_bank.mu*dynamic_bank.m/(vx*dynamic_bank.I*(dynamic_bank.lr+dynamic_bank.lf)) *(dynamic_bank.lf**2*dynamic_bank.C_Sf*(g*dynamic_bank.lr-acc*dynamic_bank.h) + dynamic_bank.lr**2*dynamic_bank.C_Sr*(g*dynamic_bank.lf + acc*dynamic_bank.h))*omega
        # two = dynamic_bank.mu*dynamic_bank.m/(dynamic_bank.I*(dynamic_bank.lr+dynamic_bank.lf))*(dynamic_bank.lr*dynamic_bank.C_Sr*(g*dynamic_bank.lf + acc*dynamic_bank.h) - dynamic_bank.lf*dynamic_bank.C_Sf*(g*dynamic_bank.lr - acc*dynamic_bank.h))*slip
        # three = (dynamic_bank.mu*dynamic_bank.m/(dynamic_bank.I*(dynamic_bank.lr+dynamic_bank.lf))*dynamic_bank.lf*dynamic_bank.C_Sf*(g*dynamic_bank.lr - acc*dynamic_bank.h)*steer)
        
        dxdt[:, 0] = vx * np.cos(psi + slip) #x
        dxdt[:, 1] = vx * np.sin(psi + slip) #y
        dxdt[:, 2] = omega # psi
        dxdt[:, 3] = acc #vx
        dxdt[:, 4] = (dynamic_bank.mu/(vx**2*(dynamic_bank.lr+dynamic_bank.lf))*(dynamic_bank.C_Sr*(g*dynamic_bank.lf + acc*dynamic_bank.h)*dynamic_bank.lr - dynamic_bank.C_Sf*(g*dynamic_bank.lr - acc*dynamic_bank.h)*dynamic_bank.lf)-1)*omega \
            -dynamic_bank.mu/(vx*(dynamic_bank.lr+dynamic_bank.lf))*(dynamic_bank.C_Sr*(g*dynamic_bank.lf + acc*dynamic_bank.h) + dynamic_bank.C_Sf*(g*dynamic_bank.lr-acc*dynamic_bank.h))*slip \
            +dynamic_bank.mu/(vx*(dynamic_bank.lr+dynamic_bank.lf))*(dynamic_bank.C_Sf*(g*dynamic_bank.lr-acc*dynamic_bank.h))*steer #slip
        dxdt[:, 5] = -dynamic_bank.mu*dynamic_bank.m/(vx*dynamic_bank.I*(dynamic_bank.lr+dynamic_bank.lf)) *(dynamic_bank.lf**2*dynamic_bank.C_Sf*(g*dynamic_bank.lr-acc*dynamic_bank.h) + dynamic_bank.lr**2*dynamic_bank.C_Sr*(g*dynamic_bank.lf + acc*dynamic_bank.h))*omega \
            +dynamic_bank.mu*dynamic_bank.m/(dynamic_bank.I*(dynamic_bank.lr+dynamic_bank.lf))*(dynamic_bank.lr*dynamic_bank.C_Sr*(g*dynamic_bank.lf + acc*dynamic_bank.h) - dynamic_bank.lf*dynamic_bank.C_Sf*(g*dynamic_bank.lr - acc*dynamic_bank.h))*slip \
            +dynamic_bank.mu*dynamic_bank.m/(dynamic_bank.I*(dynamic_bank.lr+dynamic_bank.lf))*dynamic_bank.lf*dynamic_bank.C_Sf*(g*dynamic_bank.lr - acc*dynamic_bank.h)*steer #omega
        dxdt[:, 6] = steer_rate # steer

    return dxdt


spec = [
    # Non-varying scalar parameters
    ('lf', float64),
    ('lr', float64),
    ('m', float64),
    ('I', float64),
    ('h', float64),
    ('v_switch', float64),
    ('a_max', float64),
    ('v_min', float64),
    ('v_max', float64),
    ('s_min', float64),
    ('s_max', float64),
    ('sv_min', float64),
    ('sv_max', float64),

    # Varying parameters (arrays)
    ('C_Sf', float64[:]),
    ('C_Sr', float64[:]),
    ('mu', float64[:]),
]
@jitclass(spec)
class DynamicSimBank():
    def __init__(self, 
                 lf, lr, 
                 mass, I, 
                 h, v_switch,
                 a_max, v_min, v_max,
                 s_min, s_max, 
                 sv_min, sv_max,
                 C_Sf,
                 C_Sr, mu
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

        # super().__init__(num_mod`     els, history_length, dt, cost_weights, state_size = 7, sim = True)

    def get_model_params_arr(self, index):
        return np.array([
            self.C_Sf[index],
            self.C_Sr[index],
            self.mu[index]
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

