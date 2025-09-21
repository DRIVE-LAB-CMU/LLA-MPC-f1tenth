# integrate vehicle dynamics by 1 step
import numpy as np
# from numba import njit, float64, boolean, int64
# from numba.experimental import jitclass

import jax
import jax.numpy as jnp
from functools import partial


# @njit(parallel=True)
# @njit(fastmath=True)



@jax.jit
def diffequation(
    bank_params, known_params,x, u):
    """	write dynamics as first order ODE: dxdt = f(x(t))
        x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
        u is a 2x1 vector: [acc/pwm, steer]^T
    """
    g = 9.81
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, pitch, roll = known_params
    Ffy, Frx, Fry = _calc_forces(bank_params, known_params, x, u)

    return jnp.array([
        vx*jnp.cos(psi) - vy*jnp.sin(psi),
        vx*jnp.sin(psi) + vy*jnp.cos(psi),
        omega,
        1/mass * (Frx - Ffy*jnp.sin(steer)) + vy*omega - g * pitch,
        1/mass * (Fry + Ffy*jnp.cos(steer)) - vx*omega + g * roll,
        1/Iz * (Ffy* lf*jnp.cos(steer) - Fry * lr)
    ])
    

@jax.jit
def _calc_forces(bank_params, known_params, x, u):
    acc = u[0]
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params
    mass, Iz, lf, lr, pitch, roll =  known_params

    Frx = mass * (acc * Ce - Cm * vx ) - Cro - Cd * (vx * vx)
    def small_velocity_case(_):
        alphaf = 0.0
        alphar = 0.0
        return alphaf, alphar

    def normal_case(_):
        alphaf = steer - jnp.arctan2((lf * omega + vy), jnp.abs(vx))
        alphar = jnp.arctan2((lr * omega - vy), jnp.abs(vx))
        return alphaf, alphar

    alphaf, alphar = jax.lax.cond(
        jnp.abs(vx) < 1e-4,
        small_velocity_case,
        normal_case,
        operand=None  # no additional input needed
    )

    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    return Ffy, Frx, Fry # each of these should end up being num_models long
        
# @jitclass(spec)
class DBMPacejkaBank():
    def __init__(self, 
                 lf, lr, 
                 mass, Iz, 
                 Bf, Br,
                 Cf, Cr, 
                 Df, Dr, 
                 Cro, Cd,
                 Ce, Cm, 
                 num_models
                 ):
        # non-varying parameters
        self.lf = lf
        self.lr = lr
        self.mass = mass
        self.Iz = Iz

        # varying parameters
        self.Bf = Bf
        self.Br = Br
        self.Cf = Cf
        self.Cr = Cr
        self.Df = Df
        self.Dr = Dr
        self.Cro = Cro
        self.Cd = Cd
        self.Ce = Ce
        self.Cm = Cm

        # non-sampled state parameters
        self.roll = 0
        self.pitch = 0

        self.num_models = num_models

        self.param_bank = jnp.stack([
            self.Bf, self.Br, self.Cf, self.Cr, self.Df, self.Dr,
            self.Cro, self.Cd, self.Ce, self.Cm
        ], axis=1)


    def set_roll_pitch(self, roll, pitch): 
        # non-sampled state parameters (i.e. state parameters not differentiated)
        # are updated here, as they are not given to diffiequation via x_batch
        self.roll = roll
        self.pitch = pitch

    def get_known_params(self):
        return jnp.array([self.mass, self.Iz, self.lf, self.lr, self.pitch, self.roll])

    def get_bank_params(self):
        return self.param_bank

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

    

    
    # def integrate_batch(self, x_t_batch, u_t_batch, t_start, t_end):
    #     """Batched version of _integrate"""

    #     odesol = self.odeintRK4_batch(x_t_batch, np.array([t_start, t_end]), u_t_batch)
    #     return odesol[-1]
    
    
    # def get_model_params(self, index):
    #     return {
    #         'Bf': self.Bf[index], 
    #         'Br': self.Br[index],
    #         'Cf': self.Cf[index], 
    #         'Cr': self.Cr[index], 
    #         'Df': self.Df[index], 
    #         'Dr': self.Dr[index], 
    #         'Cro':self.Cro[index], 
    #         'Cd': self.Cd[index], 
    #         'Ce': self.Ce[index], 
    #         'Cm': self.Cm[index], 
    #     }
