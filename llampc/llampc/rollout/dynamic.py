# integrate vehicle dynamics by 1 step
import numpy as np
# from numba import njit, float64, boolean, int64
# from numba.experimental import jitclass

import jax
import jax.numpy as jnp
from functools import partial


# @njit(parallel=True)
# @njit(fastmath=True)
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
# gpu = jax.devices("cpu")[0]



@jax.jit
def diffequation_unoptimized(
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

    mass, Iz, lf, lr, roll, pitch = known_params
    Ffy, Frx, Fry = _calc_forces(bank_params, known_params, x, u)

    
    return jnp.array([
        vx*jnp.cos(psi) - vy*jnp.sin(psi),
        vx*jnp.sin(psi) + vy*jnp.cos(psi),
        omega,
        1/mass * (Frx - Ffy*jnp.sin(steer)) + vy*omega - g * jnp.sin(pitch),
        1/mass * (Fry + Ffy*jnp.cos(steer)) - vx*omega + g * jnp.sin(roll),
        1/Iz * (Ffy* lf*jnp.cos(steer) - Fry * lr)
    ])#, jnp.array([Frx, Ffy, Fry])

@jax.jit
def diffequation(bank_params, known_params, x, u):
    """Optimized for GPU - no conditionals"""
    g = 9.81

    acc = u[0]
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, roll, pitch = known_params
    
    # Inline force calculation to reduce function call overhead
    Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params
    
    # Forces
    Frx = mass * acc * ( Ce - Cm * vx) - Cro - Cd * (vx * vx)
    
    vx_safe = jnp.where(jnp.abs(vx) < 1e-4, 1e-4, vx)
    alphaf = steer - jnp.arctan2((lf * omega + vy), vx_safe)
    alphar = jnp.arctan2((lr * omega - vy), vx_safe)
    
    mask = jnp.abs(vx) >= 1e-4
    alphaf = jnp.where(mask, alphaf, 0.0)
    alphar = jnp.where(mask, alphar, 0.0)
    
    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    return jnp.array([
        vx*jnp.cos(psi) - vy*jnp.sin(psi),
        vx*jnp.sin(psi) + vy*jnp.cos(psi),
        omega,
        1/mass * (Frx - Ffy*jnp.sin(steer)) + vy*omega - g * jnp.sin(pitch),
        1/mass * (Fry + Ffy*jnp.cos(steer)) - vx*omega + g * jnp.sin(roll),
        1/Iz * (Ffy * lf*jnp.cos(steer) - Fry * lr),
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
    mass, Iz, lf, lr, roll, pitch =  known_params

    Frx = mass*acc* ( Ce - Cm * vx ) - Cro - Cd * (vx * vx)
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
                 roll, pitch, 
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
        self.num_models = num_models

        self.pitch = pitch
        self.roll = roll

        self.known_params = jax.device_put(
            jnp.array([self.mass, self.Iz, self.lf, self.lr, self.roll, self.pitch]),
            device=gpu
        )

        # non-sampled state parameters

        self.num_models = num_models

        self.param_bank = jax.device_put(
            jnp.stack([
            self.Bf, self.Br, self.Cf, self.Cr, self.Df, self.Dr,
            self.Cro, self.Cd, self.Ce, self.Cm
            ], axis=1), 
        device=cpu)

    def update_known_params(self):
        pass

    def get_known_params(self):
        return self.known_params

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
            self.Cm[index],
            self.roll,
            self.pitch
        ])


    
