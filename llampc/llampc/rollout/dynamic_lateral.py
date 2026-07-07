# integrate vehicle dynamics by 1 step
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

cpu = jax.devices("cpu")[0]
#gpu = jax.devices("gpu")[0]
gpu = jax.devices("cpu")[0]


@jax.jit
def diffequation(bank_params, known_params, x, u):
    """Pure dynamic Pacejka model - no blending needed (no autodiff)"""
    g = 9.81

    pwm = u[0]
    steer = u[1]
    psi = x[2]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, vx = known_params
    Bf, Br, Cf, Cr, Df, Dr = bank_params

    vx_clamped = jnp.maximum(vx, 0.01)

    alphaf = steer - jnp.arctan((omega * lf + vy) / vx_clamped)
    alphar = jnp.arctan((omega * lr - vy) / vx_clamped)

    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    dx0 = (vx * jnp.cos(psi)) - (vy * jnp.sin(psi))
    dx1 = (vx * jnp.sin(psi)) + (vy * jnp.cos(psi))
    dx2 = omega
    dx3 = 0
    dx4 = (Fry + Ffy * jnp.cos(steer)) / mass - vx * omega
    dx5 = (lf * Ffy * jnp.cos(steer) - lr * Fry) / Iz

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


def diffequation_nojit(bank_params, known_params, x, u):
    """Pure dynamic Pacejka model - no blending needed (no autodiff)"""
    g = 9.81

    pwm = u[0]
    steer = u[1]
    psi = x[2]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, vx = known_params
    Bf, Br, Cf, Cr, Df, Dr = bank_params

    vx_clamped = jnp.maximum(vx, 0.01)

    alphaf = steer - jnp.arctan((omega * lf + vy) / vx_clamped)
    alphar = jnp.arctan((omega * lr - vy) / vx_clamped)

    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    dx0 = (vx * jnp.cos(psi)) - (vy * jnp.sin(psi))
    dx1 = (vx * jnp.sin(psi)) + (vy * jnp.cos(psi))
    dx2 = omega
    dx3 = 0
    dx4 = (Fry + Ffy * jnp.cos(steer)) / mass - vx * omega + g
    dx5 = (lf * Ffy * jnp.cos(steer) - lr * Fry) / Iz

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


class DBMPacejkaLateralBank():
    def __init__(self, 
                 lf, lr, 
                 mass, Iz, 
                 Bf, Br,
                 Cf, Cr, 
                 Df, Dr, 
                 num_models,
                 vx = 0
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
        self.num_models = num_models


        self.known_params = jax.device_put(
            jnp.array([self.mass, self.Iz, self.lf, self.lr, vx]),
            device=gpu
        )

        self.param_bank = jax.device_put(
            jnp.stack([
            self.Bf, self.Br, self.Cf, self.Cr, self.Df, self.Dr
            ], axis=1), 
        device=cpu)

    def update_known_params(self, vx=None):
        if vx:
            self.known_params = self.known_params.at[4].set(vx)
        

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
        ])
