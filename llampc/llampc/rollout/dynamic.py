# integrate vehicle dynamics by 1 step
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

cpu = jax.devices("cpu")[0]
#gpu = jax.devices("gpu")[0]
gpu = jax.devices("cpu")[0]

# @jax.jit
# def diffequation(bank_params, known_params, x, u):
#     """Optimized for GPU - Blended Kinematic/Dynamic Model matching CasADi"""
#     g = 9.81

#     acc = u[0]
#     steer = u[1]
#     psi = x[2]
#     vx = x[3]
#     vy = x[4]
#     omega = x[5]

#     mass, Iz, lf, lr, roll, pitch = known_params
    
#     # Inline force calculation to reduce function call overhead
#     Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params
    
#     # 1. Hard clamp for Jacobian conditioning (Match CasADi: ca.fmax(x[3], 0.5))
#     vx_clamped = jnp.maximum(vx, 0.5)

#     # 2. Kinematic model components
#     beta = (lr / (lf + lr)) * steer
#     kin_dx4 = 0.0
#     kin_dx5 = (vx * jnp.cos(beta) * steer) / (lf + lr)

#     # 3. Dynamic Pacejka model components
#     # Using jnp.arctan instead of arctan2 to perfectly match CasADi: ca.atan(...)
#     alphaf = steer - jnp.arctan((omega * lf + vy) / vx_clamped)
#     alphar = jnp.arctan((omega * lr - vy) / vx_clamped)

#     Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
#     Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

#     dyn_dx4 = (Fry + Ffy * jnp.cos(steer)) / mass - vx * omega + g * jnp.sin(roll)
#     dyn_dx5 = (lf * Ffy * jnp.cos(steer) - lr * Fry) / Iz

#     # 4. Blending based on actual velocity
#     weight_dyn = 0.5 * (1.0 + jnp.tanh(10.0 * (vx - 0.5)))
#     weight_kin = 1.0 - weight_dyn

#     # 5. Longitudinal Force
#     Frx = mass * acc * (Ce - Cm * vx) - Cro - Cd * (vx * vx)

#     # 6. Differential equations
#     dx0 = (vx * jnp.cos(psi)) - (vy * jnp.sin(psi))
#     dx1 = (vx * jnp.sin(psi)) + (vy * jnp.cos(psi))
#     dx2 = omega
    
#     # Note: Using weight_dyn to scale the lateral force contribution to vxdot, matching CasADi
#     dx3 = (Frx - weight_dyn * Ffy * jnp.sin(steer)) / mass + vy * omega - g * jnp.sin(pitch)
    
#     dx4 = weight_kin * kin_dx4 + weight_dyn * dyn_dx4
#     dx5 = weight_kin * kin_dx5 + weight_dyn * dyn_dx5

#     return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


# def diffequation_nojit(bank_params, known_params, x, u):
#     """Unoptimized version - Blended Kinematic/Dynamic Model matching CasADi"""
#     g = 9.81

#     acc = u[0]
#     steer = u[1]
#     psi = x[2]
#     vx = x[3]
#     vy = x[4]
#     omega = x[5]

#     mass, Iz, lf, lr, roll, pitch = known_params
#     Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params
    
#     vx_clamped = jnp.maximum(vx, 0.5)

#     beta = (lr / (lf + lr)) * steer
#     kin_dx4 = 0.0
#     kin_dx5 = (vx * jnp.cos(beta) * steer) / (lf + lr)

#     alphaf = steer - jnp.arctan((omega * lf + vy) / vx_clamped)
#     alphar = jnp.arctan((omega * lr - vy) / vx_clamped)

#     Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
#     Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

#     dyn_dx4 = (Fry + Ffy * jnp.cos(steer)) / mass - vx * omega + g * jnp.sin(roll)
#     dyn_dx5 = (lf * Ffy * jnp.cos(steer) - lr * Fry) / Iz

#     weight_dyn = 0.5 * (1.0 + jnp.tanh(10.0 * (vx - 0.5)))
#     weight_kin = 1.0 - weight_dyn

#     Frx = mass * acc * (Ce - Cm * vx) - Cro - Cd * (vx * vx)

#     dx0 = (vx * jnp.cos(psi)) - (vy * jnp.sin(psi))
#     dx1 = (vx * jnp.sin(psi)) + (vy * jnp.cos(psi))
#     dx2 = omega
#     dx3 = (Frx - weight_dyn * Ffy * jnp.sin(steer)) / mass + vy * omega - g * jnp.sin(pitch)
#     dx4 = weight_kin * kin_dx4 + weight_dyn * dyn_dx4
#     dx5 = weight_kin * kin_dx5 + weight_dyn * dyn_dx5

#     return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


@jax.jit
def diffequation(bank_params, known_params, x, u):
    """Pure dynamic Pacejka model - no blending needed (no autodiff)"""
    g = 9.81

    pwm = u[0]
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, roll, pitch = known_params
    Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params

    vx_clamped = jnp.maximum(vx, 0.01)

    alphaf = steer - jnp.arctan((omega * lf + vy) / vx_clamped)
    alphar = jnp.arctan((omega * lr - vy) / vx_clamped)

    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    Frx = mass * pwm * Ce - Cro - Cd * (vx * vx)

    dx0 = (vx * jnp.cos(psi)) - (vy * jnp.sin(psi))
    dx1 = (vx * jnp.sin(psi)) + (vy * jnp.cos(psi))
    dx2 = omega
    dx3 = (Frx - Ffy * jnp.sin(steer)) / mass + vy * omega - g * jnp.sin(pitch)
    dx4 = (Fry + Ffy * jnp.cos(steer)) / mass - vx * omega + g * jnp.sin(roll)
    dx5 = (lf * Ffy * jnp.cos(steer) - lr * Fry) / Iz

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


def diffequation_nojit(bank_params, known_params, x, u):
    """Pure dynamic Pacejka model - no blending needed (no autodiff)"""
    g = 9.81

    pwm = u[0]
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, roll, pitch = known_params
    Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params

    vx_clamped = jnp.maximum(vx, 0.01)

    alphaf = steer - jnp.arctan((omega * lf + vy) / vx_clamped)
    alphar = jnp.arctan((omega * lr - vy) / vx_clamped)

    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    Frx = mass * pwm * Ce - Cro - Cd * (vx * vx)

    dx0 = (vx * jnp.cos(psi)) - (vy * jnp.sin(psi))
    dx1 = (vx * jnp.sin(psi)) + (vy * jnp.cos(psi))
    dx2 = omega
    dx3 = (Frx - Ffy * jnp.sin(steer)) / mass + vy * omega - g * jnp.sin(pitch)
    dx4 = (Fry + Ffy * jnp.cos(steer)) / mass - vx * omega + g * jnp.sin(roll)
    dx5 = (lf * Ffy * jnp.cos(steer) - lr * Fry) / Iz

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


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
