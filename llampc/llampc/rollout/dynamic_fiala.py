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
    current = u[0]
    steer = u[1]

    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, rw, omega_w, dFz= known_params
    Cf, Cr, muf, mur, Cro = bank_params
    L = lf + lr

    Ffz = mass*g*lr/L - dFz
    Frz = mass*g*lf/L + dFz
    Ffmax = muf*Ffz
    Frmax = mur*Frz

    vx_clamped = jnp.maximum(vx, 0.01)
    vx_front = vx*jnp.cos(steer) + (vy + lf*omega)*jnp.sin(steer)
    vx_front_clamped = jnp.maximum(vx_front, .01)

    alphaf = steer - jnp.arctan((omega*lf + vy) / vx_clamped)
    alphar = jnp.arctan((omega*lr - vy) / vx_clamped)
    kappa_r = (rw*omega_w - vx) / vx_clamped
    kappa_f = (rw*omega_w - vx_front) / vx_front_clamped

    sigma_f = jnp.sqrt(jnp.tan(alphaf)**2 + kappa_f**2 + 1e-9)   # front free-rolling
    sigma_r = jnp.sqrt(jnp.tan(alphar)**2 + kappa_r**2 + 1e-9)

    sigmaf_max = 3*Ffmax/Cf
    sigmar_max = 3*Frmax/Cr

    def brush(C, s, Fmax):
        Cs = C*s
        return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)

    Ff_total = jnp.where(sigma_f < sigmaf_max, brush(Cf, sigma_f, Ffmax), Ffmax)
    Fr_total = jnp.where(sigma_r < sigmar_max, brush(Cr, sigma_r, Frmax), Frmax)

    # --- front: lateral only ---
    Ffy = Ff_total * jnp.tan(alphaf) / sigma_f
    Ffx = Ff_total * kappa_f           / sigma_f

    # --- rear: lateral and longitudinal share Fr_total via sigma_r ---
    v_eps = 0.5
    F_roll = Cro * Frz * jnp.tanh(vx / v_eps)
    Frx = Fr_total*kappa_r/sigma_r - F_roll
    Fry = Fr_total*jnp.tan(alphar)/sigma_r

    # --- chassis-frame dynamics (6 states only) ---
    dx0 = vx*jnp.cos(psi) - vy*jnp.sin(psi)
    dx1 = vx*jnp.sin(psi) + vy*jnp.cos(psi)
    dx2 = omega
    dx3 = (Frx + Ffx*jnp.cos(steer) - Ffy*jnp.sin(steer)) / mass + vy*omega
    dx4 = (Fry + Ffy*jnp.cos(steer) + Ffx*jnp.sin(steer)) / mass - vx*omega
    dx5 = (lf*(Ffy*jnp.cos(steer) + Ffx*jnp.sin(steer)) - lr*Fry) / Iz

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


def diffequation_nojit(bank_params, known_params, x, u):
    """Pure dynamic Pacejka model - no blending needed (no autodiff)"""
    g = 9.81
    current = u[0]
    steer = u[1]

    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, rw, omega_w, dFz= known_params
    Cf, Cr, muf, mur, Cro = bank_params
    L = lf + lr

    Ffz = mass*g*lr/L - dFz
    Frz = mass*g*lf/L + dFz
    Ffmax = muf*Ffz
    Frmax = mur*Frz

    vx_clamped = jnp.maximum(vx, 0.01)
    vx_front = vx*jnp.cos(steer) + (vy + lf*omega)*jnp.sin(steer)
    vx_front_clamped = jnp.maximum(vx_front, .01)

    alphaf = steer - jnp.arctan((omega*lf + vy) / vx_clamped)
    alphar = jnp.arctan((omega*lr - vy) / vx_clamped)
    kappa_r = (rw*omega_w - vx) / vx_clamped
    kappa_f = (rw*omega_w - vx_front) / vx_front_clamped

    sigma_f = jnp.sqrt(jnp.tan(alphaf)**2 + kappa_f**2 + 1e-9)   # front free-rolling
    sigma_r = jnp.sqrt(jnp.tan(alphar)**2 + kappa_r**2 + 1e-9)

    sigmaf_max = 3*Ffmax/Cf
    sigmar_max = 3*Frmax/Cr

    def brush(C, s, Fmax):
        Cs = C*s
        return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)

    Ff_total = jnp.where(sigma_f < sigmaf_max, brush(Cf, sigma_f, Ffmax), Ffmax)
    Fr_total = jnp.where(sigma_r < sigmar_max, brush(Cr, sigma_r, Frmax), Frmax)

    # --- front: lateral only ---
    Ffy = Ff_total * jnp.tan(alphaf) / sigma_f
    Ffx = Ff_total * kappa_f           / sigma_f

    # --- rear: lateral and longitudinal share Fr_total via sigma_r ---
    v_eps = 0.5
    F_roll = Cro * Frz * jnp.tanh(vx / v_eps)
    Frx = Fr_total*kappa_r/sigma_r - F_roll
    Fry = Fr_total*jnp.tan(alphar)/sigma_r

    # --- chassis-frame dynamics (6 states only) ---
    dx0 = vx*jnp.cos(psi) - vy*jnp.sin(psi)
    dx1 = vx*jnp.sin(psi) + vy*jnp.cos(psi)
    dx2 = omega
    dx3 = (Frx + Ffx*jnp.cos(steer) - Ffy*jnp.sin(steer)) / mass + vy*omega
    dx4 = (Fry + Ffy*jnp.cos(steer) + Ffx*jnp.sin(steer)) / mass - vx*omega
    dx5 = (lf*(Ffy*jnp.cos(steer) + Ffx*jnp.sin(steer)) - lr*Fry) / Iz

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5])


class DBMFialaBank():
    def __init__(self, 
                 lf, lr, 
                 mass, Iz, 
                 rw,
                 Cf, Cr, 
                 muf, mur, 
                 Cro,
                 num_models,
                 omega_w = 0,
                 dFz = 0
                 ):
        # non-varying parameters
        self.lf = lf
        self.lr = lr
        self.mass = mass
        self.Iz = Iz
        self.rw = rw

        # varying parameters
        self.Cf = Cf
        self.Cr = Cr
        self.muf = muf
        self.mur = mur
        self.Cro = Cro
        self.num_models = num_models

        self.known_params = jax.device_put(
            jnp.array([self.mass, self.Iz, self.lf, self.lr, self.rw, omega_w, dFz]),
            device=gpu
        )

        self.param_bank = jax.device_put(
            jnp.stack([
            self.Cf, self.Cr, self.muf, self.mur, self.Cro
            ], axis=1), 
        device=cpu)

    def update_known_params(self, omega_w=None, dFz = None):
        if omega_w:
            self.known_params = self.known_params.at[5].set(
                omega_w
            )
        if dFz:
            self.known_params = self.known_params.at[6].set(
                dFz
            )

    def get_known_params(self):
        return self.known_params

    def get_bank_params(self):
        return self.param_bank

    def get_model_params_arr(self, index):
        return np.array([
            self.Cf[index],
            self.Cr[index],
            self.muf[index],
            self.mur[index],
            self.Cro[index]
        ])
