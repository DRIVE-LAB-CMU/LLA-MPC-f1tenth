# integrate vehicle dynamics by 1 step
import numpy as np
from numba import njit, float64, boolean, int64
from numba.experimental import jitclass


@njit
def diffequation(dynamic_bank, t, x_batch, u_batch):
    """	write dynamics as first order ODE: dxdt = f(x(t))
        x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
        u is a 2x1 vector: [acc/pwm, steer]^T
    """
    g = 9.81
    steer = u_batch[:, 1]
    psi = x_batch[:, 2]
    vx = x_batch[:, 3]
    vy = x_batch[:, 4]
    omega = x_batch[:, 5]

    Ffy, Frx, Fry = _calc_forces(dynamic_bank, x_batch, u_batch)

    dxdt = np.empty((vx.shape[0], 6))
    dxdt[:, 0] = vx*np.cos(psi) - vy*np.sin(psi)
    dxdt[:, 1] = vx*np.sin(psi) + vy*np.cos(psi)
    dxdt[:, 2] = omega
    dxdt[:, 3] = 1/dynamic_bank.mass * (Frx - Ffy*np.sin(steer)) + vy*omega - g * dynamic_bank.pitch
    dxdt[:, 4] = 1/dynamic_bank.mass * (Fry + Ffy*np.cos(steer)) - vx*omega + g * dynamic_bank.roll
    dxdt[:, 5] = 1/dynamic_bank.Iz * (Ffy*dynamic_bank.lf*np.cos(steer) - Fry*dynamic_bank.lr)
    
    return dxdt

@njit
def _calc_forces(dynamic_bank, x_batch, u_batch):
    acc = u_batch[:, 0]
    steer = u_batch[:, 1]
    psi = x_batch[:, 2]
    vx = x_batch[:, 3]
    vy = x_batch[:, 4]
    omega = x_batch[:, 5]


    Frx = dynamic_bank.mass * (acc * dynamic_bank.Ce - dynamic_bank.Cm * vx ) - dynamic_bank.Cro - dynamic_bank.Cd * (vx ** 2)

    alphaf = np.where(np.abs(vx) < 1e-4, 0,  steer - np.arctan2((dynamic_bank.lf*omega + vy), abs(vx)))
    alphar = np.where(np.abs(vx) < 1e-4, 0, np.arctan2((dynamic_bank.lr*omega - vy), abs(vx)))
    Ffy = dynamic_bank.Df * np.sin(dynamic_bank.Cf * np.arctan(dynamic_bank.Bf * alphaf))
    Fry = dynamic_bank.Dr * np.sin(dynamic_bank.Cr * np.arctan(dynamic_bank.Br * alphar))

    return Ffy, Frx, Fry # each of these should end up being num_models long
        
spec = [
    # Non-varying parameters
    ('lf', float64),
    ('lr', float64),
    ('mass', float64),
    ('Iz', float64),

    # Varying parameters (as arrays)
    ('Bf', float64[:]),
    ('Br', float64[:]),
    ('Cf', float64[:]),
    ('Cr', float64[:]),
    ('Df', float64[:]),
    ('Dr', float64[:]),
    ('Cro', float64[:]),
    ('Cd', float64[:]),
    ('Ce', float64[:]),
    ('Cm', float64[:]),

    # Non-sampled state parameters
    ('roll', float64),
    ('pitch', float64),
]

@jitclass(spec)
class DBMPacejkaBank():
    def __init__(self, 
                 lf, lr, 
                 mass, Iz, 
                 Bf, Br,
                 Cf, Cr, 
                 Df, Dr, 
                 Cro, Cd,
                 Ce, Cm):
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


    def set_roll_pitch(self, roll, pitch): 
        # non-sampled state parameters (i.e. state parameters not differentiated)
        # are updated here, as they are not given to diffiequation via x_batch
        self.roll = roll
        self.pitch = pitch

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
