# integrate vehicle dynamics by 1 step
import numpy as np
from numba import njit, float64, boolean, int64, prange
from numba.experimental import jitclass


@njit(fastmath=True, parallel=True)
def diffequation(dynamic_bank, t, x_batch, u_batch):
    """	write dynamics as first order ODE: dxdt = f(x(t))
        x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
        u is a 2x1 vector: [acc/pwm, steer]^T
    """
    g = 9.81
   
    psi = x_batch[:, 2]
    vx = x_batch[:, 3]
    vy = x_batch[:, 4]
    omega = x_batch[:, 5]

    acc = u_batch[:, 0]
    steer = u_batch[:, 1]
    dxdt = np.empty((dynamic_bank.num_models, 6))

    for i in prange(dynamic_bank.num_models):
        Frx = dynamic_bank.mass * (acc[i] * dynamic_bank.Ce[i] - dynamic_bank.Cm[i] * vx[i] ) - dynamic_bank.Cro[i] - dynamic_bank.Cd[i] * (vx[i] * vx[i])
        if(np.abs(vx[i]) < 1e-4):
            alphaf =  0  
        else:
            alphaf = steer[i] - np.arctan2((dynamic_bank.lf*omega[i] + vy[i]), np.abs(vx[i]))

        if(np.abs(vx[i]) < 1e-4):
            alphar = 0 
        else:
            alphar = np.arctan2((dynamic_bank.lr*omega[i] - vy[i]), np.abs(vx[i]))

        Ffy = dynamic_bank.Df[i] * np.sin(dynamic_bank.Cf[i] * np.arctan(dynamic_bank.Bf[i] * alphaf))
        Fry = dynamic_bank.Dr[i] * np.sin(dynamic_bank.Cr[i] * np.arctan(dynamic_bank.Br[i] * alphar))

        dxdt[i, 0] = vx[i]*np.cos(psi[i]) - vy[i]*np.sin(psi[i])
        dxdt[i, 1] = vx[i]*np.sin(psi[i]) + vy[i]*np.cos(psi[i])
        dxdt[i, 2] = omega[i]
        dxdt[i, 3] = 1/dynamic_bank.mass * (Frx - Ffy*np.sin(steer[i])) + vy[i]*omega[i] - g * dynamic_bank.pitch
        dxdt[i, 4] = 1/dynamic_bank.mass * (Fry + Ffy*np.cos(steer[i])) - vx[i]*omega[i] + g * dynamic_bank.roll
        dxdt[i, 5] = 1/dynamic_bank.Iz * (Ffy*dynamic_bank.lf*np.cos(steer[i]) - Fry*dynamic_bank.lr)
    
    return dxdt


        
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

    ('num_models', int64),
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
