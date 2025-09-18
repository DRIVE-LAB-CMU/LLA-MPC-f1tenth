# integrate vehicle dynamics by 1 step
import numpy as np

import jax
from jax import jit
import numpy as jnp



class DBMPacejkaBank:
    def __init__(self, 
                 lf, lr, 
                 mass, Iz, param_dict ):
        # non-varying parameters
        self.lf = lf
        self.lr = lr
        self.mass = mass
        self.Iz = Iz

        # varying parameters
        for key, value in param_dict.items():
            setattr(self, key, value)

        # non-sampled state parameters

        # boolean for approximating dynamics
        # self.approx = False
        self.roll = 0
        self.pitch = 0

        # self.diffequation = jit(self.diffequation, static_argnums=(0,))
        # self._calc_forces = jit(self._calc_forces, static_argnums=(0,))

    def get_state_add(self):
        return jnp.array([self.roll, self.pitch])

    def diffequation(self, t, x_batch, u_batch, state_add):
        """	write dynamics as first order ODE: dxdt = f(x(t))
            x is a 8x1 vector: [x, y, psi, vx, vy, omega]^T
            u is a 2x1 vector: [acc/pwm, steer]^T
        """
        g = 9.81
        steer = u_batch[:, 1]
        psi = x_batch[:, 2]
        vx = x_batch[:, 3]
        vy = x_batch[:, 4]
        omega = x_batch[:, 5]

        Ffy, Frx, Fry = self._calc_forces(x_batch, u_batch)


        dxdt = jnp.stack([
            vx * jnp.cos(psi) - vy * jnp.sin(psi),                              
            vx * jnp.sin(psi) + vy * jnp.cos(psi),                              
            omega,                                                              
            1 / self.mass * (Frx - Ffy * jnp.sin(steer)) + vy * omega - g * state_add[1], # pitch, 
            1 / self.mass * (Fry + Ffy * jnp.cos(steer)) - vx * omega + g * state_add[0], # roll, 
            1 / self.Iz * (Ffy * self.lf * jnp.cos(steer) - Fry * self.lr)
        ], axis=1)
        
        return dxdt

    def _calc_forces(self, x_batch, u_batch):
        acc = u_batch[:, 0]
        steer = u_batch[:, 1]
        psi = x_batch[:, 2]
        vx = x_batch[:, 3]
        vy = x_batch[:, 4]
        omega = x_batch[:, 5]

        Frx = self.mass * (acc * self.Ce - self.Cm * vx ) - self.Cro - self.Cd * (vx ** 2)

        alphaf = jnp.where(jnp.abs(vx) < 1e-4, 0.0,  steer - jnp.arctan2((self.lf*omega + vy), jnp.abs(vx)))
        alphar = jnp.where(jnp.abs(vx) < 1e-4, 0.0, jnp.arctan2((self.lr*omega - vy), jnp.abs(vx)))
        Ffy = self.Df * jnp.sin(self.Cf * jnp.arctan(self.Bf * alphaf))
        Fry = self.Dr * jnp.sin(self.Cr * jnp.arctan(self.Br * alphar))

        return Ffy, Frx, Fry

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
    
    def get_model_params(self, index):
        return {
            'Bf': self.Bf[index], 
            'Br': self.Br[index],
            'Cf': self.Cf[index], 
            'Cr': self.Cr[index], 
            'Df': self.Df[index], 
            'Dr': self.Dr[index], 
            'Cro':self.Cro[index], 
            'Cd': self.Cd[index], 
            'Ce': self.Ce[index], 
            'Cm': self.Cm[index], 
        }
    

     # each of these should end up being num_models long
        

    # def integrate_batch(self, x_t_batch, u_t_batch, t_start, t_end):
    #     """Batched version of _integrate"""

    #     odesol = self.odeintRK4_batch(x_t_batch, np.array([t_start, t_end]), u_t_batch)
    #     return odesol[-1]
    
    
    
