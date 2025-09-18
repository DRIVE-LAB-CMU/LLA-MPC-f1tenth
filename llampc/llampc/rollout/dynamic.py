# integrate vehicle dynamics by 1 step
import numpy as np
    
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

        # boolean for approximating dynamics
        self.approx = False

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

    def diffequation(self, t, x_batch, u_batch):
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

        dxdt = np.column_stack((
            vx*np.cos(psi) - vy*np.sin(psi),
            vx*np.sin(psi) + vy*np.cos(psi),
            omega,
            1/self.mass * (Frx - Ffy*np.sin(steer)) + vy*omega - g * self.pitch,
            1/self.mass * (Fry + Ffy*np.cos(steer)) - vx*omega + g * self.roll,
            1/self.Iz * (Ffy*self.lf*np.cos(steer) - Fry*self.lr),
        ))
        return dxdt

    def _calc_forces(self, x_batch, u_batch, return_slip=False):
        acc = u_batch[:, 0]
        steer = u_batch[:, 1]
        psi = x_batch[:, 2]
        vx = x_batch[:, 3]
        vy = x_batch[:, 4]
        omega = x_batch[:, 5]

        if self.approx:
            # rolling friction and drag are ignored
            
            Frx = self.mass*acc

            # See Vehicle Dynamics and Control (Rajamani)
            alphaf = steer - (self.lf*omega + vy)/vx
            alphar = (self.lr*omega - vy)/vx
            Ffy = 2 * self.Cf * alphaf
            Fry = 2 * self.Cr * alphar

        else:
            Frx = self.mass * (acc * self.Ce - self.Cm * vx ) - self.Cro - self.Cd * (vx ** 2)

            alphaf = np.where(np.abs(vx) < 1e-4, 0,  steer - np.arctan2((self.lf*omega + vy), abs(vx)))
            alphar = np.where(np.abs(vx) < 1e-4, 0, np.arctan2((self.lr*omega - vy), abs(vx)))
            Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
            Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))
        if return_slip:
            return Ffy, Frx, Fry, alphaf, alphar
        else:
            return Ffy, Frx, Fry # each of these should end up being num_models long
        

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
