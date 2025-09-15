from llampc.rollout import DynamicSimBank
from llampc.params import F110_sim
from . import syncbank

import numpy as np

#Wra

class SynchronousBank():
    def __init__(self, env_timestep, sim_car):
        self.params_car = F110_sim()
        cost_weights = np.array([1.0, 1.0, 0, 0, 0, 0, 0])
        self.car = sim_car

        mean_dict = {
            'C_Sf': self.params_car['C_Sf'], 
            'C_Sr': self.params_car['C_Sr'],
            'mu': self.params_car['mu'],
        }

        # variation_dict = {
        #     'C_Sf': .15, 
        #     'C_Sr': .15,
        #     'mu': .15,
        # }

        variation_dict = {
            'C_Sf': 0, 
            'C_Sr': 0,
            'mu': 0,
        }

        self.bank = DynamicSimBank(
            self.params_car['lf'], self.params_car['lr'],
            self.params_car['m'], self.params_car['I'],
            self.params_car["h"], mean_dict,
            variation_dict, 1, 2, env_timestep, cost_weights
        )

        self.current_state = np.empty(7)
        self.update_state()

    def predict_states(self, controls):
        self.bank.predict_states(self.current_state, controls)
    
    def update_state_and_error(self):
        self.update_state()
        self.bank.update_lookback_error(self.get_state())

    def get_error_statistics(self):
        return self.bank.running_cost, self.bank.cost_history
    
    def get_predicted_states(self):
        return self.bank.last_predicted_states

    def get_state(self):
        return self.current_state

    def update_state(self):
        self.current_state = np.array(
                [
                    self.car.state[0], # x
                    self.car.state[1], # y
                    self.car.state[4], # yaw angle/psi
                    self.car.state[3], # vx
                    self.car.state[6],  # slip
                    self.car.state[5], # yaw rate / omega
                    self.car.state[2] # steer
                ]
            )

    def get_controls(self, speed, steer):
        return np.array(
            self.pid(
                speed, 
                steer, 
                self.current_state[3],
                self.current_state[6],
            )
        )

    def pid(self, speed, steer, current_speed, current_steer):
        """
        Basic controller for speed/steer -> accl./steer vel.

            Args:
                speed (float): desired input speed
                steer (float): desired input steering angle

            Returns:
                accl (float): desired input acceleration
                sv (float): desired input steering velocity
        """
        max_sv = self.params_car["sv_max"]
        max_a = self.params_car["a_max"]
        max_v = self.params_car["v_max"]
        min_v = self.params_car["v_min"]

        # steering
        steer_diff = steer - current_steer
        if np.fabs(steer_diff) > 1e-4:
            sv = (steer_diff / np.fabs(steer_diff)) * max_sv
        else:
            sv = 0.0

        # accl
        vel_diff = speed - current_speed
        # currently forward
        if current_speed > 0.:
            if (vel_diff > 0):
                # accelerate
                kp = 10.0 * max_a / max_v
                accl = kp * vel_diff
            else:
                # braking
                kp = 10.0 * max_a / (-min_v)
                accl = kp * vel_diff
        # currently backwards
        else:
            if (vel_diff > 0):
                # braking
                kp = 2.0 * max_a / max_v
                accl = kp * vel_diff
            else:
                # accelerating
                kp = 2.0 * max_a / (-min_v)
                accl = kp * vel_diff

        return accl, sv