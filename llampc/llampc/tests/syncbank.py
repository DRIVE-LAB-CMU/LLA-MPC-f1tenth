from llampc.rollout import odeintRK4_batch
from llampc.rollout import DynamicSimBank, LBHistory
from llampc.params import F110_sim, get_param_dict
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

        variation_dict = {
            'C_Sf': .15, 
            'C_Sr': .15,
            'mu': .15,
        }

        # variation_dict = {
        #     'C_Sf': 0, 
        #     'C_Sr': 0,
        #     'mu': 0,
        # }

        num_models = 200
        param_dict = get_param_dict(mean_dict, variation_dict, num_models, ground_truth = True)

        self.dynamics_bank = DynamicSimBank(
            self.params_car['lf'], self.params_car['lr'],
            self.params_car['m'], self.params_car['I'],
            self.params_car["h"], self.params_car['v_switch'],
            self.params_car['a_max'], self.params_car['v_min'],
            self.params_car['v_max'], self.params_car['s_min'],
            self.params_car['s_max'], self.params_car['sv_min'],
            self.params_car['sv_max'],
            param_dict
        )
        self.history = LBHistory(
            num_models, 20, env_timestep, cost_weights, self.dynamics_bank, odeintRK4_batch, 7
        )


        self.current_state = np.empty(7)
        self.update_state()

        self.steer_buffer = np.empty((0, ))
        self.steer_buffer_size = 2


    def get_selected_model_params(self, index):
        return self.dynamics_bank.get_model_params_arr(index)
    
    def get_selected_model_index(self):
        return self.history.get_best_model()
    
    def predict_states(self, controls):
        self.history.predict_states(self.current_state, controls)
    
    def update_state_and_error(self):
        self.update_state()
        self.history.update_lookback_error(self.get_state())

    def get_error_statistics(self):
        return self.history.running_cost, self.history.cost_history
    
    def get_predicted_states(self):
        return self.history.last_predicted_states

    def get_state(self):
        return self.current_state

    def update_state(self):

        self.current_state = np.array(
                [
                    self.car.state[0], # x
                    self.car.state[1], # y
                    np.unwrap([self.current_state[2], self.car.state[4]])[1], # yaw angle/psi
                    self.car.state[3], # vx
                    self.car.state[6],  # slip
                    self.car.state[5], # yaw rate / omega
                    self.car.state[2] # steer
                ]
            )
        


    def get_controls(self, speed, raw_steer):
        steer = 0.
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)

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