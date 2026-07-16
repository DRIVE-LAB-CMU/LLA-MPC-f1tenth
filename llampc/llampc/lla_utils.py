import os
import time
import numpy as np

class LLASolver():
    def __init__(self, mpc_solver, lla_p):
        self.solver = mpc_solver

        self.min_pwm = 0.1
        self.max_pwm = 0.25

        self.max_v = 3.0
        self.min_v = 0.2

        self.kd = 0.1
        self.ki = 0.1
        self.kp = 0.01

        self.lla_p = lla_p

        self.first_control = True

    def print_mpc_failed(self):
        self.solver.print_statistics()
        residuals = self.solver.get_residuals()
        print(f"Max Residuals (stat, eq, ineq, comp): {residuals}")
        
        N = self.solver.acados_ocp.dims.N
        print("\n--- NODE TRAJECTORY DUMP ---")
        for i in range(N + 1):
            x_i = self.solver.get(i, "x")
            print(f"Node {i} | State: {np.round(x_i, 3)}")

    def construct_params(self, N, selected_model_params, ref_segment):
        full_params = np.zeros((N+1, self.lla_p+6), np.float64)
        full_params[:, :self.lla_p] = selected_model_params
        # self.get_logger().info(f"{full_params}")
        full_params[:, self.lla_p:self.lla_p+6] = ref_segment[:6, :N+1].T #reference x, y, theta
        return full_params
    
    def mpc_solve(self, state, params):
        self.solver.set(0, "lbx", state)
        self.solver.set(0, "ubx", state)
        

        for i in range(self.N+1):
            self.solver.set(i, "p", params)

            if(i == 0 or self.first_control):
                self.solver.set(i, "x", state)
        
        self.first_control = False
        status = self.solver.solve()

        return self.solver, status

    def prepare_mpc_solve(self):
        if self.first_control:
            return
        
        for i in range(0, self.N - 1):
            x_next = self.solver.get(i + 1, "x")
            u_next = self.solver.get(i + 1, "u")
            self.solver.set(i, "x", x_next)
            self.solver.set(i, "u", u_next)

        # For the very last node, just duplicate the second-to-last node 
        # (the Levenberg-Marquardt damping will fix the slight error here)
        self.solver.set(self.N - 1, "u", self.solver.get(self.N - 2, "u"))
        self.solver.set(self.N, "x", self.solver.get(self.N - 1, "x"))
        
        for i in range(self.N):
            # Overwrite the saved dual variables for the dynamics
            num_pi = len(self.solver.get(i, "pi"))
            self.solver.set(i, "pi", np.zeros(num_pi))
            
            # Overwrite the saved dual variables for the bounds/constraints
            num_lam = len(self.solver.get(i, "lam"))
            self.solver.set(i, "lam", np.zeros(num_lam))
            
        # Terminal node constraints
        num_lam_e = len(self.solver.get(self.N, "lam"))
        self.solver.set(self.N, "lam", np.zeros(num_lam_e))


    def pid_long_control(self, ref_v, vx):
        if(self.current_mode):
            self.last_v_err = None
            self.v_int_err = 0.0
            self.current_mode = False


        diff = self.max_pwm - self.min_pwm
        err = ref_v - vx

        d_term = self.kd * (err - self.last_v_err) if self.last_v_err is not None else 0.0

        # compute unsaturated pid using CURRENT integral (don't add err yet)
        pid_unsat = self.kp * err + self.ki * self.v_int_err - d_term

        # check whether we're already at the limits
        saturated_high = pid_unsat > diff
        saturated_low = pid_unsat < 0

        # only integrate if not saturated in a way that this err would worsen
        if not ((saturated_high and err > 0) or (saturated_low and err < 0)):
            self.v_int_err += err

        # TODO: Potentially need to use leaky integrator here

        pid = self.kp * err + self.ki * self.v_int_err - d_term
        self.last_v_err = err

        return max(min(pid, diff), 0) + self.min_pwm
    

    def pure_pursuit_control(self, state, ref_point):
        x, y, psi, ref_v = state[0], state[1], state[2], state[3]

        dx = ref_point[0] - x
        dy = ref_point[1] - y

        # Transform goal to vehicle local frame
        local_x =  np.cos(psi) * dx + np.sin(psi) * dy
        local_y = -np.sin(psi) * dx + np.cos(psi) * dy

        goal_dist = np.sqrt(local_x**2 + local_y**2)

        if goal_dist < 0.1:
            steer = 0.0
        else:
            wheelbase   = self.params_car['lf']+self.params_car['lr']
            numerator   = 2.0 * wheelbase * local_y
            denominator = goal_dist**2

            steer = float(np.clip(
                np.arctan2(numerator, denominator),
                self.params_car['min_steer'], self.params_car['max_steer']
            ))

        # velocity-scaled cur: slow down on sharp turns
        scale = self.max_v - self.min_v   # note: velocity range, not cur range
        ref_v = self.min_v + scale * np.sqrt(1.0 - abs(steer / self.params_car['max_steer']))

        pwm = self.pid_long_control(ref_v, state[3])
        return np.array([pwm, steer])


class LLALogger():
    def __init__(self, out_file = None):
        if not out_file is None:
            ros_log_root = os.path.expanduser("~/.ros/log")
            os.makedirs(ros_log_root, exist_ok=True)
            
            timestamp = int(time.time() * 1e6)
            self.log_file = os.path.join(ros_log_root, f"{out_file}_{timestamp}.npz")
            self.log_buffer = {
                "time": [],
                "state": [],
                "params": [],
                "model_idx": [],
                "ctrl": [],
                "d_ctrl":[],
                "cmd":[],
                "mpc_rollout":[],
                "ref_trajectory":[],
                "ok_time":[],
                "predicted_state": [],
                "one_step_cost": [],
                "running_cost":[],
                "known_params": [],   # NEW
                "solve_time": [],     # NEW
            }
            self.get_logger().info(f"Logging MPC data to {self.log_file}")


    def log_lla_data(self, params, model_index, known_params, solve_time, delta_u=[],
                  mpc_rollout=[], ref_trajectory=[]):
        if self.log_data:
            now_ns = time.perf_counter_ns()
            self.log_buffer["time"].append(now_ns)
            self.log_buffer["state"].append(self.current_state.copy())
            self.log_buffer["params"].append(params.copy())
            self.log_buffer["model_idx"].append(model_index)
            self.log_buffer["ctrl"].append(self.last_control.copy())
            self.log_buffer["d_ctrl"].append(delta_u)
            self.log_buffer["cmd"].append(self.last_drive_command.copy())
            self.log_buffer["mpc_rollout"].append(np.array(mpc_rollout))
            self.log_buffer["ref_trajectory"].append(np.array(ref_trajectory))
            self.log_buffer["known_params"].append(np.array(known_params))  # NEW
            self.log_buffer["solve_time"].append(solve_time)                # NEW


    def log_rollout_data(self, lb_history, one_step_cost, ok_time):
        if(self.log_data):
            self.log_buffer["ok_time"].append(ok_time)
            # self.log_buffer["predicted_state"].append(lb_history.last_predicted_states.copy())
            # self.log_buffer["one_step_cost"].append(one_step_cost)
            # self.log_buffer["running_cost"].append(lb_history.running_cost.copy())


    def save_log(self):
        np.savez(
            self.log_file,
            time=np.array(self.log_buffer["time"]),
            state=np.array(self.log_buffer["state"]),
            params=np.array(self.log_buffer["params"]),
            model_index=np.array(self.log_buffer["model_idx"]),
            ctrl=np.array(self.log_buffer["ctrl"]), 
            d_ctrl=np.array(self.log_buffer["d_ctrl"]),
            states=np.array(self.log_buffer["predicted_state"]),
            mpc_rollout=np.array(self.log_buffer["mpc_rollout"]),
            ref_trajectory=np.array(self.log_buffer["ref_trajectory"]),
            one_step_cost=np.array(self.log_buffer["one_step_cost"]),
            running_cost=np.array(self.log_buffer["running_cost"]),
            ok_time = np.array(self.log_buffer["ok_time"]),
            cmd = np.array(self.log_buffer["cmd"]),
            known_params = np.array(self.log_buffer["known_params"]),
            solve_time = np.array(self.log_buffer["solve_time"])
        )