#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("/home/kathy/jax_cache")
import jax
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
# jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp

from llampc.nmpc_gen_fiala import setup_mpc
from llampc.params import F110, get_param_dict_random, get_param_dict_grid, param_validate_ptm
from llampc.planner import get_reference_trajectory_segment, get_lookahead_point
from llampc.utils import Track

import llampc.rollout.history as history
import llampc.rollout.dynamic_sim as dynamics_sim
import llampc.rollout.dynamic_fiala as dynamics_fiala
import llampc.rollout.dynamic_rp as dynamics_rp
import llampc.rollout.dynamic_full as dynamics_full
from llampc.rollout.rk6 import rk4Factory


from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float64MultiArray, Float64
from vesc_msgs.msg import VescStateStamped

import time


class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.get_logger().info("Initializing")

        self.lla_type = "reg"
        self.publish_trajectories = True
        self.log_data = True

        self.declare_params()
        self.initialize_mpc()
       
        # drive commands
        self.last_drive_command = np.array([0.0, 0.0]) #vx, steer
        self.last_control = np.array([0.0, 0.0]) #acceleration, steer
        self.rates = np.array([0.0, 0.0])
        self.first_control = True
        
        self.projidx = 0

        # timing
        self.count = 0
        self.time_window = 1
        self.time_index = 0
        self.checkpoints = 7
        self.time_history = np.zeros((self.checkpoints, self.time_window))
        self.maxtime = np.zeros(self.checkpoints)
        self.checkpoint = np.empty(self.checkpoints)

        self.last_v_err = None
        self.v_int_err = 0
        self.current_mode = False

        # dictionary, prefereably npy, which has waypoints_x, waypoints_y, and velocity
        track_name = self.get_parameter('track_file_name').get_parameter_value().string_value

        self.track = Track(track_name)




        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10
        )

        self.sensor_subscriber = self.create_subscription(
            VescStateStamped,
            '/sensors/core',
            self.sensor_callback,
            10
        )

        self.dfz_subscriber = self.create_subscription(
            Float64,
            '/dfz/estimate',
            self.dfz_callback,
            10
        )


        self.mpc_dfz_pub = self.create_publisher(
            Float64MultiArray,
            '/mpc/dfz_pred',
            10
        )

        self.predicted_path_pub = self.create_publisher(
            PoseArray,
            '/predicted_path',
            10
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped,
            '/mpc/drive',
            10
        )

        self.mpc_info_pub = self.create_publisher(
            Float64MultiArray,
            '/mpc_info',
            10
        )

        self.ref_pub = self.create_publisher(
            PoseArray,
            '/ref_trajectory',
            10
        )

        self.control_timer = self.create_timer(self.control_callback_speed, self.control_callback) # run 100 hz


        self.get_logger().info("F1tenth MPC Initialized")

    def declare_params(self):
        self.declare_parameter('solver_config', 'default')
        self.declare_parameter('json_file', 'f1tenth_acados_ocp.json')
        self.declare_parameter('track_file_name', 'mocap_square2slow.npz')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        #self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('out_file', 'out')

        self.N = 20 #steps (for nmpc)
        self.Tf = 0.8 # total time horizon (for nmpc)
        self.dt = self.Tf / self.N
        self.control_callback_speed = 0.04
        self.lla_predict_horizon = 0.04
        self.lla_reset_interval = 0
        self.lla_reset_counter = 0

        self.min_pwm = 0.1
        self.max_pwm = 0.25

        self.max_v = 3.0
        self.min_v = 0.2

        self.kd = 0.1
        self.ki = 0.1
        self.kp = 0.01

        # TODO: Implement thresholding to prevent hysteresis

        self.params_car = F110()


        if(self.log_data):
            out_file =  self.get_parameter('out_file').get_parameter_value().string_value

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
    
    def fiala_setup(self):
        self.get_logger().info("Regular MPC Initialized")
        params_car = F110()
        

        mean_dict = {
            'Cf': 250,
            'Cr': 225,
            'muf': 0.7,
            'mur': 0.7,
            'Cro': 0.0,
        }
        

        variation_dict = {
            'Cf': 75,   # 15% variation
            'Cr': 75,   # 15% variation
            'muf': 0.4,   # 15% variation
            'mur': 0.4,   # 15% variation
            'Cro': 0.0, # 15% variation
        }
        
        cost_weights = np.array([0.0, 0.0, 20.0, 5.0, 10.0, 0.01])# x, y, theta, vx, vy, omega
        # x, y, theta, vx, vy, omega
        
        # grid discretization
        discretization_dict = {
            'Cf': 7,   # 15% variation
            'Cr': 7,   # 15% variation
            'muf': 7,   # 15% variation
            'mur': 7,   # 15% variation
            'Cro': 0.0, # 15% variation
            
        }
        param_dict = get_param_dict_grid(mean_dict, variation_dict, 
                                         discretization=discretization_dict, ground_truth=True,
                                         noadapt=False)
        num_models = len(param_dict['Cf'])
        self.get_logger().info("Dynamics bank starting")
        
        self.state_size = 6
        self.dynamics_bank = dynamics_fiala.DBMFialaBank(
            params_car['lf'], params_car['lr'], 
            params_car['mass'], params_car['Iz'],
            params_car['rw'], 
            param_dict['Cf'], param_dict['Cr'],
            param_dict['muf'], param_dict['mur'],
            param_dict['Cro'],
            num_models
        )
        
        self.get_logger().info("History starting")
        
        history_length=40
        self.lb_history = history.LBHistory(
            num_models, history_length,
            self.lla_predict_horizon, cost_weights,
            self.state_size, rk4Factory,
            self.dynamics_bank, dynamics_fiala.diffequation,
            buffer_size = [0, 0]
        )
        
        self.get_logger().info("History generation complete")

    def initialize_mpc(self):
        variation_dict = None
        mean_dict = None
        
        self.state_size = 0
        self.fiala_setup()
            

        import multiprocessing
        self.get_logger().info(f"Devices seen: {multiprocessing.cpu_count()}")

        self.get_logger().info("Warm starting bank")
        # warm start bank
        self.lb_history.predict_states(np.zeros(self.state_size), np.zeros(2))
        self.lb_history.predict_states(self.lb_history.last_predicted_states, np.zeros(2))
        self.lb_history.update_lookback_error(np.zeros(self.state_size))
        self.lb_history.get_best_model()
        self.lb_history.reset()
        self.get_logger().info("Bank initialized")

        # start solver
        self.current_state = None
        self.omega_w = 0
        self.dFz = 0

        self.solver = setup_mpc(self.N, self.Tf, build=True)
        self.get_logger().info("SOLVER COMPILED, WARM STARTING")
        
        self.lb_history.predict_states(
            np.zeros(self.state_size), np.zeros(2)
            )
        self.lb_history.update_lookback_error(
            np.zeros(self.state_size)
        )

        self.lb_history.get_best_model()
        self.lb_history.reset()
        self.get_logger().info("LLA BANK COMPILED")

    def odom_callback(self, msg):    
        # print("hello")    
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        prev_phi = self.current_state[2] if not self.current_state is None else 0

        phi = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        phi = np.unwrap([prev_phi, phi])[1]

        # print(phi)
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z


        self.current_state = np.array([x, y, phi, vx, vy, omega])

        # self.get_logger().info(f"Logging State {self.current_state}")

    def sensor_callback(self, msg):
        erpm = msg.state.speed

        pole_pairs = self.params_car['pole_pairs']
        gear_ratio = self.params_car['gear_ratio']

        motor_rpm = erpm / pole_pairs
        motor_omega = motor_rpm * (2 * np.pi / 60.0)
        self.omega_w = motor_omega / gear_ratio

        self.last_control[0] = msg.state.avg_iq
        
    def dfz_callback(self, msg):
        self.dFz = msg.data

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

    
    def prepare_solve(self):
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
    
    def control_callback(self):
        self.checkpoint[0] = time.perf_counter_ns()
        print(f"CURSTATE: {self.current_state}")
        if self.track is None or self.current_state is None:
            return
        
        ##############################################
        ### BANK UPDATE
        one_step_cost = None
        # ok_time = self.time_history[-1, self.count] * 1e-6 < 2 * self.dt * 1000
        ok_time = True
        one_step_cost = self.lb_history.update_lookback_error(
            self.current_state
        )

        self.log_rollout_data(self.lb_history, one_step_cost, ok_time)

        x0 = self.current_state[:2]
        v0 = self.current_state[3]

        self.checkpoint[1] = time.perf_counter_ns()

        #############################################
        ### GET REF TRAJECTORY AND MODEL FOR ROLLOUT

        selected_model_params = None
        
        selected_model_index = self.lb_history.get_best_model()
        selected_model_params = self.dynamics_bank.get_model_params_arr(selected_model_index)
        
        # print(f"SELECTED PARAMS {selected_model_params}")
        
        ########################################################
        #### SETUP AND SOLVE MPC

        self.checkpoint[2]= time.perf_counter_ns()

      

        self.checkpoint[3] = time.perf_counter_ns()
        d_ctrl = []
        if self.current_state[3] < 0.1:
            ref_point, idx = get_lookahead_point(self.current_state, self.track, self.projidx, lookahead_dist = 1.2)
            self.projidx = idx

            record_ref_trajectory = [ref_point]

            u_opt = self.pure_pursuit_control(self.current_state, ref_point)
            status = 0

            self.mpc_dfz_pub.publish(Float64MultiArray(
                data=[0.0, 0.0]
            ))
            self.current_mode = False
            self.first_control = True
        else:
            # filtered_state = self.current_state.copy()
            # if( np.abs(self.current_state[3]) < 0.1):
            #     filtered_state[3] = 0.1
            aug_state = np.concatenate(
                [self.current_state, 
                [self.omega_w, self.dFz], 
                 self.last_control]
            )

            self.dynamics_bank.update_known_params(self.omega_w, self.dFz)

            # no need to copy states and trajectory in case of update b/c node is single thread
            ref_segment, idx = get_reference_trajectory_segment(x0, v0, self.track, self.N+1, self.dt, self.projidx)
            self.projidx = idx

            record_ref_trajectory = []
            if self.publish_trajectories:
                record_ref_trajectory = ref_segment.T
                # self.publish_ref_trajectory(ref_segment)
                # print(f"REF: {ref_segment}")

            
            self.prepare_solve()
            
            #print(f"aug state: {aug_state}")
            self.solver.set(0, "lbx", aug_state)
            self.solver.set(0, "ubx", aug_state)
            def construct_params(N, selected_model_params, ref_segment):
                full_params = np.zeros((N+1, 11), np.float64)
                full_params[:, :5] = selected_model_params
                # self.get_logger().info(f"{full_params}")
                full_params[:, 5:5+6] = ref_segment[:6, :N+1].T #reference x, y, theta
                return full_params
            
            full_params = construct_params(self.N, selected_model_params, ref_segment)

            for i in range(self.N+1):
                self.solver.set(i, "p", full_params[i])

                if(i == 0 or self.first_control):
                    self.solver.set(i, "x", aug_state)
            
            self.first_control = False
                
            status = self.solver.solve()
            u_opt = self.solver.get(1, "x")[-2:] # pwm, delta
            d_ctrl = self.solver.get(0, "u")[:]
            self.mpc_dfz_pub.publish(Float64MultiArray(
                data=[float(self.solver.get(1, "x")[7]), 1.0]
            ))

            self.current_mode = True


        self.checkpoint[4] = time.perf_counter_ns()

        print(f"CONTROL: {u_opt}")

        mpc_states = []
        mpc_controls = []
        
        if(self.publish_trajectories):
            for i in range(self.N + 1):
                x_pred = self.solver.get(i, "x")[:6]
                mpc_states.append(x_pred)
                
                c_pred = self.solver.get(i, "x")[-2:]
                mpc_controls.append(c_pred)

                #print(x_pred, c_pred)

            # print(f"PREDICTED STATES: {mpc_states}")
            # print(f"PREDICTED CONTROLS: {predicted_controls}")
                
            self.publish_predicted_trajectory(mpc_states) # Publish predicted trajectory

        
        if(not ok_time):
            self.lla_reset_counter = 0

        if(self.lla_reset_interval != 0):
            self.lla_reset_counter = (self.lla_reset_counter + 1) % self.lla_reset_interval

        #########################################
        ### PUBLISH MPC DATA
        residuals = self.solver.get_residuals()
        res_eq = residuals[1]
        vx = self.current_state[3]
        # eq_tol = 1e-2 if vx > 0.1 else 0.1   # much looser at low speed
        
        self.checkpoint[5] = time.perf_counter_ns()
        
        if status == 0 or (status == 2):  # Success
            # Get optimal control
            self.apply_control(u_opt) # Apply control
            # self.get_logger().info(f"Logging control {u_opt}")
            
            #version for our dynamics
            
            self.lb_history.predict_states(
                self.current_state, u_opt, self.lla_reset_counter == 0
            )
                
        else:
            print(f"\n--- SOLVER FAILED WITH STATUS {status} ---")

            self.solver.print_statistics()
            residuals = self.solver.get_residuals()
            print(f"Max Residuals (stat, eq, ineq, comp): {residuals}")
            
            N = self.solver.acados_ocp.dims.N
            print("\n--- NODE TRAJECTORY DUMP ---")
            for i in range(N + 1):
                x_i = self.solver.get(i, "x")
                print(f"Node {i} | State: {np.round(x_i, 3)}")
            print("-----------------------------------\n")

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.cmd_pub.publish(drive_msg)
            
            self.last_drive_command = np.array([0.0, 0.0])
            self.last_control = np.array([0.0, 0.0])
            self.get_logger().warn(f"MPC solver failed with status: {status}")

            self.current_mode = False
            self.first_control = True

        self.checkpoint[6] = time.perf_counter_ns()
        self.count = (self.count + 1) % self.time_window
        self.time_history[:self.checkpoints-1, self.count] = np.array(self.checkpoint[1:]-self.checkpoint[:-1])
        self.time_history[-1, self.count] = (self.checkpoint[-1] - self.checkpoint[0])
    
        if(self.count == 0):
            print(np.max(self.time_history*1e-6, axis = 1))

        known_params = np.array(self.dynamics_bank.get_known_params())
        
        self.log_lla_data(
            selected_model_params, 
            selected_model_index, 
            known_params, 
            self.time_history[-1, self.count]*1e-6, 
            d_ctrl,
            mpc_states, 
            record_ref_trajectory)

    def publish_ref_trajectory(self, ref_trajectory):
        ref_msg = PoseArray()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.header.frame_id = "map"

        # print(len(ref_trajectory))
        for x, y in (ref_trajectory[:2].T):
            point = Pose()
            point.position.x = x
            point.position.y = y
            ref_msg.poses.append(point)

        self.ref_pub.publish(ref_msg)

    def apply_control(self, u_opt):
        """Apply optimal control to the vehicle"""
        # acceleration = float(u_opt[0])
        desired_v = self.solver.get(1, 'x')[3]

        cur = float(u_opt[0])
        steer = float(u_opt[1])
        
        #print("cur")
        #print(cur)

        # sensor_velocity = np.sqrt(self.current_state[3] **2 + self.current_state[4]**2)
        
        # Create Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"

        # print(f"ORIGINAL {self.last_drive_command[0] + accel * self.dt}")
        # print(f"NEW {desired_v}")
        # print(f"NEW_INT {self.current_state[3] + accel * self.dt}")
        # new_int = self.current_state[3] + accel * self.dt
        # old = self.last_drive_command[0] + accel * self.dt
        
        # Convert acceleration to speed command (simple integration)
        # desired_speed = max(0.0, new_int)

        drive_msg.drive.speed = 0.0
        drive_msg.drive.jerk = 2.0 if self.current_mode else 1.0
        drive_msg.drive.acceleration = cur
        drive_msg.drive.steering_angle = steer

        self.cmd_pub.publish(drive_msg) 

        self.last_drive_command = np.array([cur, steer])
        # print( acceleration * self.dt)
        self.last_control = np.array([cur, steer])
        self.last_v = self.current_state[3]

        
    def publish_predicted_trajectory(self, predicted_states):
        """Publish predicted trajectory for visualization"""

        path_msg = PoseArray()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for state in predicted_states:
            pose_unstamped = Pose()
            pose_unstamped.position.x = float(state[0])
            pose_unstamped.position.y = float(state[1])

            # Convert yaw to quaternion
            yaw = float(state[2])
            pose_unstamped.orientation.w = np.cos(yaw / 2.0)
            pose_unstamped.orientation.z = np.sin(yaw / 2.0)
            
            path_msg.poses.append(pose_unstamped)

        # print(f"{predicted_states}")
        
        self.predicted_path_pub.publish(path_msg)

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

    def destroy_node(self):
        if(self.log_data):
            self.get_logger().info(f"Saving data to {self.log_file}")
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
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected")
    finally:
        node.destroy_node()  # save log here
        rclpy.shutdown()

if __name__ == '__main__':
    main()
