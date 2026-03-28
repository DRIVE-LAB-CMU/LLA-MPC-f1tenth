#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("/home/kathy/jax_cache")
import jax
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp



from llampc.nmpc_gen import setup_mpc
from llampc.params import F110, F110_sim, get_param_dict
from llampc.planner import get_reference_trajectory_segment
from llampc.utils import Track

import llampc.rollout.history as history
import llampc.rollout.dynamic_sim as dynamics_sim
import llampc.rollout.dynamic as dynamics
import llampc.rollout.dynamic_rp as dynamics_rp
import llampc.rollout.dynamic_full as dynamics_full
from llampc.rollout.rk6 import rk4Factory


from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float64MultiArray

import time


class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.get_logger().info("Initializing")

        self.sim = False
        self.lla_type = "reg"
        self.publish_trajectories = True
        self.log_data = True

        self.declare_params()
        self.initialize_mpc()
       
        self.last_drive_command = np.array([0.0, 0.0]) #vx, steer
        self.last_control = np.array([0.0, 0.0]) #acceleration, steer
        self.rates = np.array([0.0, 0.0])
        self.first_control = True
        
        self.projidx = 0

        self.count = 0
        self.time_window = 1
        self.time_index = 0
        self.checkpoints = 7
        self.time_history = np.zeros((self.checkpoints, self.time_window))
        self.maxtime = np.zeros(self.checkpoints)
        self.checkpoint = np.empty(self.checkpoints)

        # dictionary, prefereably npy, which has waypoints_x, waypoints_y, and velocity
        track_name = self.get_parameter('track_file_name').get_parameter_value().string_value

        self.track = Track(track_name)
    

        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10
        )


        self.predicted_path_pub = self.create_publisher(
            PoseArray,
            '/predicted_path',
            10
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
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
        self.declare_parameter('track_file_name', 'nshtest.npz')
        # self.declare_parameter('odom_topic', '/odom/filtered')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('out_file', 'out')

        self.N = 20 #steps (for nmpc)
        self.Tf = 1.0 # total time horizon (for nmpc)
        self.dt = self.Tf / self.N
        self.control_callback_speed = 0.04
        self.lla_predict_horizon = 0.04

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
                "cmd":[],
                "ok_time":[],
                "predicted_state": [],
                "one_step_cost": [],
                "running_cost":[],
            }
            self.get_logger().info(f"Logging MPC data to {self.log_file}")
    
    def regular_setup(self):
        self.get_logger().info("Regular MPC Initialized")
        params_car = F110()
        
        mean_dict = {
            'Bf': 15.0,
            'Br': 15.0,
            'Cf': 1.0,
            'Cr': 1.0,
            'Df': 25.0,
            'Dr': 25.0,
            'Cro': 0.0,
            'Cd': 0.0,
            'Ce': 1.0,
            'Cm': .0, 
        }

        variation_dict = {
                'Bf': 0 * mean_dict['Bf'],   # 15% variation
                'Br': 0 * mean_dict['Br'],   # 15% variation
                'Cf': 0 * mean_dict['Cf'],   # 15% variation
                'Cr': 0 * mean_dict['Cr'],   # 15% variation
                'Df': 0 ,   # 15% variation
                'Dr': 0 ,   # 15% variation
                'Cro': 0, # 15% variation
                'Cd': 0,  # assume negligible drag
                'Ce': 0.3,  # motor efficiency conversion should never be above 1
                'Cm': 0.5* mean_dict['Cm'],  # motor speed saturation
            }
        cost_weights = np.array([1.0, 1.0, 0, 0, 0, 0]) # x, y, theta, vx, vy, omega
        
        num_models = 10
        self.state_size = 6
        param_dict = get_param_dict(mean_dict, variation_dict, num_models, ground_truth=True)
        
        self.get_logger().info("Dynamics bank starting")

        self.dynamics_bank = dynamics.DBMPacejkaBank(
            params_car['lf'], params_car['lr'], 
            params_car['mass'], params_car['Iz'], 
            param_dict['Bf'], param_dict['Br'],
            param_dict['Cf'], param_dict['Cr'],
            param_dict['Df'], param_dict['Dr'],
            param_dict['Cro'], param_dict['Cd'],
            param_dict['Ce'], param_dict['Cm'],
            0, 0,
            num_models
        )
        
        self.get_logger().info("History starting")
        
        history_length=25
        self.lb_history = history.LBHistory(
            num_models, history_length,
            self.lla_predict_horizon, cost_weights,
            self.state_size, rk4Factory,
            self.dynamics_bank, dynamics.diffequation,
            buffer_size = [0, 0]
        )
        
        self.get_logger().info("History generation complete")

    def exp_setup(self):
        self.get_logger().info("novar Initialized")
        params_car = F110()
        
        mean_dict = {
            'Bf': 15.0,
            'Br': 15.0,
            'Cf': 1.0,
            'Cr': 1.0,
            'Df': 0.95,
            'Dr': 0.95,
            'Cro': 0.02,
            'Cd': 0.001,
            'Ce': 1.0,
            'Cm': .05, 
            
        }

        variation_dict = {
                'Bf': 0, # 15% variation
                'Br': 0, # 15% variation
                'Cf': 0, # 15% variation
                'Cr': 0, # 15% variation
                'Df': 0,
                'Dr': 0,
                'Cro':0, # 15% variation
                'Cd': 0, # 15% variation
                'Ce': 0, # 15% variation
                'Cm': 0, # 15% variation
            }
        cost_weights = np.array([1.0, 1.0, 0, 0, 0, 0]) # x, y, theta, vx, vy, omega
        
        num_models = 1
        self.state_size = 6
        param_dict = get_param_dict(mean_dict, variation_dict, num_models, ground_truth=True)

        self.dynamics_bank = dynamics.DBMPacejkaBank(
            params_car['lf'], params_car['lr'], 
            params_car['mass'], params_car['Iz'], 
            param_dict['Bf'], param_dict['Br'],
            param_dict['Cf'], param_dict['Cr'],
            param_dict['Df'], param_dict['Dr'],
            param_dict['Cro'], param_dict['Cd'],
            param_dict['Ce'], param_dict['Cm'],
            0, 0,
            num_models
        )
        
        history_length=25
        self.lb_history = history.LBHistory(
            num_models, history_length,
            self.lla_predict_horizon, cost_weights,
            self.state_size, rk4Factory,
            self.dynamics_bank, dynamics.diffequation
        )


    def rp_setup(self):
        self.get_logger().info("Roll Pitch MPC Initialized")
        params_car = F110()
        variation_dict = {
            'Df': .2,   
            'Dr': .2,
            'roll': np.pi/4, 
            'pitch': np.pi/4
        }

        mean_dict = {
            'Df': 0.8,
            'Dr': 0.8,
            'roll': 0, 
            'pitch': 0  
        }

        static_dict = {
            'Bf': 15.0,
            'Br': 15.0,
            'Cf': 1.0,
            'Cr': 1.0,
            'Cro': 0.02,
            'Cd': 0.001,
            'Ce': 1.0,
            'Cm': .05, 
        }

        cost_weights = np.array([1.0, 1.0, 0.1, 0.01, .01, 0]) # x, y, theta, vx, vy, omega
        
        num_models = 6000
        self.state_size = 6
        param_dict = get_param_dict(mean_dict, variation_dict, num_models)

        self.dynamics_bank = dynamics_rp.DBMPacejkaBankRP(
            params_car['lf'], params_car['lr'], 
            params_car['mass'], params_car['Iz'], 
            static_dict['Bf'], static_dict['Br'],
            static_dict['Cf'], static_dict['Cr'],
            param_dict['Df'], param_dict['Dr'],
            static_dict['Cro'], static_dict['Cd'],
            static_dict['Ce'], static_dict['Cm'],
            param_dict['roll'], param_dict['pitch'],
            num_models
        )

        history_length=25
        self.lb_history = history.LBHistory(
            num_models, history_length,
            self.lla_predict_horizon, cost_weights,
            self.state_size, rk4Factory,
            self.dynamics_bank, dynamics_rp.diffequation
        )

    def sim_setup(self):
        params_car = F110_sim()
        cost_weights = np.array([1.0, 1.0, 0, 0, 0, 0, 0])

        mean_dict = {
            'C_Sf': params_car['C_Sf'], 
            'C_Sr':params_car['C_Sr'],
            'mu': params_car['mu'],
        }

        variation_dict = {
            'C_Sf': 0, 
            'C_Sr': 0,
            'mu': 0,
        }

        num_models = 10
        self.state_size = 7

        param_dict = get_param_dict(mean_dict, variation_dict, num_models)
        self.dynamics_bank = dynamics_sim.DynamicSimBank(
            params_car['lf'], params_car['lr'],
            params_car['m'], params_car['I'],
            params_car["h"], params_car['v_switch'],
            params_car['a_max'], params_car['v_min'],
            params_car['v_max'], params_car['s_min'],
            params_car['s_max'], params_car['sv_min'],
            params_car['sv_max'],
                param_dict['C_Sf'], param_dict['C_Sr'],
                param_dict['mu'],num_models
            )
        self.lb_history = history.LBHistory(
            num_models, 20, 0.2, cost_weights, 7,
            rk4Factory, self.dynamics_bank, dynamics_sim.diffequation
        )

    def full_setup(self):
        self.get_logger().info("Regular MPC Initialized")
        params_car = F110()
        
        mean_dict = {
            'Bf': 15.0,
            'Br': 15.0,
            'Cf': 1.0,
            'Cr': 1.0,
            'Df': 0.8,
            'Dr': 0.8,
            'Cro': 0,
            'Cd': 0,
            'Ce': 1.0,
            'Cm': 0, 
            'roll': 0,
            'pitch': 0
        }

        variation_dict = {
            'Bf': .15 * mean_dict['Bf'],   # 15% variation
            'Br': .15 * mean_dict['Br'],   # 15% variation
            'Cf': .15 * mean_dict['Cf'],   # 15% variation
            'Cr': .15 * mean_dict['Cr'],   # 15% variation
            'Df': .15 * mean_dict['Df'],   # 15% variation
            'Dr': .15 * mean_dict['Dr'],   # 15% variation
            'Cro': 0.15* mean_dict['Cro'], # 15% variation
            'Cd': 0.15* mean_dict['Cd'],  # 15% variation
            'Ce': 0.15* mean_dict['Ce'],  # 15% variation
            'Cm': 0.15* mean_dict['Cm'],  # 15% variation,
            'roll': np.pi/4,
            'pitch': np.pi/4
        }
        
        cost_weights = np.array([1.0, 1.0, 0.1, 0.01, 0.01, 0]) # x, y, theta, vx, vy, omega
        
        num_models = 7000
        self.state_size = 6
        param_dict = get_param_dict(mean_dict, variation_dict, num_models)

        self.dynamics_bank = dynamics_full.DBMPacejkaBank(
            params_car['lf'], params_car['lr'], 
            params_car['mass'], params_car['Iz'], 
            param_dict['Bf'], param_dict['Br'],
            param_dict['Cf'], param_dict['Cr'],
            param_dict['Df'], param_dict['Dr'],
            param_dict['Cro'], param_dict['Cd'],
            param_dict['Ce'], param_dict['Cm'],
            param_dict['roll'], param_dict['pitch'],
            
            
        )
        
        history_length=25
        self.lb_history = history.LBHistory(
            num_models, history_length,
            self.lla_predict_horizon, cost_weights,
            self.state_size, rk4Factory,
            self.dynamics_bank, dynamics_full.diffequation
        )


    def initialize_mpc(self):
        variation_dict = None
        mean_dict = None
        
        self.state_size = 0

        if not self.sim:
            if(self.lla_type == "rp"):
                self.rp_setup()
            elif(self.lla_type == "exp"):
                self.exp_setup()
            elif(self.lla_type == "full"):
                self.full_setup()
            else:
                self.regular_setup()
            
        else:
            self.sim_setup()
            

        import multiprocessing
        self.get_logger().info(f"Devices seen: {multiprocessing.cpu_count()}")

        self.get_logger().info("Warm starting bank")
        # warm start bank
        self.lb_history.predict_states(np.zeros(self.state_size), np.zeros(2))
        self.lb_history.update_lookback_error(np.zeros(self.state_size))
        self.lb_history.get_best_model()
        self.lb_history.reset()
        self.get_logger().info("Bank initialized")

        # start solver
        self.current_state = None
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

    def control_callback(self):
        self.checkpoint[0] = time.perf_counter_ns()

        if self.track is None or self.current_state is None:
            return
        
        ##############################################
        ### BANK UPDATE
        one_step_cost = None
        ok_time = self.time_history[-1, self.count] * 1e-6 < 2 * self.dt * 1000
        # if(ok_time):
        if not self.sim:
            one_step_cost = self.lb_history.update_lookback_error(
                self.current_state
            )
        else:
            one_step_cost = self.lb_history.update_lookback_error(
                np.array(
                    [
                        self.current_state[0],
                        self.current_state[1],
                        self.current_state[2],
                        self.current_state[3],
                        np.arctan2(self.current_state[4], self.current_state[3]),
                        self.current_state[5], 
                        self.last_control[1]
                    ]
                )
            )

        self.log_rollout_data(self.lb_history, one_step_cost, ok_time)

        x0 = self.current_state[:2]
        v0 = self.current_state[3]

        self.checkpoint[1] = time.perf_counter_ns()

        #############################################
        ### GET REF TRAJECTORY AND MODEL FOR ROLLOUT

        # no need to copy states and trajectory in case of update b/c node is single thread
        ref_segment, idx = get_reference_trajectory_segment(x0, v0, self.track, self.N+1, self.dt, self.projidx)
        self.projidx = idx

        if self.publish_trajectories:
            self.publish_ref_trajectory(ref_segment)
            # print(f"REF: {ref_segment}")

        self.checkpoint[2] = time.perf_counter_ns()
        
        selected_model_params = None
        if self.sim:
            selected_model_params = np.array([
                15.0,
                15.0,
                1.0,
                1.0,
                0.8,
                0.8,
                0.02,
                0.001,
                1.0,
                .05, 
            ]
            )
        else:
            selected_model_index = self.lb_history.get_best_model()
            selected_model_params = self.dynamics_bank.get_model_params_arr(selected_model_index)

            self.log_lla_data(selected_model_params, selected_model_index)
        
        ########################################################
        #### SETUP AND SOLVE MPC
        filtered_state = self.current_state.copy()
        if( np.abs(self.current_state[3]) < 0.1):
            filtered_state[3] = 0.1

        aug_state = np.concatenate([filtered_state, self.last_control])
        self.solver.set(0, "lbx", aug_state)
        self.solver.set(0, "ubx", aug_state)
        def construct_params(N, selected_model_params, ref_segment):
            full_params = np.zeros((N+1, 20), np.float64)
            full_params[:, :12] = selected_model_params
            # self.get_logger().info(f"{full_params}")
            full_params[:, 12:14] = ref_segment[:2, :N+1].T #reference x, y
            return full_params
        
        full_params = construct_params(self.N, selected_model_params, ref_segment)

        for i in range(self.N+1):
            self.solver.set(i, "p", full_params[i])

            if(self.first_control or i == 0):
                self.solver.set(i, "x", aug_state)
            # self.solver.set(i, "yref", all_yref)

        if(self.first_control):
            self.first_control = False


        self.checkpoint[3]= time.perf_counter_ns()

        # print("DEBUG STATE:", aug_state)
        # print("DEBUG PARAMS NODE 0:", full_params[0])

        status = self.solver.solve()
        # self.solver.print_statistics()

        # if status != 0:
        #     self.get_logger().warn(f"MPC solver failed with status: {status}. Resetting memory!")
            
        #     # Rescue the solver from the death spiral by flatlining the guess
        #     for i in range(self.N + 1):
        #         self.solver.set(i, "x", aug_state)
            
        #     # Reset the controls to 0.0 to prevent steering spikes
        #     for i in range(self.N):
        #         self.solver.set(i, "u", np.zeros(2))

        # status = self.solver.solve()


        self.checkpoint[4] = time.perf_counter_ns()

        u_opt = self.solver.get(1, "x")[-2:] # acceleration, delta
        # print(f"CONTROL: {u_opt}")

        if(self.publish_trajectories):
            predicted_states = []
            for i in range(self.N + 1):
                x_pred = self.solver.get(i, "x")[:3]
                predicted_states.append(x_pred)

            # print(f"PREDICTED STATES: {predicted_states}")

            self.publish_predicted_trajectory(predicted_states) # Publish predicted trajectory

        #########################################
        ### PUBLISH MPC DATA
        if status == 0:  # Success
            # Get optimal control
            self.apply_control(u_opt) # Apply control
            # self.get_logger().info(f"Logging control {u_opt}")
            if not self.sim:
                #version for our dynamics
                self.checkpoint[5] = time.perf_counter_ns()
                self.lb_history.predict_states(
                    self.current_state, u_opt
                )
                self.checkpoint[6] = time.perf_counter_ns()
                
            else:
                steer_v = self.solver.get(0, "u")[1]
                self.lb_history.predict_states(
                    np.array(
                        [
                            self.current_state[0],
                            self.current_state[1],
                            self.current_state[2],
                            self.current_state[3],
                            np.arctan2(self.current_state[4], self.current_state[3]),
                            self.current_state[5], 
                            self.last_control[1]
                        ]
                    ),
                    np.array([u_opt[0], steer_v]) 
                )

            self.count = (self.count + 1) % self.time_window
            self.time_history[:self.checkpoints-1, self.count] = np.array(self.checkpoint[1:]-self.checkpoint[:-1])
            self.time_history[-1, self.count] = (self.checkpoint[-1] - self.checkpoint[0])
        
            if(self.count == 0):
                print(np.max(self.time_history*1e-6, axis = 1))
        else:
            self.apply_control([0, 0]) # Brake
            self.get_logger().warn(f"MPC solver failed with status: {status}")


    def publish_ref_trajectory(self, ref_trajectory):
        ref_msg = PoseArray()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.header.frame_id = "map"

        # print(len(ref_trajectory))
        for x, y in ref_trajectory.T:
            point = Pose()
            point.position.x = x
            point.position.y = y
            ref_msg.poses.append(point)

        self.ref_pub.publish(ref_msg)


    def apply_control(self, u_opt):
        """Apply optimal control to the vehicle"""
        # u_opt = [steer, acceleration]
        acceleration = float(u_opt[0])
        steer = float(u_opt[1])
        
        # Create Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        # Convert acceleration to speed command (simple integration)
        desired_speed = max(0.0, self.last_drive_command[0] + acceleration * self.dt)

        drive_msg.drive.speed = desired_speed
        drive_msg.drive.steering_angle = steer

        self.cmd_pub.publish(drive_msg) 

        self.last_drive_command = np.array([desired_speed, steer])
        # print( acceleration * self.dt)
        self.last_control = np.array([acceleration, steer])
        

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
    
    def publish_mpc_info(self, u_opt, status):
        """Publish MPC solver information"""
        info_msg = Float64MultiArray()
        info_msg.data = [
            float(u_opt[1]),  # acceleration
            #self.last_drive_command[0], # target speed
            float(u_opt[0]),  # steer
            float(status),    # solver status
            float(self.solver.get_cost())  # optimal cost
        ]
        self.mpc_info_pub.publish(info_msg)

    def log_lla_data(self, params, model_index):
        if(self.log_data):
            now_ns = time.perf_counter_ns()
            self.log_buffer["time"].append(now_ns)
            self.log_buffer["state"].append(self.current_state.copy())
            self.log_buffer["params"].append(params.copy())
            self.log_buffer["model_idx"].append(model_index)
            self.log_buffer["ctrl"].append(self.last_control.copy())
            self.log_buffer["cmd"].append(self.last_drive_command.copy())

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
                states=np.array(self.log_buffer["predicted_state"]),
                one_step_cost=np.array(self.log_buffer["one_step_cost"]),
                running_cost=np.array(self.log_buffer["running_cost"]),
                ok_time = np.array(self.log_buffer["ok_time"]),
                cmd = np.array(self.log_buffer["cmd"])
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
