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

from llampc.nmpc_gen_fiala_fixed import setup_mpc
from llampc.params import F110, get_param_dict_grid
from llampc.planner import get_reference_trajectory_segment, get_lookahead_point, ema_filter
from llampc.utils import Track

from llampc.lla_run_utils import LLASolver, LLALogger

import llampc.rollout.history as history
import llampc.rollout.dynamic_fiala_fixed as dynamics_fiala
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
        self.last_mu_est = None
        
        self.projidx = 0

        # timing
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

        self.sensor_subscriber = self.create_subscription(
            VescStateStamped,
            '/sensors/core',
            self.sensor_callback,
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
        self.with_lla = True
        self.adaptive_planning = False
        self.adaptive_control = True
        self.lla_reset_interval = 0
        self.lla_window = 60


        self.N = 30 #steps (for nmpc)
        self.hz = 40 #control frequency
        
        self.dt = 1/self.hz
        self.Tf = self.N * self.dt # total time horizon (for nmpc)
        self.control_callback_speed = self.dt
        self.lla_predict_dt = self.dt # lla


        self.declare_parameter('solver_config', 'default')
        self.declare_parameter('json_file', 'f1tenth_acados_ocp.json')
        self.declare_parameter('track_file_name', 'mocap_fig8fastbank.npz')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        # self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('out_file', 'out')
        self.lla_reset_counter = 0
        
        
        # TODO: Implement thresholding to prevent hysteresis

        self.params_car = F110()

        if(self.log_data):
            out_file =  self.get_parameter('out_file').get_parameter_value().string_value
            self.lla_logger = LLALogger(out_file)

    def fiala_setup(self):
        self.get_logger().info("Regular MPC Initialized")
        params_car = F110()

        mean_dict = {
            'Cf': 250,
            'Cr': 225,
            'muf': 0.6,
            'mur': 0.6,
            'Cro': 0.0,
        }

        
        variation_dict = {
            'Cf': 75,   
            'Cr': 75,   
            'muf': 0.5,
            'mur': 0.5,
            'Cro': 0.0, 
        }
        
        cost_weights = np.array([0.0, 0.0, 20.0, 1.0, 10.0, 0.1])
        # x, y, theta, vx, vy, omega
        
        # grid discretization
        discretization_dict = {
            'Cf': 7,   
            'Cr': 7,   
            'muf': 10,  
            'mur': 10,  
            'Cro': 0.0,
        }
        # stuff with blevel
        # discretization_dict = {
        #     'Cf': 5,   
        #     'Cr': 5,   
        #     'muf': 7,  
        #     'mur': 7,  
        #     'Cro': 0.0,
        # }

        param_dict = get_param_dict_grid(mean_dict, variation_dict, 
                                         discretization=discretization_dict, ground_truth=True,
                                         noadapt=(not self.with_lla))
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
        
        history_length=self.lla_window
        self.lb_history = history.LBHistory(
            num_models, history_length,
            self.lla_predict_dt, cost_weights,
            self.state_size, rk4Factory,
            self.dynamics_bank, dynamics_fiala.diffequation,
            buffer_size = [0, 0]
        )

        self.mean_params = self.dynamics_bank.get_model_params_arr(0)
        
        self.get_logger().info("History generation complete")

    def initialize_mpc(self):
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

        
        self.lla_solver = LLASolver(
            setup_mpc(self.N, self.Tf, build=True), lla_p=5, N = self.N)
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
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self.current_state = np.array([x, y, phi, vx, vy, omega])
        # self.get_logger().info(f"Logging MPC data to {self.log_file}")

    def sensor_callback(self, msg):
        erpm = msg.state.speed

        pole_pairs = self.params_car['pole_pairs']
        gear_ratio = self.params_car['gear_ratio']

        motor_rpm = erpm / pole_pairs
        motor_omega = motor_rpm * (2 * np.pi / 60.0)
        self.omega_w = motor_omega / gear_ratio

        self.last_control[0] = msg.state.avg_iq
        
  
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

        if(self.log_data):

            self.lla_logger.log_rollout_data(self.lb_history, one_step_cost, ok_time)

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
        mpc_states = []
        mpc_controls = []

        # self.current_state[3] = 0.2

        muf, mur = selected_model_params[2], selected_model_params[3]
        lf, lr = self.params_car['lf'], self.params_car['lr']
        mu_est =  (muf * lr + mur * lf) /( lf + lr)
        if self.last_mu_est is None:
            self.last_mu_est = mu_est
        mu_est = ema_filter(mu_est, self.last_mu_est, 0.1)
        self.last_mu_est = mu_est

        mu_est = None if not self.adaptive_planning else mu_est

        if self.current_state[3] < 0.1:
            ref_point, idx = get_lookahead_point(
                self.current_state, self.track, self.projidx, 
                lookahead_dist = 1.2, mu_est = mu_est)
            self.projidx = idx

            record_ref_trajectory = [ref_point]

            u_opt = self.lla_solver.pure_pursuit_control(self.current_state, ref_point)
            status = 0

            self.lla_solver.current_mode = False

            self.lla_solver.first_control=True
        else:
            aug_state = np.concatenate(
                [self.current_state, 
                [self.omega_w], 
                 self.last_control]
            )


            # no need to copy states and trajectory in case of update b/c node is single thread
            ref_segment, idx = get_reference_trajectory_segment(
                x0, v0, self.track, self.N+1, self.dt, 
                self.projidx, mu_est = mu_est)
            self.projidx = idx

            record_ref_trajectory = []
            if self.publish_trajectories:
                record_ref_trajectory = ref_segment.T
                # self.publish_ref_trajectory(ref_segment)
            #print(f"REF: {ref_segment}")

            self.lla_solver.prepare_mpc_solve()

            control_params = selected_model_params if self.adaptive_control else self.mean_params
            full_params = self.lla_solver.construct_params(
                control_params, ref_segment)
            
            mpc_solver, status = self.lla_solver.mpc_solve(aug_state, full_params)
            u_opt = mpc_solver.get(1, "x")[-2:] # pwm, delta
            d_ctrl = mpc_solver.get(0, "u")[:]

            self.lla_solver.current_mode = True


            print(f"CONTROL: {u_opt}")       
        
            if(self.publish_trajectories):
                for i in range(self.N + 1):
                    x_pred = mpc_solver.get(i, "x")[:]
                    mpc_states.append(x_pred[:6])
                    
                    c_pred = mpc_solver.get(i, "x")[-2:]
                    mpc_controls.append(c_pred)
                    # print(x_pred, c_pred)

                # print(f"PREDICTED STATES: {mpc_states}")
                # print(f"PREDICTED CONTROLS: {predicted_controls}")
                    
                self.publish_predicted_trajectory(mpc_states) # Publish predicted trajectory

            # residuals = mpc_solver.get_residuals()
            # res_eq = residuals[1]
            # vx = self.current_state[3]


        self.checkpoint[4] = time.perf_counter_ns()
        
        if(not ok_time):
            self.lla_reset_counter = 0

        if(self.lla_reset_interval != 0):
            self.lla_reset_counter = (self.lla_reset_counter + 1) % self.lla_reset_interval

        #########################################
        ### PUBLISH MPC DATA
        
        
        # eq_tol = 1e-2 if vx > 0.1 else 0.1   # much looser at low speed
        
        self.checkpoint[5] = time.perf_counter_ns()
        
        if status == 0 or (status == 2):  # Success
            # Get optimal control
            self.apply_control(u_opt, write_control = False) # Apply control
            if self.lla_solver.current_mode:
                self.dynamics_bank.update_known_params(self.omega_w)       
                self.lb_history.predict_states(
                    self.current_state, u_opt, self.lla_reset_counter == 0
                )
                
        else:
            print(f"\n--- SOLVER FAILED WITH STATUS {status} ---")
            self.lla_solver.print_mpc_failed()
            print("-----------------------------------\n")

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.cmd_pub.publish(drive_msg)
            
            self.last_drive_command = np.array([0.0, 0.0])
            self.last_control = np.array([0.0, 0.0])
            self.get_logger().warn(f"MPC solver failed with status: {status}")

            self.lla_solver.current_mode = False
            self.lla_solver.first_control = True
            

        self.checkpoint[6] = time.perf_counter_ns()
        self.count = (self.count + 1) % self.time_window
        self.time_history[:self.checkpoints-1, self.count] = np.array(self.checkpoint[1:]-self.checkpoint[:-1])
        self.time_history[-1, self.count] = (self.checkpoint[-1] - self.checkpoint[0])
    
        if(self.count == 0):
            print(np.max(self.time_history*1e-6, axis = 1))

        known_params = np.array(self.dynamics_bank.get_known_params())

        if self.log_data:
            self.lla_logger.log_lla_data(
                self.current_state,
                selected_model_params, 
                selected_model_index, 
                self.last_control,
                known_params, 
                self.last_drive_command,
                self.time_history[-1, self.count]*1e-6, 
                d_ctrl,
                mpc_states, 
                record_ref_trajectory,
                mu_est)

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

    def apply_control(self, u_opt, write_control = True):
        """Apply optimal control to the vehicle"""

        cur = float(u_opt[0])
        steer = float(u_opt[1])
        
        # Create Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"

        drive_msg.drive.speed = 0.0
        drive_msg.drive.jerk = 2.0 if self.lla_solver.current_mode else 1.0
        drive_msg.drive.acceleration = cur
        drive_msg.drive.steering_angle = steer

        self.cmd_pub.publish(drive_msg) 

        self.last_drive_command = np.array([cur, steer])
        if write_control:
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

        self.predicted_path_pub.publish(path_msg)

    def destroy_node(self):
        if(self.log_data):
            self.lla_logger.save_log()
            
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
