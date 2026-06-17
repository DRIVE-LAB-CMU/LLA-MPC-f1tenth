#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp

from llampc.cbf_gen import cbf_qp_pacejka

from llampc.nmpc_gen_pwm import setup_mpc
from llampc.params import F110, F110_sim, get_param_dict_random, get_param_dict_grid
from llampc.planner import get_reference_trajectory_segment, get_lookahead_point
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

# ANSI colour codes for terminal output.
_RED = "\033[91m"
_RESET = "\033[0m"


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
        self.declare_parameter('track_file_name', 'mocap_square2slow.npz')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('out_file', 'out')

        self.N = 15 #steps (for nmpc)
        self.Tf = 0.6 # total time horizon (for nmpc)
        self.dt = self.Tf / self.N
        self.control_callback_speed = 0.04
        self.lla_predict_horizon = 0.04
        self.lla_reset_interval = 0
        self.lla_reset_counter = 0
        self.r_car = 0.04

        self.min_pwm = 0.1
        self.max_pwm = 0.3
        self.params_car = F110()

        self.obstacles = [
            (np.array([1, -0.5]), 0.5),
            (np.array([-2.5, 0]), 0.5),
        ]


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
                "mpc_rollout":[],
                "ref_trajectory":[],
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
            'Bf': 6.5,
            'Br': 6.5,
            'Cf': 1.4,
            'Cr': 1.4,
            'Df': 17.0,
            'Dr': 17.0,
            'Cro': 0.0,
            'Cd': 0.0,
            'Ce': 10,
            'Cm': 0.0,
        }


        variation_dict = {
                'Bf': 5.5,   # 15% variation
                'Br': 5.5,   # 15% variation
                'Cf': 0.3,   # 15% variation
                'Cr': 0.3,   # 15% variation
                'Df': 15,   # 15% variation
                'Dr': 15,   # 15% variation
                'Cro': 0, # 15% variation
                'Cd': 0,  # assume negligible drag
                'Ce': 0,  # motor efficiency conversion should never be above 1
                'Cm': 0.0,  # motor speed saturation
            }

        cost_weights = np.array([1.0, 1.0, 0, 0, 0, 0]) # x, y, theta, vx, vy, omega

        # grid discretization
        discretization_dict = {
            'Bf': 4,   # 15% variation
            'Br': 4,   # 15% variation
            'Cf': 3,   # 15% variation
            'Cr': 3,   # 15% variation
            'Df': 4,   # 15% variation
            'Dr': 4,   # 15% variation
            'Cro':1, # 15% variation
            'Cd': 1,  # assume negligible drag
            'Ce': 1,  # motor efficiency conversion should never be above 1
            'Cm': 1,  # motor speed saturation
        }
        param_dict = get_param_dict_grid(mean_dict, variation_dict,
                                         discretization=discretization_dict, ground_truth=True,
                                         noadapt=False)
        num_models = len(param_dict['Bf'])

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

        history_length=40
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
        param_dict = get_param_dict_random(mean_dict, variation_dict, num_models, ground_truth=True)

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
        param_dict = get_param_dict_random(mean_dict, variation_dict, num_models)

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

        param_dict = get_param_dict_random(mean_dict, variation_dict, num_models)
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
        param_dict = get_param_dict_random(mean_dict, variation_dict, num_models)

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

        self.state_size = 6

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
        self.lb_history.predict_states(self.lb_history.last_predicted_states, np.zeros(2))
        self.lb_history.update_lookback_error(np.zeros(self.state_size))
        self.lb_history.get_best_model()
        self.lb_history.reset()
        self.get_logger().info("Bank initialized")

        # start solver
        self.current_state = None
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

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self.current_state = np.array([x, y, phi, vx, vy, omega])

    def check_obstacle_violation(self):
        """Return (violated, min_signed_clearance).

        Signed clearance = ||p - p_obs|| - (r_obs + r_car). This is the ACTUAL
        physical penetration of the car CENTER into each inflated keep-out
        disk at the current state -- independent of the CBF's predictive psi.
        Negative => the car is physically inside the keep-out zone right now.
        """
        px, py = self.current_state[0], self.current_state[1]
        min_clear = np.inf
        for p_obs, r_obs in self.obstacles:
            dist = np.hypot(px - p_obs[0], py - p_obs[1])
            clear = dist - (r_obs + self.r_car)
            min_clear = min(min_clear, clear)
        return (min_clear < 0.0), float(min_clear)

    # def cbf_rollout_policy(self, state):
    #     """Closed-loop control the CBF replays at each predicted state.

    #     Re-plans pure pursuit toward the raceline from the predicted state,
    #     so the CBF rollout is a genuine turn-then-straighten maneuver rather
    #     than a frozen arc. Returned in CBF [delta, d] order (pure_pursuit
    #     returns [pwm, steer] = [d, delta], so the two are swapped).

    #     projidx is passed read-only — get_lookahead_point's returned index is
    #     intentionally discarded so the rollout never mutates the node's
    #     progress along the track.
    #     """
    #     ref_pt, _ = get_lookahead_point(
    #         state, self.track, self.projidx, lookahead_dist=0.7
    #     )
    #     u_pp = self.pure_pursuit_control(state, ref_pt)   # [pwm, steer]
    #     return np.array([u_pp[1], u_pp[0]])               # [delta, d]

    def pure_pursuit_control(self, state, ref_point):
        x, y, psi = state[0], state[1], state[2]

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

        # velocity-scaled pwm: slow down on sharp turns
        scale = self.max_pwm - self.min_pwm
        pwm   = float(self.min_pwm + scale * np.sqrt(1.0 - abs(steer / self.params_car['max_steer'])))

        return np.array([pwm, steer])

    def control_callback(self):
        self.checkpoint[0] = time.perf_counter_ns()

        if self.track is None or self.current_state is None:
            return

        ##############################################
        ### BANK UPDATE
        one_step_cost = None
        ok_time = True

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

        ref_point, idx = get_lookahead_point(self.current_state, self.track, self.projidx, lookahead_dist = 1.2)
        self.projidx = idx
        print(f"IDX: {self.projidx}")

        record_ref_trajectory = [ref_point]
        if self.publish_trajectories:
            self.publish_ref_trajectory(ref_point.reshape(-1, 1))

        self.checkpoint[2] = time.perf_counter_ns()

        selected_model_params = None
        if self.sim:
            selected_model_index = -1
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

        self.checkpoint[3]= time.perf_counter_ns()

        u_opt = self.pure_pursuit_control(self.current_state, ref_point)


        self.checkpoint[4] = time.perf_counter_ns()

        u_nom_cbf = np.array([u_opt[1], u_opt[0]])  # [steer, pwm]

        # Pull the current best Pacejka params (first 6: Bf,Cf,Df,Br,Cr,Dr)
        theta_hat = selected_model_params[:6]
        u_safe, cbf_info = cbf_qp_pacejka(
            x        = self.current_state,   # [px, py, psi, vx, vy, omega]
            u_nom    = u_nom_cbf,
            lla_params = selected_model_params[:10],
            known_params = self.dynamics_bank.get_known_params(),
            obstacles= self.obstacles,
            r_car    = self.r_car,
            dt       = self.dt,
            alpha    = 1,
            N        = 1,
            # Match the CBF actuator bounds to THIS car's real command ranges.
            # The d-channel here carries pwm duty, not a normalized [-1, 1]
            # throttle, so the [-1, 1] module defaults would let the QP drive
            # pwm negative when braking near an obstacle.  Pin to the pure-
            # pursuit pwm band so the filter can only stay in forward duty.
            delta_max = self.params_car['max_steer'],
            d_min     = 0.1,
            d_max     = self.max_pwm,
            w_delta=0.1, 
            w_d=1/0.35,
            eps_fd=(0.05, 1e-3),
            # Closed-loop rollout: the CBF applies u_nom only on the first
            # step and re-evaluates pure pursuit at every later predicted
            # state, so steering has real authority over psi (a frozen-u
            # rollout gave delta a near-zero gradient and the filter could
            # only brake).
            # policy    = self.cbf_rollout_policy,
        )


        if cbf_info['active']:
            self.get_logger().warn(
                f"CBF active! psi={cbf_info['psi']:.4f}, correction={u_safe - u_nom_cbf}"
            )

        # Physical-violation check at the CURRENT state (independent of the
        # CBF's predictive psi). Print in RED whenever the car center is
        # inside an inflated keep-out disk -- this is the "ran through the
        # obstacle" event itself, so it tells us directly whether the filter
        # actually kept the car out rather than just whether psi went active.
        violated, clearance = self.check_obstacle_violation()
        if violated:
            print(f"{_RED}OBSTACLE VIOLATED! penetration={-clearance:.3f} m "
                  f"(clearance={clearance:.3f}) pos=({self.current_state[0]:.3f}, "
                  f"{self.current_state[1]:.3f}) psi={cbf_info['psi']:.4f} "
                  f"grad_psi={cbf_info['grad_psi']}{_RESET}", flush=True)

        # Swap back to [pwm, steer]
        u_safe = np.array([u_safe[1], u_safe[0]])

        # Log the CBF's predictive rollout (the trajectory it reasoned over)
        # so it can be replayed in the visualizer. rollout is (N+1, 6) with
        # row 0 = current state, rows 1..N = predicted states. Drop row 0 so
        # only the lookahead is shown.
        mpc_states = []
        mpc_controls = []
        cbf_rollout = cbf_info.get('rollout', None)
        if cbf_rollout is not None:
            mpc_states = [np.asarray(s, dtype=float) for s in cbf_rollout[1:]]

        self.log_lla_data(selected_model_params, selected_model_index, mpc_states, record_ref_trajectory)


        if(not ok_time):
            self.lla_reset_counter = 0

        if(self.lla_reset_interval != 0):
            self.lla_reset_counter = (self.lla_reset_counter + 1) % self.lla_reset_interval

        #########################################
        ### PUBLISH MPC DATA
        self.apply_control(u_safe) # Apply control
        if not self.sim:
            #version for our dynamics
            self.checkpoint[5] = time.perf_counter_ns()
            # Feed the bank the control we ACTUALLY applied (post-CBF), not the
            # pre-filter pure-pursuit command.  u_safe is [pwm, steer]; the bank
            # predict_states expects the same [pwm, steer] order as u_opt.
            self.lb_history.predict_states(
                self.current_state, u_safe, self.lla_reset_counter == 0
            )
            self.checkpoint[6] = time.perf_counter_ns()

        self.count = (self.count + 1) % self.time_window
        self.time_history[:self.checkpoints-1, self.count] = np.array(self.checkpoint[1:]-self.checkpoint[:-1])
        self.time_history[-1, self.count] = (self.checkpoint[-1] - self.checkpoint[0])


    def publish_ref_trajectory(self, ref_trajectory):
        ref_msg = PoseArray()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.header.frame_id = "map"

        for x, y in (ref_trajectory[:2].T):
            point = Pose()
            point.position.x = x
            point.position.y = y
            ref_msg.poses.append(point)

        self.ref_pub.publish(ref_msg)


    def apply_control(self, u_opt):
        """Apply optimal control to the vehicle"""
        accel = float(u_opt[0])
        pwm = float(u_opt[0])
        steer = float(u_opt[1])

        # Create Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"

        new_int = self.current_state[3] + accel * self.dt

        drive_msg.drive.speed = new_int
        drive_msg.drive.jerk = 1.0
        drive_msg.drive.acceleration = pwm
        drive_msg.drive.steering_angle = steer

        self.cmd_pub.publish(drive_msg)

        self.last_drive_command = np.array([pwm, steer])
        self.last_control = np.array([pwm, steer])


    def publish_predicted_trajectory(self, predicted_states):
        """Publish predicted trajectory for visualization"""

        path_msg = PoseArray()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for state in predicted_states:
            pose_unstamped = Pose()
            pose_unstamped.position.x = float(state[0])
            pose_unstamped.position.y = float(state[1])

            yaw = float(state[2])
            pose_unstamped.orientation.w = np.cos(yaw / 2.0)
            pose_unstamped.orientation.z = np.sin(yaw / 2.0)

            path_msg.poses.append(pose_unstamped)

        self.predicted_path_pub.publish(path_msg)

    def publish_mpc_info(self, u_opt, status):
        """Publish MPC solver information"""
        info_msg = Float64MultiArray()
        info_msg.data = [
            float(u_opt[1]),  # acceleration
            float(u_opt[0]),  # steer
            float(status),    # solver status
            # float(self.solver.get_cost())  # optimal cost
        ]
        self.mpc_info_pub.publish(info_msg)

    def log_lla_data(self, params, model_index, mpc_rollout = [], ref_trajectory = []):
        if(self.log_data):
            now_ns = time.perf_counter_ns()
            self.log_buffer["time"].append(now_ns)
            self.log_buffer["state"].append(self.current_state.copy())
            self.log_buffer["params"].append(params.copy())
            self.log_buffer["model_idx"].append(model_index)
            self.log_buffer["ctrl"].append(self.last_control.copy())
            self.log_buffer["cmd"].append(self.last_drive_command.copy())
            self.log_buffer["mpc_rollout"].append(np.array(mpc_rollout))
            self.log_buffer["ref_trajectory"].append(np.array(ref_trajectory))

    def log_rollout_data(self, lb_history, one_step_cost, ok_time):
        if(self.log_data):
            self.log_buffer["ok_time"].append(ok_time)

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
                mpc_rollout=np.array(self.log_buffer["mpc_rollout"]),
                ref_trajectory=np.array(self.log_buffer["ref_trajectory"]),
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
