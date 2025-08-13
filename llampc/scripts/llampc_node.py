import numba
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from llampc.nmpc_gen import setup_mpc_from_json
from llampc.params import F110
from llampc.planner import get_reference_trajectory_segment
from llampc.utils import Track
from llampc.rollout import DynamicBank

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.declare_params()
        self.initialize_mpc()
       

        self.last_drive_command = np.array([0.0, 0.0]) #vx, steer
        self.last_control = np.array([0.0, 0.0]) #acceleration, steer
        self.rates = np.array([0.0, 0.0])
        self.first_control = False

        self.projidx = 0

        # dictionary, prefereably npy, which has waypoints_x, waypoints_y, and velocity
        track_name = self.get_parameter('track_file_name').get_parameter_value().string_value
        self.track = Track(track_name)
    

        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )


        self.predicted_path_pub = self.create_publisher(
            Path,
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

        self.control_timer = self.create_timer(0.01, self.control_callback) # run 100 hz

        self.get_logger().info("F1tenth MPC Initialized")

    def declare_params(self):
        self.declare_parameter('solver_config', 'default')
        self.declare_parameter('json_file', 'f1tenth_acados_ocp.json')
        self.declare_parameter('track_file_name', '')

        self.N = 20 #steps (from nmpc)
        self.Tf = 2.0 # total time horizon (from nmpc)
        self.dt = self.Tf / self.N
        self.params_car = F110()
        

    def initialize_mpc(self):
        # solver_config = self.get_parameter('solver_config').get_parameter_value().string_value
        # json_file = self.get_parameter('json_file').get_parameter_value().string_value

        # self.solver = setup_mpc_from_json(
        #     json_file=json_file,
        #     solver_config=solver_config,
        #     params_car=F110
        # )

        params_car = F110()

        variation_dict = {
            'Bf': .15,   # 15% variation
            'Br': .15,   # 15% variation
            'Cf': .15,   # 15% variation
            'Cr': .15,   # 15% variation
            'Df': .15,   # 15% variation
            'Dr': .15,   # 15% variation
            'Cro': 0.15, # 15% variation
            'Cd': 0.15,  # 15% variation
            'Ce': 0.15,  # 15% variation
            'Cm': 0.15,  # 15% variation
        }

        mean_dict = {
            'Bf': 20.0,
            'Br': 20.0,
            'Cf': 1.0,
            'Cr': 1.0,
            'Df': 0.8,
            'Dr': 0.8,
            'Cro': 0.02,
            'Cd': 0.001,
            'Ce': 1.0,
            'Cm': .05, 

        }

        cost_weights = np.array([1.0, 1.0, 0, 0, 0, 0]) # x, y, theta, vx, vy, omega
        self.bank = DynamicBank(
            params_car['lf'], params_car['lr'], 
            params_car['mass'], params_car['Iz'], 
            mean_dict, variation_dict, 
            2000,
            cost_weights
        )
,
        self.current_state = None
        self.solver = setup_mpc_from_json()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        phi = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self.current_state = np.array([x, y, phi, vx, vy, omega])
        
    def control_callback(self):
        if self.track is None or self.current_state is None:
            return
        self.bank.update_lookback_error(self.current_state)
       
        x0 = self.current_state[:2]
        v0 = self.current_state[3]

        # no need to copy states and trajectory in case of update b/c node is single thread
        ref_segment, idx = get_reference_trajectory_segment(x0, v0, self.track, self.N, self.dt, self.projidx)
        # ref_segment is only 2 large, representing X and Y
        self.projidx = idx

        # set initial locked current state
        self.solver.set(0, "lbx", np.concatenate([self.current_state, self.last_control]))
        self.solver.set(0, "ubx", np.concatenate([self.current_state, self.last_control]))
        
        # Set reference trajectory and previous control for all stages
        selected_model_index = self.bank.get_best_model()
        selected_model_params = self.bank.get_model_params_arr(selected_model_index)
        for i in range(self.N):
            # Combine tire parameters with reference state and previous control
            full_params = np.concatenate([
                selected_model_params,
                ref_segment[:, i], 
                np.zeros(6)
                ]

                 # concatenate 2 for x, y, 6 to fill out rest of state
                 # make sure to weight non-defined states as 0 cost
                )
            
            self.solver.set(i, "p", full_params)
            self.solver.set(i, "yref", np.concatenate([np.zeros(6), np.zeros(2), np.zeros(2)]))

        status = self.solver.solve()

        if status == 0:  # Success
            # Get optimal control
            u_opt = self.solver.get(0, "x")[-2:]
            
            # Get predicted trajectory for visualization
            predicted_states = []
            for i in range(self.N + 1):
                x_pred = self.solver.get(i, "x")[:3]
                predicted_states.append(x_pred)
            
            self.apply_control(u_opt) # Apply control
            self.bank.predict_states(self.current_state, u_opt)


            self.publish_predicted_trajectory(predicted_states) # Publish predicted trajectory
            self.publish_mpc_info(u_opt, status) # Publish MPC info
            
        else:
            self.get_logger().warn(f"MPC solver failed with status: {status}")



    def apply_control(self, u_opt):
        """Apply optimal control to the vehicle"""
        # u_opt = [steer, acceleration]
        acceleration = float(u_opt[0])
        steer = float(u_opt[1])
        
        # Create Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock.now()
        drive_msg.header.frame_id = "base_link"
        
        # Convert acceleration to speed command (simple integration)
        desired_speed = max(0.0, self.last_drive_command[0] + acceleration * self.dt)
        
        drive_msg.drive.speed = desired_speed
        drive_msg.drive.steering_angle = steer

        self.cmd_pub.publish(drive_msg) 

        self.last_drive_command = np.array([desired_speed, steer])
        self.last_control = np.array([acceleration, steer])
        

    def publish_predicted_trajectory(self, predicted_states):
        """Publish predicted trajectory for visualization"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for state in predicted_states:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(state[0])
            pose_stamped.pose.position.y = float(state[1])
            pose_stamped.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            yaw = float(state[2])
            pose_stamped.pose.orientation.w = np.cos(yaw / 2.0)
            pose_stamped.pose.orientation.z = np.sin(yaw / 2.0)
            
            path_msg.poses.append(pose_stamped)
        
        self.predicted_path_pub.publish(path_msg)
    
    def publish_mpc_info(self, u_opt, status):
        """Publish MPC solver information"""
        info_msg = Float64MultiArray()
        info_msg.data = [
            float(u_opt[1]),  # acceleration
            self.cur_drive_command[0], # target speed
            float(u_opt[0]),  # steer
            float(status),    # solver status
            float(self.solver.get_cost())  # optimal cost
        ]
        self.mpc_info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()