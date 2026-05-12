#!/usr/bin/env python3
import numpy as np
import time
import sys, os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from tf2_ros import TransformBroadcaster

sys.path.append(os.path.join(os.path.dirname(__file__)))


def quat_to_rot(q):
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def rot_to_quat(R):
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s

        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s

        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

    return np.array([x, y, z, w])

class OptitrackSubscriber(Node):
    def __init__(self, velocity_filter_alpha=0.5, history_size=5):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('optitrack_bridge_sub')
        
        self.declare_parameter('topic', '/vrpn_mocap/f1tenth/pose')
        topic = self.get_parameter('topic').get_parameter_value().string_value

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.get_logger().info(f'Subscribing to {topic} with BEST_EFFORT reliability')
        self.suber = self.create_subscription(
            PoseStamped,
            topic,
            self.topic_callback,
            qos)
            
        # ==========================================================
        # NEW: Publisher for the EKF (PoseWithCovarianceStamped)
        # ==========================================================
        self.ekf_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/optitrack/pose_cov', 
            10
        )
        
        self.optitrack_position = np.zeros(3)
        self.optitrack_quaternion = np.zeros(4)
        self.optitrack_linear_velocity_world = np.zeros(3)
        self.optitrack_linear_velocity = np.zeros(3)
        self.optitrack_angular_velocity_world = np.zeros(3)
        self.optitrack_angular_velocity = np.zeros(3)
        
        self.velocity_filter_alpha = velocity_filter_alpha
        self.history_size = history_size
        
        self.position_history = []
        self.quaternion_history = []
        self.timestamp_history = []
        
        self.br = TransformBroadcaster(self)

    def topic_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        position = np.array([msg.pose.position.z, msg.pose.position.x, msg.pose.position.y])
        
        q_orig = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        # Convert quaternion -> rotation matrix
        R_orig = quat_to_rot(q_orig)

        # Axis permutation matrix
        P = np.array([
            [0, 0, 1],  # x_new = z_old
            [1, 0, 0],  # y_new = x_old
            [0, 1, 0]   # z_new = y_old
        ])

        # Transform rotation matrix
        R_new = P @ R_orig @ P.T

        # Convert back to quaternion
        quaternion = rot_to_quat(R_new)
        
        self.optitrack_position = position
        self.optitrack_quaternion = quaternion
        
        self.position_history.append(position)
        self.quaternion_history.append(quaternion)
        self.timestamp_history.append(timestamp)
        
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
            self.quaternion_history.pop(0)
            self.timestamp_history.pop(0)
        
        if len(self.position_history) >= 2:
            self.calculate_velocities()
            
        # ==========================================================
        # NEW: Convert and Publish PoseWithCovarianceStamped for EKF
        # ==========================================================
        ekf_msg = PoseWithCovarianceStamped()
        ekf_msg.header = msg.header
        ekf_msg.header.frame_id = 'map' # Standard frame for absolute pose
        ekf_msg.pose.pose.position.x = position[0]
        ekf_msg.pose.pose.position.y = position[1]
        ekf_msg.pose.pose.position.z = position[2]
        ekf_msg.pose.pose.orientation.x = quaternion[0]
        ekf_msg.pose.pose.orientation.y = quaternion[1]
        ekf_msg.pose.pose.orientation.z = quaternion[2]
        ekf_msg.pose.pose.orientation.w = quaternion[3]
        
        # Create a 6x6 covariance matrix (x, y, z, roll, pitch, yaw)
        # We give it a very small variance (1e-4) because Optitrack is highly accurate
        cov = np.zeros(36)
        cov[0]  = 1e-4 # X variance
        cov[7]  = 1e-4 # Y variance
        cov[14] = 1e-4 # Z variance
        cov[21] = 1e-4 # Roll variance
        cov[28] = 1e-4 # Pitch variance
        cov[35] = 1e-4 # Yaw variance
        ekf_msg.pose.covariance = cov.tolist()
        
        self.ekf_pose_pub.publish(ekf_msg)
        # ==========================================================
            
        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map' # Match the EKF world_frame
        t.child_frame_id = 'base_link'
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        self.br.sendTransform(t)
            
        self.get_logger().info(
            f"pos=({self.optitrack_position[0]:.3f}, {self.optitrack_position[1]:.3f}, {self.optitrack_position[2]:.3f})"
        )

    def calculate_velocities(self):
        p1, p0 = self.position_history[-2], self.position_history[-1]
        t1, t0 = self.timestamp_history[-2], self.timestamp_history[-1]
        
        dt = t0 - t1
        if dt > 0:
            raw_linear_vel = (p0 - p1) / dt
            self.optitrack_linear_velocity_world = raw_linear_vel.copy()
            self.optitrack_linear_velocity = raw_linear_vel 
        
        q1, q0 = self.quaternion_history[-2], self.quaternion_history[-1] 
        
        if dt > 0:
            q1_scipy = np.array([q1[3], q1[0], q1[1], q1[2]])
            q0_scipy = np.array([q0[3], q0[0], q0[1], q0[2]])
            
            q1_conj = q1_scipy.copy()
            q1_conj[1:] = -q1_conj[1:]
            
            q_diff = self.quaternion_multiply(q0_scipy, q1_conj)
            raw_angular_vel = 2.0 * q_diff[1:] / dt
            
            self.optitrack_angular_velocity_world = raw_angular_vel
            self.optitrack_angular_velocity = raw_angular_vel 
            
    def get_optitrack_angular_velocity_world(self): return self.optitrack_angular_velocity_world
    def get_optitrack_angular_velocity(self): return self.optitrack_angular_velocity
    def get_optitrack_linear_velocity_world(self): return self.optitrack_linear_velocity_world
    def get_optitrack_position(self): return self.optitrack_position
    def get_optitrack_quaternion(self): return self.optitrack_quaternion
    def get_optitrack_linear_velocity(self): return self.optitrack_linear_velocity
    
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])
    
    
    

def main(args=None):
    rclpy.init(args=args)
    node = OptitrackSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()