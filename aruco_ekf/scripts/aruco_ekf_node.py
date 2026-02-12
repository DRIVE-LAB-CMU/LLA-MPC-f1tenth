#!/usr/bin/env python3
"""
ArUco EKF Fusion Node (ROS2)

This node fuses ArUco marker detections with wheel odometry using an EKF.
Updated for: aruco_opencv_msgs/msg/ArucoDetection

Subscribed Topics:
    /aruco_detections (aruco_opencv_msgs/msg/ArucoDetection): Marker poses
    /odom (nav_msgs/Odometry): Wheel odometry

Published Topics:
    /odom/filtered (nav_msgs/Odometry): Filtered odometry estimate
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import math
from collections import defaultdict

# ROS Messages
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
# UDPATE: Importing the correct message type
from aruco_opencv_msgs.msg import ArucoDetection

# TF2
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs

class ArucoEKF(Node):
    """Extended Kalman Filter for fusing ArUco detections with odometry."""
    
    def __init__(self):
        super().__init__('aruco_ekf_node', 
                         allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Frame definitions
        self.base_frame = 'base_link'
        
        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.state = np.zeros(12)
        
        # State covariance matrix (P)
        self.P = np.eye(12) * 0.1
        
        # Process noise covariance (Q)
        self.Q = np.diag([
            0.01, 0.01, 0.02,     # Position
            0.02, 0.02, 0.01,     # Orientation
            0.1, 0.1, 0.2,        # Linear Vel
            0.2, 0.2, 0.1         # Angular Vel
        ])
        
        # Measurement noise covariance for Odometry (R_odom)
        self.R_odom = np.diag([
            0.1, 0.1, 0.15,       # Position
            0.15, 0.15, 0.1,      # Orientation
            0.05, 0.05, 0.1,      # Linear Vel
            0.1, 0.1, 0.05        # Angular Vel
        ])
        
        # Measurement noise covariance for ArUco (R_marker)
        self.R_marker = np.diag([
            0.05, 0.05, 0.05,     # Position
            0.05, 0.05, 0.05      # Orientation
        ])
        
        # Load Known Marker positions
        self.marker_map = self._load_marker_map()
        
        self.last_odom_time = None
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )
        
        # UPDATE: Subscribing to the correct topic type
        self.aruco_sub = self.create_subscription(
            ArucoDetection,
            '/aruco_detections', # Common topic name for this msg type
            self.aruco_callback,
            qos_profile
        )
        
        # Publisher
        self.filtered_odom_pub = self.create_publisher(
            Odometry,
            '/odom/filtered',
            qos_profile
        )
        
        self.get_logger().info("ArUco EKF node initialized")
        if not self.marker_map:
            self.get_logger().warn("No marker map loaded! Use params 'marker_map.<id>.x' etc.")

    def _load_marker_map(self):
        """
        Load Marker positions from parameter server.
        Expects params like: marker_map.0.x, marker_map.0.y, etc.
        """
        marker_map = {}
        # Get all params starting with "marker_map"
        # Note: In ROS2, declaring nested params can be tricky. 
        # Ideally, load a .yaml file. This is a manual parser.
        params = self.get_parameters_by_prefix('marker_map')
        
        # Helper to group "0.x", "0.y" into {0: {'x':...}}
        parsed_markers = defaultdict(dict)
        
        for key, param in params.items():
            try:
                parts = key.split('.') # e.g. "0.x"
                if len(parts) < 2: continue
                
                id_str = parts[0]
                field = parts[1]
                parsed_markers[id_str][field] = param.value
            except Exception:
                pass

        for id_str, pose in parsed_markers.items():
            try:
                mid = int(id_str)
                x = float(pose.get('x', 0.0))
                y = float(pose.get('y', 0.0))
                z = float(pose.get('z', 0.0))
                roll = float(pose.get('roll', 0.0))
                pitch = float(pose.get('pitch', 0.0))
                yaw = float(pose.get('yaw', pose.get('theta', 0.0)))
                
                marker_map[mid] = np.array([x, y, z, roll, pitch, yaw])
                self.get_logger().info(f"Loaded marker {mid}: {marker_map[mid]}")
            except ValueError:
                self.get_logger().error(f"Invalid marker ID: {id_str}")

        return marker_map

    def predict(self, dt):
        """Prediction step (Constant Velocity Model)."""
        x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz = self.state
        
        # Rotation matrix (Body -> World)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Linear velocity in world frame
        v_world = R @ np.array([vx, vy, vz])
        
        # State Prop
        x_pred = x + v_world[0] * dt
        y_pred = y + v_world[1] * dt
        z_pred = z + v_world[2] * dt
        
        roll_pred = self._normalize_angle(roll + wx * dt)
        pitch_pred = self._normalize_angle(pitch + wy * dt)
        yaw_pred = self._normalize_angle(yaw + wz * dt)
        
        self.state = np.array([
            x_pred, y_pred, z_pred, 
            roll_pred, pitch_pred, yaw_pred,
            vx, vy, vz, 
            wx, wy, wz
        ])
        
        # Simplified Jacobian
        F = np.eye(12)
        F[0:3, 6:9] = R * dt # dx/dv
        F[3:6, 9:12] = np.eye(3) * dt # dtheta/domega
        
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_odometry(self, odom_msg):
        """Update step for Odometry."""
        pos = odom_msg.pose.pose.position
        orient = odom_msg.pose.pose.orientation
        twist = odom_msg.twist.twist
        
        roll, pitch, yaw = self.get_euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        
        z = np.array([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw,
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z
        ])
        
        H = np.eye(12)
        y = z - self.state
        
        # Normalize angles
        y[3] = self._normalize_angle(y[3])
        y[4] = self._normalize_angle(y[4])
        y[5] = self._normalize_angle(y[5])
        
        S = H @ self.P @ H.T + self.R_odom
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.state[3] = self._normalize_angle(self.state[3])
        self.state[4] = self._normalize_angle(self.state[4])
        self.state[5] = self._normalize_angle(self.state[5])
        
        self.P = (np.eye(12) - K @ H) @ self.P

    def update_marker(self, marker_id, marker_pose_robot, distance=None):
        """Update step for Marker Detection (6-DOF)."""
        if marker_id not in self.marker_map:
            # Throttle warning to avoid log spam
            # self.get_logger().warn(f"Skipping unknown marker {marker_id}")
            return
        
        marker_world = self.marker_map[marker_id]
        x, y, z, roll, pitch, yaw = self.state[0:6]
        
        # R_world_robot (Transpose of Body->World)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R_world_robot = np.array([
            [cy*cp, sy*cp, -sp],
            [cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
            [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
        ])
        
        # Expected Measurement
        delta_world = marker_world[0:3] - np.array([x, y, z])
        delta_robot = R_world_robot @ delta_world
        
        expected_roll = self._normalize_angle(marker_world[3] - roll)
        expected_pitch = self._normalize_angle(marker_world[4] - pitch)
        expected_yaw = self._normalize_angle(marker_world[5] - yaw)
        
        z_expected = np.array([
            delta_robot[0], delta_robot[1], delta_robot[2],
            expected_roll, expected_pitch, expected_yaw
        ])
        
        # Jacobian H
        H = np.zeros((6, 12))
        H[0:3, 0:3] = -R_world_robot 
        H[3, 3] = -1; H[4, 4] = -1; H[5, 5] = -1
        
        # Adaptive Noise
        R_curr = self.R_marker.copy()
        if distance:
            factor = max(1.0, (distance / 2.0)**2)
            R_curr *= factor

        # EKF Update
        y_innovation = marker_pose_robot - z_expected
        y_innovation[3] = self._normalize_angle(y_innovation[3])
        y_innovation[4] = self._normalize_angle(y_innovation[4])
        y_innovation[5] = self._normalize_angle(y_innovation[5])
        
        S = H @ self.P @ H.T + R_curr
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y_innovation
        self.state[3] = self._normalize_angle(self.state[3])
        self.state[4] = self._normalize_angle(self.state[4])
        self.state[5] = self._normalize_angle(self.state[5])
        
        self.P = (np.eye(12) - K @ H) @ self.P
        
        self.get_logger().info(f"Corrected with Marker {marker_id}", throttle_duration_sec=2.0)

    def odom_callback(self, msg):
        current_time = self.get_clock().now()
        if self.last_odom_time is not None:
            dt = (current_time - self.last_odom_time).nanoseconds / 1e9
            if dt > 0:
                self.predict(dt)
        
        self.update_odometry(msg)
        self.last_odom_time = current_time
        self.publish_filtered_odom(msg.header.stamp, msg.header.frame_id, msg.child_frame_id)

    def aruco_callback(self, msg):
        """
        Handles aruco_opencv_msgs/msg/ArucoDetection.
        Field structure assumed:
          msg.header (std_msgs/Header)
          msg.markers (List of MarkerPose)
             marker.marker_id (int)
             marker.pose (geometry_msgs/Pose)
        """
        if not msg.markers:
            return

        for marker in msg.markers:
            marker_id = marker.marker_id
            pose = marker.pose
            
            # Wrap in PoseStamped for TF2
            pose_stamped_cam = PoseStamped()
            pose_stamped_cam.header = msg.header # Use the header from the main message
            pose_stamped_cam.pose = pose
            
            try:
                # Transform Camera Frame -> Base Frame (Robot Center)
                if not self.tf_buffer.can_transform(self.base_frame, msg.header.frame_id, rclpy.time.Time()):
                    return

                pose_stamped_base = self.tf_buffer.transform(
                    pose_stamped_cam, 
                    self.base_frame,
                    timeout=Duration(seconds=0.1)
                )
                
                # Extract data for EKF
                p = pose_stamped_base.pose.position
                o = pose_stamped_base.pose.orientation
                
                r_b, p_b, y_b = self.get_euler_from_quaternion(o.x, o.y, o.z, o.w)
                
                # This is the measured pose of the marker relative to the robot base
                marker_pose_robot = np.array([p.x, p.y, p.z, r_b, p_b, y_b])
                
                # Distance for adaptive noise
                distance = np.sqrt(p.x**2 + p.y**2 + p.z**2)
                
                self.update_marker(marker_id, marker_pose_robot, distance)

            except Exception as e:
                self.get_logger().debug(f"TF Transform failed: {e}")

    def publish_filtered_odom(self, stamp, frame_id, child_frame_id):
        """Publish the filtered odometry estimate."""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = child_frame_id
        
        # Position
        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.position.z = self.state[2]
        
        # Orientation
        roll, pitch, yaw = self.state[3:6]
        q = self.get_quaternion_from_euler(roll, pitch, yaw)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        
        # Twist
        odom_msg.twist.twist.linear.x = self.state[6]
        odom_msg.twist.twist.linear.y = self.state[7]
        odom_msg.twist.twist.linear.z = self.state[8]
        odom_msg.twist.twist.angular.x = self.state[9]
        odom_msg.twist.twist.angular.y = self.state[10]
        odom_msg.twist.twist.angular.z = self.state[11]
        
        # Populate Covariance (just a diagonal approximation for visualization)
        # Note: A real implementation would map the 12x12 P matrix to the 6x6 ROS covariances
        odom_msg.pose.covariance[0] = self.P[0,0]
        odom_msg.pose.covariance[7] = self.P[1,1]
        odom_msg.pose.covariance[35] = self.P[5,5]
        
        self.filtered_odom_pub.publish(odom_msg)

    @staticmethod
    def _normalize_angle(angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    @staticmethod
    def get_quaternion_from_euler(roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    @staticmethod
    def get_euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z

def main(args=None):
    rclpy.init(args=args)
    node = ArucoEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()