#!/usr/bin/env python3
"""
AprilTag EKF Fusion Node (ROS2)

This node fuses AprilTag detections with wheel odometry using an Extended Kalman Filter
to provide improved pose estimates.

This version computes 3D poses from 2D pixel detections using camera intrinsics.

Subscribed Topics:
    /apriltag/detections (apriltag_msgs/AprilTagDetectionArray): AprilTag 2D detections
    /camera_info (sensor_msgs/CameraInfo): Camera intrinsic parameters
    /odom (nav_msgs/Odometry): Wheel odometry

Published Topics:
    /odom/filtered (nav_msgs/Odometry): Filtered odometry estimate

The state vector is [x, y, theta, vx, vy, omega]
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, PoseStamped
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
from collections import defaultdict
import math

class AprilTagEKF(Node):
    """Extended Kalman Filter for fusing AprilTag detections with odometry.
    
    This version supports full 6-DOF state estimation and tag poses.
    State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    """
    
    def __init__(self):
        super().__init__('apriltag_ekf_node', 
                         allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Frame definitions
        self.base_frame = 'base_link'      # Robot center
        self.camera_frame = 'camera_link'  # Camera optical frame
        self.camera_to_base_transform = None # Will store the 4x4 matrix
        
        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        # Position (m), Orientation (rad), Linear velocity (m/s), Angular velocity (rad/s)
        self.state = np.zeros(12)
        
        # State covariance matrix
        self.P = np.eye(12) * 0.1
        
        # Process noise covariance
        # Larger noise for z, roll, pitch if primarily 2D motion
        self.Q = np.diag([
            0.01, 0.01, 0.02,     # x, y, z position noise
            0.02, 0.02, 0.01,     # roll, pitch, yaw orientation noise
            0.1, 0.1, 0.2,        # vx, vy, vz velocity noise
            0.2, 0.2, 0.1         # wx, wy, wz angular velocity noise
        ])
        
        # Measurement noise covariance for odometry (12 states)
        self.R_odom = np.diag([
            0.1, 0.1, 0.15,       # x, y, z position
            0.15, 0.15, 0.1,      # roll, pitch, yaw
            0.05, 0.05, 0.1,      # vx, vy, vz
            0.1, 0.1, 0.05        # wx, wy, wz
        ])
        
        # Measurement noise covariance for AprilTag (6-DOF pose)
        self.R_tag = np.diag([
            0.05, 0.05, 0.05,     # x, y, z position
            0.05, 0.05, 0.05      # roll, pitch, yaw orientation
        ])
        
        # Declare parameters
        self.declare_parameter('tag_size', 0.16) 
        self.declare_parameter('tag_sizes', {})
        
        # Get tag size parameter
        self.default_tag_size = self.get_parameter('tag_size').value
        self.tag_sizes = self.get_parameter('tag_sizes').value
        
        # Known AprilTag positions in world frame (tag_id: [x, y, theta])
        self.tag_map = self._load_tag_map()
        
        # Camera intrinsics (will be populated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # Timing
        self.last_time = self.get_clock().now()
        self.last_odom_time = None
        
        # QoS profile for reliable communication
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
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            qos_profile
        )
        
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/apriltag/detections',
            self.tag_callback,
            qos_profile
        )
        
        # Publisher
        self.filtered_odom_pub = self.create_publisher(
            Odometry,
            '/odom/filtered',
            qos_profile
        )
        
        # Timer for periodic publishing (optional, currently using event-driven)
        
        self.get_logger().info("AprilTag EKF node initialized")
        self.get_logger().info(f"Default tag size: {self.default_tag_size}m")
        self.get_logger().info("Waiting for camera_info...")
    
    def _load_tag_map(self):
        """
        Load AprilTag positions from parameter server or use defaults.
        
        Returns:
            dict: Mapping of tag_id to [x, y, z, roll, pitch, yaw] in world frame
        """
        tag_map = {}

        params = self.get_parameters_by_prefix('tag_positions')
        
        if not params:
            self.get_logger().warn("No tag positions found (checked prefix 'tag_positions')")
            return tag_map

        # Helper to group the flat params back into objects
        parsed_tags = defaultdict(dict)
        
        for key, param in params.items():
            # key is likely "0.x", "0.y", "1.roll", etc.
            try:
                # Split "0.x" into tag_id="0", field="x"
                parts = key.split('.')
                if len(parts) < 2: continue
                
                tag_id = parts[0]
                field = parts[1]
                
                # Store the value
                parsed_tags[tag_id][field] = param.value
            except Exception as e:
                self.get_logger().warn(f"Failed to parse param {key}: {e}")

        # Now convert the temp dict to your numpy structure
        for tag_id_str, pose in parsed_tags.items():
            try:
                tag_id = int(tag_id_str)
                x = float(pose.get('x', 0.0))
                y = float(pose.get('y', 0.0))
                z = float(pose.get('z', 0.0))
                roll = float(pose.get('roll', 0.0))
                pitch = float(pose.get('pitch', 0.0))
                # Support 'yaw' or 'theta'
                yaw = float(pose.get('yaw', pose.get('theta', 0.0)))
                
                tag_map[tag_id] = np.array([x, y, z, roll, pitch, yaw])
                self.get_logger().info(f"Loaded tag {tag_id}: {tag_map[tag_id]}")
            except ValueError:
                self.get_logger().error(f"Invalid tag ID: {tag_id_str}")

        return tag_map
    
    def camera_info_callback(self, msg):
        """
        Callback for camera info messages.
        Extract camera intrinsics matrix and distortion coefficients.
        
        Args:
            msg (CameraInfo): Camera info message
        """
        if not self.camera_info_received:
            # Camera intrinsics matrix K
            # [fx  0  cx]
            # [ 0 fy  cy]
            # [ 0  0   1]
            self.camera_matrix = np.array([
                [msg.k[0], msg.k[1], msg.k[2]],
                [msg.k[3], msg.k[4], msg.k[5]],
                [msg.k[6], msg.k[7], msg.k[8]]
            ])
            
            # Distortion coefficients
            self.dist_coeffs = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)
            
            self.camera_info_received = True
            self.get_logger().info("Camera intrinsics received")
            self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
    
    def get_tag_size(self, tag_id):
        """
        Get the size of a specific tag.
        
        Args:
            tag_id (int): Tag ID
            
        Returns:
            float: Tag size in meters
        """
        # Check if this tag has a custom size
        if self.tag_sizes and str(tag_id) in self.tag_sizes:
            return self.tag_sizes[str(tag_id)]
        return self.default_tag_size
    
    def pose_from_homography(self, homography, tag_size):
        """
        Compute 3D pose from homography matrix.
        
        This decomposes the homography to extract rotation and translation
        of the tag relative to the camera.
        
        Args:
            homography (np.array): 3x3 homography matrix
            tag_size (float): Physical size of the tag in meters
            
        Returns:
            tuple: (x, y, z, roll, pitch, yaw) or None if decomposition fails
        """
        if self.camera_matrix is None:
            return None
        
        # Normalize the homography
        h = homography / homography[2, 2]
        
        # Get camera intrinsics
        K = self.camera_matrix
        K_inv = np.linalg.inv(K)
        
        # Compute normalized homography
        H_norm = K_inv @ h
        
        # Extract the first two columns
        h1 = H_norm[:, 0]
        h2 = H_norm[:, 1]
        h3 = H_norm[:, 2]
        
        # Compute the scale factor
        lambda1 = 1.0 / np.linalg.norm(h1)
        lambda2 = 1.0 / np.linalg.norm(h2)
        lambda_avg = (lambda1 + lambda2) / 2.0
        
        # Compute rotation matrix columns
        r1 = lambda_avg * h1
        r2 = lambda_avg * h2
        r3 = np.cross(r1, r2)
        
        # Build rotation matrix
        R = np.column_stack([r1, r2, r3])
        
        # Ensure R is a proper rotation matrix using SVD
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Compute translation (scaled by tag size)
        t = lambda_avg * h3 * tag_size / 2.0
        
        # Convert rotation matrix to Euler angles
        # Using OpenCV's convention (same as tf)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        # Return pose as (x, y, z, roll, pitch, yaw)
        return (t[0], t[1], t[2], roll, pitch, yaw)
    
    def predict(self, dt):
        """
        Prediction step of the EKF using motion model.
        
        Args:
            dt (float): Time step in seconds
        """
        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz = self.state
        
        # Motion model (constant velocity in body frame)
        # Convert body frame velocities to world frame
        
        # Rotation matrix from body to world frame
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Full 3D rotation matrix (ZYX Euler angles)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Linear velocity in world frame
        v_world = R @ np.array([vx, vy, vz])
        
        # Predicted position
        x_pred = x + v_world[0] * dt
        y_pred = y + v_world[1] * dt
        z_pred = z + v_world[2] * dt
        
        # Predicted orientation (simplified - assuming small angular velocities)
        # For more accuracy, could use quaternion integration
        roll_pred = roll + wx * dt
        pitch_pred = pitch + wy * dt
        yaw_pred = yaw + wz * dt
        
        # Normalize angles
        roll_pred = self._normalize_angle(roll_pred)
        pitch_pred = self._normalize_angle(pitch_pred)
        yaw_pred = self._normalize_angle(yaw_pred)
        
        self.state = np.array([
            x_pred, y_pred, z_pred, 
            roll_pred, pitch_pred, yaw_pred,
            vx, vy, vz, 
            wx, wy, wz
        ])
        
        # Jacobian of motion model (simplified for constant velocity model)
        F = np.eye(12)
        
        # Position depends on orientation and velocity
        dR_droll = np.array([
            [0, cy*sp*cr + sy*sr, -cy*sp*sr + sy*cr],
            [0, sy*sp*cr - cy*sr, -sy*sp*sr - cy*cr],
            [0, cp*cr, -cp*sr]
        ])
        dR_dpitch = np.array([
            [-cy*sp, cy*cp*sr, cy*cp*cr],
            [-sy*sp, sy*cp*sr, sy*cp*cr],
            [-cp, -sp*sr, -sp*cr]
        ])
        dR_dyaw = np.array([
            [-sy*cp, -sy*sp*sr - cy*cr, -sy*sp*cr + cy*sr],
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [0, 0, 0]
        ])
        
        v_body = np.array([vx, vy, vz])
        
        # dx/d(orientation)
        F[0, 3] = (dR_droll @ v_body)[0] * dt
        F[0, 4] = (dR_dpitch @ v_body)[0] * dt
        F[0, 5] = (dR_dyaw @ v_body)[0] * dt
        
        F[1, 3] = (dR_droll @ v_body)[1] * dt
        F[1, 4] = (dR_dpitch @ v_body)[1] * dt
        F[1, 5] = (dR_dyaw @ v_body)[1] * dt
        
        F[2, 3] = (dR_droll @ v_body)[2] * dt
        F[2, 4] = (dR_dpitch @ v_body)[2] * dt
        F[2, 5] = (dR_dyaw @ v_body)[2] * dt
        
        # dx/dv
        F[0:3, 6:9] = R * dt
        
        # d(orientation)/d(angular velocity)
        F[3, 9] = dt
        F[4, 10] = dt
        F[5, 11] = dt
        
        # Predicted covariance
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update_odometry(self, odom_msg):
        """
        Update step using odometry measurements.
        
        Args:
            odom_msg (Odometry): Odometry message
        """
        # Extract measurement [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        pos = odom_msg.pose.pose.position
        orient = odom_msg.pose.pose.orientation
        twist = odom_msg.twist.twist
        
        # Convert quaternion to Euler angles
        quaternion = (orient.x, orient.y, orient.z, orient.w)
        roll, pitch, yaw = self.get_euler_from_quaternion(quaternion)
        
        # Measurement vector (full 6-DOF pose + velocities)
        z = np.array([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw,
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z
        ])
        
        # Measurement model is identity (we directly observe all states)
        H = np.eye(12)
        
        # Innovation
        y = z - self.state
        # Normalize angle differences
        y[3] = self._normalize_angle(y[3])  # roll
        y[4] = self._normalize_angle(y[4])  # pitch
        y[5] = self._normalize_angle(y[5])  # yaw
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_odom
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        # Normalize angles in state
        self.state[3] = self._normalize_angle(self.state[3])
        self.state[4] = self._normalize_angle(self.state[4])
        self.state[5] = self._normalize_angle(self.state[5])
        
        # Update covariance
        self.P = (np.eye(12) - K @ H) @ self.P
    
    def update_apriltag(self, tag_id, tag_pose_cam, distance=None):
        """
        Update step using AprilTag detection (6-DOF).
        
        Args:
            tag_id (int): ID of detected AprilTag
            tag_pose_cam (np.array): [x, y, z, roll, pitch, yaw] of tag relative to camera/robot
            distance (float): Distance to tag (optional, for adaptive noise)
        """
        if tag_id not in self.tag_map:
            self.get_logger().warn(f"Tag {tag_id} not in tag map, skipping", 
                                   throttle_duration_sec=5.0)
            return
        
        # Get tag pose in world frame [x, y, z, roll, pitch, yaw]
        tag_world = self.tag_map[tag_id]
        
        # Current state estimate
        x, y, z, roll, pitch, yaw = self.state[0:6]
        
        # Build rotation matrix from robot orientation
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R_world_robot = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Expected tag position in robot frame
        tag_pos_world = tag_world[0:3]
        robot_pos_world = np.array([x, y, z])
        
        # Transform tag position from world to robot frame
        delta_world = tag_pos_world - robot_pos_world
        delta_robot = R_world_robot.T @ delta_world
        
        # Expected tag orientation in robot frame
        # Relative orientation: tag_orientation - robot_orientation
        expected_roll = self._normalize_angle(tag_world[3] - roll)
        expected_pitch = self._normalize_angle(tag_world[4] - pitch)
        expected_yaw = self._normalize_angle(tag_world[5] - yaw)
        
        # Expected measurement (what we should see given current state)
        z_expected = np.array([
            delta_robot[0], delta_robot[1], delta_robot[2],
            expected_roll, expected_pitch, expected_yaw
        ])
        
        # Actual measurement
        z = tag_pose_cam
        
        # Measurement Jacobian H (6x12)
        # Measures position and orientation
        H = np.zeros((6, 12))
        
        # Simplified Jacobian (full derivation is complex for 6-DOF)
        # Position measurement depends on robot position and orientation
        H[0:3, 0:3] = -R_world_robot.T  # d(tag_robot)/d(robot_pos)
        # H[0:3, 3:6] would include rotation derivatives (complex, approximated here)
        
        # Orientation measurement depends on robot orientation
        H[3, 3] = -1  # roll
        H[4, 4] = -1  # pitch  
        H[5, 5] = -1  # yaw
        
        # Adaptive measurement noise based on distance
        R_tag = self.R_tag.copy()
        if distance is not None:
            # Increase uncertainty with distance (quadratic relationship)
            distance_factor = max(1.0, (distance / 2.0)**2)
            R_tag = R_tag * distance_factor
        
        # Innovation
        y = z - z_expected
        # Normalize angle differences
        y[3] = self._normalize_angle(y[3])
        y[4] = self._normalize_angle(y[4])
        y[5] = self._normalize_angle(y[5])
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_tag
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        # Normalize angles in state
        self.state[3] = self._normalize_angle(self.state[3])
        self.state[4] = self._normalize_angle(self.state[4])
        self.state[5] = self._normalize_angle(self.state[5])
        
        # Update covariance
        self.P = (np.eye(12) - K @ H) @ self.P
        
        dist_str = f" at {distance:.2f}m" if distance else ""
        self.get_logger().info(f"Updated with tag {tag_id}{dist_str}", 
                              throttle_duration_sec=1.0)
    
    def odom_callback(self, msg):
        """Callback for odometry messages."""
        current_time = self.get_clock().now()
        
        if self.last_odom_time is not None:
            dt = (current_time - self.last_odom_time).nanoseconds / 1e9
            if dt > 0:
                self.predict(dt)
        
        self.update_odometry(msg)
        self.last_odom_time = current_time
        
        # Publish filtered odometry
        self.publish_filtered_odom(msg.header.stamp, msg.header.frame_id, msg.child_frame_id)
    
    def get_camera_transform(self):
        """Look up the static transform from Base -> Camera."""
        try:
            # 1. Look up the transform
            # We use rclpy.time.Time() to get the latest available transform (time 0)
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time())

            t = trans.transform.translation
            
            r = trans.transform.rotation
            x, y, z, w = r.x, r.y, r.z, r.w

            self.camera_to_base_transform = np.array([
                [1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,       2*x*z + 2*y*w,       t.x],
                [2*x*y + 2*z*w,        1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,       t.y],
                [2*x*z - 2*y*w,        2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2, t.z],
                [0,                    0,                   0,                   1  ]
            ])

            self.get_logger().info(f"Transform found: Camera is at {t.x:.2f}, {t.y:.2f}, {t.z:.2f}")
            return True
            
        except:
            self.get_logger().warn(f"Could not get transform:", throttle_duration_sec=2.0)
            return False
        
    

    def tag_callback(self, msg):
        """
        1. Decode tag pose (Camera Frame)
        2. Transform to Robot Frame (using tf2_ros buffer)
        3. Update EKF
        """
        if len(msg.detections) == 0 or not self.camera_info_received:
            return

        for detection in msg.detections:
            tag_id = detection.id
            
            # 1. Get Pose in Camera Frame (x, y, z, r, p, y)
            H = np.array(detection.homography).reshape(3, 3)
            pose_cam = self.pose_from_homography(H, self.get_tag_size(tag_id))
            
            if pose_cam is None: continue
            
            x, y, z, r, p, yw = pose_cam

            # 2. Create a PoseStamped object
            # We must wrap the raw numbers in a ROS message to use the TF buffer
            pose_stamped_cam = PoseStamped()
            pose_stamped_cam.header.frame_id = msg.header.frame_id # usually 'camera_optical_frame'
            pose_stamped_cam.header.stamp = msg.header.stamp
            
            pose_stamped_cam.pose.position.x = x
            pose_stamped_cam.pose.position.y = y
            pose_stamped_cam.pose.position.z = z
            
            # Convert Euler (cam) -> Quaternion (msg)
            qx, qy, qz, qw = self.get_quaternion_from_euler(r, p, yw)
            pose_stamped_cam.pose.orientation.x = qx
            pose_stamped_cam.pose.orientation.y = qy
            pose_stamped_cam.pose.orientation.z = qz
            pose_stamped_cam.pose.orientation.w = qw

            # 3. Transform to Base Link
            try:
                # The buffer handles the math! 
                # Requires: import tf2_geometry_msgs
                pose_stamped_base = self.tf_buffer.transform(
                    pose_stamped_cam, 
                    self.target_frame,
                    timeout=Duration(seconds=0.1)
                )
                
                # 4. Extract data for EKF (which expects Euler)
                pos = pose_stamped_base.pose.position
                orient = pose_stamped_base.pose.orientation
                
                # Convert Quaternion (base) -> Euler (base)
                roll_b, pitch_b, yaw_b = self.get_euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
                
                tag_pose_robot = np.array([pos.x, pos.y, pos.z, roll_b, pitch_b, yaw_b])
                
                # Distance (frame invariant)
                distance = np.sqrt(x**2 + y**2 + z**2)
                
                # Update EKF
                self.update_apriltag(tag_id, tag_pose_robot, distance)

            except:
                # self.get_logger().warn(f"TF Error: {e}", throttle_duration_sec=1.0)
                pass

        # Publish result
        self.publish_filtered_odom(msg.header.stamp, "odom", "base_link")


    @staticmethod
    def get_quaternion_from_euler(roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]


    @staticmethod
    def get_euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        """
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
        
        # Orientation (convert Euler to quaternion)
        roll, pitch, yaw = self.state[3:6]
        quaternion = self.get_quaternion_from_euler(roll, pitch, yaw)
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]
        
        # Velocity (linear and angular)
        odom_msg.twist.twist.linear.x = self.state[6]
        odom_msg.twist.twist.linear.y = self.state[7]
        odom_msg.twist.twist.linear.z = self.state[8]
        odom_msg.twist.twist.angular.x = self.state[9]
        odom_msg.twist.twist.angular.y = self.state[10]
        odom_msg.twist.twist.angular.z = self.state[11]
        
        # Covariance (6x6 for pose, 6x6 for twist)
        # Map our state covariance to ROS covariance format
        pose_cov = np.zeros((6, 6))
        pose_cov[0:3, 0:3] = self.P[0:3, 0:3]  # position
        pose_cov[3:6, 3:6] = self.P[3:6, 3:6]  # orientation
        odom_msg.pose.covariance = pose_cov.flatten().tolist()
        
        twist_cov = np.zeros((6, 6))
        twist_cov[0:3, 0:3] = self.P[6:9, 6:9]    # linear velocity
        twist_cov[3:6, 3:6] = self.P[9:12, 9:12]  # angular velocity
        odom_msg.twist.covariance = twist_cov.flatten().tolist()
        
        self.filtered_odom_pub.publish(odom_msg)
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagEKF()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()