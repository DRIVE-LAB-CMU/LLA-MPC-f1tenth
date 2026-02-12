#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
import numpy as np
import math
from collections import defaultdict

# --- ROS Messages ---
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
# This is the critical message type for your setup
from aruco_opencv_msgs.msg import ArucoDetection

# --- TF2 for Transforms ---
from tf2_ros import TransformListener, Buffer

class ArucoEKF(Node):
    def __init__(self):
        super().__init__('aruco_ekf_node', 
                         allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)

        # --- 1. Initialization ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Frames
        self.base_frame = 'base_link'
        self.odom_frame = 'odom'
        
        # --- 2. Load Parameters ---
        self.publish_rate = self.get_parameter_or('publish_rate', 50.0).value
        self.timer_period = 1.0 / self.publish_rate
        
        # Load the map of known tags from YAML
        self.tag_map = self._load_tag_map()
        
        # --- 3. EKF State Initialization ---
        # State Vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.state = np.zeros(12)
        
        # State Covariance Matrix (P) - Initial uncertainty
        self.P = np.eye(12) * 0.1
        
        # Process Noise Covariance (Q) - Uncertainty in prediction model
        self.Q = np.diag([
            0.01, 0.01, 0.02,    # Pos
            0.02, 0.02, 0.01,    # Orient
            0.1, 0.1, 0.2,       # Lin Vel
            0.2, 0.2, 0.1        # Ang Vel
        ])
        
        # Measurement Noise (R)
        self.R_odom = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.R_marker = np.diag([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

        self.last_time = self.get_clock().now()

        # --- 4. Communication ---
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.aruco_sub = self.create_subscription(ArucoDetection, '/aruco_detections', self.aruco_callback, qos)
        
        # Publishers
        self.filtered_pub = self.create_publisher(Odometry, '/odom/filtered', qos)
        
        # Timer for Prediction Loop
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(f"ArUco EKF Initialized. Rate: {self.publish_rate}Hz. Loaded {len(self.tag_map)} tags.")

    def _load_tag_map(self):
        """
        Parses 'tag_positions' from YAML.
        Structure expected: tag_positions.<id>.x, tag_positions.<id>.y, etc.
        """
        tag_map = {}
        # Get all params under 'tag_positions'
        params = self.get_parameters_by_prefix('tag_positions')
        
        parsed_tags = defaultdict(dict)
        for key, param in params.items():
            try:
                # Key comes in as "0.x", "0.y", "5.yaw" etc.
                parts = key.split('.')
                if len(parts) < 2: continue
                
                tag_id = parts[0]
                field = parts[1]
                parsed_tags[tag_id][field] = param.value
            except Exception:
                pass

        for tag_id, data in parsed_tags.items():
            try:
                mid = int(tag_id)
                # Default to 0.0 if field missing
                x = float(data.get('x', 0.0))
                y = float(data.get('y', 0.0))
                z = float(data.get('z', 0.0))
                roll = float(data.get('roll', 0.0))
                pitch = float(data.get('pitch', 0.0))
                yaw = float(data.get('yaw', 0.0))
                
                tag_map[mid] = np.array([x, y, z, roll, pitch, yaw])
            except ValueError:
                self.get_logger().warn(f"Invalid tag ID found: {tag_id}")

        return tag_map

    # =========================================
    #               EKF LOGIC
    # =========================================

    def predict(self, dt):
        """Standard Constant Velocity Prediction Model."""
        x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz = self.state
        
        # Precompute trig
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Rotation Matrix (Body -> World)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # 1. Update State
        # Position += Velocity_World * dt
        v_world = R @ np.array([vx, vy, vz])
        self.state[0:3] += v_world * dt
        
        # Orientation += Angular_Vel * dt
        self.state[3] = self._normalize_angle(roll + wx * dt)
        self.state[4] = self._normalize_angle(pitch + wy * dt)
        self.state[5] = self._normalize_angle(yaw + wz * dt)
        
        # Velocity is constant in this model (Process noise handles changes)
        
        # 2. Update Covariance P = F*P*F' + Q
        # Jacobian F (Partial derivatives of State w.r.t State)
        F = np.eye(12)
        F[0:3, 6:9] = R * dt  # Position depends on Velocity
        F[3:6, 9:12] = np.eye(3) * dt # Orientation depends on Angular Velocity
        
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_odom(self, msg):
        """Update step using Wheel Odometry (measuring Velocities)."""
        # Extract Twist (Linear & Angular velocities)
        # Note: We are trusting the twist, but usually position in odom drifts.
        # So we fuse Twist for velocity correction, and we can fuse Position relative to start.
        # For simplicity here: We fuse everything from odom message.
        
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        twist = msg.twist.twist
        
        roll, pitch, yaw = self.get_euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        
        # Measurement Vector Z
        z_meas = np.array([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw,
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z
        ])
        
        # Measurement Matrix H (We measure state directly)
        H = np.eye(12)
        
        # Innovation y = z - Hx
        y = z_meas - self.state
        # Normalize angles in innovation
        y[3] = self._normalize_angle(y[3])
        y[4] = self._normalize_angle(y[4])
        y[5] = self._normalize_angle(y[5])
        
        # Kalman Gain K
        S = H @ self.P @ H.T + self.R_odom
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State & Covariance
        self.state = self.state + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P
        
        # Normalize angles in state
        self.state[3] = self._normalize_angle(self.state[3])
        self.state[4] = self._normalize_angle(self.state[4])
        self.state[5] = self._normalize_angle(self.state[5])

    def update_marker(self, tag_id, tag_pose_robot, distance):
        """
        Update step using Tag Detection.
        tag_pose_robot: The pose of the tag RELATIVE to the robot base (x,y,z,r,p,y).
        """
        if tag_id not in self.tag_map:
            return

        # 1. Known Tag World Position
        tag_world = self.tag_map[tag_id] # [x,y,z,r,p,y]
        
        # 2. Calculate Expected Measurement (h(x))
        # Where should the tag be relative to the robot, given our current estimated world pose?
        
        # Robot State
        rx, ry, rz, rr, rp, ryaw = self.state[0:6]
        
        # Vector from Robot to Tag in World Frame
        dx_world = tag_world[0] - rx
        dy_world = tag_world[1] - ry
        dz_world = tag_world[2] - rz
        
        # Rotate this vector into Robot Body Frame
        # R_world_to_robot is the inverse of R_body_to_world
        cr, sr = np.cos(rr), np.sin(rr)
        cp, sp = np.cos(rp), np.sin(rp)
        cy, sy = np.cos(ryaw), np.sin(ryaw)
        
        # Transpose of rotation matrix constructed in predict
        R_inv = np.array([
            [cy*cp, sy*cp, -sp],
            [cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
            [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
        ])
        
        pos_robot_frame = R_inv @ np.array([dx_world, dy_world, dz_world])
        
        # Expected Orientation (Simplified: Tag World Orient - Robot World Orient)
        expected_roll = self._normalize_angle(tag_world[3] - rr)
        expected_pitch = self._normalize_angle(tag_world[4] - rp)
        expected_yaw = self._normalize_angle(tag_world[5] - ryaw)
        
        z_expected = np.array([
            pos_robot_frame[0], pos_robot_frame[1], pos_robot_frame[2],
            expected_roll, expected_pitch, expected_yaw
        ])
        
        # 3. Innovation
        y = tag_pose_robot - z_expected
        y[3] = self._normalize_angle(y[3])
        y[4] = self._normalize_angle(y[4])
        y[5] = self._normalize_angle(y[5])
        
        # 4. Jacobian H
        # Partial derivatives of (Tag_Relative_Pos) w.r.t (Robot_World_Pos)
        # d(pos_robot)/d(pos_world) = -R_inv
        H = np.zeros((6, 12))
        H[0:3, 0:3] = -R_inv
        # Identity for orientation (simplified)
        H[3, 3] = -1; H[4, 4] = -1; H[5, 5] = -1
        
        # 5. Adaptive Noise
        # Trust distant tags less
        R_adaptive = self.R_marker.copy()
        if distance > 1.0:
            R_adaptive *= (distance**2)

        # 6. Update
        S = H @ self.P @ H.T + R_adaptive
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P
        
        # Normalize
        self.state[3] = self._normalize_angle(self.state[3])
        self.state[4] = self._normalize_angle(self.state[4])
        self.state[5] = self._normalize_angle(self.state[5])
        
        # Debug
        # self.get_logger().info(f"Updated with Tag {tag_id}. Robot X: {self.state[0]:.2f}")

    # =========================================
    #               CALLBACKS
    # =========================================

    def timer_callback(self):
        """Main Loop: Predict -> Publish"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        if dt > 0:
            self.predict(dt)
            
        self._publish_odom()

    def odom_callback(self, msg):
        """Asynchronous Odom Update"""
        self.update_odom(msg)

    def aruco_callback(self, msg):
        """Asynchronous Marker Update"""
        if not msg.markers:
            return

        for marker in msg.markers:
            try:
                # TF: Get Transform from Camera -> Base Link
                # We need the marker position relative to the ROBOT CENTER, not the camera.
                if not self.tf_buffer.can_transform(self.base_frame, msg.header.frame_id, rclpy.time.Time()):
                    continue
                
                # Wrap pose in PoseStamped for TF2
                p_cam = PoseStamped()
                p_cam.header = msg.header
                p_cam.pose = marker.pose
                
                # Transform
                p_base = self.tf_buffer.transform(p_cam, self.base_frame, timeout=Duration(seconds=0.1))
                
                # Extract
                pos = p_base.pose.position
                orient = p_base.pose.orientation
                roll, pitch, yaw = self.get_euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
                
                marker_pose_robot = np.array([pos.x, pos.y, pos.z, roll, pitch, yaw])
                distance = np.linalg.norm([pos.x, pos.y, pos.z])
                
                # Run EKF Update
                self.update_marker(marker.marker_id, marker_pose_robot, distance)
                
            except Exception as e:
                self.get_logger().debug(f"TF Error: {e}")

    def _publish_odom(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        
        # Pose
        msg.pose.pose.position.x = self.state[0]
        msg.pose.pose.position.y = self.state[1]
        msg.pose.pose.position.z = self.state[2]
        
        q = self.get_quaternion_from_euler(self.state[3], self.state[4], self.state[5])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Twist
        msg.twist.twist.linear.x = self.state[6]
        msg.twist.twist.linear.y = self.state[7]
        msg.twist.twist.linear.z = self.state[8]
        msg.twist.twist.angular.x = self.state[9]
        msg.twist.twist.angular.y = self.state[10]
        msg.twist.twist.angular.z = self.state[11]
        
        # Covariance (Simplified diagonal)
        # Map our 12x12 P to ROS 6x6
        # P indexes: 0,1,2 (pos), 3,4,5 (orient)
        msg.pose.covariance[0] = self.P[0,0]
        msg.pose.covariance[7] = self.P[1,1]
        msg.pose.covariance[35] = self.P[5,5]
        
        self.filtered_pub.publish(msg)

    # =========================================
    #               MATH UTILS
    # =========================================

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