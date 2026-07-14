#!/usr/bin/env python3
import numpy as np
import time
import sys, os
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
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


class CausalSpikeEMA:
    """Live (causal) EMA + windowed spike clamp for one scalar channel.

    Unlike the offline plotting-tool version, this only ever looks
    BACKWARD in time -- each new sample is judged against a trailing
    history buffer spanning `window_sec`, since a live callback has no
    access to "future" samples the way a post-hoc analysis does.

    Pipeline per sample: clamp to the trailing window's [pct, 100-pct]
    percentile band, THEN apply a dt-normalized EMA with time constant tau.
    tau should stay small (a few sample periods) -- this is meant to kill
    single-sample noise/outliers, not to do the EKF's smoothing job for it
    and add meaningful lag on top of what you're already fighting.
    """

    def __init__(self, tau, spike_pct, window_sec):
        self.tau = tau
        self.spike_pct = max(0.0, min(49.0, spike_pct))
        self.window_sec = window_sec
        self._hist_t = deque()
        self._hist_v = deque()
        self._ema = None
        self._last_t = None

    def update(self, t, v):
        # --- 1. maintain trailing window ---
        self._hist_t.append(t)
        self._hist_v.append(v)
        while self._hist_t and (t - self._hist_t[0]) > self.window_sec:
            self._hist_t.popleft()
            self._hist_v.popleft()

        # --- 2. windowed spike clamp (causal: only past+current samples) ---
        if self.spike_pct > 0 and len(self._hist_v) >= 3:
            arr = np.asarray(self._hist_v, dtype=float)
            lo = np.percentile(arr, self.spike_pct)
            hi = np.percentile(arr, 100.0 - self.spike_pct)
            v_clamped = min(max(v, lo), hi)
        else:
            v_clamped = v

        # --- 3. dt-normalized EMA ---
        if self._ema is None or self._last_t is None:
            self._ema = v_clamped
        else:
            dt = t - self._last_t
            if dt > 0 and self.tau > 0:
                a = dt / (self.tau + dt)
                self._ema = self._ema + a * (v_clamped - self._ema)
            elif dt > 0:
                self._ema = v_clamped  # tau<=0 -> spike clamp only, no smoothing
        self._last_t = t
        return self._ema


class OptitrackSubscriber(Node):
    def __init__(self, history_size=5):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('optitrack_bridge_sub')

        self.declare_parameter('mocap_topic', '/f1tenth/pose')
        mocap_topic = self.get_parameter('mocap_topic').get_parameter_value().string_value

        # --- filter tuning: EMA + windowed spike clamp on vx, vy, omega ---
        # tau kept deliberately small (15 ms) -- just enough to knock the
        # noise floor down without adding lag on top of an EKF that's
        # already lagging. spike_pct=15% / window=0.5s clamps outliers
        # relative to a LOCAL trailing neighborhood, not a single global
        # threshold, since vx/vy here are pure finite differences with no
        # independent velocity sensor to check them against.
        self.declare_parameter('vel_ema_tau', 0.06)
        self.declare_parameter('vel_spike_pct', 15.0)
        self.declare_parameter('vel_spike_window', 0.5)

        tau = self.get_parameter('vel_ema_tau').value
        spike_pct = self.get_parameter('vel_spike_pct').value
        spike_window = self.get_parameter('vel_spike_window').value

        self._filt_vx = CausalSpikeEMA(tau, spike_pct, spike_window)
        self._filt_vy = CausalSpikeEMA(tau, spike_pct, spike_window)
        self._filt_omega = CausalSpikeEMA(tau, spike_pct, spike_window)

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)


        self.suber = self.create_subscription(
            PoseStamped,
            mocap_topic,
            self.mocap_callback,
            qos)
        
        self.ekf_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.ekf_callback,
            qos)

        # ==========================================================
        # Publisher for the EKF (nav_msgs/Odometry: pose + twist)
        # ==========================================================
        self.ekf_odom_pub = self.create_publisher(
            Odometry,
            '/optitrack/odom',
            10
        )
        

        self.optitrack_position = np.zeros(3)
        self.optitrack_quaternion = np.zeros(4)
        self.optitrack_linear_velocity_world = np.zeros(3)
        self.optitrack_linear_velocity = np.zeros(3)
        self.optitrack_angular_velocity_world = np.zeros(3)
        self.optitrack_angular_velocity = np.zeros(3)
        
        
        self.ekf_linear_velocity = np.zeros(2)
        self.ekf_ang_velocity = 0
        
        self.history_size = history_size

        self.position_history = []
        self.quaternion_history = []
        self.timestamp_history = []

        self.br = TransformBroadcaster(self)
        
    def ekf_callback(self, msg):
        self.ekf_linear_velocity = [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
        
        self.ekf_abs_linear_velocity = np.sqrt(self.ekf_linear_velocity[0]**2 + self.ekf_linear_velocity[1]**2)
        self.ekf_ang_velocity = msg.twist.twist.angular.z
        

    def mocap_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        position = np.array([-msg.pose.position.x, msg.pose.position.z, msg.pose.position.y,])

        q_orig = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        # Convert quaternion -> rotation matrix
        R_orig = quat_to_rot(q_orig)


        theta = 0   # or -np.pi/2 -- sign found empirically, see below
        B = np.array([
            [ np.cos(theta), -np.sin(theta), 0],
            [ np.sin(theta),  np.cos(theta), 0],
            [ 0,               0,            1]
        ])

        # Axis permutation matrix
        P = np.array([
            [-1, 0, 0],
            [ 0, 0, 1],
            [ 0, 1, 0]])

        # Transform rotation matrix
        R_new = P @ R_orig @ B      # not P @ R_orig @ P.T
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

        odom_msg = Odometry()
        odom_msg.header = msg.header
        odom_msg.header.frame_id = 'map'        # pose frame
        odom_msg.child_frame_id = 'base_link'   # twist (body) frame

        # ---- Pose (world) ----
        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]

        # Pose covariance (6x6: x, y, z, roll, pitch, yaw)
        # Small variance (1e-4) because Optitrack is highly accurate
        pcov = np.zeros(36)
        pcov[0]  = 1e-6  # X variance
        pcov[7]  = 1e-6  # Y variance
        pcov[14] = 1e-6  # Z variance
        pcov[21] = 1e-4  # Roll variance
        pcov[28] = 1e-4  # Pitch variance
        pcov[35] = 1e-4  # Yaw variance
        odom_msg.pose.covariance = pcov.tolist()

        # ==========================================================
        # ---- Twist (body frame) ----
        # Finite-difference velocity from mocap, rotated world->body
        # (robot_localization treats twist as base_link-frame), then run
        # through a causal EMA + windowed spike clamp per channel before
        # publishing. Covariance is kept MODERATE-TO-HIGH, not low: the
        # signal is filtered but still finite-difference-derived with no
        # independent velocity sensor backing it up, so the EKF shouldn't
        # be told to trust it blindly -- only that it's a real, usable
        # (if somewhat noisy) correction.
        # ==========================================================
        if len(self.position_history) >= 2:
            R_body = quat_to_rot(quaternion)            # body->world
            v_body_raw = R_body.T @ self.optitrack_linear_velocity_world
            w_body_raw = R_body.T @ self.optitrack_angular_velocity_world

            vx_f = self._filt_vx.update(timestamp, v_body_raw[0])
            vy_f = self._filt_vy.update(timestamp, v_body_raw[1])
            wz_f = self._filt_omega.update(timestamp, w_body_raw[2])

            odom_msg.twist.twist.linear.x  = vx_f
            odom_msg.twist.twist.linear.y  = vy_f
            odom_msg.twist.twist.angular.z = wz_f

            tcov = np.zeros(36)
            tcov[0]  = .25  # Vx variance -- moderate/high: filtered, but
            tcov[7]  = .09  # Vy variance    still just finite-diff, no
            tcov[35] = .09  # Vyaw variance  independent velocity sensor.
            odom_msg.twist.covariance = tcov.tolist()
        # ==========================================================
        
        # if len(self.position_history) >= 2:
        #     if self.position_history[-1] == self.position_history[-2] and \
        #         self.quaternion_history[-1] == self.quaternion_history[-2]:

        #         if self.ekf_abs_linear_velocity > 0.4 or self.ekf_ang_velocity > 5:
        #             return

        self.ekf_odom_pub.publish(odom_msg)

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