#!/usr/bin/env python3
"""
dFz observer — fuses an MPC-predicted dFz (model-based, live sysID params)
with a pitch-derived measurement (pure suspension kinematics).

The MPC publishes its predicted dFz (integrated with the current adaptive
parameters). This node treats that as the prediction and corrects it with
pitch. No tire parameters live here — they live in the MPC.

Modes:
  "fused"     : pitch from mocap + IMU gyro, corrects MPC prediction
  "imu"       : pitch from IMU gyro only, corrects MPC prediction
  "open_loop" : pass MPC prediction through unchanged (trust the model)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class DFzObserver(Node):
    def __init__(self):
        super().__init__("dfz_observer")

        self.declare_parameter("mode", "fused")          # fused | imu | open_loop
        self.declare_parameter("rate_hz", 100.0)
        # ONLY suspension/measurement params live here -- NO tire params
        self.declare_parameter("k_theta", 200.0)         # load per pitch [N/rad] (identify)
        self.declare_parameter("b_theta", 0.0)           # load per pitch rate [N s/rad] (optional)
        # filter tuning
        self.declare_parameter("Q", 50.0)                # extra process noise [N^2/s]
        self.declare_parameter("R_pitch", 0.02)          # pitch meas noise [rad^2]
        self.declare_parameter("tau_comp", 0.5)
        self.declare_parameter("gyro_bias_tau", 20.0)
        self.declare_parameter("mpc_var", 25.0)          # trust in MPC prediction [N^2]
        # topics
        self.declare_parameter("imu_topic", "/sensors/imu/data")
        self.declare_parameter("mocap_topic", "/optitrack/odom")
        self.declare_parameter("mpc_dfz_topic", "/mpc/dfz_pred")     # <-- MPC's predicted dFz
        self.declare_parameter("mpc_dfz_var_topic", "/mpc/dfz_var")  # optional: MPC covariance

        gp = lambda n: self.get_parameter(n).value
        self.mode = gp("mode")
        self.dt = 1.0 / gp("rate_hz")
        self.k_theta = gp("k_theta")
        self.b_theta = gp("b_theta")
        self.Q = gp("Q")
        self.R = gp("R_pitch")
        self.tau_comp = gp("tau_comp")
        self.gyro_bias_tau = gp("gyro_bias_tau")
        self.mpc_var = gp("mpc_var")

        if self.mode not in ("fused", "imu", "open_loop"):
            self.get_logger().warn(f"bad mode {self.mode}, using open_loop")
            self.mode = "open_loop"

        # estimate
        self.dFz = 0.0
        self.P = 100.0

        # MPC prediction cache
        self.dFz_mpc = 0.0
        self.dFz_mpc_prev = 0.0
        self.have_mpc = False

        # pitch state
        self.pitch_comp = 0.0
        self.gyro_bias = 0.0
        self.pitch_rate = 0.0
        self.ax_imu = 0.0
        self.pitch_mocap = None
        self.level_count = 0

        # subs
        self.create_subscription(Imu, gp("imu_topic"), self.imu_cb, 50)
        self.create_subscription(Float64, gp("mpc_dfz_topic"), self.mpc_cb, 20)
        self.create_subscription(Float64, gp("mpc_dfz_var_topic"), self.mpc_var_cb, 20)
        if self.mode == "fused":
            self.create_subscription(Odometry, gp("mocap_topic"), self.mocap_cb, 50)

        # pubs
        self.pub_dFz = self.create_publisher(Float64, "/dfz/estimate", 10)
        self.pub_pitch = self.create_publisher(Float64, "/dfz/pitch", 10)

        self.timer = self.create_timer(self.dt, self.step)

    # ---- callbacks ----
    def imu_cb(self, m):
        self.pitch_rate = m.angular_velocity.y
        self.ax_imu = m.linear_acceleration.x

    def mpc_cb(self, m):
        # PREDICT happens here — once per MPC message, no double-counting
        if self.have_mpc:
            delta = m.data - self.dFz_mpc
            self.dFz += delta
            self.P += self.Q * self.dt_mpc      # process noise over MPC interval
        else:
            self.dFz = m.data                   # initialize on first message
        self.dFz_mpc = m.data
        self.have_mpc = True
        if self.mode == "open_loop":
            self.dFz = m.data                   # open loop: just track MPC absolute
            self.pub_dFz.publish(Float64(data=float(self.dFz)))


    def mpc_var_cb(self, m):
        self.mpc_var = max(m.data, 1e-3)

    def mocap_cb(self, m):
        q = m.pose.pose.orientation
        sinp = max(-1.0, min(1.0, 2.0 * (q.w*q.y - q.z*q.x)))
        self.pitch_mocap = np.arcsin(sinp)

    # ---- gyro bias: track when level & not accelerating ----
    def update_bias(self):
        if abs(self.ax_imu) < 0.3 and abs(self.pitch_rate) < 0.05:
            self.level_count += 1
            if self.level_count > 20:  # sustained level
                self.gyro_bias += (self.dt / self.gyro_bias_tau) * (self.pitch_rate - self.gyro_bias)
        else:
            self.level_count = 0

    def update_pitch(self):
        self.update_bias()
        rate = self.pitch_rate - self.gyro_bias
        if self.mode == "fused" and self.pitch_mocap is not None:
            a = self.tau_comp / (self.tau_comp + self.dt)
            self.pitch_comp = a*(self.pitch_comp + rate*self.dt) + (1-a)*self.pitch_mocap
        elif self.mode == "imu":
            a = self.tau_comp / (self.tau_comp + self.dt)
            self.pitch_comp = a*(self.pitch_comp + rate*self.dt)
        return self.pitch_comp, rate

    
    def step(self):
        # CORRECT only — pitch measurement at timer rate
        if not self.have_mpc or self.mode == "open_loop":
            return
        
        pitch, rate = self.update_pitch()
        z = self.k_theta * pitch + self.b_theta * rate
        S = self.P + self.R * self.k_theta**2
        K = self.P / S
        self.dFz += K * (z - self.dFz)
        self.P = (1.0 - K) * self.P
        self.pub_dFz.publish(Float64(data=float(self.dFz)))
        self.pub_pitch.publish(Float64(data=float(pitch)))


def main():
    rclpy.init()
    n = DFzObserver()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()