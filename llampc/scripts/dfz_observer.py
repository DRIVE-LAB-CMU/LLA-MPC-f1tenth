#!/usr/bin/env python3
"""
dFz observer — fuses an MPC-predicted dFz (model-based, live sysID params)
with a pitch-derived measurement (pure suspension kinematics).

The MPC publishes its predicted dFz plus an active flag (integrated with the
current adaptive parameters). This node treats that as the prediction when
active, and falls back to its own relaxation-toward-zero prediction when the
MPC is inactive (e.g. running pure pursuit at low speed) so filtering never
stalls. It always corrects with pitch. No tire parameters live here — they
live in the MPC.

Modes:
  "fused"     : pitch from mocap + IMU gyro, corrects the prediction
  "imu"       : pitch from IMU gyro only, corrects the prediction
  "open_loop" : pass MPC prediction through unchanged (trust the model);
                falls back to raw relaxation-to-zero when MPC is inactive
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray


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
        # fallback prediction (used only while MPC is inactive)
        self.declare_parameter("fallback_relax_rate", 20.0)   # 1/s, matches model's c
        # topics
        self.declare_parameter("imu_topic", "/sensors/imu/data")
        self.declare_parameter("mocap_topic", "/optitrack/odom")
        self.declare_parameter("mpc_dfz_topic", "/mpc/dfz_pred")     # Float64MultiArray: [dfz, active]
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
        self.c_fallback = gp("fallback_relax_rate")

        if self.mode not in ("fused", "imu", "open_loop"):
            self.get_logger().warn(f"bad mode {self.mode}, using open_loop")
            self.mode = "open_loop"

        # estimate
        self.dFz = 0.0
        self.P = 100.0

        # MPC prediction cache
        self.dFz_mpc = 0.0
        self.have_mpc = False       # have we ever received a message at all
        self.mpc_active = False     # is the MPC's dynamic model currently running
        self.last_mpc_time = None

        # pitch state
        self.pitch_comp = 0.0
        self.gyro_bias = 0.0
        self.pitch_rate = 0.0
        self.ax_imu = 0.0
        self.pitch_mocap = None
        self.level_count = 0

        # subs
        self.create_subscription(Imu, gp("imu_topic"), self.imu_cb, 50)
        self.create_subscription(Float64MultiArray, gp("mpc_dfz_topic"), self.mpc_cb, 20)
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
        # msg.data = [dfz_pred, active_flag]
        dfz_val, active = m.data[0], m.data[1]
        now = self.get_clock().now()

        if active < 0.5:
            self.mpc_active = False
            return

        # PREDICT happens here — once per active MPC message, no double-counting
        if self.have_mpc and self.mpc_active and self.last_mpc_time is not None:
            dt_mpc = (now - self.last_mpc_time).nanoseconds * 1e-9
            delta = dfz_val - self.dFz_mpc
            self.dFz += delta
            self.P += self.Q * dt_mpc
        else:
            # first message ever, or resuming after an inactive stretch --
            # re-initialize directly rather than diffing against a stale value
            self.dFz = dfz_val

        self.dFz_mpc = dfz_val
        self.have_mpc = True
        self.mpc_active = True
        self.last_mpc_time = now

        if self.mode == "open_loop":
            self.dFz = dfz_val
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

    def fallback_predict(self):
        """No active dynamic-model prediction available (MPC inactive).
        Treat dFz as a random walk: hold the estimate, inflate uncertainty,
        and let the pitch correction (in step()) pull toward the true value."""
        self.P += self.Q * self.dt

    def step(self):
        if not self.have_mpc:
            return   # never got a first message yet -- nothing to correct

        if not self.mpc_active:
            # keep predicting even though the MPC itself isn't running
            self.fallback_predict()
            if self.mode == "open_loop":
                # open loop trusts the model only; with no active model,
                # publish the relaxed fallback value directly
                self.pub_dFz.publish(Float64(data=float(self.dFz)))
                return
        elif self.mode == "open_loop":
            return  # already published straight through in mpc_cb

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