#!/usr/bin/env python3
"""
iw_estimator_node.py

Standalone ROS2 node: real-time estimate of wheel-side rotational inertia
Iw (as used in `dx6 = (tau_drive - Frx*rw - Ffx*rw)/Iw` in your acados
model) from live VESC `sensor_core` telemetry, with gear_ratio FIXED at
the known value 11.82.

Wheel angular velocity is computed EXACTLY the way `MPCNode.sensor_callback`
does it (same erpm -> motor_omega -> /gear_ratio chain, no odometry
needed), and the current signal used is `avg_iq` -- matching
`self.last_control[0] = msg.state.avg_iq`, which is what actually feeds
the `current` state / `tau_drive` term in your dynamics model.

Run directly on the car (no colcon package needed, as long as ROS2 +
vesc_msgs + rclpy are already sourced in your environment):

    chmod +x iw_estimator_node.py
    ./iw_estimator_node.py --ros-args -p sensor_core_topic:=/sensors/core

or:

    python3 iw_estimator_node.py --ros-args -p sensor_core_topic:=/sensors/core

--------------------------------------------------------------------------
CHECK/TUNE BEFORE TRUSTING THE NUMBER (search "TODO"):
  1. Message type assumed: vesc_msgs/msg/VescStateStamped.
  2. `motor_flux_linkage` (lambda) comes from F110 params.py (`lam`) --
     update if your car uses a different motor.
--------------------------------------------------------------------------

Method: RLS fit of
    domega_wheel/dt  =  a * avg_iq  +  b
where:
    omega_wheel = (erpm / pole_pairs) * (2*pi/60) / gear_ratio   [rad/s]
    (identical to MPCNode.sensor_callback's `self.omega_w`)

    a = gear_ratio * 1.5 * pole_pairs * lambda / Iw
    b = -load_torque_bias / Iw   (rolling resistance etc., absorbed so it
                                   doesn't bias the Iw estimate)

Iw is recovered every update as tau_per_amp / a and pushed through a
light smoothing filter for display/publish stability.
"""

import math
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

import matplotlib
matplotlib.use("TkAgg")  # switch to "Qt5Agg" if Tk isn't available on the car
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from vesc_msgs.msg import VescStateStamped
except ImportError as e:
    raise ImportError(
        "Could not import vesc_msgs.msg.VescStateStamped. If sensor_core "
        "uses a different custom message type with the same field names, "
        "update this import."
    ) from e


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RLS2:
    """2-parameter recursive least squares: y ~= a*u + b"""

    def __init__(self, forgetting_factor=0.995):
        self.lam = forgetting_factor
        self.theta = None
        self.P = None

    def update(self, u, y):
        phi = np.array([[u], [1.0]])
        if self.theta is None:
            self.theta = np.zeros((2, 1))
            self.P = np.eye(2) * 1e3

        Pphi = self.P @ phi
        denom = self.lam + float(phi.T @ Pphi)
        K = Pphi / denom
        err = y - float(phi.T @ self.theta)
        self.theta = self.theta + K * err
        self.P = (self.P - K @ phi.T @ self.P) / self.lam
        return float(self.theta[0, 0]), float(self.theta[1, 0])


class IwEstimator(Node):
    def __init__(self):
        super().__init__("iw_estimator")

        # ---- parameters ----
        self.declare_parameter("sensor_core_topic", "/sensors/core")
        self.declare_parameter("gear_ratio", 11.82)              # KNOWN, fixed (F110 params.py: gear_ratio)
        self.declare_parameter("pole_pairs", 2)                  # F110 params.py: poles/2 = 4/2
        self.declare_parameter("motor_flux_linkage", 0.000726)   # F110 params.py: lam
        self.declare_parameter("current_threshold", 1.0)         # [A] ignore near-idle noise (on avg_iq)
        self.declare_parameter("rls_forgetting_factor", 0.995)
        self.declare_parameter("deriv_filter_alpha", 0.2)        # EMA on domega/dt
        self.declare_parameter("Iw_filter_alpha", 0.1)           # EMA on displayed Iw
        self.declare_parameter("plot_window_sec", 10.0)
        # Nominal/CAD reference from F110 params.py: Iw = 0.9 * mw * r_Iw**2
        # with mw = 4*0.1 = 0.4 kg, r_Iw = 0.043 m  ->  ~6.656e-4 kg*m^2
        # Shown as a dashed reference line so you can compare the live
        # estimate against the design value. Set to 0.0 to hide it.
        self.declare_parameter("nominal_Iw_reference", 0.9 * (4 * 0.1) * (0.043 ** 2))

        gp = self.get_parameter
        self.gear_ratio = gp("gear_ratio").value
        self.pole_pairs = gp("pole_pairs").value
        self.lam_flux = gp("motor_flux_linkage").value
        self.i_thresh = gp("current_threshold").value
        self.deriv_alpha = gp("deriv_filter_alpha").value
        self.Iw_alpha = gp("Iw_filter_alpha").value
        self.plot_window = gp("plot_window_sec").value
        self.nominal_Iw = gp("nominal_Iw_reference").value

        topic = gp("sensor_core_topic").value
        self.sub = self.create_subscription(
            VescStateStamped, topic, self.on_sensor_core, 50
        )

        self.pub_Iw = self.create_publisher(Float64, "~/Iw_estimate", 10)
        self.pub_wheel_omega = self.create_publisher(Float64, "~/wheel_omega", 10)

        # state
        self._prev_t = None
        self._prev_wheel_omega = None
        self._filt_wheel_alpha = 0.0

        self.Iw_raw = None
        self.Iw_est = None  # smoothed, published/plotted value

        self.rls = RLS2(forgetting_factor=gp("rls_forgetting_factor").value)
        self.tau_per_amp = self.gear_ratio * 1.5 * self.pole_pairs * self.lam_flux

        self._lock = threading.Lock()
        maxlen = 3000
        self.t_buf = deque(maxlen=maxlen)
        self.i_buf = deque(maxlen=maxlen)
        self.Iw_buf = deque(maxlen=maxlen)
        self._t0 = None

        self.get_logger().info(
            f"Subscribed to {topic}, gear_ratio fixed at {self.gear_ratio}, "
            f"nominal Iw reference (CAD) = {self.nominal_Iw:.6g} kg*m^2, "
            f"waiting for data..."
        )

    def on_sensor_core(self, msg: VescStateStamped):
        t = stamp_to_sec(msg.header.stamp)
        avg_iq = msg.state.avg_iq
        erpm = msg.state.speed

        # Same chain as MPCNode.sensor_callback: erpm -> motor_omega -> omega_w
        motor_rpm = erpm / self.pole_pairs
        motor_omega = motor_rpm * (2.0 * math.pi / 60.0)
        wheel_omega = motor_omega / self.gear_ratio

        if self._prev_t is None:
            self._prev_t = t
            self._prev_wheel_omega = wheel_omega
            return

        dt = t - self._prev_t
        if dt <= 1e-4:
            return  # duplicate/out-of-order stamp, skip

        raw_alpha = (wheel_omega - self._prev_wheel_omega) / dt
        self._filt_wheel_alpha = (
            self.deriv_alpha * raw_alpha
            + (1 - self.deriv_alpha) * self._filt_wheel_alpha
        )

        self._prev_t = t
        self._prev_wheel_omega = wheel_omega

        # ---------------- Iw update (RLS), gated on avg_iq ----------------
        if abs(avg_iq) > self.i_thresh:
            a, _b = self.rls.update(avg_iq, self._filt_wheel_alpha)
            if abs(a) > 1e-9:
                Iw_hat = self.tau_per_amp / a
                if Iw_hat > 0:  # reject non-physical sign flips during transients
                    self.Iw_raw = Iw_hat
                    self.Iw_est = (
                        Iw_hat
                        if self.Iw_est is None
                        else self.Iw_alpha * Iw_hat + (1 - self.Iw_alpha) * self.Iw_est
                    )

        self.pub_wheel_omega.publish(Float64(data=wheel_omega))
        if self.Iw_est is not None:
            self.pub_Iw.publish(Float64(data=self.Iw_est))

        # ---- periodic console print of the estimate (throttled to 1 Hz) ----
        if self._t0 is not None:
            if (t - self._t0) - getattr(self, "_last_print_t", -1e9) >= 1.0:
                self._last_print_t = t - self._t0
                if self.Iw_est is not None:
                    print(f"[Iw estimate] t={t - self._t0:6.1f}s  "
                          f"avg_iq={avg_iq:6.2f} A  "
                          f"Iw_est={self.Iw_est:.6e} kg*m^2  "
                          f"(nominal={self.nominal_Iw:.3e})")
                else:
                    print(f"[Iw estimate] t={t - self._t0:6.1f}s  "
                          f"avg_iq={avg_iq:6.2f} A  Iw_est=not yet converged")

        with self._lock:
            if self._t0 is None:
                self._t0 = t
            self.t_buf.append(t - self._t0)
            self.i_buf.append(avg_iq)
            self.Iw_buf.append(self.Iw_est if self.Iw_est is not None else float("nan"))


def spin_ros(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = IwEstimator()

    ros_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    ros_thread.start()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    (line_i,) = ax1.plot([], [], color="tab:blue")
    ax1.set_ylabel("avg_iq [A]")
    ax1.grid(True)

    (line_Iw,) = ax2.plot([], [], color="tab:green", label="Iw_est (live)")
    ax2.set_ylabel("Iw_est [kg m^2]")
    ax2.set_xlabel("time [s]")
    ax2.grid(True)
    if node.nominal_Iw > 0:
        ax2.axhline(
            node.nominal_Iw,
            color="tab:gray",
            linestyle="--",
            linewidth=1,
            label=f"nominal (CAD) = {node.nominal_Iw:.4g}",
        )
    ax2.legend(loc="upper right", fontsize=8)

    # live text readout: raw estimate + how many "wheel-equivalents" it is
    readout_text = ax2.text(
        0.02, 0.95, "", transform=ax2.transAxes,
        fontsize=10, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    def update(_frame):
        with node._lock:
            t = list(node.t_buf)
            i = list(node.i_buf)
            Iw = list(node.Iw_buf)

        if not t:
            return line_i, line_Iw, readout_text

        t_end = t[-1]
        t_start = max(0.0, t_end - node.plot_window)

        for ax, line, y, extra_vals in (
            (ax1, line_i, i, []),
            (ax2, line_Iw, Iw, [node.nominal_Iw] if node.nominal_Iw > 0 else []),
        ):
            line.set_data(t, y)
            ax.set_xlim(t_start, t_end + 1e-3)
            finite_y = [v for v in y if v == v] + extra_vals
            if finite_y:
                ymin, ymax = min(finite_y), max(finite_y)
                pad = 0.1 * (ymax - ymin + 1e-6)
                ax.set_ylim(ymin - pad, ymax + pad)

        if node.Iw_est is not None:
            if node.nominal_Iw > 0:
                ratio = node.Iw_est / node.nominal_Iw
                readout_text.set_text(
                    f"Iw_est     = {node.Iw_est:.4e} kg*m^2\n"
                    f"wheel-only = {node.nominal_Iw:.4e} kg*m^2\n"
                    f"ratio      = {ratio:.2f}x wheel inertia\n"
                    f"(reflected drivetrain adds ~{ratio - 1:.2f}x on top of the bare wheel)"
                )
            else:
                readout_text.set_text(f"Iw_est = {node.Iw_est:.4e} kg*m^2")
        else:
            readout_text.set_text("Iw_est: not yet converged")

        return line_i, line_Iw, readout_text

    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()  # blocks here; ROS spinning continues in the background thread

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()