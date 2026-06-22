#!/usr/bin/env python3
"""
Live diagnostic plotter for /f1tenth/pose.

Subscribes to a geometry_msgs/PoseStamped topic, computes finite-difference
velocities (vx, vy in the WORLD frame, omega about z), and live-plots both the
pose (x, y, theta) and the velocities (vx, vy, omega) in a rolling time window.

This is meant for eyeballing the raw mocap stream -- rate, jitter, and how noisy
the finite-difference velocity actually is -- before deciding whether to feed any
of it to the EKF. It deliberately does NO smoothing, so what you see is the raw
1/dt amplification.

Usage:
    python3 pose_fd_plotter.py                 # defaults to /f1tenth/pose
    python3 pose_fd_plotter.py /some/other/pose

Notes:
  * Velocities are in the WORLD frame (consecutive map-frame positions
    differenced). robot_localization wants body-frame twist, so this is for
    *diagnosis*, not a drop-in twist source. See APPLY_BRIDGE_TRANSFORM below if
    you want the plotted pose to match your bridge's negated/permuted convention.
  * dt comes from msg.header.stamp (sensor time), matching your bridge. If you
    see spiky velocity with smooth position, look at the dt subplot-equivalent:
    irregular spacing is the usual culprit.
"""

import sys
import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_TOPIC = '/f1tenth/pose'
WINDOW_SEC = 10.0          # rolling time window shown on the x-axis
MAX_SAMPLES = 5000         # hard cap on buffered samples (memory guard)
APPLY_BRIDGE_TRANSFORM = False  # set True to mirror your bridge's x/y negation


def yaw_from_quat(x, y, z, w):
    """Extract yaw (rotation about z) from a quaternion. Valid for 2D heading."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """Smallest signed difference a - b, wrapped to (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class PoseFDPlotter(Node):
    def __init__(self, topic):
        super().__init__('pose_fd_plotter')

        # Match the mocap driver / your bridge: BEST_EFFORT, shallow queue.
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.sub = self.create_subscription(
            PoseStamped, topic, self.cb, qos)
        self.get_logger().info(
            f'Subscribing to {topic} (BEST_EFFORT). Close the plot window to quit.')

        # Shared rolling buffers (written in ROS thread, read in plot thread).
        self.lock = threading.Lock()
        self.t = deque(maxlen=MAX_SAMPLES)
        self.x = deque(maxlen=MAX_SAMPLES)
        self.y = deque(maxlen=MAX_SAMPLES)
        self.th = deque(maxlen=MAX_SAMPLES)
        self.vx = deque(maxlen=MAX_SAMPLES)
        self.vy = deque(maxlen=MAX_SAMPLES)
        self.om = deque(maxlen=MAX_SAMPLES)

        # Previous-sample state for finite differencing.
        self._prev = None   # (t, x, y, theta)
        self._t0 = None     # first timestamp, for a relative time axis

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        px = msg.pose.position.x
        py = msg.pose.position.y
        if APPLY_BRIDGE_TRANSFORM:
            px = -px
            py = -py

        theta = yaw_from_quat(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        if self._t0 is None:
            self._t0 = t
        t_rel = t - self._t0

        vx = vy = om = 0.0
        if self._prev is not None:
            pt, ppx, ppy, pth = self._prev
            dt = t - pt
            if dt > 1e-9:
                vx = (px - ppx) / dt
                vy = (py - ppy) / dt
                om = angle_diff(theta, pth) / dt
            else:
                # Duplicate/zero-dt sample: reuse nothing, skip velocity update.
                self._prev = (t, px, py, theta)
                return
        self._prev = (t, px, py, theta)

        with self.lock:
            self.t.append(t_rel)
            self.x.append(px)
            self.y.append(py)
            self.th.append(theta)
            self.vx.append(vx)
            self.vy.append(vy)
            self.om.append(om)

    def snapshot(self):
        """Return a consistent copy of all buffers for plotting."""
        with self.lock:
            return (list(self.t), list(self.x), list(self.y), list(self.th),
                    list(self.vx), list(self.vy), list(self.om))


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC

    rclpy.init()
    node = PoseFDPlotter(topic)

    # Spin ROS in a background thread so matplotlib owns the main thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'Finite-difference diagnostics  —  {topic}')

    specs = [
        (axes[0, 0], 'x', 'x [m]', 'tab:blue'),
        (axes[0, 1], 'y', 'y [m]', 'tab:blue'),
        (axes[0, 2], 'theta', 'theta [rad]', 'tab:blue'),
        (axes[1, 0], 'vx', 'vx [m/s]  (world)', 'tab:red'),
        (axes[1, 1], 'vy', 'vy [m/s]  (world)', 'tab:red'),
        (axes[1, 2], 'omega', 'omega [rad/s]', 'tab:red'),
    ]
    lines = {}
    for ax, key, label, color in specs:
        (ln,) = ax.plot([], [], color=color, linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)
        lines[key] = (ax, ln)

    def update(_frame):
        t, x, y, th, vx, vy, om = node.snapshot()
        if not t:
            return [ln for _, ln in lines.values()]

        t_now = t[-1]
        t_min = t_now - WINDOW_SEC
        data = {'x': x, 'y': y, 'theta': th, 'vx': vx, 'vy': vy, 'omega': om}

        for key, (ax, ln) in lines.items():
            ln.set_data(t, data[key])
            ax.set_xlim(max(0.0, t_min), max(WINDOW_SEC, t_now))
            # Autoscale y to the visible window only.
            vis = [v for ti, v in zip(t, data[key]) if ti >= t_min]
            if vis:
                lo, hi = min(vis), max(vis)
                pad = 0.1 * (hi - lo) if hi > lo else 0.5
                ax.set_ylim(lo - pad, hi + pad)
        return [ln for _, ln in lines.values()]

    # Keep a reference so the animation isn't garbage-collected.
    _anim = FuncAnimation(fig, update, interval=50, blit=False,
                          cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()   # blocks until the window is closed
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()