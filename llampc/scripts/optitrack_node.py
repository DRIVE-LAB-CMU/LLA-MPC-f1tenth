#!/usr/bin/env python3
"""
Live diagnostic plotter for the mocap pipeline. Two source modes:

  --source pose  (default)
      Subscribe to a geometry_msgs/PoseStamped (e.g. /f1tenth/pose) and compute
      finite-difference velocities ourselves. vx/vy are WORLD frame. This shows
      the raw mocap stream and the raw 1/dt noise -- good for judging jitter.

  --source odom
      Subscribe to a nav_msgs/Odometry (e.g. /optitrack/odom) and read the pose
      AND the velocities straight out of the message. The vx/vy/omega plotted
      here are exactly what your bridge computed and published in twist.twist
      (BODY frame), not a re-derivation. If the bridge's twist block is still
      commented out those fields are zero, so the velocity plots read flat 0 --
      which is the signal that your twist isn't being published yet.

Usage:
    python3 pose_fd_plotter.py                          # pose mode, /f1tenth/pose
    python3 pose_fd_plotter.py --source odom            # odom mode, /optitrack/odom
    python3 pose_fd_plotter.py --source odom --topic /optitrack/odom
    python3 pose_fd_plotter.py --source pose --topic /some/other/pose

Notes:
  * pose-mode velocities are WORLD frame; odom-mode velocities are whatever the
    bridge published (BODY frame). Axis labels update to reflect this.
  * dt / time axis use header.stamp (sensor time) in both modes.
"""

import sys
import math
import argparse
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_POSE_TOPIC = '/f1tenth/pose'
DEFAULT_ODOM_TOPIC = '/optitrack/odom'
WINDOW_SEC = 10.0          # rolling time window shown on the x-axis
MAX_SAMPLES = 5000         # hard cap on buffered samples (memory guard)
APPLY_BRIDGE_TRANSFORM = False  # pose-mode only: mirror the bridge x/y negation


def yaw_from_quat(x, y, z, w):
    """Extract yaw (rotation about z) from a quaternion. Valid for 2D heading."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """Smallest signed difference a - b, wrapped to (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class PipelinePlotter(Node):
    def __init__(self, source, topic):
        super().__init__('pose_fd_plotter')
        self.source = source

        # Shared rolling buffers (written in ROS thread, read in plot thread).
        self.lock = threading.Lock()
        self.t = deque(maxlen=MAX_SAMPLES)
        self.x = deque(maxlen=MAX_SAMPLES)
        self.y = deque(maxlen=MAX_SAMPLES)
        self.th = deque(maxlen=MAX_SAMPLES)
        self.vx = deque(maxlen=MAX_SAMPLES)
        self.vy = deque(maxlen=MAX_SAMPLES)
        self.om = deque(maxlen=MAX_SAMPLES)

        self._prev = None   # (t, x, y, theta) -- pose mode finite differencing
        self._t0 = None     # first timestamp, for a relative time axis

        if source == 'odom':
            # Bridge publishes Odometry with default (RELIABLE) QoS.
            qos = QoSProfile(depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST)
            self.sub = self.create_subscription(Odometry, topic, self.cb_odom, qos)
        else:
            # Mocap driver publishes PoseStamped BEST_EFFORT.
            qos = QoSProfile(depth=10,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST)
            self.sub = self.create_subscription(PoseStamped, topic, self.cb_pose, qos)

        self.get_logger().info(
            f'[{source}] subscribing to {topic}. Close the plot window to quit.')

    # ---- helpers -----------------------------------------------------------
    def _rel_time(self, stamp):
        t = stamp.sec + stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = t
        return t, t - self._t0

    def _store(self, t_rel, x, y, th, vx, vy, om):
        with self.lock:
            self.t.append(t_rel)
            self.x.append(x)
            self.y.append(y)
            self.th.append(th)
            self.vx.append(vx)
            self.vy.append(vy)
            self.om.append(om)

    # ---- pose mode: compute finite-difference velocity (WORLD frame) -------
    def cb_pose(self, msg):
        t_abs, t_rel = self._rel_time(msg.header.stamp)

        px = msg.pose.position.x
        py = msg.pose.position.y
        if APPLY_BRIDGE_TRANSFORM:
            px, py = -px, -py

        theta = yaw_from_quat(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w)

        vx = vy = om = 0.0
        if self._prev is not None:
            pt, ppx, ppy, pth = self._prev
            dt = t_abs - pt
            if dt > 1e-9:
                vx = (px - ppx) / dt
                vy = (py - ppy) / dt
                om = angle_diff(theta, pth) / dt
            else:
                self._prev = (t_abs, px, py, theta)
                return
        self._prev = (t_abs, px, py, theta)

        self._store(t_rel, px, py, theta, vx, vy, om)

    # ---- odom mode: read pose + bridge-computed twist directly -------------
    def cb_odom(self, msg):
        _, t_rel = self._rel_time(msg.header.stamp)

        p = msg.pose.pose
        theta = yaw_from_quat(p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w)

        # Velocities are taken verbatim from the message -- these are the values
        # your bridge calculated and published (zero if the twist block is still
        # commented out).
        v = msg.twist.twist
        self._store(t_rel, p.position.x, p.position.y, theta,
                    v.linear.x, v.linear.y, v.angular.z)

    def snapshot(self):
        with self.lock:
            return (list(self.t), list(self.x), list(self.y), list(self.th),
                    list(self.vx), list(self.vy), list(self.om))


def main():
    parser = argparse.ArgumentParser(description='Live mocap pipeline plotter.')
    parser.add_argument('--source', choices=['pose', 'odom'], default='pose',
                        help="'pose' = PoseStamped + finite difference (world); "
                             "'odom' = Odometry, read bridge twist (body).")
    parser.add_argument('--topic', default=None,
                        help='Override topic (defaults per source).')
    args = parser.parse_args()

    topic = args.topic or (DEFAULT_ODOM_TOPIC if args.source == 'odom'
                           else DEFAULT_POSE_TOPIC)

    # Velocity-frame label changes with the source.
    vframe = 'body' if args.source == 'odom' else 'world'

    rclpy.init()
    node = PipelinePlotter(args.source, topic)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'Mocap diagnostics  —  [{args.source}]  {topic}')

    specs = [
        (axes[0, 0], 'x', 'x [m]', 'tab:blue'),
        (axes[0, 1], 'y', 'y [m]', 'tab:blue'),
        (axes[0, 2], 'theta', 'theta [rad]', 'tab:blue'),
        (axes[1, 0], 'vx', f'vx [m/s]  ({vframe})', 'tab:red'),
        (axes[1, 1], 'vy', f'vy [m/s]  ({vframe})', 'tab:red'),
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
            vis = [val for ti, val in zip(t, data[key]) if ti >= t_min]
            if vis:
                lo, hi = min(vis), max(vis)
                pad = 0.1 * (hi - lo) if hi > lo else 0.5
                ax.set_ylim(lo - pad, hi + pad)
        return [ln for _, ln in lines.values()]

    _anim = FuncAnimation(fig, update, interval=50, blit=False,
                          cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()