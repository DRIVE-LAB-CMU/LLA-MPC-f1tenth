#!/usr/bin/env python3
"""
Live comparison plotter: raw mocap source  vs  EKF output (/odometry/filtered).

Each of the six axes (x, y, theta, vx, vy, omega) shows TWO lines:
    - RAW : the input source you picked (--source)
    - EKF : the same quantity read from nav_msgs/Odometry on --ekf-topic

Source modes (the RAW line):

  --source pose  (default)
      Subscribe to a geometry_msgs/PoseStamped (e.g. /f1tenth/pose) and compute
      finite-difference velocities ourselves. RAW vx/vy are WORLD frame.

  --source odom
      Subscribe to a nav_msgs/Odometry (e.g. /optitrack/odom) and read pose +
      twist straight from the message. RAW vx/vy are whatever the bridge
      published (BODY frame).

The EKF line always comes from nav_msgs/Odometry on --ekf-topic
(default /odometry/filtered): pose from pose.pose, velocities from twist.twist
(BODY frame, as robot_localization publishes them).

Usage:
    python3 ekf_compare_plotter.py                                  # raw=pose /f1tenth/pose vs /odometry/filtered
    python3 ekf_compare_plotter.py --source odom                   # raw=odom /optitrack/odom vs /odometry/filtered
    python3 ekf_compare_plotter.py --topic /f1tenth/pose --ekf-topic /odometry/filtered
    python3 ekf_compare_plotter.py --source odom --topic /optitrack/odom

Notes:
  * Frames can differ between the two lines: in pose mode RAW velocity is WORLD
    while EKF velocity is BODY, so vx/vy traces will only agree when the body
    is axis-aligned with the world. theta / x / y are directly comparable.
  * Both lines share ONE relative time axis built from header.stamp, so the
    first message (from either source) defines t=0.
"""

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
DEFAULT_EKF_TOPIC = '/odometry/filtered'
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


class _Series:
    """One rolling time-series bundle (t, x, y, theta, vx, vy, omega)."""

    def __init__(self):
        self.t = deque(maxlen=MAX_SAMPLES)
        self.x = deque(maxlen=MAX_SAMPLES)
        self.y = deque(maxlen=MAX_SAMPLES)
        self.th = deque(maxlen=MAX_SAMPLES)
        self.vx = deque(maxlen=MAX_SAMPLES)
        self.vy = deque(maxlen=MAX_SAMPLES)
        self.om = deque(maxlen=MAX_SAMPLES)

    def append(self, t, x, y, th, vx, vy, om):
        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.th.append(th)
        self.vx.append(vx)
        self.vy.append(vy)
        self.om.append(om)

    def snapshot(self):
        return {
            't': list(self.t), 'x': list(self.x), 'y': list(self.y),
            'theta': list(self.th), 'vx': list(self.vx),
            'vy': list(self.vy), 'omega': list(self.om),
        }


class ComparePlotter(Node):
    def __init__(self, source, topic, ekf_topic):
        super().__init__('ekf_compare_plotter')
        self.source = source

        self.lock = threading.Lock()
        self.raw = _Series()   # the chosen source
        self.ekf = _Series()   # /odometry/filtered

        self._prev = None      # pose-mode finite differencing state
        self._t0 = None        # shared first timestamp -> common time axis

        # --- RAW subscription -------------------------------------------------
        if source == 'odom':
            raw_qos = QoSProfile(depth=10,
                                 reliability=QoSReliabilityPolicy.RELIABLE,
                                 history=QoSHistoryPolicy.KEEP_LAST)
            self.sub_raw = self.create_subscription(
                Odometry, topic, self.cb_raw_odom, raw_qos)
        else:
            # Mocap driver publishes PoseStamped BEST_EFFORT.
            raw_qos = QoSProfile(depth=10,
                                 reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                 history=QoSHistoryPolicy.KEEP_LAST)
            self.sub_raw = self.create_subscription(
                PoseStamped, topic, self.cb_raw_pose, raw_qos)

        # --- EKF subscription -------------------------------------------------
        # robot_localization publishes /odometry/filtered with default RELIABLE QoS.
        ekf_qos = QoSProfile(depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST)
        self.sub_ekf = self.create_subscription(
            Odometry, ekf_topic, self.cb_ekf, ekf_qos)

        self.get_logger().info(
            f'RAW [{source}] <- {topic}   |   EKF <- {ekf_topic}. '
            f'Close the plot window to quit.')

    # ---- time helper -------------------------------------------------------
    def _rel_time(self, stamp):
        t = stamp.sec + stamp.nanosec * 1e-9
        with self.lock:
            if self._t0 is None:
                self._t0 = t
            t0 = self._t0
        return t, t - t0

    # ---- RAW pose mode: finite-difference velocity (WORLD frame) ------------
    def cb_raw_pose(self, msg):
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

        with self.lock:
            self.raw.append(t_rel, px, py, theta, vx, vy, om)

    # ---- RAW odom mode: pose + twist straight from message -----------------
    def cb_raw_odom(self, msg):
        _, t_rel = self._rel_time(msg.header.stamp)
        p = msg.pose.pose
        v = msg.twist.twist
        theta = yaw_from_quat(p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w)
        with self.lock:
            self.raw.append(t_rel, p.position.x, p.position.y, theta,
                            v.linear.x, v.linear.y, v.angular.z)

    # ---- EKF: pose + twist from /odometry/filtered -------------------------
    def cb_ekf(self, msg):
        _, t_rel = self._rel_time(msg.header.stamp)
        p = msg.pose.pose
        v = msg.twist.twist
        theta = yaw_from_quat(p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w)
        with self.lock:
            self.ekf.append(t_rel, p.position.x, p.position.y, theta,
                            v.linear.x, v.linear.y, v.angular.z)

    def snapshot(self):
        with self.lock:
            return self.raw.snapshot(), self.ekf.snapshot()


def main():
    parser = argparse.ArgumentParser(
        description='Live raw-vs-EKF comparison plotter.')
    parser.add_argument('--source', choices=['pose', 'odom'], default='pose',
                        help="RAW source: 'pose' = PoseStamped + finite diff "
                             "(world vel); 'odom' = Odometry, twist from msg "
                             "(body vel).")
    parser.add_argument('--topic', default=None,
                        help='RAW topic (defaults per source).')
    parser.add_argument('--ekf-topic', default=DEFAULT_EKF_TOPIC,
                        help='EKF Odometry topic (default /odometry/filtered).')
    args = parser.parse_args()

    raw_topic = args.topic or (DEFAULT_ODOM_TOPIC if args.source == 'odom'
                               else DEFAULT_POSE_TOPIC)

    raw_vframe = 'body' if args.source == 'odom' else 'world'
    ekf_vframe = 'body'

    rclpy.init()
    node = ComparePlotter(args.source, raw_topic, args.ekf_topic)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'RAW [{args.source}] {raw_topic}   vs   EKF {args.ekf_topic}')

    # Velocity titles flag the frame of each line so mismatches aren't surprising.
    vlabel = lambda comp: f'{comp} [m/s]  (raw:{raw_vframe}, ekf:{ekf_vframe})'
    specs = [
        (axes[0, 0], 'x', 'x [m]'),
        (axes[0, 1], 'y', 'y [m]'),
        (axes[0, 2], 'theta', 'theta [rad]'),
        (axes[1, 0], 'vx', vlabel('vx')),
        (axes[1, 1], 'vy', vlabel('vy')),
        (axes[1, 2], 'omega', 'omega [rad/s]'),
    ]

    lines = {}
    for ax, key, label in specs:
        (ln_raw,) = ax.plot([], [], color='tab:blue', linewidth=1.0,
                            label='raw')
        (ln_ekf,) = ax.plot([], [], color='tab:orange', linewidth=1.2,
                            linestyle='--', label='ekf')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)
        lines[key] = (ax, ln_raw, ln_ekf)

    # Single shared legend.
    handles = [lines['x'][1], lines['x'][2]]
    fig.legend(handles, ['raw', 'ekf'], loc='upper right')

    def update(_frame):
        raw, ekf = node.snapshot()
        all_lns = []
        for _, lr, le in lines.values():
            all_lns.extend((lr, le))

        # Time window driven by the latest sample across both series.
        latest = []
        if raw['t']:
            latest.append(raw['t'][-1])
        if ekf['t']:
            latest.append(ekf['t'][-1])
        if not latest:
            return all_lns
        t_now = max(latest)
        t_min = t_now - WINDOW_SEC

        for key, (ax, ln_raw, ln_ekf) in lines.items():
            ln_raw.set_data(raw['t'], raw[key])
            ln_ekf.set_data(ekf['t'], ekf[key])
            ax.set_xlim(max(0.0, t_min), max(WINDOW_SEC, t_now))

            # y-limits span the visible window of BOTH lines.
            vis = [val for ti, val in zip(raw['t'], raw[key]) if ti >= t_min]
            vis += [val for ti, val in zip(ekf['t'], ekf[key]) if ti >= t_min]
            if vis:
                lo, hi = min(vis), max(vis)
                pad = 0.1 * (hi - lo) if hi > lo else 0.5
                ax.set_ylim(lo - pad, hi + pad)
        return all_lns

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