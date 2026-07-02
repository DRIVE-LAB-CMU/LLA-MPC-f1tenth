#!/usr/bin/env python3
"""
Live comparison plotter: raw mocap source  vs  EKF output (/odometry/filtered).

Each of the six axes (x, y, theta, vx, vy, omega) shows TWO lines:
    - RAW : the input source you picked (--source)
    - EKF : the same quantity read from nav_msgs/Odometry on --ekf-topic

Source modes (the RAW line):

  --source pose  (default)
      Subscribe to a geometry_msgs/PoseStamped (e.g. /f1tenth/pose) and compute
      finite-difference velocities ourselves. RAW vx/vy are BODY frame
      (vx = longitudinal/forward, vy = lateral), obtained by rotating the
      finite-differenced map-frame velocity by -theta.

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
  * Both lines are now BODY frame (vx = longitudinal, vy = lateral), so vx/vy
    traces should track each other directly regardless of heading. theta / x / y
    are directly comparable too.
  * Both lines share ONE relative time axis built from header.stamp, so the
    first message (from either source) defines t=0.
"""

import math
import argparse
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import signal
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_POSE_TOPIC = '/f1tenth/pose'
DEFAULT_ODOM_TOPIC = '/optitrack/odom'
DEFAULT_EKF_TOPIC = '/odometry/filtered'
WINDOW_SEC = 10.0          # rolling time window shown on the x-axis
MAX_SAMPLES = 5000         # hard cap on buffered samples (memory guard)

# Channels treated as "velocity" for outlier-robust autoscaling.
VELOCITY_KEYS = {'vx', 'vy', 'omega'}
# When robust autoscale is on, y-limits use this percentile band instead of
# min/max, so extreme finite-difference spikes don't blow up the axis.
ROBUST_PCT = 2.0           # -> 2nd..98th percentile

# pose-mode only: apply the SAME frame transform the optitrack bridge applies
# before it publishes /optitrack/odom to the EKF. Must stay True so the RAW
# (pose) line and the EKF line live in the same 'map' frame and actually overlay.
APPLY_BRIDGE_TRANSFORM = True

# Axis-permutation matrix, copied verbatim from the bridge (optitrack_node.py).
# x_new = -y_old, y_new = x_old, z_new = z_old.
_BRIDGE_P = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1],
])


def yaw_from_quat(x, y, z, w):
    """Extract yaw (rotation about z) from a quaternion. Valid for 2D heading."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """Smallest signed difference a - b, wrapped to (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def quat_to_rot(q):
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix. Matches the bridge."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def bridge_transform(raw_pos, raw_quat):
    """Replicate the bridge's pose remap exactly.

    raw_pos  : (px, py, pz) from the incoming PoseStamped
    raw_quat : (x, y, z, w) from the incoming PoseStamped
    returns  : (pos_xyz in 'map', theta) so the RAW line matches /optitrack/odom.
    """
    # Position: negate x and y, keep z (bridge: [-x, -y, z]).
    pos = np.array([-raw_pos[0], -raw_pos[1], raw_pos[2]])
    # Orientation: R_new = P @ R_orig, then take yaw from the resulting rotation.
    R_new = _BRIDGE_P @ quat_to_rot(raw_quat)
    theta = math.atan2(R_new[1, 0], R_new[0, 0])
    return pos, theta


def world_to_body(vx_world, vy_world, theta):
    """Rotate a map/world-frame velocity vector into the robot's body frame.

    Standard world<-body relationship is world_vel = R(theta) @ body_vel, with
    R(theta) = [[cos, -sin], [sin, cos]] rotating body->world. Body-frame
    velocity is therefore R(theta)^T @ world_vel:
        vx_body =  cos(theta) * vx_world + sin(theta) * vy_world   (longitudinal)
        vy_body = -sin(theta) * vx_world + cos(theta) * vy_world   (lateral)
    """
    c, s = math.cos(theta), math.sin(theta)
    vx_body = c * vx_world + s * vy_world
    vy_body = -s * vx_world + c * vy_world
    return vx_body, vy_body


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

    # ---- RAW pose mode: finite-difference velocity, rotated into BODY frame ----
    def cb_raw_pose(self, msg):
        t_abs, t_rel = self._rel_time(msg.header.stamp)

        raw_pos = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        raw_quat = (msg.pose.orientation.x, msg.pose.orientation.y,
                    msg.pose.orientation.z, msg.pose.orientation.w)

        if APPLY_BRIDGE_TRANSFORM:
            # Same remap the bridge applies before /optitrack/odom -> EKF, so the
            # RAW line is in 'map' and overlays the EKF line.
            pos, theta = bridge_transform(raw_pos, raw_quat)
            px, py = pos[0], pos[1]
        else:
            px, py = raw_pos[0], raw_pos[1]
            theta = yaw_from_quat(*raw_quat)

        vx = vy = om = 0.0
        if self._prev is not None:
            pt, ppx, ppy, pth = self._prev
            dt = t_abs - pt
            if dt > 1e-9:
                # Finite-difference velocity in the (transformed) map/world
                # frame, then rotate into BODY frame (vx=longitudinal,
                # vy=lateral) using the current heading, so this matches the
                # EKF's twist convention instead of only agreeing when
                # theta ~= 0.
                vx_world = (px - ppx) / dt
                vy_world = (py - ppy) / dt
                om = angle_diff(theta, pth) / dt
                vx, vy = world_to_body(vx_world, vy_world, theta)
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
    
def save_npz(path, raw, ekf, meta):
    """Dump raw + ekf snapshots (and run metadata) into a single npz file."""
    def arrs(series, prefix):
        return {f'{prefix}_{k}': np.asarray(v, dtype=float)
                for k, v in series.items()}
    payload = {}
    payload.update(arrs(raw, 'raw'))
    payload.update(arrs(ekf, 'ekf'))
    payload['meta'] = np.array(list(meta.items()), dtype=object)
    np.savez(path, **payload)
    print(f'[ekf_compare_plotter] Saved {len(raw["t"])} raw / '
        f'{len(ekf["t"])} ekf samples -> {path}')

def main():
    parser = argparse.ArgumentParser(
        description='Live raw-vs-EKF comparison plotter.')
    parser.add_argument('--source', choices=['pose', 'odom'], default='pose',
                         help="RAW source: 'pose' = PoseStamped + finite diff "
                              "(rotated into body vel); 'odom' = Odometry, "
                              "twist from msg (body vel).")
    parser.add_argument('--topic', default=None,
                         help='RAW topic (defaults per source).')
    parser.add_argument('--ekf-topic', default=DEFAULT_EKF_TOPIC,
                         help='EKF Odometry topic (default /odometry/filtered).')
    parser.add_argument('--robust-scale', action='store_true',
                         help='Ignore extreme velocity outliers when autoscaling '
                              'the vx/vy/omega y-axes (data is still plotted, just '
                              'not allowed to blow up the axis limits).')
    parser.add_argument('--robust-pct', type=float, default=ROBUST_PCT,
                         help='Percentile band for --robust-scale (default %(default)s '
                              '-> uses the p..(100-p) range). Lower = tighter clip.')
    parser.add_argument('--output', default='ekf_compare_data.npz',
                         help='Where to save raw/ekf finite-difference data '
                              'on exit (default %(default)s).')
    args = parser.parse_args()

    raw_topic = args.topic or (DEFAULT_ODOM_TOPIC if args.source == 'odom'
                                else DEFAULT_POSE_TOPIC)
    raw_vframe = 'body'
    ekf_vframe = 'body'

    rclpy.init()
    node = ComparePlotter(args.source, raw_topic, args.ekf_topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Guard so we only write the npz once, however we exit.
    save_lock = threading.Lock()
    saved = {'done': False}

    def do_save():
        with save_lock:
            if saved['done']:
                return
            saved['done'] = True
            raw, ekf = node.snapshot()
            save_npz(args.output, raw, ekf,
                      {'source': args.source, 'raw_topic': raw_topic,
                       'ekf_topic': args.ekf_topic})

    def handle_sigint(signum, frame):
        # Runs even if matplotlib's event loop is blocking plt.show().
        do_save()
        plt.close('all')

    signal.signal(signal.SIGINT, handle_sigint)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'RAW [{args.source}] {raw_topic}   vs   EKF {args.ekf_topic}')

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

    handles = [lines['x'][1], lines['x'][2]]
    fig.legend(handles, ['raw', 'ekf'], loc='upper right')

    def update(_frame):
        raw, ekf = node.snapshot()
        all_lns = []
        for _, lr, le in lines.values():
            all_lns.extend((lr, le))

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

            vis = [val for ti, val in zip(raw['t'], raw[key]) if ti >= t_min]
            vis += [val for ti, val in zip(ekf['t'], ekf[key]) if ti >= t_min]
            if vis:
                if args.robust_scale and key in VELOCITY_KEYS and len(vis) >= 5:
                    p = max(0.0, min(49.0, args.robust_pct))
                    lo = float(np.percentile(vis, p))
                    hi = float(np.percentile(vis, 100.0 - p))
                else:
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
        # Covers window close / normal return; do_save() is a no-op if the
        # SIGINT handler already saved.
        do_save()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()