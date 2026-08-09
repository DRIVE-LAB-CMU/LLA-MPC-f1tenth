#!/usr/bin/env python3
"""
Live /sensors/core monitor for checking current behavior against the
configured Motor Current Max (e.g. the 60A limit read from VESC Tool).

Graphs current_motor, duty_cycle, and speed in a rolling window, and prints
a line to stdout every time current_motor crosses above WARN_FRACTION of
CURRENT_MAX -- tracking each such excursion as an "event" with its peak
current and DURATION, so you can tell a brief high-current spike (e.g. a
hard launch, expected/fine) apart from a sustained draw near the limit
(worth checking temps / whether the constraint is being hit routinely).

Usage:
    python3 vesc_current_limit_monitor.py
    python3 vesc_current_limit_monitor.py --current-max 60.0 --warn-fraction 0.8
    python3 vesc_current_limit_monitor.py --topic /sensors/core --window 10.0
"""

import argparse
import threading
import signal
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vesc_msgs.msg import VescStateStamped

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MAX_SAMPLES = 5000
WINDOW_SEC = 10.0


class CurrentLimitMonitor(Node):
    def __init__(self, topic, current_max, warn_fraction):
        super().__init__('vesc_current_limit_monitor')
        self.lock = threading.Lock()
        self.current_max = current_max
        self.warn_thresh = current_max * warn_fraction

        self.t = deque(maxlen=MAX_SAMPLES)
        self.current_motor = deque(maxlen=MAX_SAMPLES)
        self.duty = deque(maxlen=MAX_SAMPLES)
        self.speed = deque(maxlen=MAX_SAMPLES)

        self._t0 = None

        # event tracking: are we currently inside a "near-limit" excursion?
        self._in_event = False
        self._event_start_t = None
        self._event_peak = 0.0
        self._event_peak_duty = 0.0
        self._event_peak_speed = 0.0
        self._event_n_samples = 0

        qos = QoSProfile(depth=10,
                          reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          history=QoSHistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(VescStateStamped, topic, self.cb, qos)
        self.get_logger().info(
            f'Listening on {topic}. current_max={current_max}A, '
            f'warn_thresh={self.warn_thresh:.1f}A. Close plot or Ctrl+C to quit.')

    def cb(self, msg):
        s = msg.state
        stamp = msg.header.stamp
        t_abs = stamp.sec + stamp.nanosec * 1e-9

        cm = s.current_motor
        duty = s.duty_cycle
        speed = s.speed

        with self.lock:
            if self._t0 is None:
                self._t0 = t_abs
            t_rel = t_abs - self._t0

            self.t.append(t_rel)
            self.current_motor.append(cm)
            self.duty.append(duty)
            self.speed.append(speed)

        above = abs(cm) >= self.warn_thresh

        if above and not self._in_event:
            # entering a new excursion
            self._in_event = True
            self._event_start_t = t_rel
            self._event_peak = cm
            self._event_peak_duty = duty
            self._event_peak_speed = speed
            self._event_n_samples = 1
        elif above and self._in_event:
            self._event_n_samples += 1
            if abs(cm) > abs(self._event_peak):
                self._event_peak = cm
                self._event_peak_duty = duty
                self._event_peak_speed = speed
        elif not above and self._in_event:
            # excursion just ended -- report it
            duration = t_rel - self._event_start_t
            pct_of_max = abs(self._event_peak) / self.current_max * 100.0
            kind = 'SUSTAINED' if duration > 0.5 else 'brief spike'
            print(f'[t={self._event_start_t:8.3f}-{t_rel:8.3f}s dur={duration:5.3f}s] '
                  f'{kind:<12s} peak_current={self._event_peak:7.2f}A ({pct_of_max:5.1f}% of max) '
                  f'@ duty={self._event_peak_duty:6.3f} speed={self._event_peak_speed:8.1f}erpm '
                  f'[{self._event_n_samples} samples]')
            self._in_event = False

    def snapshot(self):
        with self.lock:
            return (list(self.t), list(self.current_motor), list(self.duty), list(self.speed))


def main():
    parser = argparse.ArgumentParser(description='Live VESC current-limit monitor.')
    parser.add_argument('--topic', default='/sensors/core')
    parser.add_argument('--current-max', type=float, default=60.0,
                         help='Configured Motor Current Max in Amps (default %(default)s, '
                              'read this from VESC Tool -- see conversation).')
    parser.add_argument('--warn-fraction', type=float, default=0.8,
                         help='Fraction of current_max that triggers event tracking/printing '
                              '(default %(default)s -> 80%% of current_max).')
    parser.add_argument('--window', type=float, default=WINDOW_SEC,
                         help='Rolling time window shown, in seconds (default %(default)s).')
    args = parser.parse_args()

    rclpy.init()
    node = CurrentLimitMonitor(args.topic, args.current_max, args.warn_fraction)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    stop_requested = threading.Event()

    def handle_sigint(signum, frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, handle_sigint)

    fig, (ax_current, ax_duty, ax_speed) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f'VESC current monitor -- current_max={args.current_max}A, '
                 f'warn={args.current_max*args.warn_fraction:.1f}A')

    (ln_current,) = ax_current.plot([], [], color='tab:blue', linewidth=1.0)
    ax_current.axhline(args.current_max, color='red', linewidth=1.0, linestyle='--', label='current_max')
    ax_current.axhline(args.current_max * args.warn_fraction, color='orange', linewidth=1.0,
                        linestyle=':', label=f'{args.warn_fraction*100:.0f}% threshold')
    ax_current.axhline(-args.current_max, color='red', linewidth=1.0, linestyle='--')
    ax_current.axhline(-args.current_max * args.warn_fraction, color='orange', linewidth=1.0, linestyle=':')
    ax_current.set_ylabel('current_motor [A]')
    ax_current.legend(loc='upper right', fontsize=8)
    ax_current.grid(True, alpha=0.3)

    (ln_duty,) = ax_duty.plot([], [], color='tab:green', linewidth=1.0)
    ax_duty.set_ylabel('duty_cycle')
    ax_duty.grid(True, alpha=0.3)

    (ln_speed,) = ax_speed.plot([], [], color='tab:purple', linewidth=1.0)
    ax_speed.set_ylabel('speed [erpm]')
    ax_speed.set_xlabel('t [s]')
    ax_speed.grid(True, alpha=0.3)

    def update(_frame):
        if stop_requested.is_set():
            plt.close(fig)
            return []

        t, current_motor, duty, speed = node.snapshot()
        artists = [ln_current, ln_duty, ln_speed]
        if not t:
            return artists

        t_now = t[-1]
        t_min = t_now - args.window

        ln_current.set_data(t, current_motor)
        ln_duty.set_data(t, duty)
        ln_speed.set_data(t, speed)

        for ax, key_vals in ((ax_current, current_motor), (ax_duty, duty), (ax_speed, speed)):
            ax.set_xlim(max(0.0, t_min), max(args.window, t_now))
            vis = [v for ti, v in zip(t, key_vals) if ti >= t_min]
            if vis:
                lo, hi = min(vis), max(vis)
                pad = 0.1 * (hi - lo) if hi > lo else 0.5
                ax.set_ylim(lo - pad, hi + pad)

        # keep the current panel's y-range wide enough to show the limit lines
        cur_lo, cur_hi = ax_current.get_ylim()
        ax_current.set_ylim(min(cur_lo, -args.current_max * 1.1), max(cur_hi, args.current_max * 1.1))

        return artists

    _anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

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