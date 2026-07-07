#!/usr/bin/env python3
"""
Live /sensors/core listener that plots the linear duty-cycle torque
relationship in real time: predicted iq (from duty, V_bus, omega_e using
the F110() Rs/lambda/pole_pairs/gear_ratio constants) vs measured avg_iq.

Two live panels, updated every message:
    Left  : v_q terms vs time -- duty*V_bus*k, back_EMF, and measured avg_vq
    Right : predicted iq (open-loop, from duty/V_bus/omega_e) vs measured
            avg_iq, scatter, with the y=x reference line -- a point ON the
            line means the linear open-loop prediction matches reality.

Uses the empirical duty->v_q scale factor k (fit from a running median of
avg_vq / (duty*V_bus) rather than assumed to be exactly 1), since the raw
equation v_q = duty*V_bus does not hold as-is (see conversation: only
~0.55-0.59 of duty*V_bus shows up as v_q for this firmware's modulation
convention). k is recomputed continuously from live data so the plot
stabilizes as more samples arrive, rather than needing an offline sweep
first.

Usage:
    python3 vesc_torque_linearity_plotter.py
    python3 vesc_torque_linearity_plotter.py --topic /sensors/core --window 10.0
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

# --- F110() constants (llampc.params) ---
POLE_PAIRS = 4 / 2          # poles/2 = 2
GEAR_RATIO = 11.82
LAMBDA = 0.000726           # Wb
RS = 0.00954                # ohms

MAX_SAMPLES = 5000
WINDOW_SEC = 10.0
K_HISTORY = 200              # samples used for the rolling median of k = vq/(duty*Vbus)


class TorqueLinearityNode(Node):
    def __init__(self, topic):
        super().__init__('vesc_torque_linearity_plotter')
        self.lock = threading.Lock()

        self.t = deque(maxlen=MAX_SAMPLES)
        self.duty = deque(maxlen=MAX_SAMPLES)
        self.vbus = deque(maxlen=MAX_SAMPLES)
        self.speed = deque(maxlen=MAX_SAMPLES)
        self.avg_iq = deque(maxlen=MAX_SAMPLES)
        self.avg_vq = deque(maxlen=MAX_SAMPLES)

        self.back_emf = deque(maxlen=MAX_SAMPLES)
        self.iq_pred = deque(maxlen=MAX_SAMPLES)
        self.k_hist = deque(maxlen=K_HISTORY)   # rolling avg_vq/(duty*Vbus) samples
        self.k = 1.0 / np.sqrt(3.0)             # SVM-style starting guess, refined live

        self._t0 = None

        qos = QoSProfile(depth=10,
                          reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          history=QoSHistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(VescStateStamped, topic, self.cb, qos)
        self.get_logger().info(f'Listening on {topic}. Close the plot window or Ctrl+C to quit.')

    def cb(self, msg):
        s = msg.state
        stamp = msg.header.stamp
        t_abs = stamp.sec + stamp.nanosec * 1e-9

        duty = s.duty_cycle
        vbus = s.voltage_input
        speed_erpm = s.speed
        avg_iq = s.avg_iq
        avg_vq = s.avg_vq

        omega_e = speed_erpm * (2.0 * np.pi / 60.0)   # speed is already electrical RPM
        back_emf = omega_e * LAMBDA

        denom = duty * vbus
        if abs(denom) > 1e-6:
            self.k_hist.append(avg_vq / denom)

        if len(self.k_hist) >= 5:
            self.k = float(np.median(self.k_hist))

        vq_pred = self.k * denom
        iq_pred = (vq_pred - back_emf) / RS

        with self.lock:
            if self._t0 is None:
                self._t0 = t_abs
            self.t.append(t_abs - self._t0)
            self.duty.append(duty)
            self.vbus.append(vbus)
            self.speed.append(speed_erpm)
            self.avg_iq.append(avg_iq)
            self.avg_vq.append(avg_vq)
            self.back_emf.append(back_emf)
            self.iq_pred.append(iq_pred)

    def snapshot(self):
        with self.lock:
            return (list(self.t), list(self.avg_vq), list(self.back_emf),
                    list(self.avg_iq), list(self.iq_pred), self.k)


def main():
    parser = argparse.ArgumentParser(description='Live VESC duty->torque linearity check.')
    parser.add_argument('--topic', default='/sensors/core')
    parser.add_argument('--window', type=float, default=WINDOW_SEC,
                         help='Rolling time window for the left (time-series) panel, seconds.')
    args = parser.parse_args()

    rclpy.init()
    node = TorqueLinearityNode(args.topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    stop_requested = threading.Event()

    def handle_sigint(signum, frame):
        stop_requested.set()   # flag only -- matplotlib/Tk touched from the anim loop, not here

    signal.signal(signal.SIGINT, handle_sigint)

    fig, (ax_v, ax_scatter) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('VESC duty-cycle torque linearity (live)')

    # --- left: v_q terms over time ---
    ax_v.set_title('v_q terms vs time', fontsize=10)
    ax_v.set_xlabel('t [s]')
    ax_v.set_ylabel('V')
    ax_v.grid(True, alpha=0.3)
    (ln_vq_meas,) = ax_v.plot([], [], color='tab:blue', linewidth=1.2, label='avg_vq (measured)')
    (ln_back_emf,) = ax_v.plot([], [], color='tab:orange', linewidth=1.2, label='back_EMF (omega_e*lam)')
    ax_v.legend(loc='upper right', fontsize=8)

    # --- right: predicted vs measured iq scatter ---
    ax_scatter.set_title('predicted iq vs measured avg_iq', fontsize=10)
    ax_scatter.set_xlabel('measured avg_iq [A]')
    ax_scatter.set_ylabel('predicted iq [A] (open-loop, duty/Vbus/omega_e)')
    ax_scatter.grid(True, alpha=0.3)
    scatter = ax_scatter.scatter([], [], s=8, alpha=0.5, color='tab:green')
    (ln_ref,) = ax_scatter.plot([], [], color='k', linewidth=1.0, linestyle='--', label='y = x (perfect prediction)')
    k_text = ax_scatter.text(0.02, 0.95, '', transform=ax_scatter.transAxes, fontsize=9, va='top')
    ax_scatter.legend(loc='lower right', fontsize=8)

    def update(_frame):
        if stop_requested.is_set():
            plt.close(fig)
            return []

        t, avg_vq, back_emf, avg_iq, iq_pred, k = node.snapshot()
        if not t:
            return [ln_vq_meas, ln_back_emf, scatter, ln_ref, k_text]

        t_now = t[-1]
        t_min = t_now - args.window
        ln_vq_meas.set_data(t, avg_vq)
        ln_back_emf.set_data(t, back_emf)
        ax_v.set_xlim(max(0.0, t_min), max(args.window, t_now))
        vis_v = [v for ti, v in zip(t, avg_vq + back_emf if False else avg_vq) if ti >= t_min]
        vis_v += [v for ti, v in zip(t, back_emf) if ti >= t_min]
        if vis_v:
            lo, hi = min(vis_v), max(vis_v)
            pad = 0.1 * (hi - lo) if hi > lo else 0.5
            ax_v.set_ylim(lo - pad, hi + pad)

        pts = np.column_stack([avg_iq, iq_pred])
        scatter.set_offsets(pts)
        if len(avg_iq) > 0:
            lo = min(min(avg_iq), min(iq_pred))
            hi = max(max(avg_iq), max(iq_pred))
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            ax_scatter.set_xlim(lo - pad, hi + pad)
            ax_scatter.set_ylim(lo - pad, hi + pad)
            ln_ref.set_data([lo - pad, hi + pad], [lo - pad, hi + pad])

        k_text.set_text(f'fitted k = avg_vq/(duty*Vbus): {k:.3f}\n(SVM guess: {1/np.sqrt(3):.3f})')

        return [ln_vq_meas, ln_back_emf, scatter, ln_ref, k_text]

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