#!/usr/bin/env python3
"""
Live /sensors/core listener that checks the duty-cycle -> torque relationship
by fitting DIRECTLY in current-space, rather than fitting a voltage-domain
scale factor k and then dividing by Rs.

Background (see conversation): the model's torque equation is linear in two
regressors:
    iq = A*(duty*V_bus) - B*omega_e      where A = k/Rs,  B = lambda/Rs

Previously this was computed by fitting k = avg_vq/(duty*V_bus) in
voltage-space and combining it with a fixed Rs/lambda. That chains two
independently-fit quantities and divides by a small Rs, which amplifies
small (~2%) voltage-domain errors into large (~10x) current-domain errors
whenever duty*V_bus and omega_e*lambda are close in magnitude (e.g. coasting
above the pure-pursuit cutoff, still in the dynamics-model regime, but with
near-zero net current).

This version instead runs an ordinary least-squares fit of measured avg_iq
directly against (duty*V_bus, omega_e) -- i.e. fits A and B jointly against
the quantity that actually matters (current/torque), which is what the
earlier analysis showed removes most of the amplified error. A small
intercept term is included to absorb any constant sensor offset.

Three live panels:
    Left   : v_q terms vs time -- avg_vq (measured) and back_EMF = omega_e*lam
             (lam taken from the CURRENT fit, not the fixed F110() constant)
    Middle : predicted iq (from the live A/B fit) vs measured avg_iq, scatter
             with a y=x reference line
    Right  : residual (predicted - measured) vs time, to see whether errors
             are getting smaller/staying centered as the fit accumulates data

The fit is refit periodically (every REFIT_EVERY new samples, once at least
MIN_FIT_SAMPLES are buffered) using the most recent FIT_WINDOW samples, so it
adapts as more of the driving envelope is observed without needing an
offline sweep first.

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

# --- F110() constants (llampc.params) -- Rs kept fixed; k/lambda re-derived from the live fit ---
RS = 0.00954                 # ohms -- used only to translate the fitted A/B back into k, lambda for display
LAMBDA_PRIOR = 0.000726      # Wb, from F110() -- used as a ridge anchor, see _refit()

MAX_SAMPLES = 5000
FIT_WINDOW = 2000            # most recent N samples used for each refit
MIN_FIT_SAMPLES = 30         # don't attempt a fit until this many samples are buffered
REFIT_EVERY = 10             # recompute the least-squares fit every N new messages
WINDOW_SEC = 10.0
RIDGE_LAMBDA = 50.0          # ridge strength on c2 (anchored toward -LAMBDA_PRIOR/RS); see _refit()

RESID_PCT_THRESHOLD = 10.0       # print samples where |resid|/|avg_iq| exceeds this, in percent
RESID_PCT_MIN_ABS_IQ = 0.3       # A -- ignore near-zero-current samples, where % error is meaningless


class TorqueLinearityNode(Node):
    def __init__(self, topic):
        super().__init__('vesc_torque_linearity_plotter')
        self.lock = threading.Lock()

        self.t = deque(maxlen=MAX_SAMPLES)
        self.duty_vbus = deque(maxlen=MAX_SAMPLES)   # feature 1: duty*V_bus
        self.omega_e = deque(maxlen=MAX_SAMPLES)     # feature 2: electrical speed
        self.avg_iq = deque(maxlen=MAX_SAMPLES)      # regression target
        self.avg_vq = deque(maxlen=MAX_SAMPLES)      # for the left panel only

        self._t0 = None
        self._since_refit = 0

        # fit state: iq = c0 + c1*(duty*Vbus) + c2*omega_e
        self.c0 = 0.0
        self.c1 = 0.0
        self.c2 = -LAMBDA_PRIOR / RS   # start at the physically-expected value, not zero
        self.n_fit = 0
        self.cond_number = 0.0

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
        duty_vbus = duty * vbus

        with self.lock:
            if self._t0 is None:
                self._t0 = t_abs
            t_rel = t_abs - self._t0

            self.t.append(t_rel)
            self.duty_vbus.append(duty_vbus)
            self.omega_e.append(omega_e)
            self.avg_iq.append(avg_iq)
            self.avg_vq.append(avg_vq)

            self._since_refit += 1
            if len(self.avg_iq) >= MIN_FIT_SAMPLES and self._since_refit >= REFIT_EVERY:
                self._since_refit = 0
                self._refit()

            c0, c1, c2, n_fit = self.c0, self.c1, self.c2, self.n_fit

        # --- print samples where the CURRENT sample's residual exceeds 10% of measured avg_iq ---
        # (done outside checking n_fit==0 so we never divide/report before a fit exists;
        # RESID_PCT_MIN_ABS_IQ guards against meaningless % on near-zero current)
        if n_fit > 0 and abs(avg_iq) > RESID_PCT_MIN_ABS_IQ:
            iq_pred_now = c0 + c1 * duty_vbus + c2 * omega_e
            resid_now = iq_pred_now - avg_iq
            pct = abs(resid_now) / abs(avg_iq) * 100.0
            if pct > RESID_PCT_THRESHOLD:
                print(f'[t={t_rel:8.3f}s] duty={duty:6.3f} vbus={vbus:5.2f} speed={speed_erpm:8.1f}erpm '
                      f'| measured avg_iq={avg_iq:7.3f}A  predicted={iq_pred_now:7.3f}A  '
                      f'resid={resid_now:+7.3f}A ({pct:6.1f}%)  [n_fit={n_fit}]')

    def _refit(self):
        """Ridge-regularized least squares: iq = c0 + c1*(duty*Vbus) + c2*omega_e,
        using the most recent FIT_WINDOW samples.

        duty*Vbus and omega_e are highly correlated in ordinary driving (duty
        tracks speed), which makes the plain (unregularized) 2-feature fit
        ill-conditioned: c1/c2 individually swing wildly between refits even
        though their combined effect on the fit is stable, producing huge,
        unphysical predictions (and residuals) whenever a point deviates even
        slightly from the exact duty<->speed relationship the fit last saw.

        Fix: add a ridge penalty pulling c2 toward the value implied by the
        known F110() lambda (c2_prior = -lambda/Rs). This breaks the
        collinearity by giving the fit a preferred direction to fall back on,
        while still letting the data override it if there's enough
        independent (non-collinear) variation to support that.
        """
        n = len(self.avg_iq)
        start = max(0, n - FIT_WINDOW)
        f1 = np.asarray(list(self.duty_vbus)[start:], dtype=float)
        f2 = np.asarray(list(self.omega_e)[start:], dtype=float)
        y = np.asarray(list(self.avg_iq)[start:], dtype=float)

        X = np.column_stack([np.ones_like(f1), f1, f2])
        self.cond_number = float(np.linalg.cond(X))

        # augment with a ridge row: sqrt(RIDGE_LAMBDA)*c2 ~= sqrt(RIDGE_LAMBDA)*c2_prior
        # (only c2 is regularized -- c0, c1 are left free)
        c2_prior = -LAMBDA_PRIOR / RS
        ridge_row = np.array([[0.0, 0.0, np.sqrt(RIDGE_LAMBDA)]])
        ridge_target = np.array([np.sqrt(RIDGE_LAMBDA) * c2_prior])

        X_aug = np.vstack([X, ridge_row])
        y_aug = np.concatenate([y, ridge_target])

        coeffs, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
        self.c0, self.c1, self.c2 = (float(v) for v in coeffs)
        self.n_fit = len(y)

    def snapshot(self):
        with self.lock:
            n = len(self.avg_iq)
            start = max(0, n - FIT_WINDOW)
            t = list(self.t)[start:]
            duty_vbus = list(self.duty_vbus)[start:]
            omega_e = list(self.omega_e)[start:]
            avg_iq = list(self.avg_iq)[start:]
            avg_vq = list(self.avg_vq)[start:]
            c0, c1, c2, n_fit, cond = self.c0, self.c1, self.c2, self.n_fit, self.cond_number

        iq_pred = [c0 + c1 * dv + c2 * we for dv, we in zip(duty_vbus, omega_e)]
        # back_EMF uses lambda implied by the CURRENT fit (lam = -c2*Rs), not a fixed constant,
        # so the left panel reflects what the live regression actually believes right now
        lam_fit = -c2 * RS
        back_emf = [we * lam_fit for we in omega_e]
        return t, avg_vq, back_emf, avg_iq, iq_pred, (c0, c1, c2, n_fit, lam_fit, cond)


def main():
    parser = argparse.ArgumentParser(description='Live VESC duty->torque linearity check (direct current-space fit).')
    parser.add_argument('--topic', default='/sensors/core')
    parser.add_argument('--window', type=float, default=WINDOW_SEC,
                         help='Rolling time window for the time-series panels, seconds.')
    args = parser.parse_args()

    rclpy.init()
    node = TorqueLinearityNode(args.topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    stop_requested = threading.Event()

    def handle_sigint(signum, frame):
        stop_requested.set()   # flag only -- matplotlib/Tk touched from the anim loop, not here

    signal.signal(signal.SIGINT, handle_sigint)

    fig, (ax_v, ax_scatter, ax_resid) = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle('VESC duty-cycle torque linearity (live, direct current-space fit)')

    # --- left: v_q terms over time ---
    ax_v.set_title('v_q vs implied back-EMF (from live fit)', fontsize=10)
    ax_v.set_xlabel('t [s]')
    ax_v.set_ylabel('V')
    ax_v.grid(True, alpha=0.3)
    (ln_vq_meas,) = ax_v.plot([], [], color='tab:blue', linewidth=1.2, label='avg_vq (measured)')
    (ln_back_emf,) = ax_v.plot([], [], color='tab:orange', linewidth=1.2, label='back_EMF (fit-implied)')
    ax_v.legend(loc='upper right', fontsize=8)

    # --- middle: predicted vs measured iq scatter ---
    ax_scatter.set_title('predicted iq (fit) vs measured avg_iq', fontsize=10)
    ax_scatter.set_xlabel('measured avg_iq [A]')
    ax_scatter.set_ylabel('predicted iq [A]')
    ax_scatter.grid(True, alpha=0.3)
    scatter = ax_scatter.scatter([], [], s=8, alpha=0.5, color='tab:green')
    (ln_ref,) = ax_scatter.plot([], [], color='k', linewidth=1.0, linestyle='--', label='y = x (perfect prediction)')
    fit_text = ax_scatter.text(0.02, 0.97, '', transform=ax_scatter.transAxes, fontsize=8, va='top')
    ax_scatter.legend(loc='lower right', fontsize=8)

    # --- right: residual over time ---
    ax_resid.set_title('residual: predicted - measured iq', fontsize=10)
    ax_resid.set_xlabel('t [s]')
    ax_resid.set_ylabel('A')
    ax_resid.grid(True, alpha=0.3)
    ax_resid.axhline(0.0, color='k', linewidth=0.8)
    (ln_resid,) = ax_resid.plot([], [], color='tab:red', linewidth=1.0)

    def update(_frame):
        if stop_requested.is_set():
            plt.close(fig)
            return []

        t, avg_vq, back_emf, avg_iq, iq_pred, (c0, c1, c2, n_fit, lam_fit, cond) = node.snapshot()
        artists = [ln_vq_meas, ln_back_emf, scatter, ln_ref, fit_text, ln_resid]
        if not t:
            return artists

        t_now = t[-1]
        t_min = t_now - args.window

        # left panel
        ln_vq_meas.set_data(t, avg_vq)
        ln_back_emf.set_data(t, back_emf)
        ax_v.set_xlim(max(0.0, t_min), max(args.window, t_now))
        vis_v = [v for ti, v in zip(t, avg_vq) if ti >= t_min]
        vis_v += [v for ti, v in zip(t, back_emf) if ti >= t_min]
        if vis_v:
            lo, hi = min(vis_v), max(vis_v)
            pad = 0.1 * (hi - lo) if hi > lo else 0.5
            ax_v.set_ylim(lo - pad, hi + pad)

        # middle panel
        pts = np.column_stack([avg_iq, iq_pred]) if avg_iq else np.empty((0, 2))
        scatter.set_offsets(pts)
        if avg_iq:
            lo = min(min(avg_iq), min(iq_pred))
            hi = max(max(avg_iq), max(iq_pred))
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            ax_scatter.set_xlim(lo - pad, hi + pad)
            ax_scatter.set_ylim(lo - pad, hi + pad)
            ln_ref.set_data([lo - pad, hi + pad], [lo - pad, hi + pad])

        k_fit = c1 * RS
        cond_flag = '  <-- ILL-CONDITIONED, collinear duty/speed' if cond > 1000 else ''
        fit_text.set_text(
            f'n_fit = {n_fit}\n'
            f'iq = c0 + c1*(duty*Vbus) + c2*omega_e  (ridge on c2)\n'
            f'c0 = {c0:.4f} A\n'
            f'c1 = {c1:.4f}  (implied k = c1*Rs = {k_fit:.3f})\n'
            f'c2 = {c2:.6f}  (implied lambda = -c2*Rs = {lam_fit:.6f} Wb)\n'
            f'cond(X) = {cond:.0f}{cond_flag}'
        )

        # right panel
        resid = [p - m for p, m in zip(iq_pred, avg_iq)]
        ln_resid.set_data(t, resid)
        ax_resid.set_xlim(max(0.0, t_min), max(args.window, t_now))
        vis_r = [r for ti, r in zip(t, resid) if ti >= t_min]
        if vis_r:
            lo, hi = min(vis_r), max(vis_r)
            pad = 0.1 * (hi - lo) if hi > lo else 0.1
            ax_resid.set_ylim(lo - pad, hi + pad)

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