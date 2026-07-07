#!/usr/bin/env python3
"""
Scrollable viewer for the .npz files saved by ekf_compare_plotter.py.

Loads raw_* / ekf_* arrays (t, x, y, theta, vx, vy, omega) and shows the
same 2x3 layout (x, y, theta, vx, vy, omega) as a static, scrollable plot.

Features:
    - Checkboxes to toggle RAW / EKF / EMA visibility independently
    - An EMA (exponential moving average) low-pass applied to the RAW
      finite-difference velocity channels (vx, vy, omega), plotted as a
      third line, with a slider to tune the EMA time constant live
    - A spike clamp built into the filter pipeline itself: outlier samples
      are clipped to a CAUSAL, trailing rolling percentile band (computed
      over the window_sec seconds immediately BEFORE each sample -- never
      anything after it) before they reach the EMA -- appropriate here
      because vx/vy are pure finite differences with no independent
      velocity measurement to check against, and this matches exactly what
      the live optitrack_node.py filter does, so tuning transfers directly

Controls:
    - "Position" slider : scrolls the visible window across the full time range
    - "Window" slider    : sets how many seconds are visible at once
    - "EMA tau [s]" slider : time constant for the EMA filter (0 = filter off)
    - "Spike clip %" slider : percentile band for the spike clamp (0 = off)
    - "Spike window [s]" slider : width of the rolling window the percentile
      band is computed over (0 = degenerates to per-sample, no clamping)
    - Checkboxes (bottom-right) : show/hide raw / ekf / ema lines
    - Mouse scroll wheel over any axis : nudges the Position slider
    - Left/Right arrow keys            : nudge Position by 10% of window
    - Home key                         : reset to the full time range

Usage:
    python3 scrollable_data_viewer.py ekf_compare_data.npz
    python3 scrollable_data_viewer.py ekf_compare_data.npz --window 5.0 --ema-tau 0.1 --spike-pct 2.0 --spike-window 0.5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons


CHANNELS = [
    ('x', 'x [m]'),
    ('y', 'y [m]'),
    ('theta', 'theta [rad]'),
    ('vx', 'vx [m/s]'),
    ('vy', 'vy [m/s]'),
    ('omega', 'omega [rad/s]'),
]

# Only these channels get an EMA overlay -- they're the finite-differenced
# (noisy) ones. x/y/theta come straight from mocap/EKF pose, nothing to filter.
EMA_CHANNELS = {'vx', 'vy', 'omega'}

# Percentile band used to cap spikes INSIDE the filter pipeline itself
# (same idea as ekf_compare_plotter.py's --robust-scale: don't let a single
# dropped/duplicated mocap frame produce a huge finite-difference spike --
# but here the cap is applied to the samples the EMA actually filters,
# not just to the plot's y-axis limits).
SPIKE_PCT = 2.0   # -> clip to the 2nd..98th percentile band by default


def load_series(npz, prefix):
    """Pull out t/x/y/theta/vx/vy/omega arrays for a given prefix ('raw' or 'ekf')."""
    out = {}
    for key, _ in [('t', None)] + CHANNELS:
        arr_key = f'{prefix}_{key}'
        out[key] = npz[arr_key] if arr_key in npz else np.array([])
    return out


def spike_clip_windowed(t, v, window_sec, pct):
    """Percentile-based spike clamp computed over a CAUSAL (trailing-only)
    rolling window -- i.e. the same thing the live optitrack_node.py filter
    actually does, so tuning here transfers directly to the live system.

    For each sample i, look only at samples with timestamps in
    (t[i] - window_sec, t[i]] -- never anything after i -- take the
    [pct, 100-pct] percentile band of that trailing neighborhood, and
    clamp v[i] into it.

    This matters specifically for finite-difference vx/vy: there's no
    independent velocity sensor to check them against, so "is this sample
    an outlier" has to be judged relative to what the signal was just
    doing (e.g. mid-corner vs. on a straight), not against a single
    percentile band computed over the whole recording -- a global band
    would either clip real fast-corner spikes or fail to catch a mocap
    glitch that happens during a slow, low-variance stretch.

    window_sec<=0 or pct<=0 -> no-op (returns v unchanged).
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if window_sec <= 0 or pct <= 0 or n < 3:
        return v.copy()
    p = max(0.0, min(49.0, pct))
    out = np.empty_like(v)
    lo_idx = 0
    for i in range(n):
        # trailing-only sliding window -- lo_idx only ever advances toward i,
        # hi bound is always i itself (never looks past the current sample)
        while t[i] - t[lo_idx] > window_sec:
            lo_idx += 1
        window = v[lo_idx:i + 1]
        if len(window) < 3:
            out[i] = v[i]
            continue
        lo = np.percentile(window, p)
        hi = np.percentile(window, 100.0 - p)
        out[i] = min(max(v[i], lo), hi)
    return out


def ema_filter(t, v, tau, window_sec=0.0, spike_pct=0.0):
    """dt-normalized exponential moving average, with an optional windowed
    spike clamp applied to the input before filtering (see
    spike_clip_windowed). tau<=0 disables the EMA itself (spike clipping
    alone is still applied if window_sec>0 and spike_pct>0)."""
    v = spike_clip_windowed(t, v, window_sec, spike_pct)
    if tau <= 0 or len(v) == 0:
        return v.copy()
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            out[i] = out[i - 1]
            continue
        a = dt / (tau + dt)
        out[i] = out[i - 1] + a * (v[i] - out[i - 1])
    return out


def main():
    parser = argparse.ArgumentParser(description='Scrollable raw-vs-EKF data viewer.')
    parser.add_argument('npz_path', help='Path to the .npz file saved by ekf_compare_plotter.py')
    parser.add_argument('--window', type=float, default=10.0,
                         help='Initial visible window width in seconds (default %(default)s).')
    parser.add_argument('--ema-tau', type=float, default=0.1,
                         help='Initial EMA time constant in seconds (default %(default)s, 0=off).')
    parser.add_argument('--spike-pct', type=float, default=0.0,
                         help='Percentile band (default %(default)s, 0=off) used to clamp '
                              'outlier finite-difference spikes BEFORE they reach the EMA '
                              'filter -- e.g. 2.0 clips to the 2nd..98th percentile.')
    parser.add_argument('--spike-window', type=float, default=0.5,
                         help='Trailing time window in seconds (default %(default)s) that the '
                              'spike-clip percentile band is computed over -- CAUSAL, i.e. only '
                              'samples up to and including the current one, matching the live '
                              'optitrack_node.py filter (no lookahead). Local/rolling rather '
                              'than a single global band, since vx/vy here are pure finite '
                              'differences (no independent velocity measurement to check '
                              'against) and a fixed regime (e.g. corner vs straight) shouldn\'t '
                              'set the outlier threshold for the whole run.')
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    raw = load_series(data, 'raw')
    ekf = load_series(data, 'ekf')

    meta = {}
    if 'meta' in data:
        try:
            meta = dict(data['meta'])
        except Exception:
            meta = {}

    t_all = np.concatenate([a for a in (raw['t'], ekf['t']) if len(a) > 0]) if \
        (len(raw['t']) or len(ekf['t'])) else np.array([0.0, 1.0])
    t_min, t_max = float(np.min(t_all)), float(np.max(t_all))
    full_span = max(t_max - t_min, 1e-3)
    init_window = min(args.window, full_span)

    # Precompute initial EMA series for the finite-diff velocity channels
    # (spike_pct clamps outliers before the EMA sees them). We also keep a
    # separately-clipped copy of the RAW signal itself so the raw line can
    # be redrawn to show what actually got clipped, not just the filtered
    # output -- makes it obvious which samples the spike clamp touched.
    clipped_raw = {k: raw[k].copy() for k, _ in CHANNELS}
    ema = {k: np.array([]) for k, _ in CHANNELS}
    for k in EMA_CHANNELS:
        clipped_raw[k] = spike_clip_windowed(raw['t'], raw[k], args.spike_window, args.spike_pct)
        ema[k] = ema_filter(raw['t'], raw[k], args.ema_tau, args.spike_window, args.spike_pct)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    plt.subplots_adjust(bottom=0.33, left=0.09, right=0.97, hspace=0.4)
    title_bits = [f"{k}={v}" for k, v in meta.items()] if meta else []
    fig.suptitle('Recorded data — ' + (' | '.join(title_bits) if title_bits else args.npz_path))

    lines = {}   # key -> (ax, ln_raw, ln_ekf, ln_ema_or_None)
    for ax, (key, label) in zip(axes.flat, CHANNELS):
        (ln_raw,) = ax.plot(raw['t'], clipped_raw[key], color='tab:blue', linewidth=1.0, label='raw')
        (ln_ekf,) = ax.plot(ekf['t'], ekf[key], color='tab:orange', linewidth=1.2,
                             linestyle='--', label='ekf')
        ln_ema = None
        if key in EMA_CHANNELS:
            (ln_ema,) = ax.plot(raw['t'], ema[key], color='tab:green', linewidth=1.4,
                                 linestyle='-', alpha=0.9, label='raw (ema)')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)
        lines[key] = (ax, ln_raw, ln_ekf, ln_ema)

    handles = [lines['vx'][1], lines['vx'][2], lines['vx'][3]]
    fig.legend(handles, ['raw', 'ekf', 'raw (ema)'], loc='upper right')

    # --- sliders ---
    ax_pos = plt.axes([0.15, 0.24, 0.55, 0.03])
    ax_win = plt.axes([0.15, 0.19, 0.55, 0.03])
    ax_tau = plt.axes([0.15, 0.14, 0.55, 0.03])
    ax_spk = plt.axes([0.15, 0.09, 0.55, 0.03])
    ax_spw = plt.axes([0.15, 0.04, 0.55, 0.03])

    max_pos = max(t_max - init_window, t_min)
    s_pos = Slider(ax_pos, 'Position', t_min, max(max_pos, t_min + 1e-6),
                   valinit=t_min, valstep=full_span / 1000.0)
    s_win = Slider(ax_win, 'Window [s]', min(0.5, full_span), full_span,
                   valinit=init_window)
    s_tau = Slider(ax_tau, 'EMA tau [s]', 0.0, 2.0, valinit=args.ema_tau)
    s_spk = Slider(ax_spk, 'Spike clip %', 0.0, 20.0, valinit=args.spike_pct)
    s_spw = Slider(ax_spw, 'Spike window [s]', 0.0, min(5.0, full_span), valinit=args.spike_window)

    # --- checkboxes: raw / ekf / ema visibility ---
    ax_chk = plt.axes([0.78, 0.03, 0.14, 0.24])
    check = CheckButtons(ax_chk, ['raw', 'ekf', 'ema'], [True, True, True])

    vis = {'raw': True, 'ekf': True, 'ema': True}

    def _autoscale_y(ax, key, t0, t1):
        vis_vals = []
        if vis['raw']:
            vis_vals += [v for ti, v in zip(raw['t'], clipped_raw[key]) if t0 <= ti <= t1]
        if vis['ekf']:
            vis_vals += [v for ti, v in zip(ekf['t'], ekf[key]) if t0 <= ti <= t1]
        if vis['ema'] and key in EMA_CHANNELS:
            vis_vals += [v for ti, v in zip(raw['t'], ema[key]) if t0 <= ti <= t1]
        if not vis_vals:
            return
        lo, hi = min(vis_vals), max(vis_vals)
        pad = 0.1 * (hi - lo) if hi > lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)

    def redraw(_=None):
        window = s_win.val
        new_max_pos = max(t_max - window, t_min)
        s_pos.valmax = new_max_pos
        s_pos.ax.set_xlim(t_min, max(new_max_pos, t_min + 1e-6))

        t0 = min(s_pos.val, new_max_pos)
        t1 = t0 + window
        for key, (ax, ln_raw, ln_ekf, ln_ema) in lines.items():
            ax.set_xlim(t0, t1)
            ln_raw.set_visible(vis['raw'])
            ln_ekf.set_visible(vis['ekf'])
            if ln_ema is not None:
                ln_ema.set_visible(vis['ema'])
            _autoscale_y(ax, key, t0, t1)
        fig.canvas.draw_idle()

    def on_filter_change(_=None):
        tau = s_tau.val
        spike_pct = s_spk.val
        spike_window = s_spw.val
        for key in EMA_CHANNELS:
            clipped_raw[key] = spike_clip_windowed(raw['t'], raw[key], spike_window, spike_pct)
            ema[key] = ema_filter(raw['t'], raw[key], tau, spike_window, spike_pct)
            lines[key][1].set_ydata(clipped_raw[key])
            lines[key][3].set_ydata(ema[key])
        redraw()

    def on_check(label):
        vis[label] = not vis[label]
        redraw()

    s_pos.on_changed(redraw)
    s_win.on_changed(redraw)
    s_tau.on_changed(on_filter_change)
    s_spk.on_changed(on_filter_change)
    s_spw.on_changed(on_filter_change)
    check.on_clicked(on_check)

    def on_scroll(event):
        step = 0.05 * s_win.val * (1 if event.button == 'down' else -1)
        s_pos.set_val(min(max(s_pos.val + step, t_min), s_pos.valmax))

    def on_key(event):
        if event.key == 'right':
            s_pos.set_val(min(s_pos.val + 0.1 * s_win.val, s_pos.valmax))
        elif event.key == 'left':
            s_pos.set_val(max(s_pos.val - 0.1 * s_win.val, t_min))
        elif event.key == 'home':
            s_win.set_val(full_span)
            s_pos.set_val(t_min)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    redraw()
    plt.show()


if __name__ == '__main__':
    main()