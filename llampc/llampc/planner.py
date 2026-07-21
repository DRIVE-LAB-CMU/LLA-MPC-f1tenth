""" Generate reference for MPC using a trajectory generator.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import numpy as np
from llampc.utils import Spline2D


def _speed_at(track, s, mu_est=None):
    """Resolve reference speed at arc-length s, handling both track modes.

    Single-profile track (spline_v is a Spline)  -> that profile.
    mu-bank track (spline_v is a list of Splines) -> interpolate by mu_est
        over track.mus; clamp to endpoints outside the baked range.
    """
    # bank mode: spline_v is a list and track.mus exists
    if isinstance(track.spline_v, (list, tuple)):
        mus = track.mus
        if mu_est is None:
            # no live estimate -> fall back to the most conservative (slowest)
            # profile, which is the smallest mu.
            return track.spline_v[0].calc(s)
        if mu_est <= mus[0]:
            return track.spline_v[0].calc(s)
        if mu_est >= mus[-1]:
            return track.spline_v[-1].calc(s)
        i = int(np.searchsorted(mus, mu_est))   # mus[i-1] < mu_est <= mus[i]
        vb = track.spline_v[i - 1].calc(s)
        va = track.spline_v[i].calc(s)
        w = (mu_est - mus[i - 1]) / (mus[i] - mus[i - 1])
        return vb * (1.0 - w) + va * w

    # single-profile mode
    return track.spline_v.calc(s)


def get_reference_trajectory_segment(x0, v0, track, N, Ts, projidx, scale=1.,
                                     wrap=True, skip=2, max_accel=9.51,
                                     mu_est=None):
    raceline = track.raceline
    num_pts = raceline.shape[1]
    search_indices = np.arange(projidx, projidx + 50) % num_pts
    search_window = raceline[:, search_indices]

    if np.allclose(search_window[:, 0], search_window[:, 1]):
        search_indices = (search_indices + 1) % num_pts
        search_window = raceline[:, search_indices]
    xy, idx_rel = track.project_fast_vectorized(x=x0[0], y=x0[1], raceline=search_window)
    new_projidx = search_indices[idx_rel]

    dist_start = track.spline.s[new_projidx]
    max_s = track.spline.s[-1]
    dist = dist_start
    v = max(v0, 0.2)

    # Generate N+1+skip points, then slice off the first `skip`
    total = N + 1 + skip
    xref_full = np.zeros([6, total])
    xref_full[:2, 0] = x0

    for idh in range(1, total):
        dist += scale * v * Ts
        if wrap:
            s_sample = dist % max_s
        else:
            if dist >= max_s:
                xref_full[:2, idh:] = track.spline.calc_position(max_s - 1e-4).reshape(2, 1)
                xref_full[2, idh:] = track.spline.calc_yaw(max_s - 1e-4)
                xref_full[3, idh:] = 0
                break
            s_sample = dist

        xref_full[:2, idh] = track.spline.calc_position(s_sample)
        xref_full[2, idh] = track.spline.calc_yaw(s_sample)

        v_next = _speed_at(track, s_sample, mu_est)          # <-- mode-aware lookup
        v = min(v_next, max_accel * Ts + v)  # velocity aware planner
        xref_full[3, idh] = v

    xref = xref_full[:, skip:skip + N + 1]

    return xref, new_projidx


def get_lookahead_point(x0, track, projidx, lookahead_dist, mu_est=None):
    """
    Walk forward along the raceline from projidx until a point is
    found at euclidean distance >= lookahead_dist from x0.
    """
    raceline = track.raceline
    num_pts = raceline.shape[1]

    search_indices = np.arange(projidx, projidx + 50) % num_pts
    search_window = raceline[:, search_indices]

    # Skip ahead if the first two points in the window are identical
    # to avoid zero-division in projection
    if np.allclose(search_window[:, 0], search_window[:, 1]):
        search_indices = (search_indices + 1) % num_pts
        search_window = raceline[:, search_indices]

    _, idx_rel = track.project_fast(x=x0[0], y=x0[1], raceline=search_window)
    new_projidx = search_indices[idx_rel]

    for i in range(num_pts):
        idx = (new_projidx + i) % num_pts
        dp = raceline[:2, idx] - x0[:2]
        if np.linalg.norm(dp) >= lookahead_dist:
            s = track.spline.s[idx] % track.spline.s[-1]
            xy = track.spline.calc_position(s)
            yaw = track.spline.calc_yaw(s)
            speed = _speed_at(track, s, mu_est)              # <-- mode-aware lookup
            return np.array([xy[0], xy[1], yaw, speed, 0.0, 0.0]), new_projidx

    # fallback: last idx examined
    s = track.spline.s[idx] % track.spline.s[-1]
    xy = track.spline.calc_position(s)
    yaw = track.spline.calc_yaw(s)
    speed = _speed_at(track, s, mu_est)                      # <-- mode-aware lookup
    return np.array([xy[0], xy[1], yaw, speed, 0.0, 0.0]), new_projidx