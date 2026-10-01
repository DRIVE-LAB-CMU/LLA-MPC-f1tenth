#!/usr/bin/env python3
"""fiala_car_3d.py - interactive 3-D before/after of the Fiala bank, for figures.

No run log involved: you set the car's state and input, and the tool draws
an F1TENTH car

  before   at the origin, heading +x, with its velocity vector, a yaw-rate
           arc, and the applied input on the wheels: a drive torque arrow on
           each rear wheel and a steering arrow on each front wheel
  after    one car per Fiala parameter set, each integrated forward by one
           step of length dt from that same state under that same input,
           coloured along the swept parameter (translucent)
  truth    where an imagined "real" car with the --truth params ends up after
           the same step, drawn solid under the translucent fan, so you can
           see which predictions land on it. The start car and every true
           state share one colour (--color truth=..., tyres tire=...)

WHAT THE INPUT IS

  dynamic_fiala.diffequation takes u = [current, steer] but never reads
  `current` - drive force comes from the rear wheel speed omega_w, which the
  bank carries in known_params. So the inputs are the steering angle and the
  rear wheel's surface speed rw*omega_w (m/s; equal to vx means
  free-rolling, above it is drive slip, below it is braking).

  The torque arrow shows what that wheel speed does: the rear tyre's
  longitudinal force Frx times the wheel radius, at the unswept (mean)
  params. Frx comes from diffequation itself, evaluated with the front tyre's
  grip set to zero so the longitudinal acceleration is the rear tyre's alone.
  The arrow's full length (the steering arrows' length) is the rear tyre's
  friction limit; it points backwards when braking.

THE STEP

  "One step" is one step of length dt, integrated with the same RK6 step the
  controller's banks use (rk6.rollout(_rk6_step, ...)), split into SUBSTEPS
  equal substeps so a long dt stays accurate. dt defaults to 0.3 s so the fan
  is visible; set it to 0.025 for the controller's actual 40 Hz step.

  --steps N (1-5) runs N steps, each with its own input (--steer and
  --wheel-speed take one value per step). The truth car rolls forward
  through all of them; at every step the bank is reset to the truth car's
  state and predicts one dt ahead under that step's input, as the controller
  would - predictions are never chained. Each new input's arrows sit on the
  truth car where it is applied. The fans and truth cars before the last
  step are the "intermediate cars" layer, at --mid-opacity times the final
  cars' opacity.

  --skip-step K draws step K (1 .. N-1) as a time skip: its fan and predicted
  paths go, the truth car it ends at becomes a dashed 3-D silhouette in the
  same place (redrawn for the view as the camera moves; u_{K+1}'s arrows
  stay on it), and the truth path into and out of it is dashed. The
  predicted paths stay solid. --no-skip-car leaves the car (and its arrows)
  out, so only the dashed path marks the skip.
  Only the drawing changes - the error graph still scores every step.

THE SWEEP AND THE TRUTH

  --sweep takes one or more of Cf Cr muf mur (or mu, which ties muf = mur).
  Each swept param takes --n values across mean +/- variation (the grid of
  fiala_setup() in scripts/llampc_fiala_fixed.py, overridable with --range);
  several swept params form a full grid. Unswept params sit at their mean.
  Cars are coloured by --color-by (default: the first swept param).
  --truth sets the ground-truth car's params (default: the mean).

THE WINDOW

  The 3-D view on the left is the export frame: it is letterboxed to the
  output size's aspect ratio (--size, or "Frame" in the panel) and shows
  nothing that won't be exported. Stroke widths (paths, grid) are real
  geometry in millimetres and label text scales with the frame height, so an
  export at any resolution looks like the preview. Exports are rendered
  offscreen at exactly the output size, from the current camera.

  The panel on the right holds every control, in sections: car state and
  input, sweep, ground truth, appearance (opacities, stroke widths, grid,
  colour map, every colour), layers, camera, frame, and export.

  Camera: elevation and azimuth about the view centre, and the view centre's
  x / y / z offset from the middle of the cars. Mouse rotate / pan in the
  view writes back into them; mouse zoom sets the distance, which they keep.
  Azimuth is measured from the car's forward axis (+x): 180 looks from
  behind, 90 from the car's left. Unless "lock camera" is ticked, changing
  the car or sweep refits the view to the cars at the current angles.

  Export (to the chosen folder):
    PNG / PNG (clear)  the frame at the output size; "clear" has a transparent
                       background (hide the ground and grid to keep just cars)
    PDF / SVG          vector export (VTK's gl2ps - translucent cars may not
                       survive; use PNG for those)
    legend             a vertical legend as .svg / .pdf / .png - the swept
                       colour bar (one band per car) plus entries for the
                       truth car, start pose and input arrows - and a .json of
                       every colour and value in it. The SVG keeps its text.
                       With it, the LLAMPC error graph (_error.*): every
                       model's error against the truth car, e = sum_i w_i
                       (x_i - x_hat_i)^2 with the controller's weights
                       (--cost-weights), inside each step (it resets with the
                       bank) and summed over the window. The model with the
                       smallest window cost - LLAMPC's pick - is drawn heavy.
                       It is exactly --graph-size W H inches (default 3 x 6;
                       PNG at 300 dpi); a tall graph puts its colour bar
                       across the bottom.
    settings           a .json of everything, camera included; "Load..." or
                       `--config that.json` reproduces it
  Closing the window caches the session (settings, camera, window size) in
  ~/.cache/fiala_car_3d/last_session.json; the next GUI launch without
  --config resumes it (flags still override it; --fresh ignores it).
  Exports are exactly what the frame shows. The colour bar and caption
  layers start hidden, since they belong in the legend; tick them under
  Layers to put them in the frame (and so in the export).

  Keys in the view: s = export PNG, k = lock / unlock camera, r = VTK reset.

HEADLESS

  --off-screen renders once to --save (.png/.pdf/.svg) and/or --legend at
  --size and exits; with --config it reproduces a saved session exactly.

Examples
  python fiala_car_3d.py --sweep mu --truth mu=0.45 --vx 3 --steer 0.3
  python fiala_car_3d.py --sweep Cf Cr --color-by Cr --layout side
  python fiala_car_3d.py --config fiala_car_3d_0_settings.json --off-screen \\
      --save poster_fan.png --legend poster_legend --size 4800 3200
"""

import argparse
import copy
import functools
import itertools
import json
import os
import re
import sys

import numpy as np
import pyvista as pv

# Make `import llampc` work when run as a plain script from the source tree.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import jax
import jax.numpy as jnp

from llampc.params import F110
from llampc.rollout.dynamic_fiala import diffequation
from llampc.rollout.rk6 import rollout, _rk6_step


# Bank order expected by diffequation.
BANK_KEYS = ['Cf', 'Cr', 'muf', 'mur', 'Cro']
# Mirrors fiala_setup() in scripts/llampc_fiala_fixed.py.
PARAM_MEAN = {'Cf': 250.0, 'Cr': 225.0, 'muf': 0.6, 'mur': 0.6, 'Cro': 0.0}
PARAM_VARIATION = {'Cf': 75.0, 'Cr': 75.0, 'muf': 0.5, 'mur': 0.5, 'Cro': 0.0}
SWEEPABLE = ['Cf', 'Cr', 'muf', 'mur', 'mu']
PARAM_TITLE = {'Cf': 'C_f  [N/rad]', 'Cr': 'C_r  [N/rad]', 'muf': 'mu_f', 'mur': 'mu_r',
               'mu': 'mu  (front = rear)'}
PARAM_TEX = {'Cf': r'$C_f$ [N/rad]', 'Cr': r'$C_r$ [N/rad]', 'muf': r'$\mu_f$',
             'mur': r'$\mu_r$', 'mu': r'$\mu$ (front = rear)'}

SUBSTEPS = 40               # RK6 substeps per dt: <= 25 ms up to dt = 1 s
# LLAMPC's lookback cost weights on (x, y, psi, vx, vy, omega), as in
# scripts/llampc_fiala_*.py.
COST_WEIGHTS = [0.0, 0.0, 20.0, 1.0, 10.0, 0.1]
G = 9.81

# F1TENTH geometry [m], body frame: x forward from the CG, y left, z up.
WHEEL_WIDTH = 0.043
WHEEL_Y = 0.135             # lateral offset of each wheel centre
CAR_LENGTH = 0.50           # bumper to bumper, for camera framing
ROOF_Z = 0.155              # top of the lidar
INPUT_LEN = 0.26            # steer arrow length = full-scale torque arrow
GLYPH_Z = ROOF_Z + 0.12     # velocity / yaw-rate glyphs float above the car

# Ground: a square of GROUND_SIZE centred ahead of the car.
GROUND_SIZE = 30.0
GROUND_CENTER = (4.5, 0.0)

# Pixel sizes (label text, colour bar) are for a 1000 px tall frame and scale
# with the frame height, so any export resolution matches the preview.
FRAME_REF_PX = 1000.0

COLORS = {
    'background': '#ffffff', 'ground': '#f4f4f2', 'grid': '#d9d9d6', 'text': '#000000',
    # the real car (start pose and every true state after it): body in
    # 'truth', tyres in 'tire'
    'truth': '#d7263d', 'tire': '#1c1c1c',
    # overlays
    'torque': '#f18f01', 'steer': '#2e9e5b',
    'velocity': '#222222', 'yaw': '#6a4c93',
    # integrated paths; 'auto' = predictions follow the colour map, the truth
    # path follows 'truth'
    'pred_path': 'auto', 'truth_path': 'auto',
}
AUTO_COLORS = ('pred_path', 'truth_path')
CAR_PARTS = ['chassis', 'deck', 'electronics', 'lidar', 'tire', 'rim']

# Session cached on closing the GUI and resumed by the next GUI launch.
SESSION_CACHE = os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
                             'fiala_car_3d', 'last_session.json')

# --view presets as (elevation, azimuth) [deg].
VIEW_PRESETS = {'iso': (31.0, -139.0), 'top': (89.5, 180.0), 'side': (14.0, -90.0),
                'front': (16.7, 0.0), 'rear': (19.3, 180.0)}

# Output aspect presets for the frame.
ASPECTS = {'3:2': (3, 2), '16:9': (16, 9), '4:3': (4, 3), '1:1': (1, 1), '2:1': (2, 1),
           '21:9': (21, 9), '2:3': (2, 3), '3:4': (3, 4), '9:16': (9, 16)}

# Scene layers, each with a "show" checkbox.
LAYERS = ['ground', 'grid', 'before car', 'after cars', 'truth car', 'intermediate cars',
          'paths', 'velocity', 'yaw rate', 'input arrows', 'input labels', 'colorbar',
          'caption']
MAX_STEPS = 5


# ===========================================================================
# Dynamics
# ===========================================================================
_rollout = rollout(_rk6_step, diffequation)


@functools.lru_cache(maxsize=None)
def _bank_rollout(n):
    """n substeps, vmapped over bank params; state, input, known params and
    h are shared. One compile per n."""
    return jax.jit(jax.vmap(lambda p, kp, x0, u, h: _rollout(p, kp, x0, u, h, n),
                            in_axes=(0, None, None, None, None)))


def known_params(car, wheel_speed):
    return jnp.array([car['mass'], car['Iz'], car['lf'], car['lr'], car['rw'],
                      wheel_speed / car['rw'], 0.0], dtype=jnp.float32)


def step_bank(bank_params, car, state, steer, wheel_speed, dt, steps=1):
    """(M, steps*SUBSTEPS+1, 6) trajectories of every bank model over `steps`
    steps of dt with the input held; step k ends at index k*SUBSTEPS."""
    x0 = jnp.asarray(state, dtype=jnp.float32)
    u = jnp.array([0.0, steer], dtype=jnp.float32)
    traj = np.asarray(_bank_rollout(steps * SUBSTEPS)(
        jnp.asarray(bank_params, jnp.float32), known_params(car, wheel_speed), x0, u,
        jnp.float32(dt / SUBSTEPS)))
    start = np.broadcast_to(np.asarray(state, float), (len(traj), 1, 6))
    return np.concatenate([start, traj], axis=1)


def rear_drive(car, state, steer, wheel_speed):
    """(Frx [N], rear friction limit [N]) at the mean params.

    With muf = 0 the front tyre carries no force, so diffequation's
    d(vx)/dt = Frx / m + vy * omega isolates the rear tyre.
    """
    p = jnp.array([PARAM_MEAN['Cf'], PARAM_MEAN['Cr'], 0.0, PARAM_MEAN['mur'],
                   PARAM_MEAN['Cro']], dtype=jnp.float32)
    dx = np.asarray(diffequation(p, known_params(car, wheel_speed),
                                 jnp.asarray(state, jnp.float32),
                                 jnp.array([0.0, steer], jnp.float32)))
    frx = car['mass'] * (dx[3] - state[4] * state[5])
    if abs(frx) < 1e-3:     # free rolling: float noise, not a signed torque
        frx = 0.0
    frz = car['mass'] * G * car['lf'] / (car['lf'] + car['lr'])
    return float(frx), PARAM_MEAN['mur'] * frz


def build_sweep(sweep, n, ranges):
    """Grid over the swept params -> (bank_params (M, 5), {swept key: (M,) values})."""
    axes = []
    for key in sweep:
        base = 'muf' if key == 'mu' else key
        lo, hi = ranges.get(key, (PARAM_MEAN[base] - PARAM_VARIATION[base],
                                  PARAM_MEAN[base] + PARAM_VARIATION[base]))
        axes.append(np.linspace(lo, hi, n))

    combos = np.array(list(itertools.product(*axes))) if axes else np.zeros((1, 0))
    bank = np.tile([PARAM_MEAN[k] for k in BANK_KEYS], (len(combos), 1)).astype(float)
    values = {}
    for j, key in enumerate(sweep):
        targets = ['muf', 'mur'] if key == 'mu' else [key]
        for t in targets:
            bank[:, BANK_KEYS.index(t)] = combos[:, j]
        values[key] = combos[:, j]
    return bank, values


# ===========================================================================
# Meshes
# ===========================================================================
def _slab(x0, x1, y0, y1, z0, z1, r, n=7):
    """A plate with rounded corners (radius r in plan)."""
    r = min(r, (x1 - x0) / 2 - 1e-4, (y1 - y0) / 2 - 1e-4)
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90),
                       (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)):
        for t in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t), z0))
    pts = np.array(pts)
    base = pv.PolyData(pts, faces=np.r_[len(pts), np.arange(len(pts))])
    return base.extrude((0, 0, z1 - z0), capping=True).triangulate()


def _box(x0, x1, y0, y1, z0, z1):
    return pv.Box(bounds=(x0, x1, y0, y1, z0, z1)).triangulate()


def _cyl(center, direction, radius, height, res=24):
    return pv.Cylinder(center=center, direction=direction, radius=radius, height=height,
                       resolution=res).triangulate()


def _pose(x, y, psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])


class F1TenthMesh:
    """An F1TENTH car as {part: mesh}, built once in the body frame and posed
    by copy + transform. Only the front wheels depend on steer, so those are
    rebuilt per steering angle."""

    def __init__(self, car):
        self.lf, self.lr, self.rw = car['lf'], car['lr'], car['rw']
        lf, lr = self.lf, self.lr
        c = (lf - lr) / 2
        parts = {k: [] for k in CAR_PARTS}

        # Chassis tub, bumpers, suspension arms, deck standoffs.
        parts['chassis'] += [_slab(-lr - 0.05, lf + 0.05, -0.07, 0.07, 0.026, 0.044, 0.035),
                             _slab(lf + 0.045, lf + 0.085, -0.09, 0.09, 0.028, 0.048, 0.018),
                             _slab(-lr - 0.085, -lr - 0.045, -0.085, 0.085, 0.030, 0.050, 0.015)]
        for ax in (lf, -lr):
            for s in (1, -1):
                ys = sorted((s * 0.065, s * 0.11))
                parts['chassis'].append(_box(ax - 0.018, ax + 0.018, *ys, 0.036, 0.044))
                parts['chassis'].append(_box(ax - 0.008, ax + 0.008, *ys, 0.068, 0.074))
        for dx in (-0.13, 0.13):
            for s in (1, -1):
                parts['chassis'].append(_cyl((c + dx, s * 0.06, 0.07), (0, 0, 1), 0.0045, 0.05, 12))

        # Upper deck; battery under it, compute + heatsink on it; lidar at the front.
        parts['deck'].append(_slab(c - 0.16, c + 0.17, -0.08, 0.08, 0.095, 0.100, 0.025))
        parts['electronics'] += [_slab(c - 0.11, c + 0.07, -0.024, 0.024, 0.044, 0.074, 0.008),
                                 _box(c - 0.10, c + 0.0, -0.045, 0.045, 0.100, 0.117)]
        for i in range(7):
            x = c - 0.093 + i * 0.0145
            parts['electronics'].append(_box(x, x + 0.004, -0.04, 0.04, 0.117, 0.132))
        lx = c + 0.125
        parts['lidar'] += [_slab(lx - 0.025, lx + 0.025, -0.025, 0.025, 0.100, 0.124, 0.006),
                           _cyl((lx, 0, 0.1395), (0, 0, 1), 0.021, 0.031, 36)]

        rear_tire, rear_rim = self._wheels(-lr, 0.0)
        parts['tire'] += rear_tire
        parts['rim'] += rear_rim
        self.static = {k: pv.merge(v) for k, v in parts.items()}
        self._front = {}

    def ghost(self, x, y, psi, steer):
        """A simplified closed car at (x, y, psi) for the time-skip silhouette:
        body, deck, compute box, lidar and wheels (front ones steered) - few
        parts, so the outline stays clean. Outward-facing, as the silhouette
        filter needs."""
        lf, lr, rw = self.lf, self.lr, self.rw
        c = (lf - lr) / 2
        lx = c + 0.125
        parts = [_slab(-lr - 0.085, lf + 0.085, -0.08, 0.08, 0.026, 0.052, 0.03),
                 _slab(c - 0.16, c + 0.17, -0.08, 0.08, 0.086, 0.100, 0.025),
                 _box(c - 0.10, c + 0.0, -0.045, 0.045, 0.100, 0.132),
                 _cyl((lx, 0, 0.1275), (0, 0, 1), 0.023, 0.055, 36)]
        for ax in (lf, -lr):
            for s in (1, -1):
                w = _cyl((0, 0, 0), (0, 1, 0), rw, WHEEL_WIDTH, 48)
                if ax == lf:
                    w.rotate_z(np.degrees(steer), point=(0, 0, 0), inplace=True)
                parts.append(w.translate((ax, s * WHEEL_Y, rw), inplace=False))
        # Box and cylinder faces carry their own copies of shared corners;
        # merge them, or every edge reads as a border and none as silhouette.
        parts = [m.clean(tolerance=1e-6).compute_normals(
                     auto_orient_normals=True, consistent_normals=True, cell_normals=False,
                     split_vertices=False) for m in parts]
        return pv.merge(parts, merge_points=False).transform(_pose(x, y, psi), inplace=False)

    def _wheels(self, ax, steer):
        tires, rims = [], []
        for s in (1, -1):
            hub = np.array([ax, s * WHEEL_Y, self.rw])
            wheel = [(_cyl((0, 0, 0), (0, 1, 0), self.rw, WHEEL_WIDTH, 40), 'tire'),
                     (_cyl((0, s * (WHEEL_WIDTH / 2 + 0.0005), 0), (0, 1, 0), 0.033, 0.002, 32), 'rim'),
                     (_cyl((0, s * (WHEEL_WIDTH / 2 + 0.003), 0), (0, 1, 0), 0.009, 0.006, 16), 'rim')]
            for mesh, kind in wheel:
                mesh.rotate_z(np.degrees(steer), point=(0, 0, 0), inplace=True)
                mesh.translate(hub, inplace=True)
                (tires if kind == 'tire' else rims).append(mesh)
        return tires, rims

    def parts(self, x, y, psi, steer):
        key = round(float(steer), 5)
        if key not in self._front:
            if len(self._front) > 64:
                self._front.clear()
            tires, rims = self._wheels(self.lf, steer)
            self._front[key] = {'tire': pv.merge(tires), 'rim': pv.merge(rims)}
        front = self._front[key]
        T = _pose(x, y, psi)
        out = {}
        for k, mesh in self.static.items():
            m = mesh.merge(front[k]) if k in front else mesh.copy()
            out[k] = m.transform(T, inplace=False)
        return out


def _arrow(start, direction, length, shaft=0.012, tip=0.035):
    """Arrow of absolute length [m] with fixed-size shaft and head."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    head = min(0.06, 0.5 * length)
    shaft_mesh = pv.Cylinder(center=np.asarray(start) + d * (length - head) / 2, direction=d,
                             radius=shaft, height=max(length - head, 1e-4), resolution=20)
    cone = pv.Cone(center=np.asarray(start) + d * (length - head / 2), direction=d,
                   height=head, radius=tip, resolution=24)
    return shaft_mesh.merge(cone)


def velocity_arrow(state, scale=0.1):
    x, y, psi, vx, vy = state[:5]
    c, s = np.cos(psi), np.sin(psi)
    v = np.array([c * vx - s * vy, s * vx + c * vy, 0.0])
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        return None
    return _arrow((x, y, GLYPH_Z), v, speed * scale, shaft=0.012, tip=0.035)


def yaw_arc(state, radius=0.22):
    x, y, psi, _, _, omega = state
    if abs(omega) < 1e-3:
        return None
    angle = min(abs(omega) * 60.0, 300.0)
    polar = (radius * np.cos(psi + np.pi), radius * np.sin(psi + np.pi), 0.0)
    arc = pv.CircularArcFromNormal(center=(x, y, GLYPH_Z), resolution=48,
                                   normal=(0, 0, np.sign(omega)), polar=polar, angle=angle)
    pts = arc.points
    tangent = pts[-1] - pts[-2]
    tip = pv.Cone(center=pts[-1] + tangent / np.linalg.norm(tangent) * 0.025,
                  direction=tangent, height=0.06, radius=0.028, resolution=24)
    return arc.tube(radius=0.009).merge(tip)


def input_arrows(car, state, steer, torque_frac, label_side=1):
    """Arrows on the wheels: steering on each front wheel, drive torque on each
    rear wheel. Each leaves the tyre's leading edge at hub height, along the
    wheel's heading (rear arrows reverse when braking).

    Returns ({'steer': mesh, 'torque': mesh or None}, {'torque': xyz, 'steer':
    xyz}) where the xyz are the arrow tips on the label_side (+1 left, -1
    right) wheels.
    """
    x, y, psi = state[:3]
    rw = car['rw']
    fwd = np.array([np.cos(psi), np.sin(psi), 0.0])
    left = np.array([-np.sin(psi), np.cos(psi), 0.0])
    sdir = np.array([np.cos(psi + steer), np.sin(psi + steer), 0.0])
    cg = np.array([x, y, rw])
    steer_parts, torque_parts, tips = [], [], {}

    length = INPUT_LEN * min(abs(torque_frac), 1.2)
    tdir = fwd * (np.sign(torque_frac) or 1.0)
    for side in (1, -1):
        front = cg + car['lf'] * fwd + side * WHEEL_Y * left
        rear = cg - car['lr'] * fwd + side * WHEEL_Y * left

        start = front + sdir * rw
        steer_parts.append(_arrow(start, sdir, INPUT_LEN, shaft=0.01, tip=0.028))
        tstart = rear + tdir * rw
        if length > 0.01:
            torque_parts.append(_arrow(tstart, tdir, length, shaft=0.01, tip=0.028))
        if side == label_side:
            tips['steer'] = start + sdir * INPUT_LEN
            tips['torque'] = tstart + tdir * max(length, 0.03)

    meshes = {'steer': pv.merge(steer_parts),
              'torque': pv.merge(torque_parts) if torque_parts else None}
    return meshes, tips



def ground_plane():
    return pv.Plane(center=(*GROUND_CENTER, 0.0), direction=(0, 0, 1), i_size=GROUND_SIZE,
                    j_size=GROUND_SIZE, i_resolution=1, j_resolution=1)


def grid_mesh(spacing, width):
    """The ground grid as flat ribbons `width` [m] wide - real geometry, so the
    stroke looks the same at every export resolution."""
    if width <= 0 or spacing <= 0:
        return None
    half = GROUND_SIZE / 2
    cx, cy = GROUND_CENTER
    ticks = np.arange(-half, half + 1e-9, spacing)
    z, w = 0.0008, width / 2
    rects = [(cx + t - w, cx + t + w, cy - half, cy + half) for t in ticks] + \
            [(cx - half, cx + half, cy + t - w, cy + t + w) for t in ticks]
    pts, faces = [], []
    for i, (x0, x1, y0, y1) in enumerate(rects):
        pts += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        faces += [4, 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
    return pv.PolyData(np.array(pts), faces=np.array(faces))


def lead_in_path(state, length, max_turn=np.pi / 2, n=400):
    """The truth's path before `state`, `length` [m] long and ending at it:
    run backwards at the state's body velocities and yaw rate for up to
    `max_turn` [rad] of heading change, then straight. xy points, far end
    first; None when the car isn't moving."""
    x, y, psi, vx, vy, omega = map(float, state)
    speed = np.hypot(vx, vy)
    if speed < 1e-6 or length <= 0:
        return None
    dt = length / n / speed
    pts, h, turned = [(x, y)], psi, 0.0
    for _ in range(n):
        c, s = np.cos(h), np.sin(h)
        pts.append((pts[-1][0] - (c * vx - s * vy) * dt, pts[-1][1] - (s * vx + c * vy) * dt))
        if turned < max_turn:
            h -= omega * dt
            turned += abs(omega) * dt
    return np.array(pts[::-1])


def path_tubes(lines, width):
    """Integrated paths as tubes `width` [m] across, sitting on the ground."""
    if width <= 0:
        return None
    r = width / 2
    tubes = []
    for pts, value in lines:
        # One polyline, so one seamless tube (a line per segment would overlap
        # at every joint and show as beads when translucent).
        line = pv.PolyData(np.c_[pts, np.full(len(pts), r + 0.002)],
                           lines=np.r_[len(pts), np.arange(len(pts))])
        if value is not None:
            line.point_data['value'] = np.full(line.n_points, value)
        tubes.append(line.tube(radius=r, n_sides=12))
    return pv.merge(tubes)


def _blocked(eye, targets, tris, chunk=256):
    """For each target point: does the segment from `eye` to it cross any of
    the (T, 3, 3) triangles `tris`? (Moller-Trumbore, vectorised.)"""
    v0 = tris[:, 0]
    e1, e2 = tris[:, 1] - v0, tris[:, 2] - v0
    s = eye - v0                                             # (T, 3)
    q = np.cross(s, e1)                                      # (T, 3)
    tq = np.einsum('tk,tk->t', e2, q)                        # (T,)
    out = np.zeros(len(targets), bool)
    for i in range(0, len(targets), chunk):
        d = targets[i:i + chunk] - eye                       # (R, 3)
        h = np.cross(d[:, None, :], e2[None])                # (R, T, 3)
        a = np.einsum('rtk,tk->rt', h, e1)
        ok = np.abs(a) > 1e-14
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        u = f * np.einsum('rtk,tk->rt', h, s)
        v = f * (d @ q.T)
        t = f * tq[None]
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9) & (t < 1)
        out[i:i + chunk] = hit.any(axis=1)
    return out


def silhouette(mesh, eye, tris=None, step=0.004):
    """The outline of `mesh` seen from the point `eye`, as (n, 3) polylines.
    With `tris` (the mesh's triangles, (T, 3, 3)) hidden lines are removed:
    the outline is sampled every `step` [m] and only runs whose sight line
    to the eye is clear are kept."""
    from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
    f = vtkPolyDataSilhouette()
    f.SetInputData(mesh)
    f.SetDirectionToSpecifiedOrigin()
    f.SetOrigin(*eye)
    f.SetEnableFeatureAngle(False)      # the outline only, not every sharp edge
    f.SetBorderEdges(False)
    f.Update()
    out = pv.wrap(f.GetOutput())
    if out.n_lines == 0:
        return []
    out = out.strip(join=True)
    cells, lines, i = out.lines, [], 0
    while i < len(cells):
        n = cells[i]
        lines.append(out.points[cells[i + 1:i + 1 + n]])
        i += n + 1
    if tris is None:
        return lines
    eye = np.asarray(eye, float)
    samples = []
    for p in lines:
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step)), 1) + 1)
        samples.append(np.column_stack([np.interp(t, s, p[:, j]) for j in range(3)]))
    allq = np.vstack(samples)
    # Stop just short of each point, so its own surface doesn't count.
    d = eye - allq
    seen_all = ~_blocked(eye, allq + 0.002 * d / np.linalg.norm(d, axis=1, keepdims=True), tris)
    visible, i = [], 0
    for q in samples:
        seen = seen_all[i:i + len(q)]
        i += len(q)
        # Split into the visible runs (a lone visible sample is dropped).
        edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > 1:
                visible.append(q[a:b])
    return visible


def dashed_tubes(lines, width, dash, gap):
    """(n, 3) polylines as dashed tubes `width` [m] across: dashes of about
    `dash` [m], `gap` [m] apart, stretched so an open line starts and ends on
    a dash and a closed one holds a whole number of them."""
    if width <= 0:
        return None
    period = max(dash + gap, 1e-4)
    pts, cells, off = [], [], 0
    for p in lines:
        p = np.asarray(p, float)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        L = s[-1]
        if len(p) < 2 or L < 1e-5:
            continue
        if np.allclose(p[0], p[-1]):
            n = max(int(round(L / period)), 1)
            k = L / (n * period)
        else:
            n = max(int(round((L + gap) / period)), 1)
            k = L / (n * dash + (n - 1) * gap)
        for a in np.arange(n) * period * k:
            b = min(a + dash * k, L)
            if b - a < 1e-5:
                continue
            t = np.r_[a, s[(s > a) & (s < b)], b]
            q = np.column_stack([np.interp(t, s, p[:, j]) for j in range(3)])
            pts.append(q)
            cells.append(np.r_[len(q), np.arange(off, off + len(q))])
            off += len(q)
    if not pts:
        return None
    lines = pv.PolyData(np.vstack(pts), lines=np.concatenate(cells))
    return lines.tube(radius=width / 2, n_sides=10, capping=True)


# ===========================================================================
# Legend
# ===========================================================================
def llampc_error(pred, truth, weights):
    """LLAMPC's per-model prediction error (rollout/history.py
    get_lookback_error): sum_i w_i (x_i - x_hat_i)^2, yaw wrapped to
    [-pi, pi)."""
    err = np.asarray(truth) - np.asarray(pred)
    err[..., 2] = (err[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return np.sum(np.square(err) * np.asarray(weights), axis=-1)


# Every piece of text in the legend and the error graph, as
# key: (group, name, default text, default size [pt]). A text of None is a
# size only (tick numbers). In a text, {name} is filled in (the names each
# one takes are in its tooltip), \\ is a line break, and $...$ is maths
# (matplotlib mathtext: \mu, x_1, \hat{x}, \frac{a}{b}, \mathrm{...}).
# An empty text hides the element.
GRAPH_TEXT = {
    'err_title': ('error graph', 'title', '', 10.0),
    'err_ylabel_top': ('error graph', 'error y label', 'prediction error', 9.0),
    'err_ylabel_bottom': ('error graph', 'cost y label', 'window cost', 9.0),
    'err_xlabel': ('error graph', 'x label', 'time [s]', 9.0),
    'err_ticks': ('error graph', 'axis numbers', None, 9.0),
    'err_steps': ('error graph', 'step labels', '$u_{{k}}$', 8.0),
    'err_pick': ('error graph', 'pick key', 'LLAMPC pick: {pick}', 7.5),
    'err_truth': ('error graph', 'truth key', 'ground truth: {truth}', 7.5),
    'err_bar_label': ('error graph', 'colour bar title', '{param}', 9.0),
    'err_bar_ticks': ('error graph', 'colour bar numbers', None, 9.0),
    'leg_bar_label': ('legend', 'colour bar title', '{param}', 10.0),
    'leg_bar_ticks': ('legend', 'colour bar numbers', None, 10.0),
    'leg_truth': ('legend', 'ground truth',
                  r'ground truth \\ $\mu_f$ {muf}  $\mu_r$ {mur} \\ $C_f$ {Cf}  $C_r$ {Cr}', 10.0),
    'leg_truth_path': ('legend', 'truth path', 'ground-truth path', 10.0),
    'leg_pred_paths': ('legend', 'predicted paths', 'predicted paths', 10.0),
    'leg_earlier': ('legend', 'earlier truth', 'truth at earlier steps', 10.0),
    'leg_skip': ('legend', 'time skip', 'time skip', 10.0),
    'leg_start': ('legend', 'start pose', 'start pose', 10.0),
    'leg_torque': ('legend', 'torque', '{torque} torque', 10.0),
    'leg_steer': ('legend', 'steer', 'steer', 10.0),
    'leg_velocity': ('legend', 'velocity', 'velocity', 10.0),
}
# The {names} each text takes, and sample values to check a text with.
GRAPH_TEXT_SUBS = {'err_steps': ['k'], 'err_pick': ['pick'], 'err_bar_label': ['param'],
                   'err_truth': ['truth', 'muf', 'mur', 'Cf', 'Cr'],
                   'leg_bar_label': ['param'], 'leg_truth': ['muf', 'mur', 'Cf', 'Cr'],
                   'leg_torque': ['torque']}
_SUB_SAMPLES = {'k': '1', 'pick': r'$\mu$ 0.35', 'param': r'$\mu$', 'muf': '0.60',
                'truth': r'$\mu_f$ 0.68, $\mu_r$ 0.54',
                'mur': '0.60', 'Cf': '250', 'Cr': '225', 'torque': 'drive'}


def graph_text(style, key, **subs):
    """(text, size) of one graph element under `style` ({'text': {key:
    text}, 'size': {key: pt}} overrides); text None = hidden (or size only)."""
    _, _, default, size = GRAPH_TEXT[key]
    size = float((style or {}).get('size', {}).get(key, size))
    if default is None:
        return None, size
    text = (style or {}).get('text', {}).get(key, default)
    for name, value in subs.items():
        text = text.replace('{%s}' % name, str(value))
    text = re.sub(r'\s*\\\\\s*', '\n', text)      # \\ = line break
    return (text if text.strip() else None), size


def graph_text_problem(key, text):
    """Why `text` can't be drawn (bad maths, say), or None if it can."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    subs = {n: _SUB_SAMPLES[n] for n in GRAPH_TEXT_SUBS.get(key, [])}
    text, _ = graph_text({'text': {key: text}}, key, **subs)
    if text is None:
        return None
    fig = Figure()
    FigureCanvasAgg(fig)
    fig.text(0, 0, text)
    try:
        fig.canvas.draw()
    except Exception as e:                  # mathtext raises ValueError on bad maths
        return str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
    return None


def _draw_colorbar(fig, cax, info, orientation='vertical', label=(None, 9.0), ticks=9.0):
    """The swept-param colour bar: one band per car (up to 24), else the
    continuous map; the truth's value marked when it is on the bar. `label`
    is its (title, size), `ticks` the size of its numbers."""
    horiz = orientation == 'horizontal'
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    values = np.asarray(info['values'], float)
    if len(values) <= 24:
        # One band per car, in exactly the colour it was drawn with.
        cmap = mcolors.ListedColormap(info['colors'])
        if len(values) > 1:
            mids = (values[1:] + values[:-1]) / 2
            edges = np.r_[values[0] - (mids[0] - values[0]), mids,
                          values[-1] + (values[-1] - mids[-1])]
        else:
            edges = np.array([values[0] - 0.5, values[0] + 0.5])
        norm = mcolors.BoundaryNorm(edges, cmap.N)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=values,
                          orientation=orientation)
        cb.set_ticklabels([f'{v:.3g}' for v in values])
    else:
        norm = mcolors.Normalize(*info['clim'])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=info['cmap']), cax=cax,
                          orientation=orientation)
    if label[0]:
        cb.set_label(label[0], fontsize=label[1])
    cb.ax.tick_params(labelsize=ticks)
    cb.outline.set_linewidth(0.6)
    # The truth: a line at each of its values, tagged (f / r) on the side
    # away from the numbers when there are two.
    for mark in info.get('truth_marks', []):
        (cax.axvline if horiz else cax.axhline)(mark['value'], color=info['truth_color'], lw=2.5)
        if mark['tag']:
            if horiz:
                cax.text(mark['value'], 1.12, mark['tag'], transform=cax.get_xaxis_transform(),
                         ha='center', va='bottom', color=info['truth_color'],
                         fontsize=0.85 * ticks, fontweight='bold')
            else:
                cax.text(-0.18, mark['value'], mark['tag'], transform=cax.get_yaxis_transform(),
                         ha='right', va='center', color=info['truth_color'],
                         fontsize=0.85 * ticks, fontweight='bold')
    return cb


def _graph_rc(font, size):
    """matplotlib settings for the legend and error graph: text kept as text
    in the SVG, fonts embedded in the PDF, and `font` for the text and the
    maths ($\\mu$, $u_1$). A font installed since matplotlib built its cache
    is looked for in the usual font folders; one not found falls back to
    DejaVu Sans, with a warning."""
    import glob
    from matplotlib import font_manager as fm
    known = lambda: {f.name for f in fm.fontManager.ttflist}
    if font not in known():
        stem = font.replace(' ', '').lower()
        for d in ('~/.fonts', '~/.local/share/fonts', '/usr/share/fonts',
                  '/usr/local/share/fonts'):
            for path in glob.glob(os.path.join(os.path.expanduser(d), '**', '*.[ot]tf'),
                                  recursive=True):
                if os.path.basename(path).replace(' ', '').lower().startswith(stem):
                    fm.fontManager.addfont(path)
    rc = {'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': size,
          'mathtext.fontset': 'dejavusans'}
    if font in known():
        rc.update({'font.family': [font, 'DejaVu Sans'], 'mathtext.fontset': 'custom',
                   'mathtext.rm': font, 'mathtext.it': f'{font}:italic',
                   'mathtext.bf': f'{font}:bold', 'mathtext.sf': font})
    else:
        print(f'warning: font {font!r} not found - using DejaVu Sans')
        rc['font.family'] = ['DejaVu Sans']
    return rc


def _svg_in_px(path):
    """Rewrite a matplotlib SVG so one unit is one CSS pixel (1/96 in).

    matplotlib draws in points and writes text as `font-size: 36px`, which
    per SVG means 36 units = 36 pt. Browsers and Inkscape read it that way,
    but PowerPoint and Word take px as 1/96 in, so the text comes in at
    3/4 size (36 pt shows as 27 pt). Scaling every length by 96/72 makes
    both readings agree, and the drawing stays the same size."""
    k = 96 / 72
    num = re.compile(r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
    fmt = lambda v: f'{v:.6g}'
    scale_all = lambda s: num.sub(lambda m: fmt(float(m.group()) * k), s)

    def transform(s):
        # translate(x y): both; rotate(a cx cy): the centre only.
        def one(m):
            name, vals = m.group(1), num.findall(m.group(2))
            if name == 'translate':
                vals = [fmt(float(v) * k) for v in vals]
            elif name == 'rotate':
                vals = vals[:1] + [fmt(float(v) * k) for v in vals[1:]]
            elif name == 'matrix':
                vals = vals[:4] + [fmt(float(v) * k) for v in vals[4:]]
            return f"{name}({' '.join(vals)})"
        return re.sub(r'(\w+)\(([^)]*)\)', one, s)

    def style(s):
        return re.sub(r'((?:font-size|stroke-width|stroke-dasharray|stroke-dashoffset):)'
                      r'([^;]*)', lambda m: m.group(1) + scale_all(m.group(2)), s)

    with open(path) as f:
        svg = f.read()
    root = re.search(r'<svg\b[^>]*>', svg)
    head = root.group()
    w, h = (float(re.search(rf'\s{a}="([\d.]+)pt"', head).group(1)) for a in ('width', 'height'))
    head = re.sub(r'\swidth="[^"]*"', f' width="{fmt(w * k)}px"', head)
    head = re.sub(r'\sheight="[^"]*"', f' height="{fmt(h * k)}px"', head)
    head = re.sub(r'\sviewBox="[^"]*"', f' viewBox="0 0 {fmt(w * k)} {fmt(h * k)}"', head)
    fix = {'d': scale_all, 'x': scale_all, 'y': scale_all, 'width': scale_all,
           'height': scale_all, 'transform': transform, 'style': style}
    body = re.sub(r'(\s)(d|x|y|width|height|transform|style)="([^"]*)"',
                  lambda m: f'{m.group(1)}{m.group(2)}="{fix[m.group(2)](m.group(3))}"',
                  svg[root.end():])
    with open(path, 'w') as f:
        f.write(svg[:root.start()] + head + body)


def _thin_bar_ticks(fig, cb, pad=0.25):
    """Keep every n-th colour bar number, the fewest dropped so that none
    overlap (`pad` [em] apart) - large text on a short bar otherwise piles up.
    The truth's marks are separate lines and always stay."""
    ticks = list(cb.get_ticks())
    if len(ticks) < 2:
        return
    horiz = cb.orientation == 'horizontal'
    labels = [t.get_text() for t in (cb.ax.get_xticklabels() if horiz
                                      else cb.ax.get_yticklabels())]
    for step in range(1, len(ticks)):
        keep = list(range(0, len(ticks), step))
        cb.set_ticks([ticks[i] for i in keep], labels=[labels[i] for i in keep])
        if not _overlaps(fig, cb.ax.get_xticklabels() if horiz else cb.ax.get_yticklabels(),
                         pad):
            return


def _overlaps(fig, texts, pad=0.25):
    """Whether any two of `texts` (drawn) overlap, with `pad` [em] between."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in texts:
        if t.get_visible() and t.get_text():
            e = t.get_fontsize() * fig.dpi / 72 * pad / 2
            boxes.append(t.get_window_extent(r).expanded(1, 1).padded(e))
    return any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def _warn_overlap(fig, texts, what):
    if _overlaps(fig, texts, pad=0.0):
        print(f'warning: the {what} overlap - make the graph wider or the text smaller')


def _warn_outside(fig, base, axes, legends):
    """Warn of anything the figure edge cuts off (more than the graph size fits)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    cut = []
    for a in list(axes) + list(legends):
        b = a.get_tightbbox(r)
        if b is not None and (b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1
                              or b.y1 > fb.y1 + 1):
            name = ('key' if a in legends else a.get_ylabel() or a.get_xlabel()
                    or 'colour bar')
            cut.append(name)
    if cut:
        print(f'warning: {os.path.basename(base)}: cut off at the edge: {", ".join(cut)} '
              f'- make the graph bigger or the text smaller')


def _split_longest(text, colon_first=False):
    """`text` with its longest line broken in two at a space outside $maths$
    (after a ':' if `colon_first`, else the one nearest its middle), or None
    if no line has such a space."""
    lines = text.split('\n')
    for li in sorted(range(len(lines)), key=lambda i: -len(lines[i])):
        line, spots, math = lines[li], [], False
        for i, ch in enumerate(line):
            if ch == '$' and (i == 0 or line[i - 1] != '\\'):
                math = not math
            elif ch == ' ' and not math:
                spots.append(i)
        if spots:
            colon = [i for i in spots if line[i - 1] == ':'] if colon_first else []
            at = colon[0] if colon else min(spots, key=lambda i: abs(i - len(line) / 2))
            lines[li:li + 1] = [line[:at], line[at + 1:]]
            return '\n'.join(lines)
    return None


def _wrap_to_width(fig, leg):
    """Break the legend's labels over more lines, the longest first, until
    it fits inside the figure's left edge (it stands at the right)."""
    while True:
        fig.canvas.draw()
        if leg.get_window_extent().x0 >= fig.bbox.x0 + 2:
            return
        texts = sorted(leg.get_texts(), key=lambda t: -max(len(l) for l in
                                                           t.get_text().split('\n')) * t.get_fontsize())
        for t in texts:
            new = _split_longest(t.get_text(), colon_first=True)
            if new:
                t.set_text(new)
                break
        else:
            return


def _wrap_axis_labels(fig, axes):
    """Break each axis label over more lines while it is longer than its plot
    (a y label taller than the plot runs into the next one)."""
    for _ in range(8):
        fig.canvas.draw()
        changed = False
        for ax in axes:
            box = ax.get_window_extent()
            for label, span in ((ax.yaxis.label, box.height), (ax.xaxis.label, box.width)):
                if not label.get_text():
                    continue
                e = label.get_window_extent()
                if (e.height if label is ax.yaxis.label else e.width) > span:
                    new = _split_longest(label.get_text())
                    if new:
                        label.set_text(new)
                        changed = True
        if not changed:
            return


def _fit_blocks(fig, key_sf, main_sf, leg, key_gap, bar_sf, cb, bar, bar_gap, below):
    """Size the error graph's blocks to their contents: the key's block to
    the key plus `key_gap` [in], the colour bar's so the bar comes out `bar`
    [in] thick with `bar_gap` [in] clear of the plots. The plots get the rest."""
    def resize(sf, amount, horiz=False):
        # Its share of its parent's grid, given `amount` [in] of the parent.
        gs = sf._subplotspec.get_gridspec()
        parent = (fig if sf is key_sf else main_sf).bbox
        total = (parent.width if horiz else parent.height) / fig.dpi
        if sf is key_sf:
            ratios = [amount, max(total - amount, 0.1)]
        else:
            ratios = [max(total - amount, 0.1), amount]
        (gs.set_width_ratios if horiz else gs.set_height_ratios)(ratios)

    dpi = fig.dpi
    for _ in range(4):
        fig.canvas.draw()
        if leg is not None:
            resize(key_sf, leg.get_window_extent().height / dpi + key_gap)
        else:
            resize(key_sf, 0.0)
        if cb is not None:
            # The bar and gap are the inner cells; the rest of the block is its
            # numbers and title. Keep that, and make the cells gap + bar.
            ax = cb.ax.get_window_extent()
            block = bar_sf.bbox
            if below:
                other = block.height / dpi - ax.height / dpi * (1 + bar_gap / bar)
                resize(bar_sf, other + bar + bar_gap)
            else:
                other = block.width / dpi - ax.width / dpi * (1 + bar_gap / bar)
                resize(bar_sf, other + bar + bar_gap, horiz=True)
    fig.canvas.draw()


def write_error_plot(base, info, err, log=False, size=(3.0, 6.0), font='Aptos', style=None,
                     panels='both', bar=0.35, bar_gap=0.25, key_gap=0.15, line=2.0):
    """LLAMPC's error of every model against the truth over the window, as
    base.svg/.pdf/.png at exactly `size` (width, height) [in] - the PNG at
    300 dpi - and the numbers as base.json.

    Top: each model's error inside each step, from the truth state it was
    reset to (0 at the reset) to the step's end - the dot, the value LLAMPC
    scores. Bottom: the window cost, the running sum of those dots, whose
    argmin is the model LLAMPC picks (drawn heavy). `panels` = 'both',
    'error' or 'cost' (one plot only; the truth key and colour bar stay).
    Text and sizes: `style` (see GRAPH_TEXT).

    The key, the plots and the colour bar are stacked blocks: the bar `bar`
    [in] thick across the whole width (under the y labels too), or the whole
    height at the right of a wide graph; `key_gap` / `bar_gap` [in] clear
    between the key / bar and the plots. `line` [pt] is the models' line
    width; the pick's is 1.8 times it, outlined, and the dots grow with it."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(_graph_rc(font, 9))
    t, e = np.asarray(err['t']), np.asarray(err['error'])        # (K, S+1), (M, K, S+1)
    step_err = np.asarray(err['step_error'])                     # (M, K)
    window = np.asarray(err['window_cost'])                      # (M, K)
    m, k = step_err.shape
    best, dt = err['best'], err['dt']
    cols = err['colors']
    has_bar = info['color_by'] is not None and len(info['values']) > 1
    shown = {'both': ['error', 'cost'], 'error': ['error'], 'cost': ['cost']}[panels]

    if log:
        # Log axes: the zeros (each reset, the window's start) sit on a floor
        # five decades under the largest value shown.
        top_val = max(float(np.max(e)) if 'error' in shown else 0.0,
                      float(np.max(window)) if 'cost' in shown else 0.0)
        floor = max(top_val, 1e-12) * 1e-5
        e, step_err, window = (np.maximum(v, floor) for v in (e, step_err, window))
        zero = floor
    else:
        zero = 0.0

    # Constrained layout fits the labels inside `size`. The figure is split
    # into blocks - key on top, then the plots, the colour bar under them
    # (a tall graph) or at their right (a wide one) - each sized once its
    # contents are drawn (see _fit_blocks).
    w_in, h_in = map(float, size)
    fig = plt.figure(figsize=(w_in, h_in), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0, wspace=0)
    n = len(shown)
    bar_below = has_bar and h_in >= 1.3 * w_in
    key_sf, main_sf = fig.subfigures(2, 1, height_ratios=[0.1 * h_in, 0.9 * h_in],
                                     hspace=0)
    plot_sf, bar_sf = main_sf, None
    if bar_below:
        plot_sf, bar_sf = main_sf.subfigures(2, 1, height_ratios=[0.8, 0.2], hspace=0)
        bar_gs = bar_sf.add_gridspec(2, 1, height_ratios=[bar_gap, bar])
    elif has_bar:
        plot_sf, bar_sf = main_sf.subfigures(1, 2, width_ratios=[0.8, 0.2], wspace=0)
        bar_gs = bar_sf.add_gridspec(1, 2, width_ratios=[bar_gap, bar])
    gs = plot_sf.add_gridspec(n, 1)
    axes = {}
    for r, name in enumerate(shown):
        axes[name] = plot_sf.add_subplot(gs[r, 0], sharex=axes.get(shown[0]))
    top_ax, bottom_ax = axes[shown[0]], axes[shown[-1]]
    # Line widths [pt]: the models at `line`, the pick heavier with a black
    # outline; dots and outline grow with them.
    line = float(line)
    pick_lw = 1.8 * line
    edge = 0.5 + 0.2 * line
    heavy = [pe.Stroke(linewidth=pick_lw + 2 * edge, foreground='black'), pe.Normal()]
    dot = max(3.5, 1.6 * line + 1.5)

    order = [i for i in range(m) if i != best] + [best]          # best on top
    for i in order:
        lw, fx, z = (pick_lw, heavy, 3) if i == best else (line, None, 2)
        ms = 1.3 * dot if i == best else dot
        if 'error' in axes:
            ax = axes['error']
            for j in range(k):
                ax.plot(t[j], e[i, j], color=cols[i], lw=lw, path_effects=fx, zorder=z,
                        solid_capstyle='round')
            ax.plot(t[:, -1], step_err[i], 'o', color=cols[i], ms=ms,
                    mec='black' if i == best else 'none', mew=edge, zorder=z + 1)
        if 'cost' in axes:
            axes['cost'].plot(np.r_[0.0, t[:, -1]], np.r_[zero, window[i]], '-o',
                              color=cols[i], lw=lw, ms=ms,
                              mec='black' if i == best else 'none', mew=edge,
                              path_effects=fx, zorder=z)
    for ax in axes.values():
        for j in range(1, k):
            ax.axvline(j * dt, color='0.6', lw=0.7, ls=':', zorder=0)
        ax.grid(True, axis='y', color='0.9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        if log:
            ax.set_yscale('log')
            ax.set_ylim(bottom=zero)
    # A little room each side so the end dots and heavy lines aren't cut.
    top_ax.set_xlim(-0.03 * k * dt, 1.03 * k * dt)
    T = functools.partial(graph_text, style)
    for ax in axes.values():
        ax.tick_params(labelsize=T('err_ticks')[1])
        if ax is not bottom_ax:
            ax.tick_params(labelbottom=False)
    labels = [(key, axes[name].set_ylabel) for key, name in
              (('err_ylabel_top', 'error'), ('err_ylabel_bottom', 'cost')) if name in axes]
    for key, setter in labels + [('err_xlabel', bottom_ax.set_xlabel)]:
        text, pt = T(key)
        if text:
            setter(text, fontsize=pt)
    text, pt = T('err_title')
    if text:
        # Over the top plot (above the step labels), under the truth key.
        top_ax.set_title(text, fontsize=pt)
    if k > 1 and T('err_steps', k=1)[0]:
        # Each step's input over its span, as a top axis (so the layout sees it).
        top = top_ax.secondary_xaxis('top')
        top.set_xticks([(j + 0.5) * dt for j in range(k)],
                       [T('err_steps', k=j + 1)[0] or '' for j in range(k)],
                       fontsize=T('err_steps')[1])
        top.tick_params(length=0)
        top.spines['top'].set_visible(False)
    # One key above the plots: the truth's colour and LLAMPC's pick. (The
    # pick inside a plot squeezed it once its text outgrew the plot.)
    tp = info['truth_params']
    key = []
    text, pt = T('err_truth', truth=info['truth_text'], muf=f"{tp['muf']:.2f}",
                 mur=f"{tp['mur']:.2f}", Cf=f"{tp['Cf']:.0f}", Cr=f"{tp['Cr']:.0f}")
    if text:
        key.append((Patch(facecolor=info['truth_color'], edgecolor='none', label=text), pt))
    text, pt = T('err_pick', pick=err['best_label'])
    if text:
        # As drawn, but no bigger than its text allows.
        f = min(1.0, 0.55 * pt / (1.3 * dot))
        key.append((Line2D([0], [0], color=cols[best], lw=pick_lw * f,
                           path_effects=[pe.Stroke(linewidth=(pick_lw + 2 * edge) * f,
                                                   foreground='black'), pe.Normal()],
                           marker='o', ms=1.3 * dot * f, mec='black', mew=edge * f,
                           solid_capstyle='butt', label=text), pt))
    if key:
        # Handles and spacing scale with the largest text, each label its own size.
        big = max(pt for _, pt in key)
        leg = key_sf.legend(handles=[h for h, _ in key], loc='upper right', fontsize=big,
                            borderaxespad=0.0,
                            handlelength=1.2, handleheight=0.9, handletextpad=0.5,
                            borderpad=0.35, labelspacing=0.3, framealpha=1.0, edgecolor='0.7',
                            fancybox=True)
        leg.get_frame().set_linewidth(0.5)
        for txt, (_, pt) in zip(leg.get_texts(), key):
            txt.set_fontsize(pt)
    else:
        leg = None
    bar_text = dict(label=T('err_bar_label', param=PARAM_TEX.get(info['color_by'], '')),
                    ticks=T('err_bar_ticks')[1])
    cb = None
    if bar_below:
        cb = _draw_colorbar(bar_sf, bar_sf.add_subplot(bar_gs[1, 0]), info, 'horizontal',
                            **bar_text)
    elif has_bar:
        cb = _draw_colorbar(bar_sf, bar_sf.add_subplot(bar_gs[0, 1]), info, **bar_text)
    _fit_blocks(fig, key_sf, main_sf, leg, key_gap, bar_sf, cb, bar, bar_gap, bar_below)
    if leg is not None:
        _wrap_to_width(fig, leg)
    _wrap_axis_labels(fig, list(axes.values()) + ([cb.ax] if cb is not None else []))
    if cb is not None:
        _thin_bar_ticks(fig, cb)
    _fit_blocks(fig, key_sf, main_sf, leg, key_gap, bar_sf, cb, bar, bar_gap, bar_below)
    if k > 1 and T('err_steps', k=1)[0]:
        _warn_overlap(fig, top.get_xticklabels(), 'step labels (err_steps)')
    _warn_outside(fig, base, list(axes.values()) + ([cb.ax] if cb is not None else []),
                  [leg] if leg is not None else [])

    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{base}.{ext}', dpi=300, transparent=True)
    plt.close(fig)
    _svg_in_px(f'{base}.svg')
    with open(f'{base}.json', 'w') as f:
        json.dump(err, f, indent=1)
    print(f'saved {base}.svg / .pdf / .png / .json')


def write_legend(base, info, font='Aptos', style=None):
    """A vertical legend - the swept colour bar, one band per car, then the
    truth / start / input entries - as base.svg/.pdf/.png, and every colour
    and value in it as base.json."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(_graph_rc(font, 10))
    T = functools.partial(graph_text, style)
    # Each entry's text and size (an emptied text drops the entry).
    entries = []
    for e in info['entries']:
        text, pt = T(e['key'], **e.get('subs', {}))
        if text:
            entries.append(dict(e, label=text, size=pt))
    values = np.asarray(info['values'], float)
    has_bar = info['color_by'] is not None and len(values) > 0

    bar_h = 3.2 if has_bar else 0.0
    # Room for each entry by its lines and size (0.42 in for one line at 10 pt).
    ent_h = sum(0.2 + 0.22 * (e['label'].count('\n') + 1) * e['size'] / 10
                for e in entries) + 0.25
    fig = plt.figure(figsize=(2.3, bar_h + ent_h + 0.3))
    H = bar_h + ent_h + 0.3

    if has_bar:
        cax = fig.add_axes([0.14, (ent_h + 0.2) / H, 0.13, (bar_h - 0.1) / H])
        cb = _draw_colorbar(fig, cax, info,
                            label=T('leg_bar_label', param=PARAM_TEX.get(info['color_by'], '')),
                            ticks=T('leg_bar_ticks')[1])
        _thin_bar_ticks(fig, cb)

    handles = []
    for e in entries:
        if e['kind'] == 'patch':
            handles.append(Patch(facecolor=e['color'], edgecolor='none',
                                 alpha=e.get('alpha', 1.0), label=e['label']))
        elif e['kind'] == 'line':
            handles.append(Line2D([0], [0], color=e['color'], lw=3, alpha=e.get('alpha', 1.0),
                                  solid_capstyle='round', label=e['label']))
        elif e['kind'] == 'dashes':
            handles.append(Line2D([0], [0], color=e['color'], lw=2.2, alpha=e.get('alpha', 1.0),
                                  linestyle=(0, (2.2, 1.3)), dash_capstyle='butt',
                                  label=e['label']))
        else:
            handles.append(Line2D([0], [0], color=e['color'], lw=2.5, marker='>',
                                  markersize=7, label=e['label']))
    if handles:
        # Handles and spacing scale with the largest text, each label its own size.
        leg = fig.legend(handles=handles, loc='upper left',
                         bbox_to_anchor=(0.02, (ent_h + 0.05) / H), frameon=False,
                         fontsize=max(e['size'] for e in entries),
                         handlelength=1.6, borderaxespad=0.1, labelspacing=0.7)
        for txt, e in zip(leg.get_texts(), entries):
            txt.set_fontsize(e['size'])

    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{base}.{ext}', dpi=300, transparent=True, bbox_inches='tight',
                    pad_inches=0.05)
    plt.close(fig)
    _svg_in_px(f'{base}.svg')
    with open(f'{base}.json', 'w') as f:
        json.dump(dict(info, entries=entries), f, indent=2)
    print(f'saved {base}.svg / .pdf / .png / .json')


# ===========================================================================
# Scene
# ===========================================================================
class FialaScene:
    """The 3-D scene and its state. Draws into `plotter` (the GUI's embedded
    view) or, if none is given, into its own offscreen pv.Plotter at
    args.size. It has no UI of its own; the Qt panel drives it through the
    set_* methods."""

    def __init__(self, args, plotter=None, on_camera=None):
        self.args = args
        self.interactive = plotter is not None
        self.on_camera = on_camera
        self.car = F110()
        self.mesh = F1TenthMesh(self.car)
        self.colors = dict(COLORS, **args.color)
        steers = _per_step(args.steer, args.steer)
        wheels = _per_step(args.wheel_speed, args.vx)
        self.s = {'vx': args.vx, 'vy': args.vy, 'omega': args.omega, 'steer': steers[0],
                  'wheel': wheels[0], 'dt': args.dt}
        # Inputs of steps 2..MAX_STEPS (only the first steps-1 are used).
        self.later = [{'steer': st, 'wheel': wh} for st, wh in zip(steers[1:], wheels[1:])]
        self.truth = {k: PARAM_MEAN[k] for k in ('Cf', 'Cr', 'muf', 'mur')}
        for k, v in args.truth.items():
            for t in (['muf', 'mur'] if k == 'mu' else [k]):
                self.truth[t] = v
        self.sweep = list(args.sweep)
        self.camera_locked = False
        elev, azim = VIEW_PRESETS[args.view]
        self.cam = {'elev': elev if args.elev is None else args.elev,
                    'azim': azim if args.azim is None else args.azim,
                    'off_x': args.offset[0], 'off_y': args.offset[1], 'off_z': args.offset[2]}
        self.fit = {}               # renderer -> (centre of its cars, camera distance)
        self.show_layer = {k: k not in args.hide for k in LAYERS}
        self.layer_actors = {k: {} for k in LAYERS}
        self.legend_info = None
        self.px = args.size[1] / FRAME_REF_PX
        self._last_end = None

        side = args.layout == 'side'
        if plotter is None:
            plotter = pv.Plotter(shape=(1, 2) if side else (1, 1), off_screen=True,
                                 window_size=list(args.size))
        self.p = plotter
        self.before_r = (0, 0)
        self.after_r = (0, 1) if side else (0, 0)
        for r in self.renderers():
            self.p.subplot(*r)
            self.p.set_background(self.colors['background'])
            # Ground and grid are unlit: flat, and exactly the colour picked.
            self._add('ground', 'ground', ground_plane(), color=self.colors['ground'],
                      lighting=False, pickable=False)
        self._draw_grid()
        self.p.enable_anti_aliasing('ssaa')
        self.p.enable_depth_peeling()
        if args.shadows:
            self.p.enable_shadows()

        self._skip_car = self._skip_lines = self._skip_tris = None
        self.p.subplot(*self.after_r)
        self._after_ren = self.p.renderer
        self._after_ren.AddObserver('StartEvent', lambda *_: self._refresh_skip_car())
        self.update(initial=True)
        if args.camera:
            self.set_camera_positions(args.camera)
        if self.interactive:
            # The interactor style (not the interactor) fires this at the end of
            # every mouse rotate / pan / zoom.
            self.p.iren.style.AddObserver('EndInteractionEvent',
                                          lambda *_: self._sync_from_camera())

    def renderers(self):
        return sorted({self.before_r, self.after_r})

    # -- layers ------------------------------------------------------------
    def _add(self, layer, name, mesh, **kw):
        """add_mesh into the current subplot, filed under a toggleable layer."""
        key = (self.p.renderer, name)
        if mesh is None:
            self.p.remove_actor(name, render=False)
            self.layer_actors[layer].pop(key, None)
            return None
        actor = self.p.add_mesh(mesh, name=name, render=False, **kw)
        actor.SetVisibility(self.show_layer[layer])
        self.layer_actors[layer][key] = actor
        return actor

    def _file(self, layer, actor, name):
        actor.SetVisibility(self.show_layer[layer])
        self.layer_actors[layer][(self.p.renderer, name)] = actor

    def set_layer(self, layer, on, render=True):
        self.show_layer[layer] = on
        for actor in self.layer_actors[layer].values():
            actor.SetVisibility(on)
        if layer == 'colorbar':
            for bar in self.p.scalar_bars.values():
                bar.SetVisibility(on)
        if render:
            self.p.render()

    def _draw_grid(self):
        mesh = grid_mesh(self.args.grid_spacing, self.args.grid_width / 1000)
        for r in self.renderers():
            self.p.subplot(*r)
            self._add('grid', 'grid', mesh, color=self.colors['grid'], lighting=False,
                      pickable=False)

    # -- setters used by the panel ------------------------------------------
    # Each returns True when the scene needs a full update() (the caller may
    # batch those), False when it has already applied the change.
    def set_value(self, group, key, value):
        value = float(value)
        if group == 'state':
            self.s[key] = value
            return True
        if group.startswith('u'):           # u2, u3, ...: a later step's input
            self.later[int(group[1:]) - 2][key] = value
            return True
        if group == 'truth':
            self.truth[key] = value
            return True
        if group == 'cam':
            if key == 'zoom':
                self.args.zoom = value
                self.refit()
            else:
                self.cam[key] = value
                self._keep_distance()
                self._apply_camera()
            self.p.render()
            return False
        # style
        setattr(self.args, key, value)
        if key in ('grid_width', 'grid_spacing'):
            self._draw_grid()
            self.p.render()
            return False
        if key in ('path_width', 'truth_path_width'):
            return True
        if key in ('path_opacity', 'truth_path_opacity'):
            name = 'after_paths' if key == 'path_opacity' else 'truth_path'
            for layer in self.layer_actors.values():
                for (_, n), actor in layer.items():
                    if n.startswith(name):          # truth_path, truth_path_skip
                        actor.GetProperty().SetOpacity(value)
            self._legend_info(*self._legend_args)
            self.p.render()
            return False
        if key in ('opacity', 'truth_opacity', 'before_opacity', 'mid_opacity'):
            layer = {'opacity': 'after cars', 'truth_opacity': 'truth car',
                     'before_opacity': 'before car', 'mid_opacity': None}[key]
            for (_, name), actor in self.layer_actors.get(layer, {}).items():
                if name != 'after_ghost' and not name.startswith('truth_path'):
                    actor.GetProperty().SetOpacity(value)
            a = self.args
            for (_, name), actor in self.layer_actors['intermediate cars'].items():
                if name == 'skip_outline':
                    actor.GetProperty().SetOpacity(a.truth_opacity)
                    continue
                if name == 'skip_fill':
                    continue
                base = a.truth_opacity if name.startswith('truth') else a.opacity
                actor.GetProperty().SetOpacity(base * a.mid_opacity)
            self.p.render()
            return False
        return True

    def set_sweep(self, key, on):
        if on and key not in self.sweep:
            self.sweep.append(key)
        elif not on and key in self.sweep:
            self.sweep.remove(key)
        return True

    def set_option(self, key, value):
        """n, color_by, cmap - anything in args that just needs an update()."""
        setattr(self.args, key, value)
        return True

    def set_color(self, key, value):
        self.colors[key] = value
        if key == 'background':
            for r in self.renderers():
                self.p.subplot(*r)
                self.p.set_background(value)
        elif key in ('ground', 'grid'):
            for actor in self.layer_actors[key].values():
                actor.GetProperty().SetColor(pv.Color(value).float_rgb)
        else:
            return True
        self.p.render()
        return False

    def set_frame_height(self, px):
        """Physical pixel height of the frame; text sizes follow it."""
        self.px = max(px, 1) / FRAME_REF_PX
        return True

    def _text_scale(self):
        """Font sizes for this frame height. VTK scales text by the window's
        DPI (a HiDPI Qt view reports more than the offscreen 72), so divide
        that back out - preview and export then agree."""
        return self.px * 72.0 / max(self.p.ren_win.GetDPI(), 1)

    def set_locked(self, on):
        self.camera_locked = bool(on)

    def toggle_camera_lock(self):
        self.camera_locked = not self.camera_locked
        print('camera', 'locked' if self.camera_locked else 'unlocked')
        return self.camera_locked

    def refit(self):
        if self._last_end is not None:
            self._fit_camera(self._last_end)
            self._apply_camera()
            self.p.render()

    # -- export ------------------------------------------------------------
    def _next_path(self, ext, tag=''):
        """fiala_car_3d_<i><tag>.<ext> for the first i not taken, so a view's
        PNG, legend and settings exported in turn share a number."""
        os.makedirs(self.args.out_dir, exist_ok=True)
        i = 0
        while os.path.exists(os.path.join(self.args.out_dir, f'fiala_car_3d_{i}{tag}.{ext}')):
            i += 1
        return os.path.join(self.args.out_dir, f'fiala_car_3d_{i}{tag}.{ext}')

    def camera_positions(self):
        cams = []
        for r in self.renderers():
            self.p.subplot(*r)
            cams.append([list(map(float, v)) for v in self.p.camera_position])
        return cams

    def set_camera_positions(self, cams):
        for r, cam in zip(self.renderers(), cams):
            self.p.subplot(*r)
            self.p.camera_position = [tuple(v) for v in cam]
            self.p.reset_camera_clipping_range()
        self.camera_locked = True
        self._sync_from_camera(apply=False)

    def _replica(self):
        """An offscreen copy of this scene at the output size, same camera."""
        a = copy.copy(self.args)
        a.vx, a.vy, a.omega = self.s['vx'], self.s['vy'], self.s['omega']
        a.steer = [self.s['steer']] + [u['steer'] for u in self.later]
        a.wheel_speed = [self.s['wheel']] + [u['wheel'] for u in self.later]
        a.dt = self.s['dt']
        a.truth = dict(self.truth)
        a.sweep = list(self.sweep)
        a.color = dict(self.colors)
        a.hide = [k for k in LAYERS if not self.show_layer[k]]
        a.elev, a.azim = self.cam['elev'], self.cam['azim']
        a.offset = [self.cam['off_x'], self.cam['off_y'], self.cam['off_z']]
        a.camera = self.camera_positions()
        a.shadows = self.args.shadows
        return FialaScene(a)

    def export(self, fmt, path=None, transparent=False):
        """The frame at args.size, exactly as shown (hidden layers stay hidden)."""
        path = path or self._next_path(fmt, '_clear' if transparent else '')
        if self.interactive:
            scene = self._replica()
            try:
                scene.export(fmt, path, transparent)
            finally:
                scene.p.close()
            return path
        self.p.render()
        if fmt == 'png':
            self.p.screenshot(path, transparent_background=transparent)
        else:
            self.p.save_graphic(path, title='Fiala bank - one step')
        print(f'saved {path}')
        return path

    def error_info(self):
        """LLAMPC's error of every bank model against the truth: inside each
        step (from the reset to the step's end), at each step's end, and
        summed over the window; best = the model LLAMPC would pick."""
        r, a = self._rollouts, self.args
        dt, weights = self.s['dt'], list(map(float, a.cost_weights))
        t = [j * dt + np.linspace(0.0, dt, SUBSTEPS + 1) for j in range(len(r['fans']))]
        e = np.stack([llampc_error(f, tr[None], weights)
                      for f, tr in zip(r['fans'], r['truths'])], axis=1)   # (M, K, S+1)
        step_err = e[:, :, -1]
        window = np.cumsum(step_err, axis=1)
        best = int(np.argmin(window[:, -1]))
        import matplotlib
        cmap = matplotlib.colormaps[a.cmap]
        lo, hi = r['clim']
        cols = [matplotlib.colors.to_hex(cmap((v - lo) / (hi - lo))) for v in r['scalars']]
        models = [dict(zip(BANK_KEYS, map(float, row))) for row in r['bank']]
        swept = {k: float(v[best]) for k, v in r['values'].items()}
        label = ',  '.join(f"{PARAM_TEX.get(k, k).split(' [')[0].split(' (')[0]} {v:.3g}"
                           for k, v in swept.items()) or 'the only model'
        truth = ', '.join(f"{PARAM_TEX[k].split(' [')[0]} {self.truth[k]:.3g}"
                          for k in ('muf', 'mur', 'Cf', 'Cr'))
        return {
            'weights': weights, 'weights_on': ['x', 'y', 'psi', 'vx', 'vy', 'omega'],
            'dt': dt, 'steps': len(t), 'inputs': [list(u) for u in self.inputs()],
            't': np.array(t).tolist(), 'error': e.tolist(),
            'step_error': step_err.tolist(), 'window_cost': window.tolist(),
            'best': best, 'best_label': label, 'truth_label': truth,
            'models': models, 'colors': cols, 'color_by': r['color_key'],
            'color_values': [float(v) for v in r['scalars']],
        }

    def graph_style(self):
        """The legend / error graph text and size overrides (see GRAPH_TEXT)."""
        return {'text': self.args.graph_text, 'size': self.args.graph_fontsize}

    def export_legend(self, base=None):
        """The legend (base.*) and the LLAMPC error graph (base_error.*)."""
        base = base or self._next_path('json', '_legend')[:-len('.json')]
        write_legend(base, self.legend_info, font=self.args.font, style=self.graph_style())
        stem = base[:-len('_legend')] if base.endswith('_legend') else base
        write_error_plot(f'{stem}_error', self.legend_info, self.error_info(),
                         log=self.args.error_log, size=self.args.graph_size,
                         font=self.args.font, style=self.graph_style(),
                         panels=self.args.error_panels, bar=self.args.graph_bar,
                         bar_gap=self.args.graph_bar_gap, key_gap=self.args.graph_key_gap,
                         line=self.args.graph_line)
        return base

    def settings(self):
        a = self.args
        return {
            'vx': self.s['vx'], 'vy': self.s['vy'], 'omega': self.s['omega'],
            'steer': [u[0] for u in self.inputs()],
            'wheel_speed': [u[1] for u in self.inputs()], 'dt': self.s['dt'],
            'steps': a.steps, 'mid_opacity': a.mid_opacity, 'skip_step': a.skip_step,
            'skip_line': a.skip_line, 'skip_dash': a.skip_dash, 'skip_gap': a.skip_gap,
            'skip_fill': a.skip_fill, 'skip_car': a.skip_car,
            'sweep': [k for k in SWEEPABLE if k in self.sweep], 'n': a.n,
            'range': [f'{k}={lo}:{hi}' for k, (lo, hi) in a.range.items()],
            'truth': [f'{k}={v}' for k, v in self.truth.items()],
            'color_by': a.color_by, 'layout': a.layout, 'view': a.view, 'zoom': a.zoom,
            'cmap': a.cmap, 'opacity': a.opacity,
            'truth_opacity': a.truth_opacity, 'before_opacity': a.before_opacity,
            'path_width': a.path_width, 'truth_path_width': a.truth_path_width,
            'path_opacity': a.path_opacity, 'cost_weights': list(a.cost_weights),
            'error_log': a.error_log, 'graph_size': list(a.graph_size), 'font': a.font,
            'error_panels': a.error_panels, 'graph_bar': a.graph_bar,
            'graph_bar_gap': a.graph_bar_gap, 'graph_key_gap': a.graph_key_gap,
            'graph_line': a.graph_line,
            'graph_text': [f'{k}={v}' for k, v in a.graph_text.items()],
            'graph_fontsize': [f'{k}={v:g}' for k, v in a.graph_fontsize.items()],
            'truth_path_opacity': a.truth_path_opacity, 'grid_width': a.grid_width,
            'lead_in': a.lead_in, 'lead_fade': a.lead_fade,
            'grid_spacing': a.grid_spacing,
            'color': [f'{k}={v}' for k, v in self.colors.items() if COLORS.get(k) != v],
            'hide': [k for k in LAYERS if not self.show_layer[k]],
            'size': list(a.size), 'elev': self.cam['elev'], 'azim': self.cam['azim'],
            'offset': [self.cam['off_x'], self.cam['off_y'], self.cam['off_z']],
            'camera': self.camera_positions(),
        }

    def save_settings(self, path=None):
        path = path or self._next_path('json', '_settings')
        with open(path, 'w') as f:
            json.dump(self.settings(), f, indent=2)
        print(f'saved {path}\n  re-render: python {os.path.basename(__file__)} '
              f'--config {path} --off-screen --save out.png --legend out_legend')
        return path

    def apply_settings(self, cfg):
        """Load a settings dict into this live scene (layout can't change)."""
        a = self.args
        if cfg.get('layout', a.layout) != a.layout:
            print(f"note: settings use layout {cfg['layout']!r}; restart with --config "
                  f"to switch - keeping {a.layout!r}")
        for k in ('vx', 'vy', 'omega', 'dt'):
            if cfg.get(k) is not None:
                self.s[k] = float(cfg[k])
        for k, sk in (('steer', 'steer'), ('wheel_speed', 'wheel')):
            if cfg.get(k) is not None:
                vals = _per_step(cfg[k], None)
                self.s[sk] = vals[0]
                for u, v in zip(self.later, vals[1:]):
                    u[sk] = v
        for k in ('n', 'steps', 'mid_opacity', 'skip_step', 'skip_line', 'skip_dash', 'skip_gap', 'skip_fill', 'skip_car',
                  'color_by', 'zoom', 'cmap', 'opacity',
                  'truth_opacity',
                  'before_opacity', 'path_width', 'truth_path_width', 'path_opacity',
                  'truth_path_opacity', 'cost_weights', 'error_log', 'graph_size', 'font',
                  'lead_in', 'lead_fade', 'error_panels', 'graph_bar', 'graph_bar_gap',
                  'graph_key_gap', 'graph_line', 'grid_width', 'grid_spacing'):
            if k in cfg:
                setattr(a, k, cfg[k])
        if 'size' in cfg:
            a.size = list(cfg['size'])
        if 'sweep' in cfg:
            self.sweep = list(cfg['sweep'])
        if 'range' in cfg:
            a.range = _parse_pairs(cfg['range'], _span, 'range')
        if 'graph_text' in cfg:
            a.graph_text = {k: v for k, v in _parse_pairs(cfg['graph_text'], str,
                                                          'graph_text').items()
                            if k in GRAPH_TEXT}
        if 'graph_fontsize' in cfg:
            a.graph_fontsize = {k: v for k, v in _parse_pairs(cfg['graph_fontsize'], float,
                                                              'graph_fontsize').items()
                                if k in GRAPH_TEXT}
        if 'truth' in cfg:
            for k, v in _parse_pairs(cfg['truth'], float, 'truth').items():
                for t in (['muf', 'mur'] if k == 'mu' else [k]):
                    self.truth[t] = v
        if 'color' in cfg:
            # Keys no longer in COLORS (old per-part car colours) are dropped.
            self.colors = dict(COLORS, **{k: v for k, v in
                                          _parse_pairs(cfg['color'], str, 'color').items()
                                          if k in COLORS})
            for r in self.renderers():
                self.p.subplot(*r)
                self.p.set_background(self.colors['background'])
            for key in ('ground', 'grid'):
                for actor in self.layer_actors[key].values():
                    actor.GetProperty().SetColor(pv.Color(self.colors[key]).float_rgb)
        if 'hide' in cfg:
            for k in LAYERS:
                self.show_layer[k] = k not in cfg['hide']
        for k in ('elev', 'azim'):
            if k in cfg:
                self.cam[k] = float(cfg[k])
        if 'offset' in cfg:
            self.cam.update(off_x=cfg['offset'][0], off_y=cfg['offset'][1],
                            off_z=cfg['offset'][2])
        self._draw_grid()
        for k in LAYERS:
            self.set_layer(k, self.show_layer[k], render=False)
        self.update()
        if cfg.get('camera'):
            self.set_camera_positions(cfg['camera'])
        self.p.render()

    # -- drawing -----------------------------------------------------------
    def state(self):
        return np.array([0.0, 0.0, 0.0, self.s['vx'], self.s['vy'], self.s['omega']])

    def car_colors(self):
        """Part colours of the real car: body in 'truth', tyres in 'tire'."""
        return {k: self.colors['tire' if k == 'tire' else 'truth'] for k in CAR_PARTS}

    def _add_car(self, layer, prefix, x, y, psi, steer, opacity, **kw):
        """One real car (see car_colors)."""
        parts = self.mesh.parts(x, y, psi, steer)
        colors = self.car_colors()
        for k, m in parts.items():
            self._add(layer, f'{prefix}_{k}', m, color=colors[k],
                      opacity=opacity, smooth_shading=True, split_sharp_edges=True, **kw)

    def inputs(self):
        """[(steer, rear wheel speed)] for each of the args.steps steps."""
        first = (self.s['steer'], self.s['wheel'])
        later = [(u['steer'], u['wheel']) for u in self.later]
        return [first] + later[:int(self.args.steps) - 1]

    def skip_step(self):
        """The step shown as a time skip, or 0: only a step that ends at an
        intermediate car (1 .. steps - 1) can be skipped."""
        k = int(self.args.skip_step)
        return k if 1 <= k < int(self.args.steps) else 0

    def _skip_outline(self, eye):
        """The skipped car's silhouette from `eye`, as dashed tubes."""
        if self._skip_car is None:
            return None
        a = self.args
        if self._skip_tris is None:
            m = self._skip_car
            self._skip_tris = np.asarray(m.points)[m.regular_faces]
        return dashed_tubes(silhouette(self._skip_car, eye, self._skip_tris), a.skip_line / 1000,
                            a.skip_dash / 1000, a.skip_gap / 1000)

    def _refresh_skip_car(self):
        """Before each render of the after panel: redraw the silhouette if
        the camera has moved since it was drawn."""
        if self._skip_car is None or self._skip_lines is None:
            return
        eye = tuple(self._after_ren.GetActiveCamera().GetPosition())
        if eye == self._skip_eye:
            return
        self._skip_eye = eye
        new = self._skip_outline(eye)
        self._skip_lines.copy_from(new if new is not None else pv.PolyData())

    def _input_arrows(self, name, starts, inputs, label_side, px, tagged):
        """Input arrows (and their labels) for each (start state, input) pair,
        merged into one actor per kind under `name`."""
        car, col = self.car, self.colors
        meshes = {'steer': [], 'torque': []}
        points, labels = [], []
        for k, (x, (steer, wheel)) in enumerate(zip(starts, inputs)):
            frx, frx_max = rear_drive(car, x, steer, wheel)
            arrows, tips = input_arrows(car, x, steer, frx / frx_max, label_side)
            for kind, mesh in arrows.items():
                if mesh is not None:
                    meshes[kind].append(mesh)
            tag = f'u{tagged[k]} ' if tagged else ''
            points += [tips['torque'], tips['steer']]
            labels += [f'{tag}torque {frx * car["rw"]:+.3f} N m ({100 * frx / frx_max:+.0f}% grip)',
                       f'{tag}steer {steer:+.3f} rad']
        for kind, parts in meshes.items():
            self._add('input arrows', f'{name}_{kind}', pv.merge(parts) if parts else None,
                      color=col[kind], smooth_shading=True)
        if not points:
            self.p.remove_actor(f'{name}_labels', render=False)
            self.layer_actors['input labels'].pop((self.p.renderer, f'{name}_labels'), None)
            return
        lab = self.p.add_point_labels(np.array(points) + [0, 0, 0.07], labels,
                                      font_size=max(6, int(round(16 * px))), bold=False,
                                      text_color=col['text'], shape='rounded_rect',
                                      shape_color='white', shape_opacity=0.8,
                                      margin=max(1, int(round(4 * px))), show_points=False,
                                      always_visible=True, name=f'{name}_labels', render=False)
        self._file('input labels', lab, f'{name}_labels')

    def update(self, initial=False):
        a, p, car, col = self.args, self.p, self.car, self.colors
        x0 = self.state()
        inputs = self.inputs()
        steps = len(inputs)
        steer = inputs[0][0]
        frx, _ = rear_drive(car, x0, steer, inputs[0][1])
        px = self._text_scale()
        label_side = 1 if np.sin(np.radians(self.cam['azim'])) >= 0 else -1

        # After: the truth car rolls forward step by step, each step under its
        # own input; at every step the bank is reset to the truth's state and
        # predicts one dt ahead under that step's input (as the controller
        # does). Bank and truth are integrated in one call per step.
        sweep = [k for k in SWEEPABLE if k in self.sweep]   # stable order
        bank, values = build_sweep(sweep, a.n, a.range)
        truth = np.array([self.truth.get(k, PARAM_MEAN[k]) for k in BANK_KEYS])
        starts, fans, truth_trajs = [x0], [], []
        for st, wh in inputs:
            traj = step_bank(np.vstack([bank, truth]), car, starts[-1], st, wh, self.s['dt'])
            fans.append(traj[:-1])
            truth_trajs.append(traj[-1])
            starts.append(traj[-1, -1])
        m = len(bank)
        # Time skip: step `skip`'s fan and paths are dropped and the truth car
        # it ends at becomes a dashed silhouette, in place (0 = none).
        skip = self.skip_step()

        # Before: the start pose, showing step 1's input.
        p.subplot(*self.before_r)
        self._add_car('before car', 'before', 0.0, 0.0, 0.0, steer,
                      1.0 if a.layout == 'side' else a.before_opacity, specular=0.3)
        self._add('velocity', 'before_v', velocity_arrow(x0), color=col['velocity'],
                  smooth_shading=True)
        self._add('yaw rate', 'before_w', yaw_arc(x0), color=col['yaw'], smooth_shading=True)
        tagged = list(range(1, steps + 1)) if steps > 1 else None
        self._input_arrows('input', starts[:1], inputs[:1], label_side, px, tagged)

        p.subplot(*self.after_r)
        # Each new actuation, on the truth car where it's applied.
        # (Not on a skipped car that isn't drawn: only its dashed path shows.)
        later = [k for k in range(1, steps) if not (k == skip and not a.skip_car)]
        self._input_arrows('input_later', [starts[k] for k in later], [inputs[k] for k in later],
                           label_side, px, [tagged[k] for k in later] if tagged else None)

        color_key = a.color_by if a.color_by in values else (sweep[0] if sweep else None)
        scalars = values[color_key] if color_key else np.zeros(m)
        clim = (scalars.min(), scalars.max()) if np.ptp(scalars) > 0 else (scalars[0] - 1, scalars[0] + 1)

        # The fan: every non-tyre part takes the car's colour; tyres the same
        # colour darkened, so a dense fan doesn't turn into a heap of black.
        # Each car's front wheels show the input of the step that moved it.
        def fan(k):
            body, tires = [], []
            for i, xe in enumerate(fans[k][:, -1]):
                parts = self.mesh.parts(xe[0], xe[1], xe[2], inputs[k][0])
                b = pv.merge([mm for kk, mm in parts.items() if kk != 'tire'])
                b.cell_data['value'] = np.full(b.n_cells, scalars[i])
                t = parts['tire']
                t.cell_data['value'] = np.full(t.n_cells, scalars[i])
                body.append(b)
                tires.append(t)
            return body, tires

        body, tires = fan(steps - 1)

        for title in list(p.scalar_bars.keys()):
            p.remove_scalar_bar(title, render=False)
        bar = dict(title=PARAM_TITLE.get(color_key, ''), color=col['text'], vertical=True,
                   position_x=0.9, position_y=0.3, height=0.45, width=0.035,
                   title_font_size=max(6, int(round(16 * px))),
                   label_font_size=max(6, int(round(14 * px))), fmt='%.3g', n_labels=5)
        op = a.opacity
        self._add('after cars', 'after_body', pv.merge(body), scalars='value', cmap=a.cmap,
                  clim=clim, opacity=op, show_scalar_bar=bool(color_key),
                  scalar_bar_args=bar, smooth_shading=True, split_sharp_edges=True)
        for b in p.scalar_bars.values():
            b.SetVisibility(self.show_layer['colorbar'])
        tire_look = dict(scalars='value', cmap=a.cmap, clim=clim, show_scalar_bar=False,
                         ambient=0.0, diffuse=0.45, specular=0.0, smooth_shading=True,
                         split_sharp_edges=True)
        self._add('after cars', 'after_tires', pv.merge(tires), opacity=op, **tire_look)

        # The fans of the earlier steps (none for one step).
        mid_body, mid_tires = [], []
        for k in range(steps - 1):
            if k + 1 == skip:
                continue
            b, t = fan(k)
            mid_body += b
            mid_tires += t
        self._add('intermediate cars', 'mid_body', pv.merge(mid_body) if mid_body else None,
                  scalars='value', cmap=a.cmap, clim=clim, opacity=op * a.mid_opacity,
                  show_scalar_bar=False, smooth_shading=True, split_sharp_edges=True)
        self._add('intermediate cars', 'mid_tires', pv.merge(mid_tires) if mid_tires else None,
                  opacity=op * a.mid_opacity, **tire_look)

        w = a.path_width / 1000
        tubes = path_tubes([(f[i, :, :2], scalars[i]) for k, f in enumerate(fans)
                            if k + 1 != skip for i in range(m)], w)
        if col['pred_path'] == 'auto':
            look = dict(scalars='value', cmap=a.cmap, clim=clim, show_scalar_bar=False)
        else:
            look = dict(color=col['pred_path'])
        # Back faces culled: a translucent tube otherwise shows its far wall
        # through the near one as stripes.
        self._add('paths', 'after_paths', tubes, opacity=a.path_opacity, smooth_shading=True,
                  culling='back', **look)

        # Truth: coloured like the start car.
        # The cars at the earlier step boundaries show the next step's input.
        te = starts[-1]
        truth_colors = self.car_colors()
        self._add_car('truth car', 'truth', te[0], te[1], te[2], inputs[-1][0], a.truth_opacity,
                      specular=0.3)
        mid_parts = [self.mesh.parts(*starts[k][:3], inputs[k][0]) for k in range(1, steps)
                     if k != skip]
        for part in CAR_PARTS:
            self._add('intermediate cars', f'truth_mid_{part}',
                      pv.merge([mp[part] for mp in mid_parts]) if mid_parts else None,
                      color=truth_colors[part], opacity=a.truth_opacity * a.mid_opacity,
                      smooth_shading=True, split_sharp_edges=True, specular=0.3)
        # The skipped car's silhouette depends on the view, so it is redrawn
        # whenever the camera moves (_refresh_skip_car, before each render).
        self._skip_car = (self.mesh.ghost(*starts[skip][:3], inputs[skip][0])
                          if skip and a.skip_car else None)
        self._skip_tris = None
        self._skip_eye = tuple(p.camera.position)
        self._skip_lines = self._skip_outline(self._skip_eye)
        actor = self._add('intermediate cars', 'skip_outline', self._skip_lines,
                          color=col['truth'], opacity=a.truth_opacity)
        if actor is not None:
            # add_mesh may draw a copy; refresh the one actually drawn.
            self._skip_lines = actor.mapper.dataset
        # A faint body behind the dashes, so the silhouette reads as a solid.
        self._add('intermediate cars', 'skip_fill',
                  self._skip_car.copy() if self._skip_car is not None and a.skip_fill > 0
                  else None, color=col['truth'],
                  opacity=a.skip_fill, ambient=0.3, specular=0.0, smooth_shading=True)
        # The truth path: dashed into the skipped car and on to the next one,
        # solid elsewhere.
        runs = [truth_trajs[:skip - 1], truth_trajs[skip + 1:]] if skip else [truth_trajs]
        joined = lambda r: np.vstack([r[0]] + [t[1:] for t in r[1:]])
        lines = [(joined(r)[:, :2], None) for r in runs if r]
        tw = a.truth_path_width / 1000
        look = dict(color=self.path_color('truth_path'), opacity=a.truth_path_opacity,
                    smooth_shading=True)
        self._add('truth car', 'truth_path', path_tubes(lines, tw) if lines else None,
                  culling='back', **look)
        # Lead-in: the truth line coming into the start car from off in the
        # distance, fading out there - part of a longer trajectory.
        lead = lead_in_path(x0, a.lead_in) if tw > 0 else None
        tube = None
        if lead is not None:
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(lead, axis=0), axis=1))]
            d = d[-1] - d                                   # distance from the car
            fade = max(a.lead_fade, 1e-6) * a.lead_in
            alpha = np.clip((a.lead_in - d) / fade, 0.0, 1.0) if a.lead_fade > 0 \
                else np.ones(len(d))
            line = pv.PolyData(np.c_[lead, np.full(len(lead), tw / 2 + 0.002)],
                               lines=np.r_[len(lead), np.arange(len(lead))])
            rgb = pv.Color(self.path_color('truth_path')).int_rgb
            line.point_data['rgba'] = np.c_[np.tile(rgb, (len(lead), 1)),
                                            np.round(255 * alpha)].astype(np.uint8)
            tube = line.tube(radius=tw / 2, n_sides=12)
        self._add('truth car', 'truth_path_lead', tube, scalars='rgba', rgba=True,
                  opacity=a.truth_path_opacity, smooth_shading=True, culling='back')
        dashed = None
        if skip:
            xy = joined(truth_trajs[skip - 1:skip + 1])[:, :2]
            dashed = dashed_tubes([np.c_[xy, np.full(len(xy), tw / 2 + 0.002)]], tw,
                                  a.skip_dash / 1000, a.skip_gap / 1000)
        self._add('truth car', 'truth_path_skip', dashed, **look)

        if a.layout == 'side':
            # Faint copy of the start pose so the after panel reads on its own.
            ghost = pv.merge(list(self.mesh.parts(0.0, 0.0, 0.0, steer).values()))
            self._add('after cars', 'after_ghost', ghost, color=col['truth'], opacity=0.2)

        self._rollouts = dict(bank=bank, values=values, fans=fans, truths=truth_trajs,
                              scalars=scalars, clim=clim, color_key=color_key)
        self._legend_args = (color_key, values, scalars, clim, frx)
        self._legend_info(*self._legend_args)
        self._caption(values, m, frx)
        self._last_end = np.vstack([f[:, -1] for f in fans] + [np.array(starts)])
        if initial or not self.camera_locked:
            self._fit_camera(self._last_end)
            self._apply_camera()
        p.render()

    def path_color(self, key):
        """A path colour with 'auto' resolved (None: the colour map)."""
        c = self.colors[key]
        if c != 'auto':
            return c
        return self.colors['truth'] if key == 'truth_path' else None

    def _legend_info(self, color_key, values, scalars, clim, frx):
        import matplotlib
        cmap = matplotlib.colormaps[self.args.cmap]
        u = np.unique(scalars) if color_key else np.array([])
        span = clim[1] - clim[0]
        cols = [matplotlib.colors.to_hex(cmap((v - clim[0]) / span)) for v in u]
        t = self.truth
        # The truth's value(s) of the colour-by param: a mark on the colour bar
        # for each (mu: front and rear, tagged f / r when they differ) and the
        # text for {truth} in the graph key.
        mu_tied = np.isclose(t['muf'], t['mur'])
        if color_key in ('mu', None):
            marks = [('', t['muf'])] if mu_tied else [('f', t['muf']), ('r', t['mur'])]
            truth_text = (rf"$\mu$ {t['muf']:.2f}" if mu_tied else
                          rf"$\mu_f$ {t['muf']:.2f}, $\mu_r$ {t['mur']:.2f}")
        else:
            marks = [('', t[color_key])]
            name = PARAM_TEX[color_key].split(' [')[0]
            truth_text = f'{name} {t[color_key]:.3g}'
        marks = [(tag, v) for tag, v in marks if color_key and clim[0] <= v <= clim[1]]
        entries = []
        if self.show_layer['truth car']:
            entries.append({'kind': 'patch', 'color': self.colors['truth'], 'key': 'leg_truth',
                            'subs': {'muf': f"{t['muf']:.2f}", 'mur': f"{t['mur']:.2f}",
                                     'Cf': f"{t['Cf']:.0f}", 'Cr': f"{t['Cr']:.0f}"}})
        if self.show_layer['truth car'] and self.args.truth_path_width > 0 and \
                self.colors['truth_path'] != 'auto':
            entries.append({'kind': 'line', 'color': self.path_color('truth_path'),
                            'alpha': self.args.truth_path_opacity,
                            'key': 'leg_truth_path'})
        if self.show_layer['paths'] and self.args.path_width > 0 and \
                self.colors['pred_path'] != 'auto':
            entries.append({'kind': 'line', 'color': self.colors['pred_path'],
                            'alpha': self.args.path_opacity, 'key': 'leg_pred_paths'})
        if self.show_layer['intermediate cars'] and int(self.args.steps) > 1:
            entries.append({'kind': 'patch', 'color': self.colors['truth'],
                            'alpha': self.args.truth_opacity * self.args.mid_opacity,
                            'key': 'leg_earlier'})
        if self.show_layer['intermediate cars'] and self.skip_step():
            entries.append({'kind': 'dashes', 'color': self.colors['truth'],
                            'alpha': self.args.truth_opacity, 'key': 'leg_skip'})
        if self.show_layer['before car']:
            entries.append({'kind': 'patch', 'color': self.colors['truth'],
                            'alpha': self.args.before_opacity, 'key': 'leg_start'})
        if self.show_layer['input arrows']:
            if frx != 0.0:
                entries.append({'kind': 'arrow', 'color': self.colors['torque'],
                                'key': 'leg_torque',
                                'subs': {'torque': 'drive' if frx > 0 else 'brake'}})
            entries.append({'kind': 'arrow', 'color': self.colors['steer'], 'key': 'leg_steer'})
        if self.show_layer['velocity']:
            entries.append({'kind': 'arrow', 'color': self.colors['velocity'],
                            'key': 'leg_velocity'})
        self.legend_info = {
            'color_by': color_key, 'cmap': self.args.cmap,
            'clim': [float(clim[0]), float(clim[1])],
            'values': [float(v) for v in u], 'colors': cols,
            'fan_opacity': self.args.opacity, 'steps': int(self.args.steps),
            'dt': float(self.s['dt']),
            'truth_color': self.colors['truth'],
            'truth_params': {k: float(v) for k, v in self.truth.items()},
            'truth_marks': [{'value': float(v), 'tag': tag} for tag, v in marks],
            'truth_text': truth_text,
            'entries': entries,
        }

    def _caption(self, values, m, frx):
        s, t = self.s, self.truth
        slip = (s['wheel'] - s['vx']) / max(s['vx'], 1e-3)
        sw = ', '.join(f'{k} {v.min():.3g}-{v.max():.3g}' for k, v in values.items()) or 'none'
        later = ''.join(f"   u{k + 2}: steer {st:+.3f} rad, wheel {wh:.2f} m/s"
                        for k, (st, wh) in enumerate(self.inputs()[1:]))
        text = (f"v_x {s['vx']:.2f} m/s   v_y {s['vy']:.2f} m/s   yaw rate {s['omega']:.2f} rad/s\n"
                f"steer {s['steer']:+.3f} rad   rear wheel {s['wheel']:.2f} m/s (slip {slip:+.2f})"
                f"   torque {frx * self.car['rw']:+.3f} N m{later}\n"
                f"{self.args.steps} x dt = {self.args.steps} x {s['dt']:.3f} s   {m} models   sweep: {sw} ({self.args.n} each)\n"
                f"truth: mu_f {t['muf']:.2f}  mu_r {t['mur']:.2f}  "
                f"C_f {t['Cf']:.0f}  C_r {t['Cr']:.0f}")
        self.p.subplot(*self.after_r)
        actor = self.p.add_text(text, position='upper_right',
                                font_size=max(5, int(round(9 * self._text_scale()))),
                                color=self.colors['text'], name='caption', render=False)
        self._file('caption', actor, 'caption')

    # -- camera ------------------------------------------------------------
    def _view_dir(self):
        e, a = np.radians(self.cam['elev']), np.radians(self.cam['azim'])
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def _offset(self):
        return np.array([self.cam['off_x'], self.cam['off_y'], self.cam['off_z']])

    def _fit_camera(self, end):
        """Per panel: the centre of the cars it shows, and the distance that fits
        them at the current elevation / azimuth."""
        pad = CAR_LENGTH / 2 + 0.05

        def bounds(pts):
            lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
            return (lo[0], hi[0], lo[1], hi[1], 0.0, GLYPH_Z)

        everything = bounds(np.vstack([[0.0, 0.0], end[:, :2]]))
        panels = {self.after_r: everything}
        panels[self.before_r] = bounds(np.zeros((1, 2))) if self.args.layout == 'side' else everything
        for r, b in panels.items():
            self.p.subplot(*r)
            centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 0.1])
            self.p.camera_position = [tuple(centre + self._view_dir()), tuple(centre), (0, 0, 1)]
            self.p.reset_camera(bounds=b, render=False)
            self.fit[r] = (centre, self.p.camera.distance / self.args.zoom)

    def _apply_camera(self):
        for r, (centre, dist) in self.fit.items():
            self.p.subplot(*r)
            focal = centre + self._offset()
            self.p.camera_position = [tuple(focal + self._view_dir() * dist), tuple(focal),
                                      (0, 0, 1)]
            self.p.reset_camera_clipping_range()

    def _keep_distance(self):
        """Adopt each panel's current distance (the mouse wheel's zoom)."""
        for r, (centre, _) in self.fit.items():
            self.p.subplot(*r)
            self.fit[r] = (centre, self.p.camera.distance)

    def _sync_from_camera(self, apply=True):
        """Read elevation / azimuth / offset back from the after panel's camera
        (after a mouse rotate or pan) and tell the panel."""
        self.p.subplot(*self.after_r)
        pos, foc, _ = self.p.camera_position
        d = np.subtract(pos, foc)
        dist = np.linalg.norm(d)
        self.cam['elev'] = float(np.degrees(np.arcsin(np.clip(d[2] / dist, -1, 1))))
        self.cam['azim'] = float(np.degrees(np.arctan2(d[1], d[0])))
        off = np.subtract(foc, self.fit[self.after_r][0])
        self.cam.update(off_x=float(off[0]), off_y=float(off[1]), off_z=float(off[2]))
        self._keep_distance()
        if apply:
            self._apply_camera()    # side layout: bring the other panel along
            self.p.render()
        if self.on_camera:
            self.on_camera(dict(self.cam))

    def show(self):
        """Headless run: write --save and/or --legend."""
        if self.args.save:
            fmt = os.path.splitext(self.args.save)[1].lstrip('.').lower() or 'png'
            self.export(fmt, path=self.args.save, transparent=self.args.transparent)
        if self.args.legend:
            self.export_legend(os.path.splitext(self.args.legend)[0])
        self.p.close()


# ===========================================================================
# GUI (Qt): the frame on the left, every control in a panel on the right
# ===========================================================================
# name, label, unit, lo, hi, decimals[, spin box max beyond the slider's]
STATE_ROWS = [('vx', 'v_x', 'm/s', 0.2, 8.0, 2), ('vy', 'v_y', 'm/s', -2.0, 2.0, 2),
              ('omega', 'yaw rate', 'rad/s', -4.0, 4.0, 2),
              ('steer', 'steer', 'rad', -0.34, 0.34, 3),
              ('wheel', 'rear wheel speed', 'm/s', 0.0, 10.0, 2),
              ('dt', 'step dt', 's', 0.025, 1.0, 3)]
TRUTH_ROWS = [('muf', 'mu_f', '', 0.05, 1.5, 2), ('mur', 'mu_r', '', 0.05, 1.5, 2),
              ('Cf', 'C_f', 'N/rad', 50.0, 500.0, 0), ('Cr', 'C_r', 'N/rad', 50.0, 500.0, 0)]
STYLE_ROWS = [('opacity', 'fan opacity', '', 0.02, 1.0, 2),
              ('truth_opacity', 'truth opacity', '', 0.05, 1.0, 2),
              ('before_opacity', 'start car opacity', '', 0.05, 1.0, 2),
              ('mid_opacity', 'intermediate cars x', '', 0.05, 1.0, 2),
              ('path_width', 'predicted path width', 'mm', 0.0, 60.0, 1, 500.0),
              ('truth_path_width', 'truth path width', 'mm', 0.0, 60.0, 1, 500.0),
              ('path_opacity', 'path opacity', '', 0.02, 1.0, 2),
              ('truth_path_opacity', 'truth path opacity', '', 0.02, 1.0, 2),
              ('lead_in', 'lead-in truth line', 'm', 0.0, 20.0, 1, 200.0),
              ('lead_fade', 'lead-in fade', '', 0.0, 1.0, 2),
              ('skip_line', 'time-skip line width', 'mm', 0.5, 20.0, 1, 100.0),
              ('skip_dash', 'time-skip dash', 'mm', 2.0, 100.0, 1, 500.0),
              ('skip_gap', 'time-skip gap', 'mm', 1.0, 100.0, 1, 500.0),
              ('skip_fill', 'time-skip fill', '', 0.0, 1.0, 2),
              ('grid_width', 'grid line width', 'mm', 0.0, 60.0, 1, 500.0),
              ('grid_spacing', 'grid spacing', 'm', 0.05, 2.0, 2, 10.0)]
CAM_ROWS = [('elev', 'elevation', 'deg', 0.0, 89.5, 1),
            ('azim', 'azimuth', 'deg', -180.0, 180.0, 1),
            ('off_x', 'offset x', 'm', -4.0, 4.0, 2), ('off_y', 'offset y', 'm', -4.0, 4.0, 2),
            ('off_z', 'offset z', 'm', -1.0, 1.0, 2), ('zoom', 'fit zoom', 'x', 0.3, 4.0, 2)]
# copper, YlOrBr, Oranges and Wistia are warm without the truth car's red/pink.
CMAPS = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'copper', 'YlOrBr', 'Oranges',
         'Wistia', 'turbo', 'coolwarm', 'RdYlBu', 'Spectral', 'twilight']


def _gui():
    """Import Qt and define the GUI classes (kept out of headless runs)."""
    os.environ.setdefault('QT_API', 'pyside6')
    if os.environ.get('DISPLAY'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')   # VTK's GL wants X, not Wayland
    from qtpy import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
    Signal = QtCore.Signal

    class ValueRow(QtWidgets.QWidget):
        """label | slider | spin box, kept in step; emits the value."""
        changed = Signal(float)
        STEPS = 1000

        def __init__(self, label, unit, lo, hi, decimals, value, spin_hi=None):
            super().__init__()
            self.lo, self.hi = lo, hi
            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            name = QtWidgets.QLabel(label)
            name.setMinimumWidth(104)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setRange(0, self.STEPS)
            self.spin = QtWidgets.QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setRange(lo, spin_hi if spin_hi is not None else hi)
            self.spin.setSingleStep(max(10 ** -decimals, round((hi - lo) / 100, decimals)))
            self.spin.setKeyboardTracking(False)
            self.spin.setMinimumWidth(84)
            if unit:
                self.spin.setSuffix(f' {unit}')
            lay.addWidget(name)
            lay.addWidget(self.slider, 1)
            lay.addWidget(self.spin)
            self.set_value(value)
            self.slider.valueChanged.connect(self._from_slider)
            self.spin.valueChanged.connect(self._from_spin)

        def _to_slider(self, v):
            return int(round((np.clip(v, self.lo, self.hi) - self.lo)
                             / (self.hi - self.lo) * self.STEPS))

        def set_value(self, v):
            for w in (self.slider, self.spin):
                w.blockSignals(True)
            self.spin.setValue(float(v))
            self.slider.setValue(self._to_slider(v))
            for w in (self.slider, self.spin):
                w.blockSignals(False)

        def value(self):
            return self.spin.value()

        def _from_slider(self, i):
            v = self.lo + (self.hi - self.lo) * i / self.STEPS
            self.spin.blockSignals(True)
            self.spin.setValue(v)
            self.spin.blockSignals(False)
            self.changed.emit(self.spin.value())

        def _from_spin(self, v):
            self.slider.blockSignals(True)
            self.slider.setValue(self._to_slider(v))
            self.slider.blockSignals(False)
            self.changed.emit(v)

    class Section(QtWidgets.QWidget):
        """A titled, collapsible block of the panel."""

        def __init__(self, title, expanded=True):
            super().__init__()
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 4)
            outer.setSpacing(2)
            self.button = QtWidgets.QToolButton(text=title, checkable=True, checked=expanded)
            self.button.setStyleSheet('QToolButton { border: none; font-weight: bold; }')
            self.button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.body = QtWidgets.QFrame()
            self.body.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            self.lay = QtWidgets.QVBoxLayout(self.body)
            self.lay.setContentsMargins(8, 6, 8, 6)
            self.lay.setSpacing(4)
            outer.addWidget(self.button)
            outer.addWidget(self.body)
            self.button.toggled.connect(self._toggle)
            self._toggle(expanded)

        def _toggle(self, on):
            self.button.setArrowType(QtCore.Qt.ArrowType.DownArrow if on
                                     else QtCore.Qt.ArrowType.RightArrow)
            self.body.setVisible(on)

        def add(self, w):
            self.lay.addWidget(w)
            return w

    class AspectFrame(QtWidgets.QWidget):
        """Holds the 3-D view at the output aspect ratio, letterboxed on grey,
        so the view is exactly the export frame."""
        resized = Signal(int)

        def __init__(self):
            super().__init__()
            self.child = None
            self.aspect = 1.5
            self.setAutoFillBackground(True)
            pal = self.palette()
            pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor('#3c3f44'))
            self.setPalette(pal)
            self.setMinimumSize(320, 240)

        def set_child(self, w):
            self.child = w
            self._place()

        def set_aspect(self, w, h):
            self.aspect = w / h
            self._place()

        def resizeEvent(self, ev):
            super().resizeEvent(ev)
            self._place()

        def _place(self):
            if self.child is None:
                return
            m = 10
            W, H = max(self.width() - 2 * m, 10), max(self.height() - 2 * m, 10)
            if W / H > self.aspect:
                h, w = H, int(round(H * self.aspect))
            else:
                w, h = W, int(round(W / self.aspect))
            self.child.setGeometry(m + (W - w) // 2, m + (H - h) // 2, w, h)
            self.resized.emit(int(round(h * self.devicePixelRatioF())))

    class Panel(QtWidgets.QWidget):
        def __init__(self, win):
            super().__init__()
            self.win = win
            sc = win.scene
            a = sc.args
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            self.rows = {}

            def rows(section, group, specs, store):
                for spec in specs:
                    key, label, unit, lo, hi, dec = spec[:6]
                    spin_hi = spec[6] if len(spec) > 6 else None
                    row = ValueRow(label, unit, lo, hi, dec, store(key), spin_hi)
                    row.changed.connect(lambda v, g=group, k=key: win.set_value(g, k, v))
                    self.rows[(group, key)] = row
                    section.add(row)

            # Car state and input.
            s = Section('Car state and input')
            rows(s, 'state', STATE_ROWS, lambda k: sc.s[k])
            form = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(form)
            f.setContentsMargins(0, 0, 0, 0)
            self.steps_spin = QtWidgets.QSpinBox()
            self.steps_spin.setRange(1, MAX_STEPS)
            self.steps_spin.setValue(a.steps)
            self.steps_spin.setToolTip('steps of dt, each with its own input; the bank '
                                       'resets to the truth car at every step. Earlier '
                                       'steps are the "intermediate cars" layer')
            self.steps_spin.valueChanged.connect(self._steps_changed)
            f.addRow('steps', self.steps_spin)
            self.skip_spin = QtWidgets.QSpinBox()
            self.skip_spin.setSpecialValueText('none')
            self.skip_spin.setToolTip('draw this step as a time skip: no fan or predicted '
                                      'paths, and the truth car it ends at as a dashed '
                                      'silhouette in place; the truth path into and out '
                                      'of it is dashed')
            self.skip_spin.valueChanged.connect(
                lambda k: (self.win.set_option('skip_step', int(k)), self._skip_range()))
            f.addRow('time skip at step', self.skip_spin)
            self.skip_car = QtWidgets.QCheckBox('draw the skipped car')
            self.skip_car.setToolTip('off: only the dashed truth path marks the skip (no '
                                     'silhouette, no input arrows there)')
            self.skip_car.toggled.connect(lambda on: self.win.set_option('skip_car', bool(on)))
            f.addRow('', self.skip_car)
            self._skip_range()
            s.add(form)
            self.step1_note = QtWidgets.QLabel('steer / rear wheel speed above: step 1 (u1)')
            self.step1_note.setStyleSheet('color: #777;')
            s.add(self.step1_note)
            # Each later step's input, shown for the steps in use.
            self.step_boxes = []
            later_rows = [r for r in STATE_ROWS if r[0] in ('steer', 'wheel')]
            for k in range(2, MAX_STEPS + 1):
                box = QtWidgets.QWidget()
                v = QtWidgets.QVBoxLayout(box)
                v.setContentsMargins(0, 4, 0, 0)
                v.setSpacing(2)
                head = QtWidgets.QLabel(f'step {k} input (u{k})')
                head.setStyleSheet('font-weight: bold;')
                v.addWidget(head)
                rows(argparse.Namespace(add=v.addWidget), f'u{k}', later_rows,
                     lambda key, k=k: sc.later[k - 2][key])
                s.add(box)
                self.step_boxes.append(box)
            self._show_step_boxes()
            lay.addWidget(s)

            # Sweep.
            s = Section('Sweep')
            box = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            self.sweep_checks = {}
            for key in SWEEPABLE:
                cb = QtWidgets.QCheckBox(key)
                cb.setChecked(key in sc.sweep)
                cb.toggled.connect(lambda on, k=key: win.set_sweep(k, on))
                self.sweep_checks[key] = cb
                h.addWidget(cb)
            s.add(box)
            form = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(form)
            f.setContentsMargins(0, 0, 0, 0)
            self.n_spin = QtWidgets.QSpinBox()
            self.n_spin.setRange(1, 41)
            self.n_spin.setValue(a.n)
            self.n_spin.valueChanged.connect(lambda v: win.set_option('n', int(v)))
            self.color_by = QtWidgets.QComboBox()
            self.color_by.addItems(['(first swept)'] + SWEEPABLE)
            self.color_by.setCurrentText(a.color_by or '(first swept)')
            self.color_by.currentTextChanged.connect(
                lambda t: win.set_option('color_by', None if t.startswith('(') else t))
            self.cmap = QtWidgets.QComboBox()
            self.cmap.setEditable(True)
            self.cmap.addItems(CMAPS if a.cmap in CMAPS else [a.cmap] + CMAPS)
            self.cmap.setCurrentText(a.cmap)
            self.cmap.activated.connect(lambda _: self._cmap(self.cmap.currentText()))
            self.cmap.lineEdit().editingFinished.connect(
                lambda: self._cmap(self.cmap.currentText()))
            f.addRow('values per param', self.n_spin)
            f.addRow('colour by', self.color_by)
            f.addRow('colour map', self.cmap)
            s.add(form)
            lay.addWidget(s)

            # Truth.
            s = Section('Ground truth car')
            rows(s, 'truth', TRUTH_ROWS, lambda k: sc.truth[k])
            lay.addWidget(s)

            # Appearance.
            s = Section('Appearance')
            rows(s, 'style', STYLE_ROWS, lambda k: getattr(a, k))
            grid = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(grid)
            g.setContentsMargins(0, 4, 0, 0)
            g.setSpacing(3)
            self.color_buttons, self.color_edits = {}, {}
            hex_re = QtCore.QRegularExpression(r'#?[0-9A-Fa-f]{0,6}')
            for i, key in enumerate(COLORS):
                # Two per row: name, swatch (opens the picker), hex field.
                r, c = i // 2, 3 * (i % 2)
                b = QtWidgets.QPushButton()
                b.setFixedSize(26, 20)
                b.setToolTip(f'pick the {key} colour')
                b.clicked.connect(lambda _=False, k=key: self._pick_color(k))
                e = QtWidgets.QLineEdit()
                e.setFixedWidth(72)
                e.setValidator(QtGui.QRegularExpressionValidator(hex_re, e))
                e.setToolTip('hex: #rrggbb, rrggbb or #rgb')
                if key in AUTO_COLORS:
                    e.setPlaceholderText('auto')
                    e.setToolTip('hex: #rrggbb, rrggbb or #rgb; empty = auto ('
                                 + ('the colour map' if key == 'pred_path' else 'the truth colour')
                                 + ')')
                e.editingFinished.connect(lambda k=key: self._hex_edited(k))
                self.color_buttons[key], self.color_edits[key] = b, e
                g.addWidget(QtWidgets.QLabel(key), r, c)
                g.addWidget(b, r, c + 1)
                g.addWidget(e, r, c + 2)
            g.setColumnStretch(0, 1)
            g.setColumnStretch(3, 1)
            self._paint_color_buttons()
            s.add(QtWidgets.QLabel('colours (click a swatch, or type a hex;\n'
                                   'empty path colours follow the colour map / truth)'))
            s.add(grid)
            lay.addWidget(s)

            # Layers.
            s = Section('Layers')
            grid = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(grid)
            g.setContentsMargins(0, 0, 0, 0)
            self.layer_checks = {}
            for i, layer in enumerate(LAYERS):
                cb = QtWidgets.QCheckBox(layer)
                cb.setChecked(sc.show_layer[layer])
                cb.toggled.connect(lambda on, k=layer: win.set_layer(k, on))
                self.layer_checks[layer] = cb
                g.addWidget(cb, i // 2, i % 2)
            s.add(grid)
            lay.addWidget(s)

            # Camera.
            s = Section('Camera')
            rows(s, 'cam', CAM_ROWS, lambda k: a.zoom if k == 'zoom' else sc.cam[k])
            box = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            self.view_combo = QtWidgets.QComboBox()
            self.view_combo.addItems(['preset...'] + list(VIEW_PRESETS))
            self.view_combo.activated.connect(lambda _: self._preset())
            self.lock = QtWidgets.QCheckBox('lock camera')
            self.lock.setChecked(sc.camera_locked)
            self.lock.toggled.connect(win.set_locked)
            fit = QtWidgets.QPushButton('Fit to cars')
            fit.clicked.connect(win.refit)
            h.addWidget(self.view_combo)
            h.addWidget(self.lock)
            h.addWidget(fit)
            s.add(box)
            lay.addWidget(s)

            # Frame.
            s = Section('Frame (output size)')
            form = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(form)
            f.setContentsMargins(0, 0, 0, 0)
            self.aspect = QtWidgets.QComboBox()
            self.aspect.addItems(list(ASPECTS) + ['custom'])
            self.w_spin = QtWidgets.QSpinBox()
            self.h_spin = QtWidgets.QSpinBox()
            for sp in (self.w_spin, self.h_spin):
                sp.setRange(100, 16000)
                sp.setSingleStep(100)
                sp.setSuffix(' px')
                sp.setKeyboardTracking(False)
            self.w_spin.setValue(a.size[0])
            self.h_spin.setValue(a.size[1])
            self._match_aspect_combo()
            self.aspect.activated.connect(lambda _: self._aspect_preset())
            self.w_spin.valueChanged.connect(lambda _: self._size_edited())
            self.h_spin.valueChanged.connect(lambda _: self._size_edited())
            f.addRow('aspect', self.aspect)
            f.addRow('width', self.w_spin)
            f.addRow('height', self.h_spin)
            s.add(form)
            lay.addWidget(s)

            # Export.
            s = Section('Export')
            box = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            self.out_dir = QtWidgets.QLineEdit(os.path.abspath(a.out_dir))
            self.out_dir.editingFinished.connect(
                lambda: setattr(a, 'out_dir', self.out_dir.text()))
            browse = QtWidgets.QPushButton('...')
            browse.setFixedWidth(32)
            browse.clicked.connect(self._browse)
            h.addWidget(QtWidgets.QLabel('folder'))
            h.addWidget(self.out_dir, 1)
            h.addWidget(browse)
            s.add(box)
            s.add(QtWidgets.QLabel('Exports are the frame exactly as shown, at the frame size.'))
            grid = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(grid)
            g.setContentsMargins(0, 0, 0, 0)
            buttons = [('PNG', lambda: win.export('png')),
                       ('PNG (clear)', lambda: win.export('png', transparent=True)),
                       ('PDF', lambda: win.export('pdf')), ('SVG', lambda: win.export('svg')),
                       ('Legend + error graph', win.export_legend),
                       ('Save settings', win.save_settings),
                       ('Load settings...', win.load_settings)]
            for i, (label, fn) in enumerate(buttons):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(lambda _=False, fn=fn: fn())
                g.addWidget(b, i // 2, i % 2)
            s.add(grid)
            # The error graph's LLAMPC weights and axis scale.
            form = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(form)
            f.setContentsMargins(0, 6, 0, 0)
            self.weights = QtWidgets.QLineEdit()
            self.weights.setToolTip('LLAMPC error weights on x y psi vx vy omega '
                                    '(the controller uses 0 0 20 1 10 0.1)')
            self.weights.editingFinished.connect(self._weights_edited)
            self.error_log = QtWidgets.QCheckBox('log y axes')
            self.error_log.toggled.connect(lambda on: setattr(a, 'error_log', on))
            f.addRow('error weights', self.weights)
            f.addRow('error graph', self.error_log)
            self.error_panels = QtWidgets.QComboBox()
            self.panel_choices = [('both', 'error + window cost'), ('error', 'prediction error'),
                                  ('cost', 'window cost')]
            self.error_panels.addItems([label for _, label in self.panel_choices])
            self.error_panels.setToolTip('which plots the error graph has; the truth key '
                                         'and colour bar stay either way')
            self.error_panels.currentIndexChanged.connect(
                lambda i: setattr(a, 'error_panels', self.panel_choices[i][0]))
            f.addRow('graph plots', self.error_panels)
            size = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(size)
            h.setContentsMargins(0, 0, 0, 0)
            self.graph_w, self.graph_h = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
            for i, sp in enumerate((self.graph_w, self.graph_h)):
                sp.setRange(1.0, 30.0)
                sp.setDecimals(2)
                sp.setSingleStep(0.25)
                sp.setSuffix(' in')
                sp.setToolTip('error graph width / height [in]; the PNG is 300 dpi')
                sp.valueChanged.connect(lambda v, i=i: a.graph_size.__setitem__(i, float(v)))
            h.addWidget(self.graph_w)
            h.addWidget(QtWidgets.QLabel('x'))
            h.addWidget(self.graph_h)
            f.addRow('graph size', size)
            # Colour bar thickness and the gaps key / bar <-> plots.
            self.graph_spacing = {}
            for attr, label, tip in (
                    ('graph_line', 'graph line width', 'error graph model line width [pt]; the '
                                                       'pick is drawn 1.8x as thick'),
                    ('graph_bar', 'colour bar thickness', 'error graph colour bar thickness [in]'),
                    ('graph_bar_gap', 'colour bar gap', 'space between the error graph plots '
                                                        'and its colour bar [in]'),
                    ('graph_key_gap', 'key gap', 'space between the error graph key and '
                                                 'its plots [in]')):
                sp = QtWidgets.QDoubleSpinBox()
                pt = attr == 'graph_line'
                sp.setRange(0.1 if pt else 0.0 if attr != 'graph_bar' else 0.05,
                            20.0 if pt else 5.0)
                sp.setDecimals(2)
                sp.setSingleStep(0.25 if pt else 0.05)
                sp.setSuffix(' pt' if pt else ' in')
                sp.setToolTip(tip)
                sp.valueChanged.connect(lambda v, attr=attr: setattr(a, attr, float(v)))
                self.graph_spacing[attr] = sp
                f.addRow(label, sp)
            self.font_edit = QtWidgets.QLineEdit()
            self.font_edit.setToolTip('font of the legend and error graph, e.g. Aptos; '
                                 'falls back to DejaVu Sans if not installed')
            self.font_edit.editingFinished.connect(
                lambda: setattr(a, 'font', self.font_edit.text().strip() or 'Aptos'))
            f.addRow('graph font', self.font_edit)
            s.add(form)
            self._show_error_opts()
            lay.addWidget(s)

            # Every text in the legend and error graph: its words and size.
            s = Section('Graph text', expanded=False)
            note = QtWidgets.QLabel('$...$ is maths (e.g. $\\mu_f$, $\\hat{x}$), '
                                    '\\\\ a line break, {name} a value; empty hides it.')
            note.setWordWrap(True)
            note.setStyleSheet('color: #777;')
            s.add(note)
            grid = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(grid)
            g.setContentsMargins(0, 0, 0, 0)
            g.setHorizontalSpacing(4)
            g.setVerticalSpacing(2)
            self.text_edits, self.text_sizes = {}, {}
            row, group = 0, None
            for key, (grp, name, default, pt) in GRAPH_TEXT.items():
                if grp != group:
                    head = QtWidgets.QLabel(grp)
                    head.setStyleSheet('font-weight: bold;')
                    g.addWidget(head, row, 0, 1, 4)
                    row, group = row + 1, grp
                g.addWidget(QtWidgets.QLabel(name), row, 0)
                subs = GRAPH_TEXT_SUBS.get(key, [])
                if default is not None:
                    e = QtWidgets.QLineEdit()
                    e.setToolTip(f'default: {default}' + (
                        '\nfilled in: ' + ', '.join('{%s}' % n for n in subs) if subs else ''))
                    e.editingFinished.connect(lambda k=key: self._text_edited(k))
                    g.addWidget(e, row, 1)
                    self.text_edits[key] = e
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(3.0, 72.0)
                sp.setDecimals(1)
                sp.setSingleStep(0.5)
                sp.setSuffix(' pt')
                sp.setToolTip(f'size (default {pt:g} pt)')
                sp.valueChanged.connect(lambda v, k=key: self._text_size_edited(k, v))
                g.addWidget(sp, row, 2)
                self.text_sizes[key] = sp
                reset = QtWidgets.QToolButton(text='\u21ba')
                reset.setToolTip('back to the default text and size')
                reset.clicked.connect(lambda _=False, k=key: self._text_reset(k))
                g.addWidget(reset, row, 3)
                row += 1
            g.setColumnStretch(1, 1)
            s.add(grid)
            self._show_graph_text()
            lay.addWidget(s)
            lay.addStretch(1)

        # -- helpers -------------------------------------------------------
        def _cmap(self, name):
            import matplotlib
            if name in matplotlib.colormaps:
                self.win.set_option('cmap', name)
                self._paint_color_buttons()
            else:
                self.win.status(f'unknown colour map {name!r}')

        def _color_text(self, key):
            v = self.win.scene.colors[key]
            return '' if v == 'auto' else QtGui.QColor(v).name()

        def _paint_color_buttons(self):
            import matplotlib
            sc = self.win.scene
            for key, b in self.color_buttons.items():
                c = sc.path_color(key) if key in AUTO_COLORS else sc.colors[key]
                if c is None:
                    # auto prediction paths: a strip of the colour map
                    cmap = matplotlib.colormaps[sc.args.cmap]
                    stops = ', '.join(f'stop:{t} {matplotlib.colors.to_hex(cmap(t))}'
                                      for t in (0, 0.25, 0.5, 0.75, 1))
                    bg = f'qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops})'
                else:
                    bg = QtGui.QColor(c).name()
                style = 'dashed' if sc.colors[key] == 'auto' else 'solid'
                b.setStyleSheet(f'QPushButton {{ background: {bg}; '
                                f'border: 1px {style} #888; }}')
                e = self.color_edits[key]
                if not e.hasFocus():
                    e.setText(self._color_text(key))

        def _hex_edited(self, key):
            e = self.color_edits[key]
            text = e.text().strip()
            if not text and key in AUTO_COLORS:
                if self.win.scene.colors[key] != 'auto':
                    self.win.set_color(key, 'auto')
            else:
                text = text if text.startswith('#') else '#' + text
                # #rgb or #rrggbb only; anything else snaps back to the current colour.
                c = QtGui.QColor(text)
                if len(text) in (4, 7) and c.isValid():
                    if c.name() != self._color_text(key):
                        self.win.set_color(key, c.name())
                else:
                    self.win.status(f'{key}: {e.text()!r} is not a hex colour')
            e.setText(self._color_text(key))
            self._paint_color_buttons()

        def _pick_color(self, key):
            sc = self.win.scene
            start = (sc.path_color(key) or '#808080') if key in AUTO_COLORS else sc.colors[key]
            c = QtWidgets.QColorDialog.getColor(QtGui.QColor(start),
                                                self, f'{key} colour')
            if c.isValid():
                self.win.set_color(key, c.name())
                self._paint_color_buttons()

        def _preset(self):
            name = self.view_combo.currentText()
            if name in VIEW_PRESETS:
                elev, azim = VIEW_PRESETS[name]
                self.win.set_value('cam', 'elev', elev)
                self.win.set_value('cam', 'azim', azim)
                self.set_camera(self.win.scene.cam)
            self.view_combo.setCurrentIndex(0)

        def _match_aspect_combo(self):
            w, h = self.w_spin.value(), self.h_spin.value()
            name = next((k for k, (aw, ah) in ASPECTS.items() if abs(w * ah - h * aw) <= aw),
                        'custom')
            self.aspect.blockSignals(True)
            self.aspect.setCurrentText(name)
            self.aspect.blockSignals(False)

        def _aspect_preset(self):
            name = self.aspect.currentText()
            if name in ASPECTS:
                aw, ah = ASPECTS[name]
                self.h_spin.blockSignals(True)
                self.h_spin.setValue(int(round(self.w_spin.value() * ah / aw)))
                self.h_spin.blockSignals(False)
                self._size_edited()

        def _size_edited(self):
            self._match_aspect_combo()
            self.win.set_size(self.w_spin.value(), self.h_spin.value())

        def _browse(self):
            d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Export folder',
                                                           self.out_dir.text())
            if d:
                self.out_dir.setText(d)
                self.win.scene.args.out_dir = d

        def set_camera(self, cam):
            for key in ('elev', 'azim', 'off_x', 'off_y', 'off_z'):
                self.rows[('cam', key)].set_value(cam[key])

        def _row_value(self, group, key):
            sc, a = self.win.scene, self.win.scene.args
            if group.startswith('u'):
                return sc.later[int(group[1:]) - 2][key]
            return {'state': lambda: sc.s[key], 'truth': lambda: sc.truth[key],
                    'style': lambda: getattr(a, key),
                    'cam': lambda: a.zoom if key == 'zoom' else sc.cam[key]}[group]()

        def _show_graph_text(self):
            a = self.win.scene.args
            for key, e in self.text_edits.items():
                e.setText(a.graph_text.get(key, GRAPH_TEXT[key][2]))
            for key, sp in self.text_sizes.items():
                sp.blockSignals(True)
                sp.setValue(float(a.graph_fontsize.get(key, GRAPH_TEXT[key][3])))
                sp.blockSignals(False)

        def _text_edited(self, key):
            a = self.win.scene.args
            text = self.text_edits[key].text()
            problem = graph_text_problem(key, text)
            if problem:
                self.win.status(f'{GRAPH_TEXT[key][1]}: {problem}')
            elif text == GRAPH_TEXT[key][2]:
                a.graph_text.pop(key, None)
            else:
                a.graph_text[key] = text
            self._show_graph_text()

        def _text_size_edited(self, key, v):
            a = self.win.scene.args
            if abs(v - GRAPH_TEXT[key][3]) < 1e-9:
                a.graph_fontsize.pop(key, None)
            else:
                a.graph_fontsize[key] = float(v)

        def _text_reset(self, key):
            a = self.win.scene.args
            a.graph_text.pop(key, None)
            a.graph_fontsize.pop(key, None)
            self._show_graph_text()

        def _show_error_opts(self):
            a = self.win.scene.args
            self.weights.setText(' '.join(f'{v:g}' for v in a.cost_weights))
            self.error_log.blockSignals(True)
            self.error_log.setChecked(bool(a.error_log))
            self.error_log.blockSignals(False)
            self.error_panels.blockSignals(True)
            self.error_panels.setCurrentIndex(
                [key for key, _ in self.panel_choices].index(a.error_panels))
            self.error_panels.blockSignals(False)
            self.font_edit.setText(a.font)
            for sp, v in zip((self.graph_w, self.graph_h), a.graph_size):
                sp.blockSignals(True)
                sp.setValue(float(v))
                sp.blockSignals(False)
            for attr, sp in self.graph_spacing.items():
                sp.blockSignals(True)
                sp.setValue(float(getattr(a, attr)))
                sp.blockSignals(False)

        def _weights_edited(self):
            try:
                w = [float(v) for v in self.weights.text().replace(',', ' ').split()]
                if len(w) != 6:
                    raise ValueError
                self.win.scene.args.cost_weights = w
            except ValueError:
                self.win.status('error weights: six numbers, for x y psi vx vy omega')
            self._show_error_opts()

        def _skip_range(self):
            """Only steps ending at an intermediate car can be skipped."""
            a = self.win.scene.args
            self.skip_spin.blockSignals(True)
            self.skip_spin.setRange(0, max(int(a.steps) - 1, 0))
            self.skip_spin.setValue(self.win.scene.skip_step())
            self.skip_spin.setEnabled(int(a.steps) > 1)
            self.skip_spin.blockSignals(False)
            self.skip_car.blockSignals(True)
            self.skip_car.setChecked(bool(a.skip_car))
            self.skip_car.setEnabled(self.win.scene.skip_step() > 0)
            self.skip_car.blockSignals(False)

        def _show_step_boxes(self):
            n = self.win.scene.args.steps
            self.step1_note.setVisible(n > 1)
            for k, box in enumerate(self.step_boxes, start=2):
                box.setVisible(k <= n)

        def _steps_changed(self, n):
            """A newly shown step starts with the previous step's input."""
            sc = self.win.scene
            for k in range(sc.args.steps + 1, n + 1):
                prev = sc.inputs()[-1] if k == 2 else \
                    (sc.later[k - 3]['steer'], sc.later[k - 3]['wheel'])
                sc.later[k - 2].update(steer=prev[0], wheel=prev[1])
                for key in ('steer', 'wheel'):
                    self.rows[(f'u{k}', key)].set_value(sc.later[k - 2][key])
            self.win.set_option('steps', int(n))
            if self.win.scene.skip_step() == 0:
                self.win.scene.args.skip_step = 0      # dropped with its step
            self._skip_range()
            self._show_step_boxes()

        def refresh(self):
            """Every control from the scene (after loading settings)."""
            sc, a = self.win.scene, self.win.scene.args
            for (group, key), row in self.rows.items():
                row.set_value(self._row_value(group, key))
            widgets = list(self.sweep_checks.values()) + list(self.layer_checks.values()) + \
                [self.n_spin, self.steps_spin, self.color_by, self.cmap, self.lock,
                 self.w_spin, self.h_spin]
            for w in widgets:
                w.blockSignals(True)
            for k, cb in self.sweep_checks.items():
                cb.setChecked(k in sc.sweep)
            for k, cb in self.layer_checks.items():
                cb.setChecked(sc.show_layer[k])
            self.n_spin.setValue(a.n)
            self.steps_spin.setValue(a.steps)
            self.color_by.setCurrentText(a.color_by or '(first swept)')
            self.cmap.setCurrentText(a.cmap)
            self.lock.setChecked(sc.camera_locked)
            self.w_spin.setValue(a.size[0])
            self.h_spin.setValue(a.size[1])
            for w in widgets:
                w.blockSignals(False)
            self._match_aspect_combo()
            self._paint_color_buttons()
            self._show_step_boxes()
            self._skip_range()
            self._show_error_opts()
            self._show_graph_text()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, args):
            super().__init__()
            self.setWindowTitle('Fiala bank - one step')
            self.frame = AspectFrame()
            self.view = QtInteractor(self.frame, shape=(1, 2) if args.layout == 'side' else (1, 1),
                                     auto_update=False)
            self.scene = FialaScene(args, plotter=self.view, on_camera=self._camera_moved)
            self.frame.set_child(self.view.interactor)
            self.frame.set_aspect(*args.size)
            self.setCentralWidget(self.frame)

            self.panel = Panel(self)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidget(self.panel)
            scroll.setWidgetResizable(True)
            scroll.setMinimumWidth(430)
            dock = QtWidgets.QDockWidget('Controls')
            dock.setWidget(scroll)
            dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                             | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

            # Heavy changes (anything that re-integrates or rebuilds the cars)
            # are batched: slider drags queue one update per 40 ms at most.
            self.timer = QtCore.QTimer(self, singleShot=True, interval=40)
            self.timer.timeout.connect(self.scene.update)
            self.frame.resized.connect(self._frame_resized)
            self.view.add_key_event('s', lambda: self.export('png'))
            self.view.add_key_event('k', self._toggle_lock)
            self.resize(*args.window)
            self.status('ready')

        # -- plumbing from the panel -----------------------------------------
        def _apply(self, heavy):
            if heavy:
                self.timer.start()

        def set_value(self, group, key, v):
            self._apply(self.scene.set_value(group, key, v))

        def set_sweep(self, key, on):
            self._apply(self.scene.set_sweep(key, on))

        def set_option(self, key, v):
            self._apply(self.scene.set_option(key, v))

        def set_color(self, key, v):
            self._apply(self.scene.set_color(key, v))

        def set_layer(self, layer, on):
            self.scene.set_layer(layer, on)

        def set_locked(self, on):
            self.scene.set_locked(on)

        def refit(self):
            self.scene.refit()
            self.panel.set_camera(self.scene.cam)

        def set_size(self, w, h):
            self.scene.args.size = [int(w), int(h)]
            self.frame.set_aspect(w, h)

        def _frame_resized(self, h):
            self._apply(self.scene.set_frame_height(h))

        def _camera_moved(self, cam):
            if hasattr(self, 'panel'):
                self.panel.set_camera(cam)

        def _toggle_lock(self):
            self.panel.lock.setChecked(self.scene.toggle_camera_lock())

        def status(self, text):
            self.statusBar().showMessage(text)

        # -- export ----------------------------------------------------------
        def export(self, fmt, transparent=False):
            self.timer.stop()
            self.scene.update()                 # flush any queued change first
            self.status(f'rendering {fmt.upper()} at {self.scene.args.size[0]} x '
                        f'{self.scene.args.size[1]} ...')
            QtWidgets.QApplication.processEvents()
            path = self.scene.export(fmt, transparent=transparent)
            self.status(f'saved {path}')

        def export_legend(self):
            base = self.scene.export_legend()
            stem = base[:-len('_legend')]
            self.status(f'saved {base}.* and {stem}_error.* (svg / pdf / png / json)')

        def save_settings(self):
            self.status(f'saved {self.scene.save_settings()}')

        def load_settings(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Load settings', self.scene.args.out_dir, 'Settings (*.json)')
            if path:
                self.apply_settings_file(path)

        def closeEvent(self, event):
            """Cache the session so the next launch picks up where this left off."""
            try:
                cfg = dict(self.scene.settings(), window=[self.width(), self.height()])
                os.makedirs(os.path.dirname(SESSION_CACHE), exist_ok=True)
                tmp = SESSION_CACHE + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(cfg, f, indent=2)
                os.replace(tmp, SESSION_CACHE)
            except Exception as e:                  # never block closing
                print(f'warning: could not cache the session: {e}')
            super().closeEvent(event)

        def apply_settings_file(self, path):
            with open(path) as f:
                self.scene.apply_settings(json.load(f))
            self.frame.set_aspect(*self.scene.args.size)
            self.panel.refresh()
            self.panel.set_camera(self.scene.cam)
            self.status(f'loaded {path}')

    return QtWidgets, MainWindow


def run_gui(args):
    QtWidgets, MainWindow = _gui()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow(args)
    win.show()
    return app.exec()


# ===========================================================================
def _parse_pairs(items, parse, what):
    out = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f'bad {what} {item!r}: expected KEY=VALUE')
        key, val = item.split('=', 1)
        out[key] = parse(val)
    return out


def _per_step(val, default):
    """A per-step input (a number or a list, one per step) as MAX_STEPS
    values: missing steps repeat the last given one."""
    if val is None:
        val = default
    vals = [float(v) for v in (val if isinstance(val, (list, tuple)) else [val])]
    return (vals + vals[-1:] * MAX_STEPS)[:MAX_STEPS]


def _span(val):
    lo, hi = val.split(':')
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='settings .json from "Save settings"; '
                                     'other flags override it')
    ap.add_argument('--fresh', action='store_true',
                    help=f'ignore the cached last session ({SESSION_CACHE})')
    st = ap.add_argument_group('state / input')
    st.add_argument('--vx', type=float, default=3.0)
    st.add_argument('--vy', type=float, default=0.0)
    st.add_argument('--omega', type=float, default=0.0, help='yaw rate [rad/s]')
    st.add_argument('--steer', type=float, nargs='+', default=[0.25],
                    help='[rad], one per step (missing steps repeat the last)')
    st.add_argument('--wheel-speed', type=float, nargs='+', default=None,
                    help='rear wheel surface speed rw*omega_w [m/s], one per step '
                         '(default: vx, free rolling)')
    st.add_argument('--dt', type=float, default=0.3, help='length of one step [s]')
    st.add_argument('--steps', type=int, default=1,
                    help=f'steps of dt to take, each with its own input (1-{MAX_STEPS})')
    st.add_argument('--skip-step', type=int, default=0, metavar='K',
                    help='draw step K (1 .. steps-1) as a time skip: no fan or predicted '
                         'paths, and the truth car it ends at as a dashed silhouette in '
                         'place; the truth path into and out of it is dashed (0 = none)')
    st.add_argument('--no-skip-car', dest='skip_car', action='store_false',
                    help='time skip: leave out the skipped car (and its input arrows); '
                         'only the dashed truth path shows')

    sw = ap.add_argument_group('sweep / truth')
    sw.add_argument('--sweep', nargs='*', default=['mu'], choices=SWEEPABLE)
    sw.add_argument('--n', type=int, default=7, help='values per swept param')
    sw.add_argument('--range', nargs='*', metavar='KEY=LO:HI',
                    help='override a swept range, e.g. mu=0.2:1.0 Cf=150:350')
    sw.add_argument('--color-by', choices=SWEEPABLE, default=None,
                    help='default: first swept param')
    sw.add_argument('--truth', nargs='*', metavar='KEY=VALUE',
                    help='ground-truth car params (Cf Cr muf mur, or mu for both); '
                         'the rest stay at the mean')

    vw = ap.add_argument_group('view')
    vw.add_argument('--layout', choices=['overlay', 'side'], default='overlay')
    vw.add_argument('--view', choices=list(VIEW_PRESETS), default='iso')
    vw.add_argument('--elev', type=float, default=None,
                    help='camera elevation [deg] (default: from --view)')
    vw.add_argument('--azim', type=float, default=None,
                    help='camera azimuth [deg] from the car\'s +x (default: from --view)')
    vw.add_argument('--offset', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=('X', 'Y', 'Z'), help='view centre offset from the cars [m]')
    vw.add_argument('--zoom', type=float, default=1.15, help='camera zoom after fitting')
    vw.add_argument('--cmap', default='viridis')
    vw.add_argument('--opacity', type=float, default=0.2, help='the predicted fan')
    vw.add_argument('--truth-opacity', type=float, default=1.0)
    vw.add_argument('--before-opacity', type=float, default=1.0)
    vw.add_argument('--mid-opacity', type=float, default=0.6,
                    help='intermediate-step cars, as a multiple of the final cars\' opacity')
    vw.add_argument('--skip-line', type=float, default=5.0,
                    help='time-skip silhouette line width [mm]')
    vw.add_argument('--skip-dash', type=float, default=22.0,
                    help='time-skip dash length [mm] (silhouette and truth path)')
    vw.add_argument('--skip-gap', type=float, default=14.0,
                    help='time-skip gap between dashes [mm]')
    vw.add_argument('--skip-fill', type=float, default=0.12,
                    help='opacity of the faint body inside the time-skip silhouette '
                         '(0 = outline only)')
    vw.add_argument('--color', nargs='*', metavar='KEY=COLOR',
                    help=f'override colours (hex or name); keys: {", ".join(COLORS)}. '
                         f'{" and ".join(AUTO_COLORS)} also take "auto"')
    vw.add_argument('--path-width', type=float, default=8.0,
                    help='predicted paths\' stroke [mm]')
    vw.add_argument('--truth-path-width', type=float, default=12.0,
                    help='ground-truth path\'s stroke [mm]')
    vw.add_argument('--path-opacity', type=float, default=1.0, help='predicted paths')
    vw.add_argument('--truth-path-opacity', type=float, default=1.0)
    vw.add_argument('--lead-in', type=float, default=0.0, metavar='M',
                    help='draw the truth line coming into the start car from M metres back '
                         '(its current velocities and yaw rate, turning at most 90 deg), '
                         'to show it is part of a longer run (0 = off)')
    vw.add_argument('--lead-fade', type=float, default=0.6,
                    help='fraction of the lead-in, from its far end, that fades out')
    vw.add_argument('--grid-width', type=float, default=5.0, help='ground grid stroke [mm]')
    vw.add_argument('--grid-spacing', type=float, default=0.5, help='ground grid pitch [m]')
    vw.add_argument('--hide', nargs='*', default=['colorbar', 'caption'], metavar='LAYER',
                    help=f'layers to start hidden (default: colorbar caption - they belong in '
                         f'the legend): {", ".join(repr(k) for k in LAYERS)}')
    vw.add_argument('--shadows', action='store_true')
    vw.add_argument('--window', type=int, nargs=2, default=[1750, 1050], metavar=('W', 'H'),
                    help='GUI window size')

    out = ap.add_argument_group('output')
    out.add_argument('--size', type=int, nargs=2, default=[2400, 1600], metavar=('W', 'H'),
                     help='export frame size [px]; its aspect ratio is the preview\'s')
    out.add_argument('--off-screen', action='store_true',
                     help='no GUI: render --save (.png/.pdf/.svg) and/or --legend, then exit')
    out.add_argument('--save', help='scene output path for --off-screen')
    out.add_argument('--legend', help='legend output path (no extension) for --off-screen; '
                                      'the error graph goes beside it as <path>_error.*')
    out.add_argument('--transparent', action='store_true', help='transparent PNG background')
    out.add_argument('--out-dir', default='.', help='where the export buttons write')
    out.add_argument('--cost-weights', type=float, nargs=6, default=COST_WEIGHTS,
                     metavar=('X', 'Y', 'PSI', 'VX', 'VY', 'OMEGA'),
                     help='LLAMPC error weights for the error graph (default: the '
                          'llampc_fiala_* scripts\')')
    out.add_argument('--error-log', action='store_true', help='log y axes on the error graph')
    out.add_argument('--error-panels', choices=['both', 'error', 'cost'], default='both',
                     help='error graph plots: both, prediction error only, or window cost '
                          'only (the truth key and colour bar stay)')
    out.add_argument('--graph-size', type=float, nargs=2, default=[3.0, 6.0],
                     metavar=('W', 'H'), help='error graph size [in] (PNG at 300 dpi)')
    out.add_argument('--graph-line', type=float, default=2.0,
                     help='error graph model line width [pt] (the pick 1.8x)')
    out.add_argument('--graph-bar', type=float, default=0.35,
                     help='error graph colour bar thickness [in]')
    out.add_argument('--graph-bar-gap', type=float, default=0.25,
                     help='space between the error graph plots and colour bar [in]')
    out.add_argument('--graph-key-gap', type=float, default=0.15,
                     help='space between the error graph key and plots [in]')
    out.add_argument('--font', default='Aptos',
                     help='font of the legend and error graph (falls back to DejaVu Sans)')
    out.add_argument('--graph-text', nargs='*', metavar='KEY=TEXT',
                     help='legend / error graph text; $...$ is maths, \\\\ a line break, '
                          'empty hides it. Keys: ' + ', '.join(GRAPH_TEXT))
    out.add_argument('--graph-fontsize', nargs='*', metavar='KEY=PT',
                     help='legend / error graph text sizes [pt], same keys as --graph-text')

    pre, _ = ap.parse_known_args()
    camera = None
    cfg = None
    if pre.config:
        with open(pre.config) as f:
            cfg = json.load(f)
        cfg.pop('window', None)
    elif not (pre.off_screen or pre.fresh) and os.path.exists(SESSION_CACHE):
        # The GUI resumes the last session; headless runs never do, so
        # they depend only on their flags.
        try:
            with open(SESSION_CACHE) as f:
                cfg = json.load(f)
            print(f'resuming the last session ({SESSION_CACHE}); --fresh ignores it')
        except (OSError, ValueError) as e:
            print(f'warning: ignoring the cached session: {e}')
    if cfg is not None:
        camera = cfg.pop('camera', None)
        for old in ('export_overlays', 'scale'):   # keys from older versions
            cfg.pop(old, None)
        # Per-part car colours from older versions: the car now uses 'truth'.
        cfg['color'] = [c for c in cfg.get('color', [])
                        if c.split('=', 1)[0].strip() in COLORS]
        ap.set_defaults(**cfg)
    args = ap.parse_args()
    args.camera = camera
    args.hide = ['input arrows' if k == 'input frames' else k for k in args.hide]

    bad = [k for k in args.hide if k not in LAYERS]
    if bad:
        ap.error(f'unknown layer(s) {bad}; choose from {LAYERS}')
    args.range = _parse_pairs(args.range, _span, 'range')
    args.truth = _parse_pairs(args.truth, float, 'truth')
    bad = [k for k in args.truth if k not in SWEEPABLE]
    if bad:
        ap.error(f'unknown truth param(s) {bad}; choose from {SWEEPABLE}')
    args.graph_text = _parse_pairs(args.graph_text, str, 'graph text')
    args.graph_fontsize = _parse_pairs(args.graph_fontsize, float, 'graph font size')
    for what, d in (('graph text', args.graph_text), ('graph font size', args.graph_fontsize)):
        bad = [k for k in d if k not in GRAPH_TEXT]
        if bad:
            ap.error(f'unknown {what} key(s) {bad}; choose from {list(GRAPH_TEXT)}')
    for k, v in args.graph_text.items():
        if GRAPH_TEXT[k][2] is None:
            ap.error(f'{k} is a size only (use --graph-fontsize)')
        problem = graph_text_problem(k, v)
        if problem:
            ap.error(f'graph text {k}: {problem}')
    args.color = _parse_pairs(args.color, str, 'color')
    bad = [k for k in args.color if k not in COLORS]
    if bad:
        ap.error(f'unknown colour key(s) {bad}; choose from {list(COLORS)}')
    for k, v in args.color.items():
        if not (v == 'auto' and k in AUTO_COLORS):
            try:
                pv.Color(v)
            except ValueError:
                ap.error(f'bad colour {k}={v!r}')
    if not 1 <= args.steps <= MAX_STEPS:
        ap.error(f'--steps must be 1-{MAX_STEPS}')
    if args.skip_step and not 1 <= args.skip_step < args.steps:
        ap.error(f'--skip-step must be 1-{args.steps - 1} with --steps {args.steps} (0 = none)')
    if args.off_screen and not (args.save or args.legend):
        ap.error('--off-screen needs --save and/or --legend')
    if args.n ** len(args.sweep) > 600:
        print(f'warning: {args.n ** len(args.sweep)} cars - rendering will be slow')

    if args.off_screen:
        FialaScene(args).show()
    else:
        sys.exit(run_gui(args))


if __name__ == '__main__':
    main()
