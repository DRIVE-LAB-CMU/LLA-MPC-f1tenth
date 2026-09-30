#!/usr/bin/env python3
"""compare_render.py - two runs, side by side, with the reference overlaid.

Companion to camera_align.py / trim_marker.py. Where camera_align.py is for
*fitting* a camera pose to one video, this one is for *presenting*: take two
already-calibrated clips of the same track - typically a nominal controller
and an LLAMPC one - put them next to each other, overlay the reference
raceline on both, label them, stamp a clock on each, and render the result
to a file small enough to submit.

WHAT ONE JOB LOOKS LIKE

  a JOB is one output video, built from two CLIPS (left/right, or top/bottom).
  Each clip is:

    video        the mp4/mov, any resolution - the two don't have to match
    calibration  a "<name>.camalign.json" written by camera_align.py; supplies
                 the camera/track pose (including keyframes for a moving
                 camera), the reference line's color/width/segments, and the
                 legend rows. Nothing else is read from it - the video and
                 path files it names are ignored, exactly as camera_align.py
                 itself ignores them.
    reference    the path file to draw (.npz with x/y or state, or csv). Only
                 the reference is drawn - there's no rollout trace and no
                 current-position marker.
    label        free text drawn top-center over that clip ("Nominal MPC",
                 "LLA-MPC", LaTeX allowed)
    npz          optional; if set, a state plot is rendered as its own little
                 panel welded to that clip - to the right, left, above or
                 below it - showing the driven path with a trail and a
                 moving dot.

  Clips run on a shared composite clock. Each has a time_shift (when it
  starts, in composite seconds) and its own trim window. WHEN ONE CLIP RUNS
  OUT THE OTHER KEEPS GOING: the finished panel holds its last frame (or dims
  it, or goes black - `end_behavior`), and its timestamp freezes at its own
  final time. The job's length is the longest clip unless you pin it.

  Labels and timestamps can be typeset with LaTeX, per job: tick the label
  and/or timestamp LaTeX boxes, and turn on usetex to hand the strings to
  your own `latex` + `dvipng` rather than matplotlib's mathtext. usetex
  understands text-mode markup - \\textbf, \\texttt, \\textcolor - and
  whatever your preamble's packages provide; mathtext handles maths mode
  only ($v_{max}$, $\\tau$), but needs nothing installed. The setting
  covers the label, the timestamp, the legend and the npz panel's title
  together, and "Check LaTeX" typesets a sample and reports what happened,
  since a failed render otherwise falls back to plain text in silence.

  The legend can be built here rather than inherited: rows carry a label
  (LaTeX allowed), a line or marker swatch, a colour and a stroke width, and
  "Seed from clips" fills in one row per clip from what that clip actually
  draws. With no rows defined, the legend falls back to the one saved in
  each clip's .camalign.json - and asking for it here draws it whether or
  not camera_align.py had it toggled on, since the switch that matters is
  the one in front of you.

  The npz panel's colours and strokes are all separately pickable behind
  "Panel style…": the driven trajectory, the full log behind it, the
  reference line (colour, width and pen style), the position marker, the
  backdrop, border and text.

  Each clip's reference line has a "custom" switch: on, the colour, stroke
  width and pen style set on the clip are used exactly as typed, the width
  in output panel pixels; off, all three come from the calibration (whose
  width is in source-video pixels and gets rescaled). Ticking the switch
  seeds the three from the calibration first, so nothing jumps. The same line is
  drawn inside the npz panel too (`ref` toggle), in the clip's colour and
  style, so the plot and the video agree about what the planned path is.

  `align` = fill removes the backdrop margins entirely: instead of centring
  a shorter unit in the taller one's slot, each unit is scaled until it
  spans the cross axis, so two clips of different aspect ratios - or one
  with an npz panel and one without - meet edge to edge. Nothing is cropped
  or distorted; the units just differ in length along the main axis. Any
  remaining gap is the `gap` setting and the per-clip `aux_gap`.

  The npz panel's `fit` decides how the path uses the space it's given:
  stretch (default) fills the panel edge to edge so it lines up with the
  video's width with no dead margin, aspect keeps the track's true shape
  and centres it, width spans the full width at true aspect and clips
  anything taller.

  Labels and timestamps are sized in PANEL pixels, so both panels match no
  matter what the two sources' resolutions were - what you type is what you
  get. The two things that come out of a calibration - the reference line's
  width and the legend's geometry - were tuned in camera_align.py against
  full-res frames, so they're rescaled by (panel width / source width) to
  keep the size they had there. If the two clips were shot at different
  resolutions their legends will then disagree; set a legend size on the job
  to pin both in panel pixels instead, or set legend scope so only one panel
  carries it.

LAYOUT

    layout       horizontal (side by side) or vertical (stacked)
    panel_size   the height every video is normalized to when horizontal,
                 or the width when vertical - this is how two clips of
                 different resolutions end up the same size
    align        start / center / end, for the cross axis, when the two
                 units come out different sizes (different aspect ratios,
                 or only one has an npz panel)
    gap, bg      spacing and backdrop between the units
    max_width    hard ceiling; the whole plan is recomputed smaller rather
                 than resampled twice, so text stays crisp

EXPORT

  Renders straight into ffmpeg over a pipe (libx264, yuv420p) with progress
  printed as it goes:

    [ 42.3%]  frame 508/1200   31.7 fps   elapsed 00:16   eta 00:21

  target_mb enforces a size budget (default 20 MB): if the CRF encode lands
  over it, the file is re-encoded two-pass at the bitrate that fits, up to
  three tightening attempts. No re-render - only the cheap transcode repeats.
  Without ffmpeg it falls back to OpenCV's mp4v writer, which honors neither
  CRF nor the budget.

  "All -> one video" renders every job back to back into a single file
  instead of one file each. Jobs keep their own sizes and frame rates; the
  reel picks one canvas (the largest job, or a size you set) and one clock
  (the fastest job, or a rate you set), fits each job into it - centred and
  letterboxed if it's smaller - and can drop a gap of blank frames between
  them. The size budget then applies to the whole reel rather than per job,
  which is the number that matters for a single submission. Only one job's
  videos are open at a time, so a long batch doesn't hold every decoder
  open at once.

  Jobs are a list: queue up as many pairs as you like and render them in
  succession, from the GUI or headless:

    python compare_render.py                       # GUI, empty
    python compare_render.py batch.compare.json    # GUI, batch loaded
    python compare_render.py batch.compare.json --render          # all jobs
    python compare_render.py batch.compare.json --render --only 2 # just #2
    python compare_render.py batch.compare.json --render --combined # one reel

Requires: PyQt6, opencv-python, numpy.  Optional: matplotlib (LaTeX labels),
ffmpeg on PATH (quality + size control).
    pip install PyQt6 opencv-python numpy matplotlib
"""
from __future__ import annotations

import io
import os
import re
import sys
import csv
import copy
import json
import time
import bisect
import shutil
import tempfile
import subprocess

import numpy as np
import cv2

from PyQt6 import QtCore, QtGui, QtWidgets

# matplotlib is optional and used only to rasterize LaTeX labels.
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext as _mathtext
    from matplotlib.font_manager import FontProperties as _FontProperties
    HAVE_MATHTEXT = True
except Exception:                                   # pragma: no cover
    HAVE_MATHTEXT = False

SCHEMA = "vcompare/1"
SUFFIX = ".compare.json"
DEFAULT_LATEX_PREAMBLE = r"\usepackage{xcolor}"


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def latex_available():
    return HAVE_MATHTEXT


def usetex_available():
    return bool(HAVE_MATHTEXT and shutil.which("latex") and shutil.which("dvipng"))


def ffmpeg_available():
    return bool(shutil.which("ffmpeg"))


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_CAM = {
    "cam_x": 0.0, "cam_y": -10.0, "cam_z": 3.0,
    "azimuth": 90.0, "elevation": 15.0, "roll": 0.0,
    "fov": 60.0, "pp_x": 0.0, "pp_y": 0.0,
    "path_scale": 1.0,
    "path_yaw": 0.0, "path_pitch": 0.0, "path_roll": 0.0,
    "path_offset_x": 0.0, "path_offset_y": 0.0, "path_z": 0.0,
}

PEN_STYLES = {
    "solid": QtCore.Qt.PenStyle.SolidLine,
    "dash": QtCore.Qt.PenStyle.DashLine,
    "dot": QtCore.Qt.PenStyle.DotLine,
    "dashdot": QtCore.Qt.PenStyle.DashDotLine,
    "dashdotdot": QtCore.Qt.PenStyle.DashDotDotLine,
}
CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"]
POSITIONS = CORNERS + ["top-center", "bottom-center"]
LEGEND_SCOPES = ["both", "left", "right", "none"]
TIMESTAMP_SOURCES = ["video", "trajectory"]
AUX_POSITIONS = ["none", "right", "left", "above", "below"]
# How the state plot uses the panel it's given:
#   aspect  uniform scale, whole path inside - true track shape, but a wide
#           panel leaves dead space either side
#   stretch x and y scaled independently so the path fills the panel edge to
#           edge - fits horizontally, at the cost of distorting the shape
#   width   uniform scale set by the width alone, so the path spans the full
#           width; anything taller than the panel is clipped
AUX_FITS = ["stretch", "aspect", "width"]
LAYOUTS = ["horizontal", "vertical"]
# "fill" scales each unit up until it spans the cross axis, instead of
# centring it and painting backdrop either side - see Layout.plan.
ALIGNMENTS = ["center", "fill", "start", "end"]
# "inherit" = whatever the clip's .camalign.json says
REF_STYLES = ["inherit"] + list(PEN_STYLES.keys())
END_BEHAVIORS = ["hold", "hold-dim", "black"]

DEFAULT_REF_COLOR = "#dc46c8ff"        # falls back to a magenta-ish cyan pair
DEFAULT_REF_COLOR = "#dc46c8ff"

DEFAULT_LEGEND = {
    "enabled": True,
    "corner": "top-right",
    "margin": 40.0, "padding": 24.0,
    "text_size": 64.0, "bold": True,
    "sample_len": 140.0, "gap": 24.0, "row_gap": 14.0,
    "bg_alpha": 150, "columns": 1, "col_gap": 60.0,
}

DEFAULT_TIMESTAMP = {
    "enabled": True,
    "corner": "bottom-left",
    "source": "video",
    "prefix": "t = ", "suffix": " s",
    "decimals": 1,
    "text_size": 40.0, "bold": True,
    "margin": 26.0, "padding": 12.0,
    "bg_alpha": 150, "latex": False,
    "text_color": "#ffffffff", "bg_color": "#ff000000",
    "freeze_at_end": True,
}

DEFAULT_LABEL = {
    "enabled": True,
    "position": "top-center",
    "text_size": 46.0, "bold": True,
    "margin": 22.0, "padding": 14.0,
    "bg_alpha": 160, "latex": False,
    "text_color": "#ffffffff", "bg_color": "#ff000000",
}

DEFAULT_AUXSTYLE = {
    "bg_color": "#ff14141a",
    "border_color": "#ff3c3c46",
    "path_color": "#ff5a5a6e",
    "trail_color": "#ffffd200",
    "point_color": "#ffff3c3c",
    "text_color": "#ffdcdce6",
    "line_width": 2.0,
    "trail_width": 3.0,
    "point_size": 7.0,
    "margin": 22.0,
    "text_size": 26.0,
    "show_readout": True,
    "show_heading": True,
    # the planned line, drawn in the panel underneath the driven one, the
    # same way camera_align.py draws it over the video
    "show_ref": True,
    "ref_color": "",          # blank -> follow the clip's reference colour
    "ref_width": 2.0,
    "ref_style": "dot",
    "title": "",
    "fit": "stretch",
}

DEFAULT_CLIP = {
    "video": None,
    "calib": None,
    "reference": None,
    "label": "",
    "time_shift": 0.0,
    "trim_start": 0.0,
    "trim_end": 0.0,          # 0 -> to the end of the clip
    "overlay": True,
    # Reference line overrides. Left empty/zero, each falls back to the
    # calibration's own setting (which is in SOURCE video pixels and gets
    # rescaled); set here, the width is in panel pixels like everything
    # else in this tool, so both clips can be given one matched look.
    "ref_custom": False,      # False -> take all three from the calibration
    "ref_color": "#ff46c8ff",
    "ref_width": 3.0,         # panel pixels, used as typed
    "ref_style": "dot",
    "legend": True,
    "timestamp": True,
    "npz": None,
    "npz_offset": 0.0,
    "aux": "none",
    "aux_frac": 0.6,
    "aux_gap": 8.0,
    "aux_style": dict(DEFAULT_AUXSTYLE),
}

DEFAULT_JOB = {
    "name": "",
    "left": copy.deepcopy(DEFAULT_CLIP),
    "right": copy.deepcopy(DEFAULT_CLIP),
    "layout": "horizontal",
    "align": "center",
    "gap": 16.0,
    "bg": "#ff0a0a0e",
    "panel_size": 720,
    "max_width": 1920,
    "max_height": 0,
    "fps": 30.0,
    "duration": 0.0,          # 0 -> longest clip
    "end_behavior": "hold",
    "output": "",
    "label": dict(DEFAULT_LABEL),
    "timestamp": dict(DEFAULT_TIMESTAMP),
    # entries: this tool's own legend rows. Empty -> fall back to whatever
    # legend the clip's .camalign.json carries (which may be none at all,
    # if you never built one in camera_align.py).
    "legend": {"scope": "both", "size": 0.0, "corner": "inherit",
               "columns": 1, "bg_alpha": 150, "entries": [],
               "text_color": "#ffffffff", "bg_color": "#ff000000"},
    # Applies to the label, the timestamp and the legend alike. usetex
    # shells out to a real `latex` + `dvipng` on PATH (text-mode commands,
    # \textcolor, your own packages via the preamble); with it off, LaTeX-
    # flagged strings go through matplotlib's mathtext instead.
    "latex": {"usetex": False, "preamble": DEFAULT_LATEX_PREAMBLE},
    "encode": {"use_ffmpeg": True, "crf": 20, "preset": "veryfast",
               "target_mb": 20.0},
}


# One file containing every job in the batch, back to back. Jobs keep their
# own sizes and frame rates; the reel picks a single canvas and clock and
# fits each job into it, so a 1278x356 pair and a 660x830 stack can share
# one output.
DEFAULT_COMBINED = {
    "output": "",             # blank -> comparisons/<batch name>_all.mp4
    "fps": 0.0,               # 0 -> the fastest job's rate
    "width": 0, "height": 0,  # 0 -> big enough for the largest job
    "gap_s": 0.0,             # blank frames between jobs
    "bg": "#ff000000",
    "encode": {"use_ffmpeg": True, "crf": 20, "preset": "veryfast",
               "target_mb": 20.0},
}


def merged(default, loaded):
    """Recursively fill `loaded` with anything missing from `default`."""
    out = copy.deepcopy(default)
    if not isinstance(loaded, dict):
        return out
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def qcolor(spec, fallback="#ffffffff"):
    c = QtGui.QColor(spec if spec else fallback)
    return c if c.isValid() else QtGui.QColor(fallback)


def hexargb(color):
    return color.name(QtGui.QColor.NameFormat.HexArgb)


def _css_rgba(color):
    return (f"rgba({color.red()},{color.green()},{color.blue()},"
            f"{color.alphaF():.3f})")


def _rel(path, base):
    if not path:
        return None
    try:
        return os.path.relpath(os.path.abspath(path),
                               os.path.abspath(base)).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def _resolve(path, base):
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base or ".", path))


def _fmt_clock(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--"
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ===========================================================================
# LaTeX label rasterization (mathtext, or a real TeX install)
# ===========================================================================
_LATEX_CACHE = {}
_LATEX_CACHE_MAX = 256
_LATEX_WARNED = [False]
_LATEX_LAST_ERROR = [""]


def latex_last_error():
    """Why the last LaTeX render fell back to plain text ('' if none).

    Without this a bad preamble or a missing package just silently draws
    the raw source text and you're left guessing.
    """
    return _LATEX_LAST_ERROR[0]


def latex_status():
    """One line on what's available: no matplotlib / mathtext only / usetex."""
    if not HAVE_MATHTEXT:
        return "matplotlib not installed - LaTeX labels unavailable"
    latex, dvipng = shutil.which("latex"), shutil.which("dvipng")
    if latex and dvipng:
        return f"usetex available: {latex} + {dvipng}"
    missing = " and ".join(n for n, p in (("latex", latex), ("dvipng", dvipng))
                           if not p)
    return f"mathtext only - {missing} not on PATH"


def _keyed_qimage(png_bytes, color):
    raw = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("could not decode the rendered png")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    bgr = raw[:, :, :3].astype(np.float32)
    coverage = np.clip(255.0 - bgr.min(axis=2), 0.0, 255.0) * color.alphaF()
    hh, ww = coverage.shape
    bgra = np.empty((hh, ww, 4), dtype=np.uint8)
    bgra[:, :, 0] = color.blue()
    bgra[:, :, 1] = color.green()
    bgra[:, :, 2] = color.red()
    bgra[:, :, 3] = coverage.astype(np.uint8)
    bgra = np.ascontiguousarray(bgra)
    return QtGui.QImage(bgra.data, ww, hh, 4 * ww,
                        QtGui.QImage.Format.Format_ARGB32).copy()


def _render_mathtext(text, px, color, bold):
    prop = _FontProperties(size=px, weight="bold" if bold else "normal")
    buf = io.BytesIO()
    _mathtext.math_to_image(text, buf, prop=prop, dpi=72, format="png",
                            color="black")
    return _keyed_qimage(buf.getvalue(), color)


def _tex_error_summary(output, max_lines=6):
    try:
        lines = output.decode("utf-8", "replace").splitlines()
    except Exception:
        return "latex failed"
    bad = [ln for ln in lines if ln.startswith("!")]
    return " / ".join(bad[:max_lines]) if bad else "latex failed"


def _render_usetex(text, px, color, bold, preamble):
    doc = "\n".join([
        r"\documentclass{article}", r"\usepackage{type1cm}",
        r"\usepackage{xcolor}", preamble or "", r"\pagestyle{empty}",
        r"\definecolor{cmpfg}{RGB}{%d,%d,%d}" % (color.red(), color.green(),
                                                 color.blue()),
        r"\begin{document}",
        r"\fontsize{%dpt}{%dpt}\selectfont" % (px, int(px * 1.2)),
        r"\color{cmpfg}", r"\bfseries" if bold else "", text,
        r"\end{document}",
    ])
    tmp = tempfile.mkdtemp(prefix="cmp_tex_")
    try:
        with open(os.path.join(tmp, "job.tex"), "w") as f:
            f.write(doc)

        def run(cmd):
            return subprocess.run(cmd, cwd=tmp, timeout=30,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        r = run(["latex", "-interaction=nonstopmode", "-halt-on-error",
                 "job.tex"])
        if r.returncode != 0 or not os.path.exists(os.path.join(tmp, "job.dvi")):
            raise RuntimeError(_tex_error_summary(r.stdout))
        r = run(["dvipng", "-T", "tight", "-D", "72", "-bg", "Transparent",
                 "-z", "9", "-o", "out.png", "job.dvi"])
        png = os.path.join(tmp, "out.png")
        if r.returncode != 0 or not os.path.exists(png):
            raise RuntimeError(_tex_error_summary(r.stdout))
        with open(png, "rb") as f:
            data = f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("could not decode the dvipng output")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
    elif raw.shape[2] == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
    bgra = raw.copy()
    bgra[:, :, 3] = np.clip(bgra[:, :, 3].astype(np.float32) * color.alphaF(),
                            0, 255).astype(np.uint8)
    bgra = np.ascontiguousarray(bgra)
    hh, ww = bgra.shape[:2]
    return QtGui.QImage(bgra.data, ww, hh, 4 * ww,
                        QtGui.QImage.Format.Format_ARGB32).copy()


def render_latex(text, pixel_size, color, bold=False, usetex=False,
                 preamble=DEFAULT_LATEX_PREAMBLE):
    """Rasterize `text` as LaTeX -> transparent QImage, or None on failure."""
    if not text or not HAVE_MATHTEXT:
        return None
    if usetex and not usetex_available():
        usetex = False
    px = max(4, int(round(float(pixel_size))))
    key = (text, px, int(color.rgba()), bool(bold), bool(usetex),
           preamble if usetex else "")
    if key in _LATEX_CACHE:
        return _LATEX_CACHE[key]
    img = None
    try:
        img = (_render_usetex(text, px, color, bold, preamble) if usetex
               else _render_mathtext(text, px, color, bold))
        _LATEX_LAST_ERROR[0] = ""
    except Exception as exc:
        _LATEX_LAST_ERROR[0] = f"{'usetex' if usetex else 'mathtext'}: {exc}"
        if not _LATEX_WARNED[0]:
            log(f"LaTeX render failed ({exc}) - falling back to plain text")
            _LATEX_WARNED[0] = True
        img = None
    if len(_LATEX_CACHE) > _LATEX_CACHE_MAX:
        _LATEX_CACHE.clear()
    _LATEX_CACHE[key] = img
    return img


# ===========================================================================
# Projection (identical maths to camera_align.py, so poses transfer exactly)
# ===========================================================================
def _camera_basis(azimuth_deg, elevation_deg, roll_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    roll = np.radians(roll_deg)
    forward = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        -np.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    n = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if n < 1e-8 else right / n
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-8)
    right_r = np.cos(roll) * right + np.sin(roll) * down
    down_r = -np.sin(roll) * right + np.cos(roll) * down
    return right_r, down_r, forward


def _track_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return Rz @ Rx @ Ry


def project_points(xy, cam, image_w, image_h):
    """World ground-plane (x, y) -> (u, v, valid) pixel coordinates."""
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    scale = cam.get("path_scale", 1.0)
    local = np.column_stack([xy[:, 0] * scale, xy[:, 1] * scale,
                             np.zeros(len(xy))])
    R = _track_rotation_matrix(cam.get("path_yaw", cam.get("path_rotation", 0.0)),
                               cam.get("path_pitch", 0.0),
                               cam.get("path_roll", 0.0))
    P = local @ R.T + np.array([cam.get("path_offset_x", 0.0),
                                cam.get("path_offset_y", 0.0),
                                cam.get("path_z", 0.0)])
    right, down, forward = _camera_basis(cam["azimuth"], cam["elevation"],
                                         cam["roll"])
    rel = P - np.array([cam["cam_x"], cam["cam_y"], cam["cam_z"]])
    Xc, Yc, Zc = rel @ right, rel @ down, rel @ forward
    f = (image_w / 2.0) / np.tan(np.radians(cam["fov"]) / 2.0)
    valid = Zc > 1e-3
    u = np.full(len(P), np.nan)
    v = np.full(len(P), np.nan)
    u[valid] = f * Xc[valid] / Zc[valid] + image_w / 2.0 + cam.get("pp_x", 0.0)
    v[valid] = f * Yc[valid] / Zc[valid] + image_h / 2.0 + cam.get("pp_y", 0.0)
    return u, v, valid


def _lerp_angle(a, b, t):
    return a + (((b - a + 180) % 360) - 180) * t


def interpolate_camera(keyframes, frame_idx):
    if not keyframes:
        return dict(DEFAULT_CAM)
    frames = sorted(keyframes.keys())
    if frame_idx <= frames[0]:
        return dict(keyframes[frames[0]])
    if frame_idx >= frames[-1]:
        return dict(keyframes[frames[-1]])
    i = bisect.bisect_right(frames, frame_idx) - 1
    f0, f1 = frames[i], frames[i + 1]
    t = (frame_idx - f0) / float(f1 - f0)
    c0, c1 = keyframes[f0], keyframes[f1]
    angle_keys = ("azimuth", "elevation", "roll",
                  "path_yaw", "path_pitch", "path_roll")
    return {k: (_lerp_angle(c0[k], c1[k], t) if k in angle_keys
                else c0[k] + (c1[k] - c0[k]) * t) for k in DEFAULT_CAM}


def scaled_cam(cam, s):
    """The same pose expressed for an image scaled by `s` (uniform)."""
    out = dict(cam)
    out["pp_x"] = cam.get("pp_x", 0.0) * s
    out["pp_y"] = cam.get("pp_y", 0.0) * s
    return out


def _star_polygon(cx, cy, r_outer, n_points=5, inner_ratio=0.45):
    poly = QtGui.QPolygonF()
    r_inner = r_outer * inner_ratio
    for i in range(n_points * 2):
        ang = -np.pi / 2.0 + i * np.pi / n_points
        r = r_outer if i % 2 == 0 else r_inner
        poly.append(QtCore.QPointF(cx + r * np.cos(ang), cy + r * np.sin(ang)))
    return poly


def _corner_origin(corner, w, h, box_w, box_h, margin):
    x = margin if corner.endswith("left") else w - margin - box_w
    y = margin if corner.startswith("top") else h - margin - box_h
    return x, y


# ===========================================================================
# Sources
# ===========================================================================
class VideoFrameSource:
    """Random access with a fast path for forward, near-sequential reads."""

    MAX_GRAB_AHEAD = 12

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {path}")
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            n = 0
            while self.cap.grab():
                n += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.n_frames = max(1, n)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 1e-3 else 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cached_idx = -1
        self._last = None

    @property
    def duration(self):
        return self.n_frames / max(self.fps, 1e-6)

    def frame_at(self, idx):
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        if idx == self._cached_idx and self._last is not None:
            return self._last
        ahead = idx - self._cached_idx
        if ahead < 0 or ahead > self.MAX_GRAB_AHEAD:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        else:
            for _ in range(ahead - 1):
                self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
        if not ok:
            return self._last
        self._cached_idx = idx
        self._last = frame
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class PathSource:
    """A ground-plane path: 'timeseries' (with a clock) or 'static'."""

    def __init__(self, path):
        self.path = path
        self.kind = None
        self.xy = None
        self.state = None
        self.time = None
        self.dt = None
        self._load()
        self.n = 0 if self.xy is None else len(self.xy)

    def _load(self):
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".npz":
            raw = np.load(self.path, allow_pickle=True)
            keys = set(raw.files)
            if "state" in keys:
                state = np.asarray(raw["state"], dtype=float)
                if state.ndim != 2 or state.shape[1] < 2:
                    raise RuntimeError("'state' array must be (N, >=2)")
                self.state = state
                self.xy = state[:, :2]
                self.kind = "timeseries"
                if "time" in keys:
                    t = np.asarray(raw["time"], dtype=float)
                    if t.size:
                        t = t - t[0]
                        if np.nanmax(t) > 1e6:      # perf_counter_ns
                            t = t * 1e-9
                    self.time = t
                    self.dt = (float(np.median(np.diff(t))) if len(t) > 1
                               else 1 / 25.0)
                else:
                    self.dt = 1 / 25.0
            elif "x" in keys and "y" in keys:
                x = np.asarray(raw["x"], dtype=float).reshape(-1)
                y = np.asarray(raw["y"], dtype=float).reshape(-1)
                self.xy = np.column_stack([x, y])
                self.kind = "static"
            else:
                raise RuntimeError("npz has neither 'state' nor 'x'/'y'")
        elif ext in (".csv", ".txt"):
            rows = []
            with open(self.path, newline="") as f:
                sniff = f.read(2048)
                f.seek(0)
                has_header = any(c.isalpha() for c in sniff.split("\n")[0])
                reader = csv.reader(f)
                ix, iy, it = 0, 1, None
                if has_header:
                    header = [h.strip().lower() for h in next(reader)]
                    if "x" in header and "y" in header:
                        ix, iy = header.index("x"), header.index("y")
                        it = header.index("t") if "t" in header else None
                for row in reader:
                    if not row:
                        continue
                    rows.append((float(row[ix]), float(row[iy]),
                                 float(row[it]) if it is not None else None))
            self.xy = np.array([(r[0], r[1]) for r in rows], dtype=float)
            ts = [r[2] for r in rows]
            if rows and all(t is not None for t in ts) and len(ts) > 1:
                t = np.asarray(ts, dtype=float)
                self.time = t - t[0]
                self.dt = float(np.median(np.diff(self.time)))
                self.kind = "timeseries"
                self.state = np.column_stack([self.xy])
            else:
                self.kind = "static"
        else:
            raise RuntimeError(f"unsupported path file type: {ext}")

    @property
    def is_timeseries(self):
        return self.kind == "timeseries"

    def bounds(self):
        if self.xy is None or len(self.xy) == 0:
            return (0.0, 0.0), (0.0, 0.0)
        return ((float(self.xy[:, 0].min()), float(self.xy[:, 1].min())),
                (float(self.xy[:, 0].max()), float(self.xy[:, 1].max())))

    def full_time_range(self):
        if not self.is_timeseries or self.n == 0:
            return 0.0, 0.0
        if self.time is not None and len(self.time):
            return float(self.time[0]), float(self.time[-1])
        return 0.0, float((self.n - 1) * self.dt)

    def sample_at(self, t_seconds):
        if self.n == 0:
            return None
        if not self.is_timeseries:
            return 0
        if self.time is not None and len(self.time):
            idx = int(np.searchsorted(self.time, t_seconds))
        else:
            idx = int(round(t_seconds / max(self.dt or 1e-9, 1e-9)))
        return int(np.clip(idx, 0, self.n - 1))


# ===========================================================================
# Calibration: what we take from a camera_align.py .camalign.json
# ===========================================================================
class Calibration:
    """Pose + reference line style + legend, read from a .camalign.json.

    Deliberately partial, and deliberately file-agnostic: the video and path
    names inside the calibration are ignored entirely, same as
    camera_align.py's own loader treats them as provenance only.
    """

    def __init__(self, path=None):
        self.path = path
        self.cam = dict(DEFAULT_CAM)
        self.keyframes = {}
        self.offset_s = 0.0
        self.image_w = None
        self.image_h = None
        self.ref_color = qcolor("#ff46c8ff")
        self.ref_width = 2.0
        self.ref_segments = []
        self.ref_dotted = True
        self.legend = dict(DEFAULT_LEGEND)
        self.legend_entries = []
        self.legend_text_color = qcolor("#ffffffff")
        self.legend_bg_color = qcolor("#ff000000")
        self.usetex = False
        self.preamble = DEFAULT_LATEX_PREAMBLE
        if path:
            self.load(path)

    @staticmethod
    def _migrate(c):
        c = dict(c or {})
        if "path_yaw" not in c and "path_rotation" in c:
            c["path_yaw"] = c["path_rotation"]
        return c

    def load(self, path):
        with open(path) as f:
            spec = json.load(f)
        self.path = path
        cam = self._migrate(spec.get("camera"))
        self.cam = {k: float(cam.get(k, DEFAULT_CAM[k])) for k in DEFAULT_CAM}
        self.keyframes = {
            int(f): {k: float(self._migrate(c).get(k, DEFAULT_CAM[k]))
                     for k in DEFAULT_CAM}
            for f, c in (spec.get("keyframes") or {}).items()}
        self.offset_s = float((spec.get("sync") or {}).get("offset_s", 0.0))
        self.image_w = spec.get("image_w")
        self.image_h = spec.get("image_h")

        disp = spec.get("display") or {}
        if disp.get("ref_color"):
            self.ref_color = qcolor(disp["ref_color"])
        if disp.get("ref_width"):
            self.ref_width = float(disp["ref_width"])
        self.ref_segments = []
        for s in disp.get("ref_segments") or []:
            try:
                self.ref_segments.append({"start": float(s["start"]),
                                          "end": float(s["end"]),
                                          "color": qcolor(s["color"])})
            except (KeyError, TypeError, ValueError):
                continue

        self.legend = merged(DEFAULT_LEGEND, disp.get("legend") or {})
        self.legend_entries = []
        for e in disp.get("legend_entries") or []:
            try:
                self.legend_entries.append({
                    "label": str(e.get("label", "")),
                    "kind": e.get("kind", "line"),
                    "style": e.get("style", "solid"),
                    "width": float(e.get("width", 8.0)),
                    "latex": bool(e.get("latex", False)),
                    "color": qcolor(e["color"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        if disp.get("legend_text_color"):
            self.legend_text_color = qcolor(disp["legend_text_color"])
        if disp.get("legend_bg_color"):
            self.legend_bg_color = qcolor(disp["legend_bg_color"])
        self.usetex = bool(disp.get("usetex", False))
        self.preamble = str(disp.get("latex_preamble", DEFAULT_LATEX_PREAMBLE))

    def cam_at(self, frame_idx):
        if self.keyframes:
            return interpolate_camera(self.keyframes, frame_idx)
        return dict(self.cam)


# ===========================================================================
# Drawing primitives
# ===========================================================================
def _draw_polyline(painter, u, v, valid):
    path = QtGui.QPainterPath()
    open_ = False
    for x, y, ok in zip(u, v, valid):
        if not ok or not np.isfinite(x) or not np.isfinite(y):
            open_ = False
            continue
        if not open_:
            path.moveTo(x, y)
            open_ = True
        else:
            path.lineTo(x, y)
    painter.drawPath(path)


def _segment_index_bounds(src, a, b):
    """A segment's [start, end] -> inclusive sample indices.

    Timeseries paths read them as seconds on the path's own clock (same as
    camera_align.py). Static paths - a raceline has no clock - read them as
    fractions of the path's length, so you can still recolor "the last
    third" or "just the hairpin".
    """
    if src is None or src.n == 0:
        return 0, -1
    a, b = sorted((float(a), float(b)))
    if src.is_timeseries:
        if src.time is not None and len(src.time):
            i0 = int(np.searchsorted(src.time, a, side="left"))
            i1 = int(np.searchsorted(src.time, b, side="right")) - 1
        else:
            dt = max(src.dt or 0.0, 1e-9)
            i0, i1 = int(np.ceil(a / dt)), int(np.floor(b / dt))
    else:
        i0 = int(np.floor(np.clip(a, 0.0, 1.0) * (src.n - 1)))
        i1 = int(np.ceil(np.clip(b, 0.0, 1.0) * (src.n - 1)))
    return int(np.clip(i0, 0, src.n - 1)), int(np.clip(i1, 0, src.n - 1))


def draw_reference(painter, w, h, cam, src, color, width, segments,
                   pen_style="dot"):
    """Project and stroke the reference path, honoring color segments."""
    if src is None or src.n == 0:
        return

    def stroke(xy, col):
        u, v, valid = project_points(xy, cam, w, h)
        pen = QtGui.QPen(col)
        pen.setWidthF(max(0.3, width))
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        pen.setStyle(PEN_STYLES.get(pen_style, QtCore.Qt.PenStyle.DotLine))
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        _draw_polyline(painter, u, v, valid)

    if not segments:
        stroke(src.xy, color)
        return
    stroke(src.xy, color)
    for seg in segments:
        i0, i1 = _segment_index_bounds(src, seg["start"], seg["end"])
        if i1 > i0:
            stroke(src.xy[i0:i1 + 1], seg["color"])


def draw_legend(painter, w, h, legend, entries, text_color, bg_color, s,
                usetex=False, preamble=DEFAULT_LATEX_PREAMBLE):
    """Legend box, with every geometry value scaled by `s`."""
    if not entries:
        return
    painter.save()
    font = QtGui.QFont()
    font.setPixelSize(max(1, int(round(legend["text_size"] * s))))
    font.setBold(bool(legend["bold"]))
    painter.setFont(font)
    fm = QtGui.QFontMetricsF(font)

    pad = legend["padding"] * s
    gap = legend["gap"] * s
    sample = legend["sample_len"] * s
    row_gap = legend["row_gap"] * s
    col_gap = legend.get("col_gap", 60.0) * s

    n = len(entries)
    ncols = int(np.clip(int(legend.get("columns", 1) or 1), 1, 8))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / float(ncols)))

    columns = []
    for c in range(ncols):
        rows = []
        for e in entries[c * nrows:(c + 1) * nrows]:
            img = None
            if e.get("latex"):
                img = render_latex(e["label"], legend["text_size"] * s,
                                   text_color, bool(legend["bold"]),
                                   usetex=usetex, preamble=preamble)
            if img is not None:
                lw, lh = float(img.width()), float(img.height())
            else:
                lw, lh = fm.horizontalAdvance(e["label"]), fm.height()
            wpx = e["width"] * s
            marker_h = wpx if e["kind"] == "line" else wpx * 2.0
            rows.append((e, img, lw, max(lh, marker_h), wpx))
        if rows:
            columns.append(rows)

    col_w = [sample + gap + max(r[2] for r in rows) for rows in columns]
    col_h = [sum(r[3] for r in rows) + row_gap * (len(rows) - 1)
             for rows in columns]
    box_w = pad * 2 + sum(col_w) + col_gap * (len(columns) - 1)
    box_h = pad * 2 + (max(col_h) if col_h else 0.0)
    x, y = _corner_origin(legend["corner"], w, h, box_w, box_h,
                          legend["margin"] * s)

    if legend["bg_alpha"] > 0:
        bg = QtGui.QColor(bg_color)
        bg.setAlpha(int(np.clip(legend["bg_alpha"], 0, 255)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(QtCore.QRectF(x, y, box_w, box_h),
                                pad * 0.5, pad * 0.5)

    cx0 = x + pad
    for rows, colw in zip(columns, col_w):
        cy = y + pad
        text_w = colw - sample - gap
        for e, img, lw, rh, wpx in rows:
            mid = cy + rh / 2.0
            if e["kind"] == "line":
                pen = QtGui.QPen(e["color"])
                pen.setWidthF(max(0.5, wpx))
                pen.setStyle(PEN_STYLES.get(e["style"],
                                            QtCore.Qt.PenStyle.SolidLine))
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawLine(QtCore.QPointF(cx0, mid),
                                 QtCore.QPointF(cx0 + sample, mid))
            else:
                r = max(1.0, wpx)
                mx = cx0 + sample / 2.0
                painter.setBrush(QtGui.QBrush(e["color"]))
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0),
                                          max(1.0, r * 0.12)))
                if e["kind"] == "star":
                    painter.drawPolygon(_star_polygon(mx, mid, r))
                else:
                    painter.drawEllipse(QtCore.QPointF(mx, mid), r, r)
            tx = cx0 + sample + gap
            if img is not None:
                painter.drawImage(QtCore.QPointF(tx, mid - img.height() / 2.0),
                                  img)
            else:
                painter.setPen(QtGui.QPen(text_color))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawText(
                    QtCore.QRectF(tx, cy, max(text_w, lw), rh),
                    int(QtCore.Qt.AlignmentFlag.AlignVCenter |
                        QtCore.Qt.AlignmentFlag.AlignLeft), e["label"])
            cy += rh + row_gap
        cx0 += colw + col_gap
    painter.restore()


def legend_entries_from(raw):
    """JSON legend rows -> drawable ones (colour strings become QColors)."""
    out = []
    for e in raw or []:
        try:
            out.append({
                "label": str(e.get("label", "")),
                "kind": e.get("kind", "line"),
                "style": e.get("style", "solid"),
                "width": float(e.get("width", 8.0)),
                "latex": bool(e.get("latex", False)),
                "color": qcolor(e.get("color", "#ffffffff")),
            })
        except (TypeError, ValueError):
            continue
    return out


def job_legend_geometry(lg, cal=None):
    """Legend box geometry for this tool's own rows.

    Only the text size is asked for; every other measurement scales with
    it, so one number gives a legend that stays in proportion.
    """
    geom = dict(DEFAULT_LEGEND)
    size = float(lg.get("size", 0.0) or 0.0) or 40.0
    k = size / DEFAULT_LEGEND["text_size"]
    geom["text_size"] = size
    for key in ("margin", "padding", "sample_len", "gap", "row_gap",
                "col_gap"):
        geom[key] = DEFAULT_LEGEND[key] * k
    corner = lg.get("corner", "inherit")
    if corner not in CORNERS:
        corner = (cal.legend.get("corner", "top-right") if cal is not None
                  else "top-right")
    geom["corner"] = corner
    geom["columns"] = int(lg.get("columns", 1) or 1)
    geom["bg_alpha"] = float(lg.get("bg_alpha", 150))
    return geom


def _draw_text_box(painter, w, h, text, style, s, origin, usetex=False,
                   preamble=DEFAULT_LATEX_PREAMBLE):
    """Shared text-in-a-rounded-box renderer for labels and timestamps.

    `origin` is either a corner name or "top-center"/"bottom-center".
    """
    if not text:
        return
    painter.save()
    text_color = qcolor(style.get("text_color", "#ffffffff"))
    bg_color = qcolor(style.get("bg_color", "#ff000000"))
    font = QtGui.QFont()
    font.setPixelSize(max(1, int(round(style["text_size"] * s))))
    font.setBold(bool(style.get("bold", True)))
    painter.setFont(font)
    fm = QtGui.QFontMetricsF(font)

    img = None
    if style.get("latex"):
        img = render_latex(text, style["text_size"] * s, text_color,
                           bool(style.get("bold", True)), usetex=usetex,
                           preamble=preamble)
    if img is not None:
        tw, th = float(img.width()), float(img.height())
    else:
        tw, th = fm.horizontalAdvance(text), fm.height()

    pad = style["padding"] * s
    margin = style["margin"] * s
    box_w, box_h = tw + pad * 2, th + pad * 2
    if origin == "top-center":
        x, y = (w - box_w) / 2.0, margin
    elif origin == "bottom-center":
        x, y = (w - box_w) / 2.0, h - margin - box_h
    else:
        x, y = _corner_origin(origin, w, h, box_w, box_h, margin)

    if style.get("bg_alpha", 0) > 0:
        bg = QtGui.QColor(bg_color)
        bg.setAlpha(int(np.clip(style["bg_alpha"], 0, 255)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(QtCore.QRectF(x, y, box_w, box_h),
                                pad * 0.6, pad * 0.6)
    if img is not None:
        painter.drawImage(QtCore.QPointF(x + pad, y + pad), img)
    else:
        painter.setPen(QtGui.QPen(text_color))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawText(QtCore.QRectF(x + pad, y + pad, tw, th),
                         int(QtCore.Qt.AlignmentFlag.AlignVCenter |
                             QtCore.Qt.AlignmentFlag.AlignHCenter), text)
    painter.restore()


# ===========================================================================
# The npz side panel
# ===========================================================================
class NpzPlot:
    """A small XY state plot: full path, trail so far, current pose.

    Cheap on purpose - plain QPainter, no matplotlib - because it's redrawn
    once per output frame.
    """

    MAX_POINTS = 6000

    def __init__(self, src, style, ref=None):
        self.src = src
        # the planned line lives in the same track frame as the state log,
        # so it can share the plot's transform directly
        self.ref = ref
        self.style = merged(DEFAULT_AUXSTYLE, style or {})
        self._size = None
        self._px = self._py = None
        self._full = None
        self._ref_poly = None
        self._deci = 1

    def _prepare(self, w, h):
        if self._size == (w, h) or self.src is None or self.src.n == 0:
            return
        self._size = (w, h)
        st = self.style
        # A panel welded to a clip can end up small, so the margin and text
        # are ceilinged by the panel itself: the style's numbers are an
        # upper bound, not a demand. Without this the reserved text bands
        # can eat the whole panel and squash the plot to a dot.
        m = min(float(st["margin"]), min(w, h) * 0.08)
        n_bands = bool(st.get("title")) + bool(st.get("show_readout"))
        ts = min(float(st["text_size"]), h * 0.11, w * 0.09)
        if n_bands:
            ts = min(ts, 0.40 * h / (1.6 * n_bands))
        ts = max(ts, 6.0)
        self._m, self._ts = m, ts
        # keep the plot clear of the title and the readout rather than
        # letting the path run underneath them
        top = ts * 1.6 if st.get("title") else 0.0
        bot = ts * 1.6 if st.get("show_readout") else 0.0
        x_lo, x_hi = m, w - m
        y_lo, y_hi = m + top, h - m - bot
        aw, ah = max(x_hi - x_lo, 8.0), max(y_hi - y_lo, 8.0)
        (x0, y0), (x1, y1) = self.src.bounds()
        if self.ref is not None and self.ref.n and st.get("show_ref", True):
            (rx0, ry0), (rx1, ry1) = self.ref.bounds()
            x0, y0 = min(x0, rx0), min(y0, ry0)
            x1, y1 = max(x1, rx1), max(y1, ry1)
        dx, dy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        fit = st.get("fit", "stretch")
        if fit == "aspect":
            scx = scy = max(min(aw / dx, ah / dy), 1e-9)
        elif fit == "width":
            scx = scy = max(aw / dx, 1e-9)
        else:                                       # stretch
            scx, scy = max(aw / dx, 1e-9), max(ah / dy, 1e-9)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self._px = (x_lo + x_hi) / 2.0 + (self.src.xy[:, 0] - cx) * scx
        self._py = (y_lo + y_hi) / 2.0 - (self.src.xy[:, 1] - cy) * scy
        self._scale = min(scx, scy)
        self._sx, self._sy = scx, scy
        self._deci = max(1, int(np.ceil(self.src.n / self.MAX_POINTS)))
        poly = QtGui.QPolygonF()
        for x, y in zip(self._px[::self._deci], self._py[::self._deci]):
            poly.append(QtCore.QPointF(float(x), float(y)))
        self._full = poly

        self._ref_poly = None
        if self.ref is not None and self.ref.n and st.get("show_ref", True):
            rx = (x_lo + x_hi) / 2.0 + (self.ref.xy[:, 0] - cx) * scx
            ry = (y_lo + y_hi) / 2.0 - (self.ref.xy[:, 1] - cy) * scy
            step = max(1, int(np.ceil(self.ref.n / self.MAX_POINTS)))
            rpoly = QtGui.QPolygonF()
            for x, y in zip(rx[::step], ry[::step]):
                rpoly.append(QtCore.QPointF(float(x), float(y)))
            self._ref_poly = rpoly

    def draw(self, painter, w, h, idx, t_text=None):
        st = self.style
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(QtCore.QRectF(0, 0, w, h), qcolor(st["bg_color"]))
        painter.setPen(QtGui.QPen(qcolor(st["border_color"]), 1.5))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(QtCore.QRectF(0.75, 0.75, w - 1.5, h - 1.5))

        if self.src is None or self.src.n == 0:
            painter.restore()
            return
        self._prepare(w, h)
        st = self.style
        # a 'width' fit can run taller than the panel - keep it inside
        painter.setClipRect(QtCore.QRectF(0, 0, w, h))

        if self._ref_poly is not None:
            pen = QtGui.QPen(qcolor(st.get("ref_color") or st["trail_color"]))
            pen.setWidthF(max(0.3, float(st.get("ref_width", 2.0))))
            pen.setStyle(PEN_STYLES.get(st.get("ref_style", "dot"),
                                        QtCore.Qt.PenStyle.DotLine))
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPolyline(self._ref_poly)

        pen = QtGui.QPen(qcolor(st["path_color"]))
        pen.setWidthF(st["line_width"])
        pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawPolyline(self._full)

        idx = int(np.clip(idx if idx is not None else 0, 0, self.src.n - 1))
        if idx > 0:
            trail = QtGui.QPolygonF()
            step = self._deci
            for x, y in zip(self._px[:idx + 1:step], self._py[:idx + 1:step]):
                trail.append(QtCore.QPointF(float(x), float(y)))
            trail.append(QtCore.QPointF(float(self._px[idx]),
                                        float(self._py[idx])))
            pen = QtGui.QPen(qcolor(st["trail_color"]))
            pen.setWidthF(st["trail_width"])
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(trail)

        cxp, cyp = float(self._px[idx]), float(self._py[idx])
        state = self.src.state
        if (st.get("show_heading") and state is not None
                and state.shape[1] >= 3):
            th = float(state[idx, 2])
            L = st["point_size"] * 3.0
            # under a stretched fit the on-screen heading is skewed the same
            # way the path is, so the tick has to be skewed to match
            dxp, dyp = np.cos(th) * self._sx, np.sin(th) * self._sy
            n = max(float(np.hypot(dxp, dyp)), 1e-9)
            painter.setPen(QtGui.QPen(qcolor(st["point_color"]),
                                      max(1.5, st["point_size"] * 0.35)))
            painter.drawLine(QtCore.QPointF(cxp, cyp),
                             QtCore.QPointF(cxp + L * dxp / n,
                                            cyp - L * dyp / n))
        painter.setBrush(QtGui.QBrush(qcolor(st["point_color"])))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1.2))
        painter.drawEllipse(QtCore.QPointF(cxp, cyp),
                            st["point_size"], st["point_size"])

        ts = getattr(self, "_ts", st["text_size"])
        pad = getattr(self, "_m", st["margin"]) * 0.5
        painter.setPen(QtGui.QPen(qcolor(st["text_color"])))
        avail = w - 2 * pad
        if st.get("title"):
            timg = None
            if st.get("title_latex"):
                timg = render_latex(str(st["title"]), ts,
                                    qcolor(st["text_color"]), True,
                                    usetex=bool(st.get("usetex")),
                                    preamble=st.get("preamble",
                                                    DEFAULT_LATEX_PREAMBLE))
            if timg is not None:
                if timg.width() > avail:            # too wide for the panel
                    timg = timg.scaledToWidth(
                        int(avail),
                        QtCore.Qt.TransformationMode.SmoothTransformation)
                painter.drawImage(QtCore.QPointF(pad, pad), timg)
            else:
                self._fit_text(painter, str(st["title"]), avail, ts,
                               QtCore.QRectF(pad, pad, avail, ts * 1.4))
        if st.get("show_readout"):
            bits = []
            if t_text:
                bits.append(t_text)
            if state is not None and state.shape[1] >= 4:
                bits.append(f"vx={float(state[idx, 3]):.2f}")
            if bits:
                self._fit_text(
                    painter, "   ".join(bits), avail, ts,
                    QtCore.QRectF(pad, h - pad - ts * 1.4, avail, ts * 1.4))
        painter.restore()

    @staticmethod
    def _fit_text(painter, text, avail, size, rect):
        """Shrink (then elide) so a readout never runs off a narrow panel."""
        font = QtGui.QFont()
        font.setBold(True)
        px = max(6, int(round(size)))
        for _ in range(8):
            font.setPixelSize(px)
            if QtGui.QFontMetricsF(font).horizontalAdvance(text) <= avail:
                break
            if px <= 8:
                break
            px = max(8, int(px * 0.88))
        painter.setFont(font)
        fm = QtGui.QFontMetricsF(font)
        shown = fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight,
                              int(avail))
        painter.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignLeft |
                                   QtCore.Qt.AlignmentFlag.AlignVCenter),
                         shown)


# ===========================================================================
# Image plumbing
# ===========================================================================
def bgr_to_qimage(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w,
                        QtGui.QImage.Format.Format_RGB888).copy()


def qimage_to_bgr(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())
    arr = arr[:, :w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def resize_to(frame, w, h):
    ih, iw = frame.shape[:2]
    if (iw, ih) == (w, h):
        return frame
    interp = cv2.INTER_AREA if (w < iw or h < ih) else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


def fit_into(frame, W, H, bg_bgr):
    """Centre `frame` in a W x H canvas, scaled to fit, letterboxed on the
    backdrop. Used only by the combined reel, where jobs of different
    shapes have to share one frame size."""
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    sc = min(W / float(w), H / float(h))
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:, :] = bg_bgr
    paste(canvas, resize_to(frame, nw, nh), (W - nw) // 2, (H - nh) // 2)
    return canvas


def paste(canvas, img, x, y):
    h, w = img.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]


# ===========================================================================
# One clip: video + calibration + reference + optional npz panel
# ===========================================================================
class ClipSource:

    def __init__(self, spec, base_dir="."):
        self.spec = merged(DEFAULT_CLIP, spec)
        self.base_dir = base_dir
        self.video = None
        self.cal = Calibration()
        self.ref = None
        self.npz = None
        self.plot = None
        self.notes = []

        vp = _resolve(self.spec.get("video"), base_dir)
        if vp and os.path.exists(vp):
            self.video = VideoFrameSource(vp)
        elif vp:
            raise RuntimeError(f"video not found: {vp}")

        cp = _resolve(self.spec.get("calib"), base_dir)
        if cp and os.path.exists(cp):
            try:
                self.cal.load(cp)
            except Exception as exc:
                self.notes.append(f"calibration {os.path.basename(cp)}: {exc}")
        elif cp:
            self.notes.append(f"calibration not found: {cp}")

        rp = _resolve(self.spec.get("reference"), base_dir)
        if rp and os.path.exists(rp):
            try:
                self.ref = PathSource(rp)
            except Exception as exc:
                self.notes.append(f"reference {os.path.basename(rp)}: {exc}")
        elif rp:
            self.notes.append(f"reference not found: {rp}")

        np_ = _resolve(self.spec.get("npz"), base_dir)
        if np_ and os.path.exists(np_) and self.spec.get("aux", "none") != "none":
            try:
                self.npz = PathSource(np_)
                raw = self.spec.get("aux_style") or {}
                style = merged(DEFAULT_AUXSTYLE, raw)
                if not style.get("title"):
                    style["title"] = self.spec.get("label", "")
                # unless the style names one explicitly, the trail picks up
                # this clip's reference colour so the panel reads as its own
                if "trail_color" not in raw and self.cal.ref_color.isValid():
                    style["trail_color"] = hexargb(self.cal.ref_color)
                # unless the panel style names its own, the reference in
                # the plot matches the one drawn over the video - it is the
                # same planned line, so it should read as the same line
                rcol, _rw, rsty = self.ref_style_for()
                if "ref_color" not in raw:
                    style["ref_color"] = hexargb(rcol)
                if "ref_style" not in raw:
                    style["ref_style"] = rsty
                self.plot = NpzPlot(self.npz, style, ref=self.ref)
            except Exception as exc:
                self.notes.append(f"npz {os.path.basename(np_)}: {exc}")
        elif np_ and self.spec.get("aux", "none") != "none":
            self.notes.append(f"npz not found: {np_}")
        elif np_:
            self.notes.append(
                f"npz {os.path.basename(np_)} is loaded but its panel "
                "position is 'none', so no panel is drawn")
        if self.npz is not None and self.npz.n == 0:
            self.notes.append(f"npz {os.path.basename(np_)} has no points")

    def ref_style_for(self, s=1.0):
        """(colour, width, pen style) for this clip's reference line.

        A width set on the clip is taken as panel pixels and used as-is; a
        width inherited from the calibration is in source-video pixels and
        so gets scaled by `s`. Pass s=1.0 for anything already drawn in
        panel space, like the npz plot.
        """
        spec = self.spec
        cal_sty = "dot" if self.cal.ref_dotted else "solid"
        # older batches had no flag and used "set = override, blank/0 =
        # inherit", so honour that shape too
        custom = spec.get("ref_custom")
        if custom is None:
            custom = bool(spec.get("ref_color")
                          or float(spec.get("ref_width", 0.0) or 0.0) > 0)
        if not custom:
            return (QtGui.QColor(self.cal.ref_color),
                    max(0.3, self.cal.ref_width * s), cal_sty)

        col = (qcolor(spec["ref_color"]) if spec.get("ref_color")
               else QtGui.QColor(self.cal.ref_color))
        # a width set here is a panel-pixel number, used exactly as typed -
        # no rescaling, no falling back to the calibration
        wid = float(spec.get("ref_width", 0.0) or 0.0)
        if wid <= 0:
            wid = max(0.5, self.cal.ref_width * s)
        sty = spec.get("ref_style", "inherit")
        if sty not in PEN_STYLES:
            sty = cal_sty
        return col, max(0.3, wid), sty

    # -- timing ----------------------------------------------------------
    @property
    def ok(self):
        return self.video is not None

    @property
    def trim_start(self):
        return max(0.0, float(self.spec.get("trim_start", 0.0)))

    @property
    def trim_end(self):
        te = float(self.spec.get("trim_end", 0.0) or 0.0)
        full = self.video.duration if self.video else 0.0
        return full if te <= 0 else min(te, full)

    @property
    def play_length(self):
        return max(0.0, self.trim_end - self.trim_start)

    @property
    def time_shift(self):
        return float(self.spec.get("time_shift", 0.0))

    @property
    def end_time(self):
        return self.time_shift + self.play_length

    def local_time(self, t):
        """(source time, state) for composite time t.

        state is 'before' | 'live' | 'after' - the panel holds/blacks in the
        outer two, which is what lets a short clip sit frozen while the
        other one keeps running.
        """
        tau = t - self.time_shift
        if tau < 0:
            return self.trim_start, "before"
        if tau > self.play_length:
            return self.trim_end, "after"
        return self.trim_start + tau, "live"

    def release(self):
        if self.video is not None:
            self.video.release()


# ===========================================================================
# Layout planning
# ===========================================================================
class Layout:
    """Where every panel lands, computed once per job."""

    def __init__(self, job, clips):
        self.job = job
        self.clips = clips
        self.plan(int(job.get("panel_size", 720)))
        max_w = int(job.get("max_width", 0) or 0)
        max_h = int(job.get("max_height", 0) or 0)
        sc = 1.0
        if max_w and self.W > max_w:
            sc = min(sc, max_w / float(self.W))
        if max_h and self.H > max_h:
            sc = min(sc, max_h / float(self.H))
        if sc < 1.0:
            self.plan(max(64, int(job.get("panel_size", 720) * sc)))

    def _unit(self, clip, base):
        """(video w,h), (aux w,h) or None, unit w,h for one clip."""
        vw, vh = (clip.video.w, clip.video.h) if clip.ok else (16, 9)
        if self.job.get("layout", "horizontal") == "horizontal":
            h = max(16, int(base))
            w = max(16, int(round(h * vw / float(max(vh, 1)))))
        else:
            w = max(16, int(base))
            h = max(16, int(round(w * vh / float(max(vw, 1)))))
        pos = clip.spec.get("aux", "none")
        if clip.plot is None or pos == "none":
            return (w, h), None, (w, h), 0
        frac = float(np.clip(clip.spec.get("aux_frac", 0.6), 0.05, 3.0))
        gap = int(round(float(clip.spec.get("aux_gap", 8.0))))
        if pos in ("left", "right"):
            aw, ah = max(16, int(round(w * frac))), h
            return (w, h), (aw, ah), (w + gap + aw, h), gap
        ah, aw = max(16, int(round(h * frac))), w
        return (w, h), (aw, ah), (w, h + gap + ah), gap

    def plan(self, base):
        job = self.job
        gap = int(round(float(job.get("gap", 16.0))))
        horizontal = job.get("layout", "horizontal") == "horizontal"
        align = job.get("align", "center")
        units = [self._unit(c, base) for c in self.clips]

        # "fill": rather than centring a short unit in the taller one's slot
        # and painting backdrop either side, give each unit its own base so
        # they all span the cross axis exactly. Two clips of different aspect
        # ratios, or one carrying an npz panel and one not, then meet edge to
        # edge with no black margin. Nothing is cropped or distorted - the
        # units just end up different lengths along the main axis.
        if align == "fill" and len(units) > 1:
            cross = 1 if horizontal else 0
            target = max(u[2][cross] for u in units)
            bases = [base * target / max(u[2][cross], 1) for u in units]
            units = [self._unit(c, b) for c, b in zip(self.clips, bases)]

        sizes = [u[2] for u in units]
        if horizontal:
            W = sum(s[0] for s in sizes) + gap * (len(sizes) - 1)
            H = max(s[1] for s in sizes)
        else:
            W = max(s[0] for s in sizes)
            H = sum(s[1] for s in sizes) + gap * (len(sizes) - 1)
        W += W % 2
        H += H % 2

        def off(free):
            if align in ("start", "fill"):
                return 0
            return free if align == "end" else free // 2

        self.W, self.H = W, H
        self.video_rects = []
        self.aux_rects = []
        cur = 0
        for (vsz, asz, usz, agap), clip in zip(units, self.clips):
            uw, uh = usz
            if horizontal:
                ux, uy = cur, off(H - uh)
                cur += uw + gap
            else:
                ux, uy = off(W - uw), cur
                cur += uh + gap
            pos = clip.spec.get("aux", "none") if clip.plot is not None else "none"
            vw, vh = vsz
            if asz is None:
                self.video_rects.append((ux, uy, vw, vh))
                self.aux_rects.append(None)
                continue
            aw, ah = asz
            if pos == "right":
                vr = (ux, uy, vw, vh)
                ar = (ux + vw + agap, uy, aw, ah)
            elif pos == "left":
                ar = (ux, uy, aw, ah)
                vr = (ux + aw + agap, uy, vw, vh)
            elif pos == "above":
                ar = (ux, uy, aw, ah)
                vr = (ux, uy + ah + agap, vw, vh)
            else:                                   # below
                vr = (ux, uy, vw, vh)
                ar = (ux, uy + vh + agap, aw, ah)
            self.video_rects.append(vr)
            self.aux_rects.append(ar)

        if align == "fill":
            self._absorb_slack(horizontal)
            # W/H are rounded up to even for H.264; that spare pixel belongs
            # to the last unit, otherwise it shows as a hairline at the far
            # edge of the frame
            self._absorb_slack(not horizontal, only=len(self.clips) - 1)

    def _absorb_slack(self, horizontal, only=None):
        """Integer rounding leaves each unit a pixel or two short of the
        cross axis. Extend whichever of its panels sits at the far edge, so
        'fill' really reaches the edge instead of leaving a hairline of
        backdrop."""
        target = self.H if horizontal else self.W
        for i, (vr, ar) in enumerate(zip(self.video_rects, self.aux_rects)):
            if only is not None and i != only:
                continue
            # (start, size) along the cross axis, video first
            def cross(r):
                return (r[1], r[3]) if horizontal else (r[0], r[2])
            rects = [("v", vr)] + ([("a", ar)] if ar is not None else [])
            # Every panel sitting at the unit's far edge gets the slack, not
            # just one: when the aux is stacked along the MAIN axis (below a
            # video in a vertical layout, say) both panels span the cross
            # axis and both would otherwise fall a pixel short.
            far = max(cross(r)[0] for _k, r in rects)
            for key, rect in rects:
                cs, cz = cross(rect)
                if cs != far:
                    continue
                slack = target - (cs + cz)
                if slack <= 0 or slack > 4:
                    continue
                x, y, w, h = rect
                grown = ((x, y, w, h + slack) if horizontal
                         else (x, y, w + slack, h))
                if key == "v":
                    self.video_rects[i] = grown
                else:
                    self.aux_rects[i] = grown


# ===========================================================================
# Job renderer
# ===========================================================================
class JobRenderer:

    def __init__(self, job, base_dir="."):
        self.job = merged(DEFAULT_JOB, job)
        self.base_dir = base_dir
        self.clips = []
        self.notes = []
        for key in ("left", "right"):
            spec = self.job.get(key) or {}
            if not spec.get("video"):
                continue
            clip = ClipSource(spec, base_dir)
            self.notes.extend(f"{key}: {n}" for n in clip.notes)
            self.clips.append(clip)
        if not self.clips:
            raise RuntimeError("no clips - set at least one video")
        lg = merged(DEFAULT_JOB["legend"], self.job.get("legend"))
        if lg.get("scope", "both") != "none" and not lg.get("entries"):
            if not any(c.cal.legend_entries for c in self.clips):
                self.notes.append(
                    "legend is switched on but there are no rows to draw - "
                    "add some in the Legend section, or open calibrations "
                    "that carry one")
        self.layout = Layout(self.job, self.clips)
        self.bg = qcolor(self.job.get("bg", "#ff0a0a0e"))
        self._bg_bgr = np.array([self.bg.blue(), self.bg.green(),
                                 self.bg.red()], dtype=np.uint8)

    # -- timing ----------------------------------------------------------
    @property
    def fps(self):
        return max(1.0, float(self.job.get("fps", 30.0) or 30.0))

    @property
    def duration(self):
        d = float(self.job.get("duration", 0.0) or 0.0)
        if d > 0:
            return d
        return max([c.end_time for c in self.clips] + [0.0])

    @property
    def n_frames(self):
        return max(1, int(round(self.duration * self.fps)))

    @property
    def size(self):
        return self.layout.W, self.layout.H

    def _latex_settings(self):
        """(usetex, preamble) for this job - one engine for every string."""
        tex = merged(DEFAULT_JOB["latex"], self.job.get("latex"))
        return (bool(tex.get("usetex")) and usetex_available(),
                tex.get("preamble") or DEFAULT_LATEX_PREAMBLE)

    # -- one panel -------------------------------------------------------
    def _video_panel(self, clip, t, w, h, side=0):
        src_t, state = clip.local_time(t)
        end_behavior = self.job.get("end_behavior", "hold")
        if state == "after" and end_behavior == "black":
            return np.zeros((h, w, 3), np.uint8)

        idx = int(round(src_t * clip.video.fps))
        idx = int(np.clip(idx, 0, clip.video.n_frames - 1))
        frame = clip.video.frame_at(idx)
        if frame is None:
            return np.zeros((h, w, 3), np.uint8)
        panel = resize_to(frame, w, h)
        if panel.base is not None or not panel.flags["C_CONTIGUOUS"]:
            panel = np.ascontiguousarray(panel)

        s = w / float(max(clip.video.w, 1))
        img = bgr_to_qimage(panel)
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        cal = clip.cal
        job_tex, preamble = self._latex_settings()
        # One LaTeX engine for the whole job. A calibration that was tuned
        # with usetex keeps it for its own legend even if the job leaves it
        # off, so old .camalign.json files still render as they did.
        if clip.spec.get("overlay", True) and clip.ref is not None:
            cam = scaled_cam(cal.cam_at(idx), s)
            col, wid, sty = clip.ref_style_for(s)
            draw_reference(painter, w, h, cam, clip.ref, col, wid,
                           cal.ref_segments, pen_style=sty)

        # Legend geometry came out of camera_align.py in SOURCE video
        # pixels, so it's rescaled by s to keep its apparent size. Give the
        # job a legend size and it's pinned in panel pixels instead, which
        # is what you want when the two sources aren't the same resolution.
        lg = merged(DEFAULT_JOB["legend"], self.job.get("legend"))
        scope = lg.get("scope", "both")
        allowed = (scope == "both"
                   or (scope == "left" and side == 0)
                   or (scope == "right" and side == 1))
        if clip.spec.get("legend", True) and allowed:
            entries = legend_entries_from(lg.get("entries"))
            if entries:
                # rows defined here: sizes are panel pixels, like the label
                # and timestamp, so both clips match whatever their sources
                # were shot at
                geom = job_legend_geometry(lg, cal)
                draw_legend(painter, w, h, geom, entries,
                            qcolor(lg.get("text_color", "#ffffffff")),
                            qcolor(lg.get("bg_color", "#ff000000")), 1.0,
                            usetex=job_tex, preamble=preamble)
            elif cal.legend_entries:
                # nothing defined here, so show the calibration's own legend
                # (drawn whether or not camera_align.py had it toggled on -
                # asking for it in this tool is the switch that matters)
                forced = float(lg.get("size", 0.0) or 0.0)
                ls = (s if forced <= 0
                      else forced / max(float(cal.legend["text_size"]), 1e-6))
                draw_legend(painter, w, h, cal.legend, cal.legend_entries,
                            cal.legend_text_color, cal.legend_bg_color, ls,
                            usetex=job_tex or cal.usetex,
                            preamble=preamble if job_tex else cal.preamble)

        ts = merged(DEFAULT_TIMESTAMP, self.job.get("timestamp"))
        if clip.spec.get("timestamp", True) and ts.get("enabled", True):
            shown = src_t
            if ts.get("source") == "trajectory":
                shown = src_t - cal.offset_s
            if not ts.get("freeze_at_end", True) and state == "after":
                shown = src_t + (t - clip.end_time)
            dec = int(np.clip(int(ts.get("decimals", 1)), 0, 6))
            text = (f"{ts.get('prefix', '')}{shown:.{dec}f}"
                    f"{ts.get('suffix', '')}")
            _draw_text_box(painter, w, h, text, ts, 1.0,
                           ts.get("corner", "bottom-left"),
                           usetex=job_tex, preamble=preamble)

        lab = merged(DEFAULT_LABEL, self.job.get("label"))
        if lab.get("enabled", True) and clip.spec.get("label"):
            _draw_text_box(painter, w, h, str(clip.spec["label"]), lab, 1.0,
                           lab.get("position", "top-center"),
                           usetex=job_tex, preamble=preamble)
        painter.end()

        out = qimage_to_bgr(img)
        if state == "after" and end_behavior == "hold-dim":
            out = (out.astype(np.float32) * 0.45).astype(np.uint8)
        return out

    def _aux_panel(self, clip, t, w, h):
        img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGB888)
        img.fill(qcolor(clip.plot.style["bg_color"]))
        painter = QtGui.QPainter(img)
        # the panel title is auto-copied from the clip label, so if the
        # label is LaTeX the title has to be typeset too - otherwise it
        # shows the raw source next to a nicely set one
        lab = merged(DEFAULT_LABEL, self.job.get("label"))
        job_tex, preamble = self._latex_settings()
        clip.plot.style["title_latex"] = bool(lab.get("latex"))
        clip.plot.style["usetex"] = job_tex
        clip.plot.style["preamble"] = preamble
        src_t, _state = clip.local_time(t)
        npz_t = src_t - float(clip.spec.get("npz_offset", 0.0))
        idx = clip.npz.sample_at(npz_t) if clip.npz is not None else 0
        ts = merged(DEFAULT_TIMESTAMP, self.job.get("timestamp"))
        dec = int(np.clip(int(ts.get("decimals", 1)), 0, 6))
        clip.plot.draw(painter, w, h, idx, t_text=f"t={npz_t:.{dec}f}s")
        painter.end()
        return qimage_to_bgr(img)

    # -- one composite frame ---------------------------------------------
    def frame_at_time(self, t):
        W, H = self.size
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:, :] = self._bg_bgr
        for clip, vr, ar in zip(self.clips, self.layout.video_rects,
                                self.layout.aux_rects):
            x, y, w, h = vr
            side = self.clips.index(clip)
            paste(canvas, self._video_panel(clip, t, w, h, side), x, y)
            if ar is not None and clip.plot is not None:
                ax, ay, aw, ah = ar
                paste(canvas, self._aux_panel(clip, t, aw, ah), ax, ay)
        return canvas

    def frame_at_index(self, i):
        return self.frame_at_time(i / self.fps)

    def describe(self):
        W, H = self.size
        bits = [f"{W}x{H} @ {self.fps:g}fps", f"{self.duration:.2f}s",
                f"{self.n_frames} frames"]
        for name, clip in zip(("left", "right"), self.clips):
            bits.append(f"{name}: {os.path.basename(clip.video.path)} "
                        f"{clip.video.w}x{clip.video.h} "
                        f"{clip.play_length:.2f}s"
                        f"{' +ref' if clip.ref is not None else ''}"
                        f"{' +npz' if clip.plot is not None else ''}")
        return "   ·   ".join(bits)

    def close(self):
        for c in self.clips:
            c.release()
        self.clips = []


# ===========================================================================
# Export
# ===========================================================================
class Cancelled(Exception):
    pass


def _open_writer(path, W, H, fps, enc):
    """(writer_object, kind) where kind is 'ffmpeg' or 'cv2'."""
    if enc.get("use_ffmpeg", True) and ffmpeg_available():
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{W}x{H}", "-framerate", f"{fps}",
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", str(enc.get("preset", "veryfast")),
            "-crf", str(int(enc.get("crf", 20))),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        return proc, "ffmpeg"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"could not open a writer for {path}")
    return vw, "cv2"


def _shrink_to_budget(path, duration, target_mb, preset, log=print):
    """Two-pass re-encode until the file fits, without re-rendering."""
    if not target_mb or target_mb <= 0 or not ffmpeg_available():
        return
    # decimal MB, matching how the sizes are reported and how upload limits
    # are usually quoted - MiB here would quietly allow 20.97 MB past a
    # "20 MB" cap
    budget = target_mb * 1e6
    size = os.path.getsize(path)
    if size <= budget:
        log(f"  size {size / 1e6:.3f} MB - within the {target_mb:g} MB budget")
        return
    log(f"  size {size / 1e6:.2f} MB is over the {target_mb:g} MB budget "
        f"- re-encoding to fit")
    headroom = 0.95
    for attempt in range(3):
        kbps = max(80, int(budget * 8 * headroom / max(duration, 1e-6) / 1000))
        tmpdir = tempfile.mkdtemp(prefix="cmp_pass_")
        out = os.path.join(tmpdir, "out.mp4")
        plog = os.path.join(tmpdir, "pass")
        null = "NUL" if os.name == "nt" else "/dev/null"
        try:
            base = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-an",
                    "-c:v", "libx264", "-preset", preset, "-b:v", f"{kbps}k",
                    "-passlogfile", plog]
            subprocess.run(base + ["-pass", "1", "-f", "mp4", null], check=True)
            subprocess.run(base + ["-pass", "2", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", out], check=True)
            new = os.path.getsize(out)
            log(f"  attempt {attempt + 1}: {kbps} kbps -> {new / 1e6:.3f} MB "
                f"(budget {target_mb:g} MB)")
            if new <= budget or attempt == 2:
                shutil.move(out, path)
                return
            headroom *= budget / float(new) * 0.97
        except subprocess.CalledProcessError as exc:
            log(f"  re-encode failed ({exc}) - keeping the CRF version")
            return
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def export_job(job, base_dir=".", log=print, progress=None, cancel=None,
               job_label=""):
    """Render one job to disk. Returns the output path."""
    r = JobRenderer(job, base_dir)
    for note in r.notes:
        log(f"  ! {note}")
    W, H = r.size
    fps = r.fps
    n = r.n_frames
    out = _resolve(r.job.get("output") or default_output_name(r.job), base_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    enc = merged(DEFAULT_JOB["encode"], r.job.get("encode"))
    log(f"{job_label}{r.describe()}")
    log(f"  -> {out}")
    if not (enc.get("use_ffmpeg", True) and ffmpeg_available()):
        log("  ! ffmpeg not in use - falling back to OpenCV mp4v "
            "(no CRF, no size budget)")

    writer, kind = _open_writer(out, W, H, fps, enc)
    cancelled = False
    t0 = time.time()
    step = max(1, n // 200)
    try:
        for i in range(n):
            if cancel is not None and cancel():
                raise Cancelled()
            frame = r.frame_at_index(i)
            if kind == "ffmpeg":
                writer.stdin.write(frame.tobytes())
            else:
                writer.write(frame)
            done = i + 1
            if done % step == 0 or done == n:
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                log(f"  [{100.0 * done / n:5.1f}%]  frame {done}/{n}   "
                    f"{rate:5.1f} fps   elapsed {_fmt_clock(el)}   "
                    f"eta {_fmt_clock(eta)}")
                if progress is not None:
                    progress(done, n)
    except Cancelled:
        cancelled = True
        log("  cancelled")
        raise
    finally:
        if kind == "ffmpeg":
            try:
                writer.stdin.close()
            except Exception:
                pass
            err = writer.stderr.read().decode("utf-8", "replace").strip()
            writer.wait()
            if writer.returncode not in (0, None) and err:
                log(f"  ffmpeg: {err.splitlines()[-1]}")
        else:
            writer.release()
        r.close()
        if cancelled and os.path.exists(out):
            # a half-written mp4 is worse than none - it looks like a result
            try:
                os.remove(out)
            except OSError:
                pass

    _shrink_to_budget(out, n / fps, float(enc.get("target_mb", 0.0) or 0.0),
                      str(enc.get("preset", "veryfast")), log=log)
    log(f"  done in {_fmt_clock(time.time() - t0)}  "
        f"({os.path.getsize(out) / 1e6:.2f} MB)")
    return out


def export_combined(jobs, base_dir=".", combined=None, log=print,
                    progress=None, cancel=None, batch_name="batch"):
    """Render every job into ONE file, one after another.

    Jobs are measured first, then rendered one at a time - only one job's
    videos are open at once, so a long batch doesn't hold dozens of decoders
    open. Each job keeps its own duration; its frames are resampled onto the
    reel's clock and letterboxed into the reel's canvas, so mixed sizes and
    frame rates concatenate cleanly.
    """
    cfg = merged(DEFAULT_COMBINED, combined)
    enc = merged(DEFAULT_COMBINED["encode"], cfg.get("encode"))

    # -- pass 1: measure, so the canvas can hold the largest job ---------
    plans = []
    for i, job in enumerate(jobs):
        try:
            r = JobRenderer(job, base_dir)
        except Exception as exc:
            log(f"  ! job {i + 1} skipped: {exc}")
            continue
        plans.append({"job": job, "index": i, "size": r.size, "fps": r.fps,
                      "duration": r.duration,
                      "name": job.get("name") or f"job {i + 1}"})
        for note in r.notes:
            log(f"  ! job {i + 1}: {note}")
        r.close()
    if not plans:
        raise RuntimeError("nothing to render - every job failed to open")

    W = int(cfg.get("width", 0) or 0) or max(p["size"][0] for p in plans)
    H = int(cfg.get("height", 0) or 0) or max(p["size"][1] for p in plans)
    W += W % 2
    H += H % 2
    fps = float(cfg.get("fps", 0.0) or 0.0) or max(p["fps"] for p in plans)
    gap_frames = max(0, int(round(float(cfg.get("gap_s", 0.0)) * fps)))
    bg = qcolor(cfg.get("bg", "#ff000000"))
    bg_bgr = np.array([bg.blue(), bg.green(), bg.red()], np.uint8)

    counts = [max(1, int(round(p["duration"] * fps))) for p in plans]
    total = sum(counts) + gap_frames * max(0, len(plans) - 1)

    out = _resolve(cfg.get("output") or os.path.join(
        "comparisons", f"{batch_name}_all.mp4"), base_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    log(f"combined reel: {len(plans)} job(s)   {W}x{H} @ {fps:g}fps   "
        f"{total / fps:.2f}s   {total} frames")
    for p, n in zip(plans, counts):
        log(f"    {p['name']}: {p['size'][0]}x{p['size'][1]} @ "
            f"{p['fps']:g}fps -> {n} frames")
    log(f"  -> {out}")

    writer, kind = _open_writer(out, W, H, fps, enc)
    blank = np.empty((H, W, 3), np.uint8)
    blank[:, :] = bg_bgr
    cancelled = False
    done = 0
    t0 = time.time()
    step = max(1, total // 200)

    def emit(frame):
        nonlocal done
        if kind == "ffmpeg":
            writer.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
        done += 1
        if done % step == 0 or done == total:
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            log(f"  [{100.0 * done / total:5.1f}%]  frame {done}/{total}   "
                f"{rate:5.1f} fps   elapsed {_fmt_clock(el)}   "
                f"eta {_fmt_clock((total - done) / max(rate, 1e-6))}")
            if progress is not None:
                progress(done, total)

    try:
        for k, (plan, n) in enumerate(zip(plans, counts)):
            log(f"  [{k + 1}/{len(plans)}] {plan['name']}")
            r = JobRenderer(plan["job"], base_dir)
            try:
                for i in range(n):
                    if cancel is not None and cancel():
                        raise Cancelled()
                    emit(fit_into(r.frame_at_time(i / fps), W, H, bg_bgr))
            finally:
                r.close()
            if gap_frames and k < len(plans) - 1:
                for _ in range(gap_frames):
                    if cancel is not None and cancel():
                        raise Cancelled()
                    emit(blank)
    except Cancelled:
        cancelled = True
        log("  cancelled")
        raise
    finally:
        if kind == "ffmpeg":
            try:
                writer.stdin.close()
            except Exception:
                pass
            err = writer.stderr.read().decode("utf-8", "replace").strip()
            writer.wait()
            if writer.returncode not in (0, None) and err:
                log(f"  ffmpeg: {err.splitlines()[-1]}")
        else:
            writer.release()
        if cancelled and os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass

    _shrink_to_budget(out, total / fps,
                      float(enc.get("target_mb", 0.0) or 0.0),
                      str(enc.get("preset", "veryfast")), log=log)
    log(f"  done in {_fmt_clock(time.time() - t0)}  "
        f"({os.path.getsize(out) / 1e6:.2f} MB)")
    return out


def default_output_name(job):
    def stem(side):
        v = (job.get(side) or {}).get("video")
        return os.path.splitext(os.path.basename(v))[0] if v else side
    name = job.get("name") or f"{stem('left')}__vs__{stem('right')}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "comparison"
    return os.path.join("comparisons", name + ".mp4")


def run_batch(jobs, base_dir=".", log=print, progress=None, cancel=None,
              only=None):
    outs = []
    todo = list(enumerate(jobs))
    if only is not None:
        todo = [(i, j) for i, j in todo if i == only]
    for k, (i, job) in enumerate(todo):
        label = f"[job {k + 1}/{len(todo)}]  "
        try:
            outs.append(export_job(job, base_dir, log=log, progress=progress,
                                   cancel=cancel, job_label=label))
        except Cancelled:
            raise
        except Exception as exc:
            log(f"{label}FAILED: {exc}")
    return outs


# ===========================================================================
# Batch file io
# ===========================================================================
def save_batch(path, jobs, src_base=".", combined=None):
    """Write a batch, rewriting every path to be relative to the new file.

    `src_base` is the folder the jobs' current relative paths resolve
    against - normally the batch they were loaded from. Without it, saving a
    loaded batch into a different folder silently re-anchors "media/x.mp4"
    to the process's working directory and the paths go stale. Returns the
    rewritten jobs so the caller can keep working against the new base.
    """
    dst = os.path.dirname(os.path.abspath(path))
    out = []
    for job in jobs:
        j = copy.deepcopy(job)
        for side in ("left", "right"):
            clip = j.get(side) or {}
            for key in ("video", "calib", "reference", "npz"):
                if clip.get(key):
                    clip[key] = _rel(_resolve(clip[key], src_base), dst)
            j[side] = clip
        if j.get("output"):
            j["output"] = _rel(_resolve(j["output"], src_base), dst)
        out.append(j)
    doc = {"schema": SCHEMA,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "jobs": out}
    if combined:
        c = copy.deepcopy(combined)
        if c.get("output"):
            c["output"] = _rel(_resolve(c["output"], src_base), dst)
        doc["combined"] = c
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return out


def _migrate_job(job):
    """Bring an older job dict up to the current shape.

    Reference styling used to be "set a colour or a width and that means
    override". Filling defaults in first would bury that, so the flag is
    derived here, before the merge.
    """
    for side in ("left", "right"):
        clip = job.get(side)
        if not isinstance(clip, dict) or "ref_custom" in clip:
            continue
        if (clip.get("ref_color")
                or float(clip.get("ref_width", 0.0) or 0.0) > 0
                or clip.get("ref_style") in PEN_STYLES):
            clip["ref_custom"] = True
    return job


def load_batch(path, with_combined=False):
    with open(path) as f:
        spec = json.load(f)
    if spec.get("schema") != SCHEMA:
        log(f"note: schema is {spec.get('schema')!r}, expected {SCHEMA!r}")
    jobs = [merged(DEFAULT_JOB, _migrate_job(j)) for j in spec.get("jobs", [])]
    if not with_combined:
        return jobs
    return jobs, merged(DEFAULT_COMBINED, spec.get("combined"))


# ===========================================================================
# GUI
# ===========================================================================
class FileRow(QtWidgets.QWidget):
    """Label + path box + browse + clear, emitting `changed`."""

    changed = QtCore.pyqtSignal()

    def __init__(self, label, filt, tooltip="", parent=None):
        super().__init__(parent)
        self.filt = filt
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(66)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("(none)")
        self.edit.editingFinished.connect(self.changed.emit)
        btn = QtWidgets.QToolButton()
        btn.setText("…")
        btn.clicked.connect(self._browse)
        clr = QtWidgets.QToolButton()
        clr.setText("✕")
        clr.clicked.connect(self._clear)
        if tooltip:
            for w in (lbl, self.edit):
                w.setToolTip(tooltip)
        lay.addWidget(lbl)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)
        lay.addWidget(clr)

    def _browse(self):
        start = os.path.dirname(self.edit.text()) or ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose file",
                                                     start, self.filt)
        if p:
            self.edit.setText(p)
            self.changed.emit()

    def _clear(self):
        self.edit.clear()
        self.changed.emit()

    def path(self):
        return self.edit.text().strip() or None

    def set_path(self, p):
        self.edit.setText(p or "")


class ColorButton(QtWidgets.QPushButton):
    """A small swatch that opens a colour picker, alpha included."""

    picked = QtCore.pyqtSignal()

    def __init__(self, color="#ffffffff", title="Pick colour", parent=None):
        super().__init__(parent)
        self.title = title
        self.color = qcolor(color)
        self.setFixedSize(38, 22)
        self.clicked.connect(self._pick)
        self._refresh()

    def _pick(self):
        c = QtWidgets.QColorDialog.getColor(
            self.color, self, self.title,
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self.color = c
            self._refresh()
            self.picked.emit()

    def _refresh(self):
        self.setStyleSheet(
            f"background:{_css_rgba(self.color)}; border:1px solid #333;")

    def set_color(self, color):
        self.color = qcolor(color)
        self._refresh()

    def hex(self):
        return hexargb(self.color)


class LegendEntryDialog(QtWidgets.QDialog):
    """One legend row: a line swatch or a marker, plus its label."""

    def __init__(self, parent, entry=None):
        super().__init__(parent)
        self.setWindowTitle("Legend entry")
        e = entry or {}
        lay = QtWidgets.QFormLayout(self)

        self.edit_label = QtWidgets.QLineEdit(str(e.get("label", "")))
        lay.addRow("Label", self.edit_label)

        self.chk_latex = QtWidgets.QCheckBox("Label is LaTeX")
        self.chk_latex.setChecked(bool(e.get("latex", False)))
        self.chk_latex.setEnabled(latex_available())
        self.chk_latex.setToolTip(
            'e.g. "$v_{max}$" or, with usetex on, "\\textbf{Nominal}"')
        lay.addRow("", self.chk_latex)

        self.combo_kind = QtWidgets.QComboBox()
        self.combo_kind.addItems(["line", "star", "circle"])
        self.combo_kind.setCurrentText(e.get("kind", "line"))
        self.combo_kind.currentTextChanged.connect(
            lambda k: self.combo_style.setEnabled(k == "line"))
        lay.addRow("Kind", self.combo_kind)

        self.combo_style = QtWidgets.QComboBox()
        self.combo_style.addItems(list(PEN_STYLES.keys()))
        self.combo_style.setCurrentText(e.get("style", "solid"))
        lay.addRow("Line style", self.combo_style)

        self.btn_color = ColorButton(e.get("color", "#ffffffff"))
        lay.addRow("Colour / opacity", self.btn_color)

        self.spin_width = QtWidgets.QDoubleSpinBox()
        self.spin_width.setRange(0.5, 200.0)
        self.spin_width.setDecimals(1)
        self.spin_width.setValue(float(e.get("width", 6.0)))
        self.spin_width.setSuffix(" px")
        lay.addRow("Stroke width / marker radius", self.spin_width)

        note = QtWidgets.QLabel(
            "Sizes are in output panel pixels, so what you set is what the "
            "rendered frame gets.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:11px;")
        lay.addRow("", note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)
        self.combo_style.setEnabled(self.combo_kind.currentText() == "line")

    def values(self):
        return {"label": self.edit_label.text(),
                "kind": self.combo_kind.currentText(),
                "style": self.combo_style.currentText(),
                "color": self.btn_color.hex(),
                "width": float(self.spin_width.value()),
                "latex": bool(self.chk_latex.isChecked())}


class AuxStyleDialog(QtWidgets.QDialog):
    """Every colour and stroke in one npz panel, individually pickable."""

    def __init__(self, parent, style=None):
        super().__init__(parent)
        self.setWindowTitle("NPZ panel style")
        st = merged(DEFAULT_AUXSTYLE, style)
        lay = QtWidgets.QFormLayout(self)

        def stroke_row(color_key, width_key, lo=0.0, hi=200.0):
            btn = ColorButton(st.get(color_key) or "#ffffffff")
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setValue(float(st.get(width_key, 2.0)))
            spin.setSuffix(" px")
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(btn)
            row.addWidget(spin)
            row.addStretch(1)
            host = QtWidgets.QWidget()
            host.setLayout(row)
            return btn, spin, host

        # driven trajectory: the trail that grows as the clip plays
        (self.btn_trail, self.spin_trail,
         row_trail) = stroke_row("trail_color", "trail_width")
        lay.addRow("Trajectory (driven)", row_trail)

        # the same trajectory ahead of the playhead, i.e. the whole log
        (self.btn_path, self.spin_path,
         row_path) = stroke_row("path_color", "line_width")
        lay.addRow("Trajectory (full log)", row_path)

        # the planned line
        (self.btn_ref, self.spin_ref,
         row_ref) = stroke_row("ref_color", "ref_width")
        lay.addRow("Reference", row_ref)

        self.combo_ref_style = QtWidgets.QComboBox()
        self.combo_ref_style.addItems(list(PEN_STYLES.keys()))
        self.combo_ref_style.setCurrentText(st.get("ref_style", "dot"))
        lay.addRow("Reference style", self.combo_ref_style)

        self.chk_ref = QtWidgets.QCheckBox("Draw the reference in the panel")
        self.chk_ref.setChecked(bool(st.get("show_ref", True)))
        lay.addRow("", self.chk_ref)

        (self.btn_point, self.spin_point,
         row_point) = stroke_row("point_color", "point_size", 1.0, 100.0)
        lay.addRow("Current position", row_point)

        self.chk_heading = QtWidgets.QCheckBox("Heading tick")
        self.chk_heading.setChecked(bool(st.get("show_heading", True)))
        lay.addRow("", self.chk_heading)

        self.btn_bg = ColorButton(st.get("bg_color"))
        self.btn_border = ColorButton(st.get("border_color"))
        self.btn_text = ColorButton(st.get("text_color"))
        row_misc = QtWidgets.QHBoxLayout()
        row_misc.setContentsMargins(0, 0, 0, 0)
        for b in (self.btn_bg, self.btn_border, self.btn_text):
            row_misc.addWidget(b)
        row_misc.addStretch(1)
        host_misc = QtWidgets.QWidget()
        host_misc.setLayout(row_misc)
        lay.addRow("Backdrop / border / text", host_misc)

        self.spin_text = QtWidgets.QDoubleSpinBox()
        self.spin_text.setRange(4.0, 200.0)
        self.spin_text.setValue(float(st.get("text_size", 26.0)))
        self.spin_text.setSuffix(" px")
        self.spin_text.setToolTip(
            "Upper bound - a small panel scales its text down to fit")
        lay.addRow("Text size", self.spin_text)

        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setRange(0.0, 200.0)
        self.spin_margin.setValue(float(st.get("margin", 22.0)))
        self.spin_margin.setSuffix(" px")
        lay.addRow("Inner margin", self.spin_margin)

        self.edit_title = QtWidgets.QLineEdit(str(st.get("title", "")))
        self.edit_title.setPlaceholderText("(defaults to the clip label)")
        lay.addRow("Title", self.edit_title)

        self.chk_readout = QtWidgets.QCheckBox("Time / speed readout")
        self.chk_readout.setChecked(bool(st.get("show_readout", True)))
        lay.addRow("", self.chk_readout)

        self.combo_fit = QtWidgets.QComboBox()
        self.combo_fit.addItems(AUX_FITS)
        self.combo_fit.setCurrentText(st.get("fit", "stretch"))
        lay.addRow("Fit", self.combo_fit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)
        self._base = st

    def values(self):
        st = dict(self._base)
        st.update({
            "trail_color": self.btn_trail.hex(),
            "trail_width": self.spin_trail.value(),
            "path_color": self.btn_path.hex(),
            "line_width": self.spin_path.value(),
            "ref_color": self.btn_ref.hex(),
            "ref_width": self.spin_ref.value(),
            "ref_style": self.combo_ref_style.currentText(),
            "show_ref": self.chk_ref.isChecked(),
            "point_color": self.btn_point.hex(),
            "point_size": self.spin_point.value(),
            "show_heading": self.chk_heading.isChecked(),
            "bg_color": self.btn_bg.hex(),
            "border_color": self.btn_border.hex(),
            "text_color": self.btn_text.hex(),
            "text_size": self.spin_text.value(),
            "margin": self.spin_margin.value(),
            "title": self.edit_title.text(),
            "show_readout": self.chk_readout.isChecked(),
            "fit": self.combo_fit.currentText(),
        })
        return st


class CombinedExportDialog(QtWidgets.QDialog):
    """Settings for rendering every job into one file."""

    def __init__(self, parent, cfg=None, plans=None):
        super().__init__(parent)
        self.setWindowTitle("Export all jobs as one video")
        c = merged(DEFAULT_COMBINED, cfg)
        enc = merged(DEFAULT_COMBINED["encode"], c.get("encode"))
        lay = QtWidgets.QFormLayout(self)

        self.edit_out = QtWidgets.QLineEdit(str(c.get("output", "")))
        self.edit_out.setPlaceholderText("(auto: comparisons/<batch>_all.mp4)")
        btn_out = QtWidgets.QToolButton()
        btn_out.setText("…")
        btn_out.clicked.connect(self._browse)
        row_out = QtWidgets.QHBoxLayout()
        row_out.setContentsMargins(0, 0, 0, 0)
        row_out.addWidget(self.edit_out, 1)
        row_out.addWidget(btn_out)
        host_out = QtWidgets.QWidget()
        host_out.setLayout(row_out)
        lay.addRow("File", host_out)

        self.spin_fps = QtWidgets.QDoubleSpinBox()
        self.spin_fps.setRange(0, 240)
        self.spin_fps.setSpecialValueText("fastest job")
        self.spin_fps.setValue(float(c.get("fps", 0.0)))
        self.spin_fps.setToolTip(
            "One clock for the whole reel; each job's frames are resampled "
            "onto it, so durations are preserved either way")
        lay.addRow("FPS", self.spin_fps)

        self.spin_w = QtWidgets.QSpinBox()
        self.spin_w.setRange(0, 8192)
        self.spin_w.setSpecialValueText("largest job")
        self.spin_w.setValue(int(c.get("width", 0)))
        self.spin_h = QtWidgets.QSpinBox()
        self.spin_h.setRange(0, 8192)
        self.spin_h.setSpecialValueText("largest job")
        self.spin_h.setValue(int(c.get("height", 0)))
        row_sz = QtWidgets.QHBoxLayout()
        row_sz.setContentsMargins(0, 0, 0, 0)
        row_sz.addWidget(self.spin_w)
        row_sz.addWidget(self.spin_h)
        host_sz = QtWidgets.QWidget()
        host_sz.setLayout(row_sz)
        lay.addRow("Canvas w / h", host_sz)

        self.spin_gap = QtWidgets.QDoubleSpinBox()
        self.spin_gap.setRange(0, 60)
        self.spin_gap.setDecimals(2)
        self.spin_gap.setSingleStep(0.25)
        self.spin_gap.setSuffix(" s")
        self.spin_gap.setValue(float(c.get("gap_s", 0.0)))
        self.spin_gap.setToolTip("Blank frames inserted between jobs")
        lay.addRow("Gap between jobs", self.spin_gap)

        self.btn_bg = ColorButton(c.get("bg", "#ff000000"),
                                  "Reel backdrop colour")
        self.btn_bg.setToolTip(
            "Fills the gaps, and the letterbox around any job smaller than "
            "the canvas")
        lay.addRow("Backdrop", self.btn_bg)

        self.spin_crf = QtWidgets.QSpinBox()
        self.spin_crf.setRange(0, 51)
        self.spin_crf.setValue(int(enc.get("crf", 20)))
        lay.addRow("CRF", self.spin_crf)

        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItems(["ultrafast", "veryfast", "fast", "medium",
                                    "slow"])
        self.combo_preset.setCurrentText(str(enc.get("preset", "veryfast")))
        lay.addRow("Preset", self.combo_preset)

        self.spin_target = QtWidgets.QDoubleSpinBox()
        self.spin_target.setRange(0, 5000)
        self.spin_target.setSuffix(" MB")
        self.spin_target.setSpecialValueText("no budget")
        self.spin_target.setValue(float(enc.get("target_mb", 20.0)))
        self.spin_target.setToolTip(
            "Applies to the whole reel, not to each job - a batch of five "
            "pairs has to fit in this one number")
        lay.addRow("Size budget", self.spin_target)

        if plans:
            summary = QtWidgets.QLabel(
                f"{len(plans)} job(s), "
                + " + ".join(f"{d:.1f}s" for _n, d in plans)
                + f" = {sum(d for _n, d in plans):.1f}s of footage.")
            summary.setWordWrap(True)
            summary.setStyleSheet("color:#666; font-size:11px;")
            lay.addRow("", summary)

        note = QtWidgets.QLabel(
            "Jobs run back to back in list order. Any job smaller than the "
            "canvas is centred and letterboxed on the backdrop; per-job "
            "output files are not written.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:11px;")
        lay.addRow("", note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)

    def _browse(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Combined output", self.edit_out.text(),
            "MP4 video (*.mp4);;All files (*)")
        if p:
            self.edit_out.setText(p)

    def values(self):
        return {"output": self.edit_out.text().strip(),
                "fps": self.spin_fps.value(),
                "width": self.spin_w.value(),
                "height": self.spin_h.value(),
                "gap_s": self.spin_gap.value(),
                "bg": self.btn_bg.hex(),
                "encode": {"use_ffmpeg": True,
                           "crf": self.spin_crf.value(),
                           "preset": self.combo_preset.currentText(),
                           "target_mb": self.spin_target.value()}}


class CollapsibleBox(QtWidgets.QWidget):
    """A titled section that folds away, so the editor column fits on a
    laptop screen instead of demanding ~1000px of height all at once."""

    def __init__(self, title, widget, expanded=True, parent=None):
        super().__init__(parent)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded
                                 else QtCore.Qt.ArrowType.RightArrow)
        self.toggle.setStyleSheet(
            "QToolButton { border:none; font-weight:600; padding:3px 2px; }")
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        self.toggle.clicked.connect(self._on_toggled)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color:#333;")

        self.body = widget
        self.body.setVisible(expanded)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(line)
        lay.addWidget(self.body)

    def _on_toggled(self, checked):
        self.toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked
                                 else QtCore.Qt.ArrowType.RightArrow)
        self.body.setVisible(checked)


class ClipEditor(QtWidgets.QWidget):

    changed = QtCore.pyqtSignal()

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._loading = False
        self._seeded_ref = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 2, 4)
        lay.setSpacing(3)

        self.row_video = FileRow("Video", "Video (*.mp4 *.mov *.avi *.mkv "
                                 "*.m4v *.webm);;All files (*)")
        self.row_calib = FileRow("Calibration", "Camera alignment "
                                 "(*.camalign.json);;JSON (*.json)",
                                 "camera_align.py calibration: pose, "
                                 "keyframes, reference line style, legend")
        self.row_ref = FileRow("Reference", "Path (*.npz *.csv *.txt);;"
                               "All files (*)",
                               "The planned/ideal raceline drawn over this "
                               "clip")
        for r in (self.row_video, self.row_calib, self.row_ref):
            r.changed.connect(self._emit)
            lay.addWidget(r)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.edit_label = QtWidgets.QLineEdit()
        self.edit_label.setPlaceholderText("Nominal MPC")
        self.edit_label.editingFinished.connect(self._emit)
        form.addRow("Label", self.edit_label)

        row_t = QtWidgets.QHBoxLayout()
        self.spin_shift = self._spin(-1e5, 1e5, 0.05, 3, " s")
        self.spin_shift.setToolTip(
            "Composite time at which this clip starts. Stagger the two to "
            "line up an event that happens at different wall-clock times.")
        self.spin_trim0 = self._spin(0.0, 1e5, 0.1, 3, " s")
        self.spin_trim1 = self._spin(0.0, 1e5, 0.1, 3, " s")
        self.spin_trim1.setToolTip("0 = play to the end of the clip")
        for w, cap in ((self.spin_shift, "start at"), (self.spin_trim0, "from"),
                       (self.spin_trim1, "to")):
            row_t.addWidget(QtWidgets.QLabel(cap))
            row_t.addWidget(w)
        form.addRow("Timing", self._wrap(row_t))

        row_ref = QtWidgets.QHBoxLayout()
        self.chk_ref_custom = QtWidgets.QCheckBox("custom")
        self.chk_ref_custom.setToolTip(
            "On: the colour, stroke width and pen style set here are used "
            "as typed.\nOff: all three come from this clip's calibration, "
            "exactly as camera_align.py drew them.")
        self.chk_ref_custom.toggled.connect(self._on_ref_custom)
        self.btn_ref_color = QtWidgets.QPushButton()
        self.btn_ref_color.setFixedSize(34, 22)
        self.btn_ref_color.setToolTip("Reference line colour and opacity")
        self.btn_ref_color.clicked.connect(self._pick_ref_color)
        self.spin_ref_width = QtWidgets.QDoubleSpinBox()
        self.spin_ref_width.setRange(0.5, 200.0)
        self.spin_ref_width.setDecimals(1)
        self.spin_ref_width.setSingleStep(0.5)
        self.spin_ref_width.setValue(3.0)
        self.spin_ref_width.setSuffix(" px wide")
        self.spin_ref_width.setToolTip(
            "Stroke width of the reference line, in output panel pixels - "
            "used exactly as typed on both the video and the npz panel")
        self.spin_ref_width.valueChanged.connect(self._emit)
        self.combo_ref_style = QtWidgets.QComboBox()
        self.combo_ref_style.addItems(REF_STYLES)
        self.combo_ref_style.currentTextChanged.connect(self._emit)
        for w in (self.chk_ref_custom, self.btn_ref_color,
                  self.spin_ref_width, self.combo_ref_style):
            row_ref.addWidget(w)
        row_ref.addStretch(1)
        form.addRow("Reference line", self._wrap(row_ref))

        row_flags = QtWidgets.QHBoxLayout()
        self.chk_overlay = QtWidgets.QCheckBox("reference")
        self.chk_legend = QtWidgets.QCheckBox("legend")
        self.chk_ts = QtWidgets.QCheckBox("timestamp")
        for c in (self.chk_overlay, self.chk_legend, self.chk_ts):
            c.setChecked(True)
            c.toggled.connect(self._emit)
            row_flags.addWidget(c)
        row_flags.addStretch(1)
        form.addRow("Draw", self._wrap(row_flags))
        lay.addLayout(form)

        self.row_npz = FileRow("NPZ panel", "NumPy archive (*.npz);;"
                               "Path (*.csv *.txt);;All files (*)",
                               "Optional state plot rendered as its own "
                               "panel welded to this clip")
        self.row_npz.changed.connect(self._on_npz_changed)
        lay.addWidget(self.row_npz)

        row_aux = QtWidgets.QHBoxLayout()
        self.combo_aux = QtWidgets.QComboBox()
        self.combo_aux.addItems(AUX_POSITIONS)
        self.combo_aux.currentTextChanged.connect(self._emit)
        self.spin_frac = self._spin(0.05, 3.0, 0.05, 2, "")
        self.spin_frac.setToolTip("Panel size as a fraction of the video's "
                                  "width (left/right) or height (above/below)")
        self.spin_npz_off = self._spin(-1e5, 1e5, 0.05, 3, " s")
        self.spin_npz_off.setToolTip("npz time = clip time minus this")
        self.combo_aux_fit = QtWidgets.QComboBox()
        self.combo_aux_fit.addItems(AUX_FITS)
        self.combo_aux_fit.setToolTip(
            "How the path uses the panel.\n"
            "stretch: fills it edge to edge horizontally and vertically - "
            "no dead space, but the track's shape is distorted.\n"
            "aspect: true shape, centred, with empty space on the long "
            "axis.\n"
            "width: spans the full width at true aspect; anything taller "
            "than the panel is clipped.")
        self.combo_aux_fit.currentTextChanged.connect(self._emit)
        self.btn_aux_style = QtWidgets.QPushButton("Panel style…")
        self.btn_aux_style.setToolTip(
            "Colours and stroke widths inside the npz panel: the driven "
            "trajectory, the full log, the reference line, the position "
            "marker, backdrop and text - each pickable on its own.")
        self.btn_aux_style.clicked.connect(self._edit_aux_style)
        for cap, w in (("position", self.combo_aux), ("size", self.spin_frac),
                       ("fit", self.combo_aux_fit),
                       ("offset", self.spin_npz_off),
                       ("", self.btn_aux_style)):
            row_aux.addWidget(QtWidgets.QLabel(cap))
            row_aux.addWidget(w)
        row_aux.addStretch(1)
        lay.addLayout(row_aux)

        self.lbl_aux_hint = QtWidgets.QLabel("")
        self.lbl_aux_hint.setWordWrap(True)
        self.lbl_aux_hint.setStyleSheet("color:#a33; font-size:11px;")
        self.lbl_aux_hint.setVisible(False)
        lay.addWidget(self.lbl_aux_hint)

    def _edit_aux_style(self):
        dlg = AuxStyleDialog(self, getattr(self, "_style", None))
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._style = dlg.values()
            self.combo_aux_fit.blockSignals(True)
            self.combo_aux_fit.setCurrentText(self._style.get("fit", "stretch"))
            self.combo_aux_fit.blockSignals(False)
            self._emit()

    def _pick_ref_color(self):
        c = QtWidgets.QColorDialog.getColor(
            getattr(self, "_ref_color", qcolor("#ff46c8ff")), self,
            "Reference line colour",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not c.isValid():
            return
        self._ref_color = c
        self.chk_ref_custom.setChecked(True)     # picking one means using it
        self._refresh_ref_swatch()
        self._emit()

    def _on_ref_custom(self, on):
        if on:
            self._seed_ref_from_calib()
        self._refresh_ref_swatch()
        self._emit()

    def _seed_ref_from_calib(self):
        """Switching to custom starts from what the calibration was drawing,
        so the line doesn't jump the moment the box is ticked."""
        path = self.row_calib.path()
        if not path or not os.path.exists(path) or self._seeded_ref:
            return
        try:
            cal = Calibration(path)
        except Exception:
            return
        self._seeded_ref = True
        self._ref_color = QtGui.QColor(cal.ref_color)
        self.spin_ref_width.blockSignals(True)
        self.spin_ref_width.setValue(max(0.5, float(cal.ref_width)))
        self.spin_ref_width.blockSignals(False)
        self.combo_ref_style.blockSignals(True)
        self.combo_ref_style.setCurrentText(
            "dot" if cal.ref_dotted else "solid")
        self.combo_ref_style.blockSignals(False)

    def _refresh_ref_swatch(self):
        col = getattr(self, "_ref_color", qcolor("#ff46c8ff"))
        on = self.chk_ref_custom.isChecked()
        for w in (self.btn_ref_color, self.spin_ref_width,
                  self.combo_ref_style):
            w.setEnabled(on)
        self.btn_ref_color.setStyleSheet(
            f"background:{_css_rgba(col)}; border:1px solid #333;" if on
            else "background:transparent; border:1px dashed #777;")

    def _on_npz_changed(self):
        """Choosing an npz with the position still on 'none' used to load
        the file and draw nothing, silently. Picking a file now turns the
        panel on; clearing the file turns it back off."""
        if self.row_npz.path() and self.combo_aux.currentText() == "none":
            self.combo_aux.blockSignals(True)
            self.combo_aux.setCurrentText("right")
            self.combo_aux.blockSignals(False)
        elif not self.row_npz.path():
            self.combo_aux.blockSignals(True)
            self.combo_aux.setCurrentText("none")
            self.combo_aux.blockSignals(False)
        self._emit()

    def _refresh_aux_hint(self):
        has_file = bool(self.row_npz.path())
        pos = self.combo_aux.currentText()
        if has_file and pos == "none":
            self.lbl_aux_hint.setText(
                "An npz is loaded but its position is 'none', so no panel is "
                "drawn - pick right / left / above / below.")
            self.lbl_aux_hint.setVisible(True)
        elif pos != "none" and not has_file:
            self.lbl_aux_hint.setText(
                "A panel position is set but no npz is loaded, so there's "
                "nothing to plot.")
            self.lbl_aux_hint.setVisible(True)
        else:
            self.lbl_aux_hint.setVisible(False)
        for w in (self.spin_frac, self.spin_npz_off, self.combo_aux_fit,
                  self.btn_aux_style):
            w.setEnabled(pos != "none")

    @staticmethod
    def _wrap(layout):
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _spin(self, lo, hi, step, dec, suffix):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(dec)
        sb.setSuffix(suffix)
        sb.valueChanged.connect(self._emit)
        return sb

    def _emit(self, *_):
        self._refresh_aux_hint()
        if not self._loading:
            self.changed.emit()

    def set_clip(self, clip):
        self._loading = True
        c = merged(DEFAULT_CLIP, clip)
        self.row_video.set_path(c.get("video"))
        self.row_calib.set_path(c.get("calib"))
        self.row_ref.set_path(c.get("reference"))
        self.row_npz.set_path(c.get("npz"))
        self.edit_label.setText(c.get("label", ""))
        self.spin_shift.setValue(float(c.get("time_shift", 0.0)))
        self.spin_trim0.setValue(float(c.get("trim_start", 0.0)))
        self.spin_trim1.setValue(float(c.get("trim_end", 0.0)))
        self.chk_overlay.setChecked(bool(c.get("overlay", True)))
        self.chk_legend.setChecked(bool(c.get("legend", True)))
        self.chk_ts.setChecked(bool(c.get("timestamp", True)))
        self.combo_aux.setCurrentText(c.get("aux", "none"))
        self.spin_frac.setValue(float(c.get("aux_frac", 0.6)))
        self.spin_npz_off.setValue(float(c.get("npz_offset", 0.0)))
        self._ref_color = qcolor(c.get("ref_color") or "#ff46c8ff")
        custom = c.get("ref_custom")
        if custom is None:      # legacy batch: any override meant custom
            custom = bool(c.get("ref_color")
                          or float(c.get("ref_width", 0.0) or 0.0) > 0)
        self._seeded_ref = bool(custom)
        self.chk_ref_custom.blockSignals(True)
        self.chk_ref_custom.setChecked(bool(custom))
        self.chk_ref_custom.blockSignals(False)
        self.spin_ref_width.setValue(
            max(0.5, float(c.get("ref_width", 3.0) or 3.0)))
        self.combo_ref_style.setCurrentText(
            c.get("ref_style", "dot") if c.get("ref_style") in REF_STYLES
            else "dot")
        self._refresh_ref_swatch()
        self._style = merged(DEFAULT_AUXSTYLE, c.get("aux_style"))
        self.combo_aux_fit.setCurrentText(self._style.get("fit", "stretch"))
        self._refresh_aux_hint()
        self._loading = False

    def clip(self):
        c = copy.deepcopy(DEFAULT_CLIP)
        c.update({
            "video": self.row_video.path(),
            "calib": self.row_calib.path(),
            "reference": self.row_ref.path(),
            "npz": self.row_npz.path(),
            "label": self.edit_label.text().strip(),
            "ref_custom": self.chk_ref_custom.isChecked(),
            "ref_color": hexargb(self._ref_color),
            "ref_width": self.spin_ref_width.value(),
            "ref_style": self.combo_ref_style.currentText(),
            "time_shift": self.spin_shift.value(),
            "trim_start": self.spin_trim0.value(),
            "trim_end": self.spin_trim1.value(),
            "overlay": self.chk_overlay.isChecked(),
            "legend": self.chk_legend.isChecked(),
            "timestamp": self.chk_ts.isChecked(),
            "aux": self.combo_aux.currentText(),
            "aux_frac": self.spin_frac.value(),
            "npz_offset": self.spin_npz_off.value(),
            "aux_style": merged(getattr(self, "_style", DEFAULT_AUXSTYLE),
                                {"fit": self.combo_aux_fit.currentText()}),
        })
        return c


class ExportThread(QtCore.QThread):

    message = QtCore.pyqtSignal(str)
    frame_progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(bool, str)

    def __init__(self, jobs, base_dir, only=None, combined=None,
                 batch_name="batch", parent=None):
        super().__init__(parent)
        self.jobs = copy.deepcopy(jobs)
        self.base_dir = base_dir
        self.only = only
        # not None -> render everything into one file instead of one each
        self.combined = copy.deepcopy(combined) if combined else None
        self.batch_name = batch_name
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        def emit(msg=""):
            print(msg, flush=True)
            self.message.emit(str(msg))
        try:
            if self.combined is not None:
                out = export_combined(
                    self.jobs, self.base_dir, self.combined, log=emit,
                    progress=lambda a, b: self.frame_progress.emit(a, b),
                    cancel=lambda: self._cancel, batch_name=self.batch_name)
                self.done.emit(True, f"one reel written: "
                                     f"{os.path.basename(out)}")
                return
            outs = run_batch(self.jobs, self.base_dir, log=emit,
                             progress=lambda a, b: self.frame_progress.emit(a, b),
                             cancel=lambda: self._cancel, only=self.only)
            self.done.emit(True, f"{len(outs)} file(s) written")
        except Cancelled:
            self.done.emit(False, "cancelled")
        except Exception as exc:
            emit(f"ERROR: {exc}")
            self.done.emit(False, str(exc))


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("compare_render - side-by-side comparison renderer")
        # never open bigger than the screen: on a laptop the old fixed
        # 1620x980 pushed the export buttons off the bottom edge
        screen = QtGui.QGuiApplication.primaryScreen()
        avail = (screen.availableGeometry() if screen is not None
                 else QtCore.QRect(0, 0, 1280, 800))
        self.resize(min(1620, int(avail.width() * 0.94)),
                    min(980, int(avail.height() * 0.90)))
        self.move(avail.left() + 20, avail.top() + 20)
        self.jobs = []
        self.batch_path = None
        self.base_dir = os.getcwd()
        self._renderer = None
        self._thread = None
        self._loading = False
        self._legend_entries = []
        self._combined = copy.deepcopy(DEFAULT_COMBINED)
        self._build()
        self.add_job()
        self.statusBar().showMessage(
            "Add a job, point each side at a video + its .camalign.json + the "
            "reference path, then Export. Panels hold their last frame when "
            "one clip finishes first.")

    # -- layout ----------------------------------------------------------
    def _build(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)

        # jobs list
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("<b>Jobs</b>  (one output each)"))
        self.list_jobs = QtWidgets.QListWidget()
        self.list_jobs.currentRowChanged.connect(self._select_job)
        left.addWidget(self.list_jobs, 1)
        for text, fn in (("+ job", self.add_job),
                         ("Duplicate", self.duplicate_job),
                         ("Remove", self.remove_job),
                         ("▲", lambda: self.move_job(-1)),
                         ("▼", lambda: self.move_job(+1))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            left.addWidget(b)
        left.addSpacing(8)
        for text, fn in (("Load batch…", self.load_batch_dialog),
                         ("Save batch…", self.save_batch_dialog)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            left.addWidget(b)
        left_host = QtWidgets.QWidget()
        left_host.setLayout(left)
        left_host.setMaximumWidth(210)
        self.list_jobs.setMinimumHeight(80)

        # editor
        self.clip_a = ClipEditor("Left / top clip")
        self.clip_b = ClipEditor("Right / bottom clip")
        for c in (self.clip_a, self.clip_b):
            c.changed.connect(self._on_edit)

        mid = QtWidgets.QVBoxLayout()
        mid.setContentsMargins(4, 2, 4, 2)
        mid.setSpacing(2)
        self.sections = [
            CollapsibleBox("Left / top clip", self.clip_a, True),
            CollapsibleBox("Right / bottom clip", self.clip_b, True),
            CollapsibleBox("Layout", self._layout_box(), True),
            CollapsibleBox("Labels & timestamp", self._text_box(), False),
            CollapsibleBox("Legend", self._legend_box(), False),
            CollapsibleBox("Output & encoding", self._output_box(), False),
        ]
        for box in self.sections:
            mid.addWidget(box)
        mid.addStretch(1)
        mid_content = QtWidgets.QWidget()
        mid_content.setLayout(mid)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(mid_content)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(600)

        # preview + log, split so either can be dragged out of the way
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(2, 2, 2, 2)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(240, 160)
        self.view.setStyleSheet("background:#101014; border:1px solid #333;")
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Expanding)
        right.addWidget(self.view, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(lambda *_: self._refresh_preview())
        right.addWidget(self.slider)

        row = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("Reload preview")
        btn_refresh.setToolTip("Reopen the media and redraw (after editing "
                               "files on disk)")
        btn_refresh.clicked.connect(self._rebuild_preview)
        self.lbl_plan = QtWidgets.QLabel("")
        # without wrapping, this one label's text sets a ~870px floor on the
        # whole window's minimum width and the layout stops fitting on a
        # laptop screen
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet("font-family:monospace; color:#444; "
                                    "font-size:11px;")
        row.addWidget(btn_refresh)
        row.addWidget(self.lbl_plan, 1)
        right.addLayout(row)

        # missing files / silent-no-op settings, said out loud instead of
        # only whispered into the log
        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color:#a33; font-size:11px;")
        self.lbl_warn.setVisible(False)
        right.addWidget(self.lbl_warn)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export this job")
        self.btn_export.setStyleSheet("font-weight:bold;")
        self.btn_export.clicked.connect(lambda: self.export(only_current=True))
        self.btn_export_all = QtWidgets.QPushButton("Export all jobs")
        self.btn_export_all.setToolTip("One output file per job")
        self.btn_export_all.clicked.connect(lambda: self.export(False))
        self.btn_export_one = QtWidgets.QPushButton("All → one video…")
        self.btn_export_one.setToolTip(
            "Render every job back to back into a single file - mixed sizes "
            "and frame rates are fitted onto one canvas and clock")
        self.btn_export_one.clicked.connect(self.export_combined)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_export)
        for b in (self.btn_export, self.btn_export_all, self.btn_cancel):
            row2.addWidget(b)
        right.addLayout(row2)
        # its own row: four buttons abreast pushed the window's minimum
        # width up by ~120px, back past what fits on a small screen
        right.addWidget(self.btn_export_one)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        right.addWidget(self.bar)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(48)
        self.log_box.setStyleSheet("font-family:monospace; font-size:11px;")

        right_top = QtWidgets.QWidget()
        right_top.setLayout(right)
        right_host = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_host.addWidget(right_top)
        right_host.addWidget(self.log_box)
        right_host.setStretchFactor(0, 1)
        right_host.setSizes([700, 140])

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(left_host)
        split.addWidget(scroll)
        split.addWidget(right_host)
        split.setStretchFactor(2, 1)
        w = self.width()
        split.setSizes([int(w * 0.14), int(w * 0.31), int(w * 0.55)])
        split.setCollapsible(0, True)
        split.setCollapsible(1, True)
        root.addWidget(split)
        self.setCentralWidget(central)
        self.setAcceptDrops(True)

    def _layout_box(self):
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(6, 2, 2, 4)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.combo_layout = QtWidgets.QComboBox()
        self.combo_layout.addItems(LAYOUTS)
        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(ALIGNMENTS)
        self.combo_align.setToolTip(
            "How the two units line up on the cross axis when they come out "
            "different sizes")
        self.combo_end = QtWidgets.QComboBox()
        self.combo_end.addItems(END_BEHAVIORS)
        self.combo_end.setToolTip(
            "What a panel does once its own clip has finished while the "
            "other keeps playing")
        self.spin_panel = QtWidgets.QSpinBox()
        self.spin_panel.setRange(64, 4320)
        self.spin_panel.setToolTip(
            "Every video is normalized to this height (horizontal layout) or "
            "width (vertical) - this is how clips of different resolutions "
            "end up matched")
        self.spin_maxw = QtWidgets.QSpinBox()
        self.spin_maxw.setRange(0, 8192)
        self.spin_maxw.setSpecialValueText("no limit")
        self.spin_maxh = QtWidgets.QSpinBox()
        self.spin_maxh.setRange(0, 8192)
        self.spin_maxh.setSpecialValueText("no limit")
        self.spin_gap = QtWidgets.QDoubleSpinBox()
        self.spin_gap.setRange(0, 500)
        self.spin_gap.setSuffix(" px")
        self.btn_bg = QtWidgets.QPushButton()
        self.btn_bg.setFixedSize(40, 22)
        self.btn_bg.clicked.connect(lambda: self._pick_color("bg"))
        for cap, w in (("Arrangement", self.combo_layout),
                       ("Cross-axis align", self.combo_align),
                       ("When a clip ends", self.combo_end),
                       ("Panel size", self.spin_panel),
                       ("Max width", self.spin_maxw),
                       ("Max height", self.spin_maxh),
                       ("Gap", self.spin_gap),
                       ("Backdrop", self.btn_bg)):
            form.addRow(cap, w)
        for w in (self.combo_layout, self.combo_align, self.combo_end):
            w.currentTextChanged.connect(self._on_edit)
        for w in (self.spin_panel, self.spin_maxw, self.spin_maxh,
                  self.spin_gap):
            w.valueChanged.connect(self._on_edit)
        return box

    def _text_box(self):
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(6, 2, 2, 4)
        self.spin_lab_size = QtWidgets.QDoubleSpinBox()
        self.spin_lab_size.setRange(4, 400)
        self.spin_lab_size.setSuffix(" px")
        self.spin_lab_size.setToolTip("In panel pixels - what you type is "
                                      "what the output gets")
        self.chk_lab_latex = QtWidgets.QCheckBox("Label is LaTeX")
        self.chk_lab_latex.setEnabled(latex_available())
        self.chk_lab_latex.setToolTip(
            "Render the label through LaTeX, e.g. \"LLA-MPC $v_{max}$\" or "
            "\"\\textbf{Nominal} MPC\" (the latter needs usetex below).")
        self.chk_ts_latex = QtWidgets.QCheckBox("Timestamp is LaTeX")
        self.chk_ts_latex.setEnabled(latex_available())
        self.chk_ts_latex.setToolTip(
            "The assembled prefix + number + suffix goes through LaTeX, so "
            "a prefix like \"$t=$\" or \"$\\tau=$\" typesets as maths.")
        self.chk_usetex = QtWidgets.QCheckBox(
            "Use a real LaTeX install (usetex)")
        self.chk_usetex.setEnabled(usetex_available())
        self.chk_usetex.setToolTip(
            "Off: LaTeX-flagged strings go through matplotlib's mathtext - "
            "no external dependencies, maths mode only.\n"
            "On: they're typeset by your own `latex` and rasterised by "
            "`dvipng`, so text-mode commands work as written - \\textbf, "
            "\\texttt, \\textcolor - along with anything your preamble's "
            "packages provide.\n"
            "Applies to the label, the timestamp and the legend.")
        self.edit_preamble = QtWidgets.QLineEdit(DEFAULT_LATEX_PREAMBLE)
        self.edit_preamble.setEnabled(usetex_available())
        self.edit_preamble.setToolTip(
            "Preamble used in usetex mode - add \\usepackage lines for any "
            "packages your labels need.")
        self.btn_tex_check = QtWidgets.QPushButton("Check LaTeX")
        self.btn_tex_check.setToolTip(
            "Typeset a sample with the current settings and report what "
            "happened - a failed render otherwise just falls back to plain "
            "text without saying so.")
        self.btn_tex_check.clicked.connect(self._check_latex)
        self.lbl_tex = QtWidgets.QLabel(latex_status())
        self.lbl_tex.setWordWrap(True)
        self.lbl_tex.setStyleSheet(
            ("color:#666;" if usetex_available() else "color:#a33;")
            + " font-size:11px;")
        self.combo_lab_pos = QtWidgets.QComboBox()
        self.combo_lab_pos.addItems(POSITIONS)

        self.spin_ts_size = QtWidgets.QDoubleSpinBox()
        self.spin_ts_size.setRange(4, 400)
        self.spin_ts_size.setSuffix(" px")
        self.combo_ts_corner = QtWidgets.QComboBox()
        self.combo_ts_corner.addItems(CORNERS)
        self.combo_ts_src = QtWidgets.QComboBox()
        self.combo_ts_src.addItems(TIMESTAMP_SOURCES)
        self.combo_ts_src.setToolTip(
            "'video' = the clip's own clock; 'trajectory' = that minus the "
            "sync offset stored in its calibration")
        self.edit_ts_prefix = QtWidgets.QLineEdit()
        self.edit_ts_suffix = QtWidgets.QLineEdit()
        self.spin_ts_dec = QtWidgets.QSpinBox()
        self.spin_ts_dec.setRange(0, 6)
        for cap, w in (("Label size", self.spin_lab_size),
                       ("Label position", self.combo_lab_pos),
                       ("Timestamp size", self.spin_ts_size),
                       ("Corner", self.combo_ts_corner),
                       ("Clock", self.combo_ts_src),
                       ("Prefix", self.edit_ts_prefix),
                       ("Suffix", self.edit_ts_suffix),
                       ("Decimals", self.spin_ts_dec),
                       ("LaTeX", self.chk_lab_latex),
                       ("", self.chk_ts_latex),
                       ("", self.chk_usetex),
                       ("Preamble", self.edit_preamble),
                       ("", self.btn_tex_check),
                       ("", self.lbl_tex)):
            form.addRow(cap, w)
        for w in (self.spin_lab_size, self.spin_ts_size, self.spin_ts_dec):
            w.valueChanged.connect(self._on_edit)
        for w in (self.combo_ts_corner, self.combo_ts_src,
                  self.combo_lab_pos):
            w.currentTextChanged.connect(self._on_edit)
        for w in (self.edit_ts_prefix, self.edit_ts_suffix):
            w.editingFinished.connect(self._on_edit)
        for w in (self.chk_lab_latex, self.chk_ts_latex):
            w.toggled.connect(self._on_edit)
        self.chk_usetex.toggled.connect(self._on_usetex_toggled)
        self.edit_preamble.editingFinished.connect(self._on_preamble_changed)
        return box

    def _on_usetex_toggled(self, on):
        self.edit_preamble.setEnabled(bool(on) and usetex_available())
        _LATEX_CACHE.clear()          # engine changed, cached bitmaps stale
        self._on_edit()

    def _on_preamble_changed(self):
        _LATEX_CACHE.clear()
        self._on_edit()

    def _check_latex(self):
        """Typeset a sample with the current settings and say what happened."""
        if not latex_available():
            self.lbl_tex.setText(latex_status())
            return
        usetex = self.chk_usetex.isChecked() and usetex_available()
        sample = (self.clip_a.edit_label.text().strip()
                  or self.edit_ts_prefix.text().strip() or "$v_{max} = 3.4$")
        _LATEX_CACHE.clear()
        img = render_latex(sample, self.spin_lab_size.value(),
                           QtGui.QColor(255, 255, 255), True, usetex=usetex,
                           preamble=self.edit_preamble.text())
        engine = "usetex" if usetex else "mathtext"
        if img is None:
            err = latex_last_error() or "render returned nothing"
            self.lbl_tex.setText(f"{engine} FAILED on {sample!r} - {err}")
            self.lbl_tex.setStyleSheet("color:#a33; font-size:11px;")
        else:
            self.lbl_tex.setText(
                f"{engine} OK: {sample!r} -> {img.width()}x{img.height()}px."
                f"  {latex_status()}")
            self.lbl_tex.setStyleSheet("color:#276; font-size:11px;")

    def _legend_box(self):
        box = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(box)
        outer.setContentsMargins(6, 2, 2, 4)
        outer.setSpacing(3)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self.combo_leg_scope = QtWidgets.QComboBox()
        self.combo_leg_scope.addItems(LEGEND_SCOPES)
        self.combo_leg_scope.setToolTip(
            "Which panels carry the legend. One copy is usually enough "
            "when both clips share a key.")
        self.spin_leg_size = QtWidgets.QDoubleSpinBox()
        self.spin_leg_size.setRange(0, 400)
        self.spin_leg_size.setSuffix(" px")
        self.spin_leg_size.setSpecialValueText("from calibration")
        self.spin_leg_size.setToolTip(
            "Text size in panel pixels; everything else in the box scales "
            "with it. Only meaningful for rows defined below - with none, "
            "0 keeps each calibration's own size, rescaled.")
        self.combo_leg_corner = QtWidgets.QComboBox()
        self.combo_leg_corner.addItems(["inherit"] + CORNERS)
        self.spin_leg_cols = QtWidgets.QSpinBox()
        self.spin_leg_cols.setRange(1, 8)
        self.spin_leg_cols.setToolTip(
            "Rows are dealt out column by column, in list order")
        self.spin_leg_bg = QtWidgets.QDoubleSpinBox()
        self.spin_leg_bg.setRange(0, 255)
        self.spin_leg_bg.setToolTip("0 = no backdrop box behind the rows")
        self.btn_leg_text = ColorButton("#ffffffff", "Legend text colour")
        self.btn_leg_bg = ColorButton("#ff000000", "Legend backdrop colour")
        row_col = QtWidgets.QHBoxLayout()
        row_col.setContentsMargins(0, 0, 0, 0)
        row_col.addWidget(self.btn_leg_text)
        row_col.addWidget(self.btn_leg_bg)
        row_col.addStretch(1)
        host_col = QtWidgets.QWidget()
        host_col.setLayout(row_col)
        for cap, w in (("Draw on", self.combo_leg_scope),
                       ("Text size", self.spin_leg_size),
                       ("Corner", self.combo_leg_corner),
                       ("Columns", self.spin_leg_cols),
                       ("Backdrop alpha", self.spin_leg_bg),
                       ("Text / backdrop", host_col)):
            form.addRow(cap, w)
        outer.addLayout(form)

        self.list_legend = QtWidgets.QListWidget()
        self.list_legend.setMaximumHeight(110)
        self.list_legend.setToolTip(
            "Legend rows drawn by this tool. Leave the list empty to fall "
            "back to the legend saved in each clip's calibration. "
            "Double-click to edit.")
        self.list_legend.itemDoubleClicked.connect(
            lambda *_: self._edit_legend_entry())
        outer.addWidget(self.list_legend)

        row_b = QtWidgets.QHBoxLayout()
        for text, fn in (("+ row", self._add_legend_entry),
                         ("Edit…", self._edit_legend_entry),
                         ("Remove", self._remove_legend_entry),
                         ("▲", lambda: self._move_legend_entry(-1)),
                         ("▼", lambda: self._move_legend_entry(+1))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row_b.addWidget(b)
        outer.addLayout(row_b)

        btn_seed = QtWidgets.QPushButton("Seed from clips")
        btn_seed.setToolTip(
            "One row per clip, using its label and its reference line's "
            "actual colour, width and style - a starting point you can "
            "then edit freely.")
        btn_seed.clicked.connect(self._seed_legend)
        outer.addWidget(btn_seed)

        hint = QtWidgets.QLabel(
            "With no rows here, the legend comes from each clip's "
            ".camalign.json - and if that has none either, nothing is "
            "drawn. Sizes here are output panel pixels.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        outer.addWidget(hint)

        for w in (self.spin_leg_size, self.spin_leg_cols, self.spin_leg_bg):
            w.valueChanged.connect(self._on_edit)
        for w in (self.combo_leg_scope, self.combo_leg_corner):
            w.currentTextChanged.connect(self._on_edit)
        for w in (self.btn_leg_text, self.btn_leg_bg):
            w.picked.connect(self._on_edit)
        return box

    # -- legend rows ------------------------------------------------------
    def _refresh_legend_list(self):
        self.list_legend.clear()
        for i, e in enumerate(self._legend_entries):
            kind = e.get("kind", "line")
            desc = e.get("style", "solid") if kind == "line" else kind
            tex = "  TeX" if e.get("latex") else ""
            item = QtWidgets.QListWidgetItem(
                f"{e.get('label') or '(no label)'}   "
                f"[{kind}/{desc}  w={e.get('width', 0):g}{tex}]")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            item.setForeground(QtGui.QBrush(qcolor(e.get("color"))))
            self.list_legend.addItem(item)

    def _add_legend_entry(self):
        dlg = LegendEntryDialog(self, {"label": "", "kind": "line",
                                       "style": "solid", "width": 6.0,
                                       "color": "#ffffffff"})
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._legend_entries.append(dlg.values())
            self._refresh_legend_list()
            self._on_edit()

    def _edit_legend_entry(self):
        item = self.list_legend.currentItem()
        if item is None:
            return
        i = item.data(QtCore.Qt.ItemDataRole.UserRole)
        dlg = LegendEntryDialog(self, self._legend_entries[i])
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._legend_entries[i] = dlg.values()
            self._refresh_legend_list()
            self._on_edit()

    def _remove_legend_entry(self):
        item = self.list_legend.currentItem()
        if item is None:
            return
        self._legend_entries.pop(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self._refresh_legend_list()
        self._on_edit()

    def _move_legend_entry(self, d):
        item = self.list_legend.currentItem()
        if item is None:
            return
        i = item.data(QtCore.Qt.ItemDataRole.UserRole)
        j = i + d
        if not (0 <= j < len(self._legend_entries)):
            return
        self._legend_entries[i], self._legend_entries[j] = \
            self._legend_entries[j], self._legend_entries[i]
        self._refresh_legend_list()
        self.list_legend.setCurrentRow(j)
        self._on_edit()

    def _seed_legend(self):
        """Prefill a row per clip from what that clip actually draws."""
        for editor in (self.clip_a, self.clip_b):
            clip = editor.clip()
            if not clip.get("video"):
                continue
            col, wid, sty = "#ff46c8ff", 6.0, "dot"
            cal_path = clip.get("calib")
            if cal_path and os.path.exists(_resolve(cal_path, self.base_dir)):
                try:
                    cal = Calibration(_resolve(cal_path, self.base_dir))
                    col, sty = hexargb(cal.ref_color), (
                        "dot" if cal.ref_dotted else "solid")
                    wid = max(cal.ref_width, 4.0)
                except Exception:
                    pass
            if clip.get("ref_custom"):
                col = clip.get("ref_color") or col
                wid = float(clip.get("ref_width", 0) or 0) or wid
                if clip.get("ref_style") in PEN_STYLES:
                    sty = clip["ref_style"]
            self._legend_entries.append(
                {"label": clip.get("label") or "reference", "kind": "line",
                 "style": sty, "width": max(wid, 3.0), "color": col,
                 "latex": False})
        self._refresh_legend_list()
        self._on_edit()

    def _output_box(self):
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(6, 2, 2, 4)
        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("job name (used for the filename)")
        self.edit_out = QtWidgets.QLineEdit()
        self.edit_out.setPlaceholderText("(auto: comparisons/<name>.mp4)")
        self.spin_fps = QtWidgets.QDoubleSpinBox()
        self.spin_fps.setRange(1, 240)
        self.spin_dur = QtWidgets.QDoubleSpinBox()
        self.spin_dur.setRange(0, 1e5)
        self.spin_dur.setSuffix(" s")
        self.spin_dur.setSpecialValueText("auto (longest clip)")
        self.spin_crf = QtWidgets.QSpinBox()
        self.spin_crf.setRange(0, 51)
        self.spin_target = QtWidgets.QDoubleSpinBox()
        self.spin_target.setRange(0, 5000)
        self.spin_target.setSuffix(" MB")
        self.spin_target.setSpecialValueText("no budget")
        self.spin_target.setToolTip(
            "If the encode lands over this, it's re-encoded two-pass at a "
            "bitrate that fits. No re-render.")
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItems(["ultrafast", "veryfast", "fast", "medium",
                                    "slow"])
        for cap, w in (("Job name", self.edit_name), ("File", self.edit_out),
                       ("FPS", self.spin_fps), ("Duration", self.spin_dur),
                       ("CRF", self.spin_crf), ("Preset", self.combo_preset),
                       ("Size budget", self.spin_target)):
            form.addRow(cap, w)
        if not ffmpeg_available():
            warn = QtWidgets.QLabel("ffmpeg not found on PATH - falling back "
                                    "to OpenCV mp4v (CRF and the size budget "
                                    "are ignored).")
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#a33; font-size:11px;")
            form.addRow("", warn)
        for w in (self.spin_fps, self.spin_dur, self.spin_crf,
                  self.spin_target):
            w.valueChanged.connect(self._on_edit)
        self.combo_preset.currentTextChanged.connect(self._on_edit)
        for w in (self.edit_name, self.edit_out):
            w.editingFinished.connect(self._on_edit)
        return box

    # -- job list --------------------------------------------------------
    def _job_title(self, job, i):
        name = job.get("name") or ""
        if not name:
            def stem(side):
                v = (job.get(side) or {}).get("video")
                return os.path.basename(v) if v else "—"
            name = f"{stem('left')}  vs  {stem('right')}"
        return f"{i + 1}. {name}"

    def _refresh_job_list(self, keep=None):
        row = self.list_jobs.currentRow() if keep is None else keep
        self.list_jobs.blockSignals(True)
        self.list_jobs.clear()
        for i, job in enumerate(self.jobs):
            self.list_jobs.addItem(self._job_title(job, i))
        self.list_jobs.setCurrentRow(int(np.clip(row, 0,
                                                 len(self.jobs) - 1)))
        self.list_jobs.blockSignals(False)

    def add_job(self):
        self.jobs.append(copy.deepcopy(DEFAULT_JOB))
        self._refresh_job_list(keep=len(self.jobs) - 1)
        self._select_job(len(self.jobs) - 1)

    def duplicate_job(self):
        i = self.list_jobs.currentRow()
        if i < 0:
            return
        self._pull_job()
        self.jobs.insert(i + 1, copy.deepcopy(self.jobs[i]))
        self._refresh_job_list(keep=i + 1)
        self._select_job(i + 1)

    def remove_job(self):
        i = self.list_jobs.currentRow()
        if i < 0 or not self.jobs:
            return
        self.jobs.pop(i)
        if not self.jobs:
            self.jobs.append(copy.deepcopy(DEFAULT_JOB))
        self._refresh_job_list(keep=min(i, len(self.jobs) - 1))
        self._select_job(self.list_jobs.currentRow())

    def move_job(self, d):
        i = self.list_jobs.currentRow()
        j = i + d
        if i < 0 or not (0 <= j < len(self.jobs)):
            return
        self._pull_job()
        self.jobs[i], self.jobs[j] = self.jobs[j], self.jobs[i]
        self._refresh_job_list(keep=j)

    def _select_job(self, i):
        if not (0 <= i < len(self.jobs)):
            return
        self._push_job(self.jobs[i])
        self._rebuild_preview()

    # -- job <-> widgets --------------------------------------------------
    def _push_job(self, job):
        self._loading = True
        j = merged(DEFAULT_JOB, job)
        self.clip_a.set_clip(j["left"])
        self.clip_b.set_clip(j["right"])
        self.combo_layout.setCurrentText(j["layout"])
        self.combo_align.setCurrentText(j["align"])
        self.combo_end.setCurrentText(j["end_behavior"])
        self.spin_panel.setValue(int(j["panel_size"]))
        self.spin_maxw.setValue(int(j["max_width"]))
        self.spin_maxh.setValue(int(j["max_height"]))
        self.spin_gap.setValue(float(j["gap"]))
        self._bg = qcolor(j["bg"])
        self.btn_bg.setStyleSheet(f"background:{_css_rgba(self._bg)}; "
                                  "border:1px solid #333;")
        lab, ts = merged(DEFAULT_LABEL, j["label"]), merged(DEFAULT_TIMESTAMP,
                                                            j["timestamp"])
        self._label_style, self._ts_style = lab, ts
        self.spin_lab_size.setValue(float(lab["text_size"]))
        self.combo_lab_pos.setCurrentText(lab.get("position", "top-center"))
        self.chk_lab_latex.setChecked(bool(lab["latex"]))
        jtex = merged(DEFAULT_JOB["latex"], j.get("latex"))
        self.chk_usetex.setChecked(bool(jtex.get("usetex"))
                                   and usetex_available())
        self.edit_preamble.setText(str(jtex.get("preamble",
                                                DEFAULT_LATEX_PREAMBLE)))
        self.edit_preamble.setEnabled(self.chk_usetex.isChecked()
                                      and usetex_available())
        lgj = merged(DEFAULT_JOB["legend"], j.get("legend"))
        self.combo_leg_scope.setCurrentText(lgj.get("scope", "both"))
        self.spin_leg_size.setValue(float(lgj.get("size", 0.0)))
        self.combo_leg_corner.setCurrentText(lgj.get("corner", "inherit"))
        self.spin_leg_cols.setValue(int(lgj.get("columns", 1) or 1))
        self.spin_leg_bg.setValue(float(lgj.get("bg_alpha", 150)))
        self.btn_leg_text.set_color(lgj.get("text_color", "#ffffffff"))
        self.btn_leg_bg.set_color(lgj.get("bg_color", "#ff000000"))
        self._legend_entries = [dict(e) for e in (lgj.get("entries") or [])]
        self._refresh_legend_list()
        self.spin_ts_size.setValue(float(ts["text_size"]))
        self.combo_ts_corner.setCurrentText(ts["corner"])
        self.combo_ts_src.setCurrentText(ts["source"])
        self.edit_ts_prefix.setText(ts["prefix"])
        self.edit_ts_suffix.setText(ts["suffix"])
        self.spin_ts_dec.setValue(int(ts["decimals"]))
        self.chk_ts_latex.setChecked(bool(ts.get("latex"))
                                     and latex_available())
        enc = merged(DEFAULT_JOB["encode"], j["encode"])
        self.edit_name.setText(j.get("name", ""))
        self.edit_out.setText(j.get("output", ""))
        self.spin_fps.setValue(float(j["fps"]))
        self.spin_dur.setValue(float(j["duration"]))
        self.spin_crf.setValue(int(enc["crf"]))
        self.combo_preset.setCurrentText(enc["preset"])
        self.spin_target.setValue(float(enc["target_mb"]))
        self._loading = False

    def _pull_job(self):
        i = self.list_jobs.currentRow()
        if not (0 <= i < len(self.jobs)):
            return None
        lab = merged(getattr(self, "_label_style", DEFAULT_LABEL), {
            "text_size": self.spin_lab_size.value(),
            "position": self.combo_lab_pos.currentText(),
            "latex": self.chk_lab_latex.isChecked()})
        ts = merged(getattr(self, "_ts_style", DEFAULT_TIMESTAMP), {
            "text_size": self.spin_ts_size.value(),
            "corner": self.combo_ts_corner.currentText(),
            "source": self.combo_ts_src.currentText(),
            "prefix": self.edit_ts_prefix.text(),
            "suffix": self.edit_ts_suffix.text(),
            "decimals": self.spin_ts_dec.value(),
            "latex": self.chk_ts_latex.isChecked()})
        job = {
            "name": self.edit_name.text().strip(),
            "left": self.clip_a.clip(),
            "right": self.clip_b.clip(),
            "layout": self.combo_layout.currentText(),
            "align": self.combo_align.currentText(),
            "end_behavior": self.combo_end.currentText(),
            "panel_size": self.spin_panel.value(),
            "max_width": self.spin_maxw.value(),
            "max_height": self.spin_maxh.value(),
            "gap": self.spin_gap.value(),
            "bg": hexargb(getattr(self, "_bg", qcolor("#ff0a0a0e"))),
            "fps": self.spin_fps.value(),
            "duration": self.spin_dur.value(),
            "output": self.edit_out.text().strip(),
            "label": lab, "timestamp": ts,
            "legend": {"scope": self.combo_leg_scope.currentText(),
                       "size": self.spin_leg_size.value(),
                       "corner": self.combo_leg_corner.currentText(),
                       "columns": self.spin_leg_cols.value(),
                       "bg_alpha": self.spin_leg_bg.value(),
                       "text_color": self.btn_leg_text.hex(),
                       "bg_color": self.btn_leg_bg.hex(),
                       "entries": [dict(e) for e in self._legend_entries]},
            "latex": {"usetex": self.chk_usetex.isChecked(),
                      "preamble": self.edit_preamble.text()},
            "encode": {"use_ffmpeg": True, "crf": self.spin_crf.value(),
                       "preset": self.combo_preset.currentText(),
                       "target_mb": self.spin_target.value()},
        }
        self.jobs[i] = merged(DEFAULT_JOB, job)
        self.list_jobs.blockSignals(True)
        item = self.list_jobs.item(i)
        if item is not None:
            item.setText(self._job_title(self.jobs[i], i))
        self.list_jobs.blockSignals(False)
        return self.jobs[i]

    def _pick_color(self, which):
        c = QtWidgets.QColorDialog.getColor(
            getattr(self, "_bg", qcolor("#ff0a0a0e")), self, "Backdrop",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._bg = c
            self.btn_bg.setStyleSheet(f"background:{_css_rgba(c)}; "
                                      "border:1px solid #333;")
            self._on_edit()

    # -- preview ---------------------------------------------------------
    def _on_edit(self, *_):
        if self._loading:
            return
        self._pull_job()
        self._rebuild_preview()

    def _rebuild_preview(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        job = self._pull_job()
        if job is None:
            return
        try:
            self._renderer = JobRenderer(job, self.base_dir)
        except Exception as exc:
            self.lbl_plan.setText(str(exc))
            self.lbl_warn.setText(str(exc))
            self.lbl_warn.setVisible(True)
            self.view.setText(str(exc))
            self.slider.setRange(0, 0)
            return
        notes = list(self._renderer.notes)
        for note in notes:
            self._log(f"! {note}")
        self.lbl_warn.setText("  ·  ".join(notes))
        self.lbl_warn.setVisible(bool(notes))
        n = self._renderer.n_frames
        pos = min(self.slider.value(), n - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(max(0, pos))
        self.slider.blockSignals(False)
        self.lbl_plan.setText(self._renderer.describe())
        self._refresh_preview()

    def _refresh_preview(self):
        if self._renderer is None:
            return
        try:
            frame = self._renderer.frame_at_index(self.slider.value())
        except Exception as exc:
            self.view.setText(f"preview failed: {exc}")
            return
        pm = QtGui.QPixmap.fromImage(bgr_to_qimage(frame))
        self.view.setPixmap(pm.scaled(
            self.view.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refresh_preview()

    # -- batch io --------------------------------------------------------
    def load_batch_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load batch", "", f"Comparison batch (*{SUFFIX});;"
            "JSON (*.json)")
        if p:
            self.load_batch(p)

    def load_batch(self, path):
        try:
            jobs, combined = load_batch(path, with_combined=True)
            self._combined = combined
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load batch", str(exc))
            return
        if not jobs:
            QtWidgets.QMessageBox.warning(self, "Load batch",
                                          "that file has no jobs in it")
            return
        self.jobs = jobs
        self.batch_path = path
        self.base_dir = os.path.dirname(os.path.abspath(path))
        self._refresh_job_list(keep=0)
        self._select_job(0)
        self.statusBar().showMessage(
            f"Loaded {len(jobs)} job(s) from {os.path.basename(path)}", 6000)

    def save_batch_dialog(self):
        self._pull_job()
        default = self.batch_path or os.path.join(self.base_dir,
                                                  "batch" + SUFFIX)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save batch", default, f"Comparison batch (*{SUFFIX});;"
            "JSON (*.json)")
        if not p:
            return
        if not p.endswith(".json"):
            p += SUFFIX
        try:
            self.jobs = save_batch(p, self.jobs, self.base_dir,
                                   combined=self._combined)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save batch", str(exc))
            return
        self.batch_path = p
        self.base_dir = os.path.dirname(os.path.abspath(p))
        self._refresh_job_list()
        self._log(f"wrote {p}")
        self.statusBar().showMessage(f"Saved {os.path.basename(p)}", 6000)

    # -- export ----------------------------------------------------------
    def _log(self, msg):
        self.log_box.appendPlainText(str(msg))
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def export(self, only_current=True):
        if self._thread is not None and self._thread.isRunning():
            return
        self._pull_job()
        only = self.list_jobs.currentRow() if only_current else None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.log_box.clear()
        self._start_thread(ExportThread(self.jobs, self.base_dir, only=only))

    def export_combined(self):
        """Ask for reel settings, then render every job into one file."""
        if self._thread is not None and self._thread.isRunning():
            return
        self._pull_job()
        # durations for the dialog's summary, cheap enough to measure here
        plans = []
        for i, job in enumerate(self.jobs):
            try:
                r = JobRenderer(job, self.base_dir)
                plans.append((job.get("name") or f"job {i + 1}", r.duration))
                r.close()
            except Exception:
                continue
        if not plans:
            QtWidgets.QMessageBox.warning(
                self, "Export all as one",
                "No job opens cleanly yet - give at least one of them a "
                "video first.")
            return
        dlg = CombinedExportDialog(self, self._combined, plans)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._combined = dlg.values()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.log_box.clear()
        self._start_thread(ExportThread(
            self.jobs, self.base_dir, combined=self._combined,
            batch_name=self._batch_name()))

    def _batch_name(self):
        if self.batch_path:
            base = os.path.basename(self.batch_path)
            for suffix in (SUFFIX, ".json"):
                if base.endswith(suffix):
                    return base[:-len(suffix)]
            return base
        return "batch"

    def _start_thread(self, thread):
        self._thread = thread
        thread.message.connect(self._log)
        thread.frame_progress.connect(
            lambda a, b: self.bar.setValue(int(100 * a / max(b, 1))))
        thread.done.connect(self._export_done)
        for b in (self.btn_export, self.btn_export_all, self.btn_export_one):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        thread.start()

    def _cancel_export(self):
        if self._thread is not None:
            self._thread.cancel()

    def _export_done(self, ok, msg):
        for b in (self.btn_export, self.btn_export_all, self.btn_export_one):
            b.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.bar.setValue(100 if ok else 0)
        self.statusBar().showMessage(
            ("Export finished - " if ok else "Export stopped - ") + msg, 8000)
        self._rebuild_preview()

    # -- drag & drop ------------------------------------------------------
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        """Drop files onto the window: a batch loads, a video fills the first
        empty video slot, a calibration/path fills the matching slot of
        whichever side already has a video."""
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if not os.path.exists(p):
                continue
            low = p.lower()
            if low.endswith(SUFFIX):
                self.load_batch(p)
                return
            if low.endswith(".camalign.json"):
                target = (self.clip_a if not self.clip_a.row_calib.path()
                          else self.clip_b)
                target.row_calib.set_path(p)
            elif low.endswith((".npz", ".csv", ".txt")):
                target = (self.clip_a if not self.clip_a.row_ref.path()
                          else self.clip_b)
                target.row_ref.set_path(p)
            else:
                target = (self.clip_a if not self.clip_a.row_video.path()
                          else self.clip_b)
                target.row_video.set_path(p)
        self._on_edit()

    def closeEvent(self, ev):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(3000)
        if self._renderer is not None:
            self._renderer.close()
        super().closeEvent(ev)


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    args = sys.argv[1:]
    render = "--render" in args
    only = None
    if "--only" in args:
        k = args.index("--only")
        try:
            only = int(args[k + 1]) - 1
        except (IndexError, ValueError):
            only = None
    files = [a for a in args if not a.startswith("--") and os.path.exists(a)]
    if "--only" in args and len(args) > args.index("--only") + 1:
        files = [f for f in files if f != args[args.index("--only") + 1]]

    if render:
        if not files:
            print("--render needs a batch file, e.g. "
                  f"python compare_render.py batch{SUFFIX} --render")
            return 2
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtGui.QGuiApplication([])                   # fonts need this
        _ = app
        base = os.path.dirname(os.path.abspath(files[0]))
        jobs, combined = load_batch(files[0], with_combined=True)
        log(f"{len(jobs)} job(s) from {os.path.basename(files[0])}")
        if "--combined" in args:
            name = os.path.basename(files[0])
            for suffix in (SUFFIX, ".json"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            out = export_combined(jobs, base, combined, log=log,
                                  batch_name=name)
            log(f"wrote {out}")
            return 0
        outs = run_batch(jobs, base, log=log, only=only)
        log(f"wrote {len(outs)} file(s)")
        return 0

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    for f in files:
        if f.lower().endswith(SUFFIX):
            win.load_batch(f)
            break
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())