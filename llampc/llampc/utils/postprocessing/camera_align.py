"""camera_align.py - align a ground-plane path (recorded rollout or planned
raceline) onto a video by eye, whether the camera is fixed or moving.

The problem: you have a video, and any number of 2D (x, y) paths in some
world/track frame:

  TRAJECTORIES  any number of time-varying recorded rollouts (or static
              planned paths) - all managed from one list, each drawn as
              its own solid line. Add as many as you like to overlay
              several rollouts on the same video/track at once. Each is
              independently styled - color, line width, visibility,
              start/end star colors - and, if it has a time axis, can
              independently be sectioned down to a start/end time window
              (only that stretch of its line gets drawn) and given its
              own color/opacity segments. They all share one sync offset
              (video time -> path time) and the global current-position-
              marker toggle. The video itself stays fully scrubbable
              regardless of any trajectory's time window.
  REFERENCE   an optional second kind of path - e.g. an ideal/planned
              raceline to compare the trajectories against - always drawn
              in full, as a dotted line, with no time-window trimming.

Either can be loaded on its own; trajectories and the reference are
assumed to share one coordinate frame (the same track), so a single
TRACK pose (scale/yaw/pitch/roll/offset, see below) places all of them in
the world together - there's no separate transform per path.

VIDEO LAYERS: any number of extra videos can be composited on top of the
base video (each at reduced opacity) before any of the above gets drawn -
useful for ghosting another run's footage from the same physical camera
setup behind the current one. Each layer contributes exactly one fixed
frame, picked in its own little scrubber window (so several can be left
open side by side), not a synced playback position. The
grid/trajectories/reference/legend/timestamp always draw on top of this
composite, regardless of how many video layers are stacked under them -
that's the "final composition" saved by "Save frame...".

You want to see the path(s) drawn on top of the video, in the right
place, at the right scale, in colors/thicknesses you choose - even if the
camera pans, tilts, zooms, or is handheld and drifts around. Each path
has one base color (its alpha channel sets a uniform opacity) and,
optionally, a list of color/opacity segments - overrides for stretches of
the path given as a [start, end] fraction (0-1) of its length, e.g. to
fade a rollout from green to red over its length or gray out everything
but the corner you're studying. Segments are independent per path and
don't have to cover the whole length or even differ from each other -
leave the list empty for one solid color.

This is single-view camera calibration against known ground-plane points -
solved here interactively rather than automatically. There are two
independent things being fit at once, each with its own little pose:

  CAMERA (where you're standing/filming from)
    position     cam_x, cam_y, cam_z      world units; cam_z is height
                                           above the world's z=0 plane
    orientation  azimuth, elevation, roll degrees; azimuth/elevation aim
                                           the optical axis, roll tilts
                                           the horizon
    zoom         fov                      degrees, horizontal field of view
    fine tune    pp_x, pp_y               principal point offset, pixels -
                                           corrects lens/sensor asymmetry,
                                           rarely needs touching

  TRACK (how the path files' own coordinates sit in the world - useful
  when they were logged in some arbitrary/local frame, on a plane that
  isn't level, or just need flipping/rescaling to match reality)
    scale        path_scale              path units -> world units
    orientation  path_yaw, path_pitch,   degrees; yaw spins the track
                 path_roll               about vertical (heading), pitch
                                          tilts it uphill/downhill, roll
                                          banks it side-to-side - applied
                                          in that order (yaw, then pitch,
                                          then roll) before placing it
    position     path_offset_x,          world units; path_z is the
                 path_offset_y, path_z   track's height off the ground

Nudge azimuth/elevation to rotate the camera, fov to zoom, cam_x/y/z to
move it, until the projected path lands on the path visible in the video.
If the path is roughly right but skewed or tilted in a way camera moves
can't fix, that's the track's own yaw/pitch/roll to adjust instead. A
ground grid can be toggled on for extra perspective cues even before the
path itself lines up.

If the camera moves during the clip (handheld, or a pan/tilt/zoom shot),
set keyframes: tune the pose at one frame, jump ahead to wherever it's
visibly moved, retune, set another keyframe. Camera pose is linearly
interpolated between keyframes (angle-aware, so 350deg -> 10deg takes the
short way). A clip with 0 or 1 keyframes is just a fixed camera.

Two path formats are understood, for either the trajectory or the
reference slot:
  - .npz with a 'state' array (N, D>=2): first two columns are x, y and the
    path is treated as a time-varying recorded rollout. If a 'time' array
    is present it's used for playback sync (and trajectory time-window
    trimming), otherwise a flat dt is assumed.
  - .npz with 'x' and 'y' arrays (as written by Track-style raceline
    files): treated as a static planned path with no time axis - the whole
    curve is drawn, with no separate "current position" marker and no
    time-window trim (there's no time to section by).
  - .csv/.txt with an 'x','y' header (and optional 't'): same rules as
    above, csv version.

ANNOTATION
  LEGEND     an optional box of free-form rows (line swatches, stars,
             circles + labels) in any corner. Labels can be plain text or
             LaTeX/mathtext ("$v_{max}$", "$\\theta = 30^\\circ$") when
             matplotlib is installed - tick the entry's LaTeX box. The
             legend can be laid out in several columns, filled column by
             column, which is handy once you have more than a few rows.
  TIMESTAMP  an optional "t=x.x" readout in a corner (bottom-left by
             default), reading either raw video time or the trajectory's
             own clock (video time minus the sync offset). Prefix/suffix
             and decimal places are configurable, and it can also be
             rendered as LaTeX.

Calibration (camera/track pose, keyframes, sync offset, line
colors/widths/opacity segments, legend, timestamp, and the trajectory trim
window) saves to a "<name>.camalign.json" next to the video, and can be
reloaded to keep refining it later. Loading a calibration is agnostic of
which video or path files are open: it only applies pose/keyframes/sync/
display/trim settings to whatever video/trajectory/reference are already
loaded (or get loaded afterward) - it does not itself open, require, or
check for the video/trajectory/reference files that were open when it was
saved. This lets one calibration be reused across a re-encoded video, a
renamed file, or a different rollout recorded from the same physical
camera setup. (The file paths that were open at save time are still
recorded in the json purely as provenance/notes - they're just never read
back.) It can also bootstrap its sync offset from a trim_marker.py
".trimspec.json", if you already used that tool to align video and rollout
in time.

Requires: PyQt6, opencv-python, numpy.
Optional: matplotlib (only for LaTeX/mathtext labels - everything else
works without it; LaTeX labels quietly fall back to plain text).
    pip install PyQt6 opencv-python numpy matplotlib
"""
from __future__ import annotations

import io
import os
import sys
import csv
import json
import time
import bisect

import numpy as np
import cv2

from PyQt6 import QtCore, QtGui, QtWidgets

# matplotlib is an optional dependency: it's used purely as a LaTeX
# (mathtext) renderer for legend/timestamp labels. Without it, entries
# marked as LaTeX just draw as their literal source text.
try:
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext as _mathtext
    from matplotlib.figure import Figure as _Figure
    from matplotlib.font_manager import FontProperties as _FontProperties
    HAVE_MATHTEXT = True
except Exception:      # pragma: no cover - depends on the environment
    HAVE_MATHTEXT = False

# Real LaTeX (matplotlib's text.usetex) needs an actual TeX install on PATH:
# `latex` to typeset and `dvipng` to rasterize. Unlike mathtext this handles
# text-mode commands - \textbf, \texttt, \textcolor, \mathrm, custom
# packages via the preamble - at the cost of a hard external dependency and
# a slower first render per unique string.
DEFAULT_LATEX_PREAMBLE = r"\usepackage{xcolor}"


def usetex_missing_reason():
    """None if usetex is fully available, else a short string naming what's
    missing. usetex needs matplotlib AND both the `latex` and `dvipng`
    executables specifically visible on PATH to the process running this
    script - not just "LaTeX is installed" in some general sense. Common
    surprises: dvipng is often a separate package from the base TeX
    install (e.g. Ubuntu's texlive-font-utils) and can be missing even
    when `latex`/`pdflatex` works fine; and a GUI launched from a desktop
    icon, systemd unit, or a different shell/conda env can see a
    completely different PATH than an interactive terminal where `which
    latex` succeeds."""
    if not HAVE_MATHTEXT:
        return "matplotlib isn't installed"
    missing = [name for name in ("latex", "dvipng") if not shutil.which(name)]
    if missing:
        return f"`{'` and `'.join(missing)}` not found on this process's PATH"
    return None


def usetex_available():
    return usetex_missing_reason() is None


SCHEMA = "camalign/8"
SUFFIX = ".camalign.json"

# NOTE on naming: "azimuth/elevation/roll" pose the CAMERA; the parallel
# "path_yaw/path_pitch/path_roll" pose the TRACK. They're independent -
# e.g. panning the camera and rotating the track can look similar on
# screen for small angles, but only track rotation stays put as you
# scrub through a moving-camera clip. The TRACK pose is shared by both
# the trajectory and the reference path - they're assumed to live in the
# same coordinate frame.
DEFAULT_CAM = {
    "cam_x": 0.0, "cam_y": -10.0, "cam_z": 3.0,
    "azimuth": 90.0, "elevation": 15.0, "roll": 0.0,
    "fov": 60.0, "pp_x": 0.0, "pp_y": 0.0,
    "path_scale": 1.0,
    "path_yaw": 0.0, "path_pitch": 0.0, "path_roll": 0.0,
    "path_offset_x": 0.0, "path_offset_y": 0.0, "path_z": 0.0,
}

# (key, label, min, max, step, decimals)
CAM_FIELDS = [
    # -- camera pose --
    ("cam_x",      "Cam X",          -100000.0, 100000.0, 0.1,  3),
    ("cam_y",      "Cam Y",          -100000.0, 100000.0, 0.1,  3),
    ("cam_z",      "Cam height Z",       -500.0,    500.0, 0.05, 3),
    ("azimuth",    "Azimuth (deg)",      -360.0,    360.0, 0.5,  2),
    ("elevation",  "Elevation (deg)",     -89.0,     89.0, 0.5,  2),
    ("roll",       "Roll (deg)",          -45.0,     45.0, 0.2,  2),
    ("fov",        "FOV / zoom (deg)",      5.0,     150.0, 0.5,  2),
    ("pp_x",       "Principal pt X (px)", -4000.0,  4000.0, 1.0,  1),
    ("pp_y",       "Principal pt Y (px)", -4000.0,  4000.0, 1.0,  1),
    # -- track pose (how the path files sit in the world) --
    ("path_scale",    "Path scale",           0.001,   1000.0, 0.01, 4),
    ("path_yaw",       "Track yaw (deg)",     -360.0,    360.0, 0.5,  2),
    ("path_pitch",      "Track pitch (deg)",   -89.0,     89.0, 0.2,  2),
    ("path_roll",        "Track roll (deg)",   -89.0,     89.0, 0.2,  2),
    ("path_offset_x", "Path offset X",    -100000.0, 100000.0, 0.1,  3),
    ("path_offset_y", "Path offset Y",    -100000.0, 100000.0, 0.1,  3),
    ("path_z",     "Path height Z",       -500.0,    500.0, 0.1,  3),
]

NUDGE_STEP = {"azimuth": 1.0, "elevation": 1.0, "roll": 0.5, "fov": 2.0}
NUDGE_STEP_FINE = {k: v / 5.0 for k, v in NUDGE_STEP.items()}

DEFAULT_TRAJ_COLOR = (255, 210, 0, 230)   # amber, matches the old fixed color
DEFAULT_REF_COLOR = (70, 200, 255, 220)   # cyan, visually distinct + dotted

DEFAULT_START_COLOR = (40, 200, 80, 255)   # green star at the visible start
DEFAULT_END_COLOR = (220, 40, 40, 255)     # red star at the visible end

# Cycled through, in order, as trajectories are added - the first one keeps
# the classic amber color, and later ones stay visually distinct from each
# other and from the reference/start/end colors above.
DEFAULT_TRAJ_PALETTE = [
    DEFAULT_TRAJ_COLOR,
    (255, 80, 80, 230), (80, 160, 255, 230), (255, 170, 0, 230),
    (170, 90, 255, 230), (0, 210, 140, 230), (255, 105, 180, 230),
    (120, 220, 220, 230), (200, 200, 60, 230),
]

# Named pen styles, for both the legend entries and (later) any line that
# wants something other than solid/dotted.
PEN_STYLES = {
    "solid":   QtCore.Qt.PenStyle.SolidLine,
    "dash":    QtCore.Qt.PenStyle.DashLine,
    "dot":     QtCore.Qt.PenStyle.DotLine,
    "dashdot": QtCore.Qt.PenStyle.DashDotLine,
    "dashdotdot": QtCore.Qt.PenStyle.DashDotDotLine,
}
LEGEND_KINDS = ["line", "star", "circle"]
LEGEND_CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"]
TIMESTAMP_SOURCES = ["video", "trajectory"]

# Every legend/marker size below is in FULL-RESOLUTION video pixels. A 4K
# frame shown in a ~900px-wide widget is downscaled >4x, so a legend that
# reads comfortably on screen needs text in the 60-200px range and line
# widths in the 8-30px range. The spinbox ranges are set wide enough for
# that; don't be surprised if sane-looking values (12px text) vanish.
DEFAULT_LEGEND = {
    "enabled": False,
    "corner": "top-right",
    "margin": 40.0,        # gap from the frame edge
    "padding": 24.0,       # inside the box, around the contents
    "text_size": 64.0,     # font pixel size
    "bold": True,
    "sample_len": 140.0,   # length of the little line/marker swatch
    "gap": 24.0,           # swatch -> label gap
    "row_gap": 14.0,       # vertical gap between entries
    "bg_alpha": 150,       # 0 = no background box
    "columns": 1,          # entries are dealt out column by column
    "col_gap": 60.0,       # horizontal gap between columns
}

# The "t=x.x" readout. Same pixel convention as the legend: sizes are in
# full-resolution video pixels.
DEFAULT_TIMESTAMP = {
    "enabled": False,
    "corner": "bottom-left",
    "source": "video",     # 'video' = raw video clock,
                           # 'trajectory' = video time minus the sync offset
    "prefix": "t=",
    "suffix": " s",
    "decimals": 1,
    "text_size": 64.0,
    "bold": True,
    "margin": 40.0,
    "padding": 16.0,
    "bg_alpha": 150,       # 0 = no background box
    "latex": False,        # render the assembled string through mathtext
}


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# LaTeX (mathtext) label rendering
# ---------------------------------------------------------------------------
# matplotlib's mathtext parser handles a useful subset of LaTeX math -
# "$v_{max}$", "$\theta$", "$\frac{dx}{dt}$", "$^\circ$" - and plain text
# outside the $...$ delimiters, so a mixed label like "rollout $v_x$" works
# too. We rasterize to a transparent PNG once per (text, size, color, bold)
# and cache it, since _redraw() runs on every frame/scrub/nudge.
_LATEX_CACHE = {}
_LATEX_CACHE_MAX = 256
_LATEX_WARNED = [False]
# (text, reason) for the most recent render that fell back to plain text -
# surfaced in the main window's status bar (see MainWindow._redraw), since
# the console log below is easy to miss when the app isn't run from a
# visible terminal.
_LATEX_LAST_ISSUE = [None]


def latex_available():
    return HAVE_MATHTEXT


def _keyed_qimage(png_bytes, color):
    """White-background PNG -> transparent QImage.

    mathtext's math_to_image always writes an OPAQUE white background, so
    the glyph coverage is just the inverted luminance; every pixel is then
    tinted to `color`. (The usetex path doesn't need this - dvipng hands
    back a real alpha channel.)
    """
    raw = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("could not decode the rendered png")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    bgr = raw[:, :, :3].astype(np.float32)
    # coverage: how far this pixel is from the white page
    coverage = np.clip(255.0 - bgr.min(axis=2), 0.0, 255.0) * color.alphaF()

    hh, ww = coverage.shape
    # Format_ARGB32 is B, G, R, A in memory on little-endian machines.
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
    # dpi=72 makes 1 point == 1 pixel, so `size=px` really is px tall.
    _mathtext.math_to_image(text, buf, prop=prop, dpi=72, format="png",
                            color="black")
    return _keyed_qimage(buf.getvalue(), color)


def _render_usetex(text, px, color, bold, preamble):
    r"""Typeset `text` with a real LaTeX install: latex -> dvi -> dvipng.

    Handles text-mode markup mathtext can't - \textbf, \texttt, \textcolor,
    \mathrm - and anything else your preamble's packages provide.

    We drive latex and dvipng ourselves rather than going through
    matplotlib's usetex, because matplotlib flattens the DVI to a greyscale
    mask and tints it with the artist color, which throws away any
    \textcolor in the source. dvipng with `-bg Transparent` gives us the
    colors as typeset plus a real alpha channel.

    The entry's own color is installed as the document's default text
    color, so the swatch still controls plain labels while an explicit
    \textcolor overrides it locally.
    """
    import shlex
    import tempfile
    import subprocess

    # fix-cm (part of the LaTeX kernel, so always present) lets \fontsize
    # pick arbitrary sizes rather than snapping to Computer Modern's design
    # sizes; type1cm does the same but lives in texlive-latex-extra, which
    # a typical install lacks. Latin Modern, when installed, adds faces CM
    # lacks - notably bold typewriter, so \textbf{\texttt{..}} is bold.
    doc = "\n".join([
        r"\RequirePackage{fix-cm}",
        r"\documentclass{article}",
        r"\IfFileExists{lmodern.sty}"
        r"{\usepackage[T1]{fontenc}\usepackage{lmodern}}{}",
        r"\usepackage{xcolor}",
        preamble or "",
        r"\pagestyle{empty}",
        r"\definecolor{camalignfg}{RGB}{%d,%d,%d}" % (
            color.red(), color.green(), color.blue()),
        r"\begin{document}",
        r"\fontsize{%dpt}{%dpt}\selectfont" % (px, int(px * 1.2)),
        r"\color{camalignfg}",
        r"\bfseries" if bold else "",
        text,
        r"\end{document}",
    ])

    tmp = tempfile.mkdtemp(prefix="camalign_tex_")
    try:
        tex = os.path.join(tmp, "job.tex")
        with open(tex, "w") as f:
            f.write(doc)
        run = lambda cmd: subprocess.run(
            cmd, cwd=tmp, timeout=30, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        r = run(["latex", "-interaction=nonstopmode", "-halt-on-error",
                 "job.tex"])
        if r.returncode != 0 or not os.path.exists(os.path.join(tmp, "job.dvi")):
            raise RuntimeError(_tex_error_summary(r.stdout))
        # -D 72 makes 1pt == 1px, matching the mathtext path; -T tight crops
        # to the inked area so layout can measure the real label extent.
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
    # the swatch's alpha rides on top of dvipng's antialiasing coverage
    bgra[:, :, 3] = np.clip(
        bgra[:, :, 3].astype(np.float32) * color.alphaF(), 0, 255).astype(np.uint8)
    bgra = np.ascontiguousarray(bgra)
    hh, ww = bgra.shape[:2]
    return QtGui.QImage(bgra.data, ww, hh, 4 * ww,
                        QtGui.QImage.Format.Format_ARGB32).copy()


def _tex_error_summary(output, max_lines=6):
    """Pull the '! ...' complaint out of a LaTeX log, for the status bar."""
    try:
        lines = output.decode("utf-8", "replace").splitlines()
    except Exception:
        return "latex failed"
    bad = [ln for ln in lines if ln.startswith("!")]
    return " / ".join(bad[:max_lines]) if bad else "latex failed"


def render_latex(text, pixel_size, color, bold=False, usetex=False,
                 preamble=DEFAULT_LATEX_PREAMBLE):
    """Rasterize `text` as LaTeX -> QImage (transparent background).

    With usetex=False this is matplotlib's mathtext: a subset of LaTeX
    *math* mode, no external dependencies, and the glyphs get tinted to
    `color`. With usetex=True it shells out to a real LaTeX install, which
    understands text-mode commands and honors any \\textcolor in the source
    (then `color` only sets the overall opacity).

    Returns None if the render fails - matplotlib missing, no TeX install,
    a syntax error in the string - in which case callers fall back to
    drawing it as plain text.
    """
    if not text or not HAVE_MATHTEXT:
        return None
    if usetex and not usetex_available():
        _LATEX_LAST_ISSUE[0] = (
            text, f"usetex is on but unavailable ({usetex_missing_reason()})")
        return None
    px = max(4, int(round(float(pixel_size))))
    key = (text, px, int(color.rgba()), bool(bold), bool(usetex),
           preamble if usetex else "")
    if key in _LATEX_CACHE:
        return _LATEX_CACHE[key]
    img = None
    try:
        if usetex:
            img = _render_usetex(text, px, color, bold, preamble)
        else:
            img = _render_mathtext(text, px, color, bold)
    except Exception as exc:      # unparseable latex, missing fonts, ...
        reason = f"{'usetex' if usetex else 'mathtext'} render failed: {exc}"
        _LATEX_LAST_ISSUE[0] = (text, reason)
        if not _LATEX_WARNED[0]:
            log(f"{reason} - falling back to plain text")
            _LATEX_WARNED[0] = True
        img = None
    if len(_LATEX_CACHE) > _LATEX_CACHE_MAX:
        _LATEX_CACHE.clear()
    _LATEX_CACHE[key] = img
    return img


def _star_polygon(cx, cy, r_outer, n_points=5, inner_ratio=0.45):
    """An n-pointed star centered on (cx, cy), first point straight up.

    r_outer is the tip radius, so the star's visual size is ~2*r_outer
    across - matching how the old endpoint circles used a radius.
    """
    poly = QtGui.QPolygonF()
    r_inner = r_outer * inner_ratio
    for i in range(n_points * 2):
        ang = -np.pi / 2.0 + i * np.pi / n_points
        r = r_outer if i % 2 == 0 else r_inner
        poly.append(QtCore.QPointF(cx + r * np.cos(ang),
                                   cy + r * np.sin(ang)))
    return poly


def _css_rgba(color):
    """QColor -> a CSS rgba(...) string, so opacity actually shows up in a
    Qt stylesheet swatch (plain hex names like '#RRGGBB' drop alpha)."""
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alphaF():.3f})"


def _corner_origin(corner, w, h, box_w, box_h, margin):
    """Top-left pixel of a box_w x box_h box pinned to `corner`."""
    x = margin if corner.endswith("left") else w - margin - box_w
    y = margin if corner.startswith("top") else h - margin - box_h
    return x, y


def _rel(path, base):
    if not path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(base))
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")
    return rel.replace("\\", "/")


def _resolve(path, base):
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


# ===========================================================================
# Pinhole projection of ground-plane points onto a camera
# ===========================================================================
def _camera_basis(azimuth_deg, elevation_deg, roll_deg):
    """Right-handed camera basis (right, down, forward) in world coords.

    forward: direction the optical axis points.
    right/down: image-plane axes (down, because image row index increases
    downward - matches how the overlay is drawn).
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    roll = np.radians(roll_deg)

    forward = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        -np.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    n = np.linalg.norm(right)
    if n < 1e-8:
        # forward ~parallel to world up (looking straight down/up) - pick
        # an arbitrary right vector to avoid dividing by zero.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / n
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-8)

    right_r = np.cos(roll) * right + np.sin(roll) * down
    down_r = -np.sin(roll) * right + np.cos(roll) * down
    return right_r, down_r, forward


def _track_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """3x3 rotation matrix that poses the track (ground-plane path) in the
    world: yaw about world Z (heading), then pitch about the track's local
    X (uphill/downhill tilt), then roll about the track's local Y (side-to-
    side bank). Applied in that order - intrinsic yaw -> pitch -> roll -
    which matches "point the track the right way, then tip it downhill,
    then bank it" and keeps yaw acting like the old path_rotation did."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cp, -sp],
                   [0.0,  sp,  cp]])
    Ry = np.array([[cr, 0.0, sr],
                   [0.0, 1.0, 0.0],
                   [-sr, 0.0, cr]])
    return Rz @ Rx @ Ry


def project_points(xy, cam, image_w, image_h):
    """World (x, y) ground-plane points -> (u, v, valid) pixel coords.

    xy: (N, 2) array. cam: dict with the CAM_FIELDS keys. valid[i] is False
    where the point falls behind the camera (can't be projected sanely).
    """
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool))

    scale = cam.get("path_scale", 1.0)
    z = cam.get("path_z", 0.0)
    ox = cam.get("path_offset_x", 0.0)
    oy = cam.get("path_offset_y", 0.0)
    yaw = cam.get("path_yaw", cam.get("path_rotation", 0.0))  # back-compat
    pitch = cam.get("path_pitch", 0.0)
    roll = cam.get("path_roll", 0.0)

    # scale in the path's own flat (x, y, 0) frame, then rotate the whole
    # plane into the world (pitch/roll can lift points off z=0), then
    # translate into place.
    local = np.column_stack([xy[:, 0] * scale, xy[:, 1] * scale,
                             np.zeros(len(xy))])
    R = _track_rotation_matrix(yaw, pitch, roll)
    world = local @ R.T
    P = world + np.array([ox, oy, z])

    right, down, forward = _camera_basis(cam["azimuth"], cam["elevation"],
                                         cam["roll"])

    C = np.array([cam["cam_x"], cam["cam_y"], cam["cam_z"]])
    rel = P - C
    Xc = rel @ right
    Yc = rel @ down
    Zc = rel @ forward

    f = (image_w / 2.0) / np.tan(np.radians(cam["fov"]) / 2.0)
    valid = Zc > 1e-3

    u = np.full(len(P), np.nan)
    v = np.full(len(P), np.nan)
    u[valid] = f * Xc[valid] / Zc[valid] + image_w / 2.0 + cam["pp_x"]
    v[valid] = f * Yc[valid] / Zc[valid] + image_h / 2.0 + cam["pp_y"]
    return u, v, valid


def ground_grid_lines(bounds, spacing, pad_cells=3, samples_per_line=40):
    """Grid lines on the ground plane around `bounds`, as a list of (N,2)
    polylines (finely sampled so the perspective-behind-camera clip looks
    smooth instead of a straight line snapping off)."""
    (xmin, ymin), (xmax, ymax) = bounds
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or spacing <= 0:
        return []
    xmin -= spacing * pad_cells
    xmax += spacing * pad_cells
    ymin -= spacing * pad_cells
    ymax += spacing * pad_cells

    def _rng(lo, hi, step):
        n0 = int(np.floor(lo / step))
        n1 = int(np.ceil(hi / step))
        return [k * step for k in range(n0, n1 + 1)]

    lines = []
    for gx in _rng(xmin, xmax, spacing):
        ys = np.linspace(ymin, ymax, samples_per_line)
        lines.append(np.column_stack([np.full_like(ys, gx), ys]))
    for gy in _rng(ymin, ymax, spacing):
        xs = np.linspace(xmin, xmax, samples_per_line)
        lines.append(np.column_stack([xs, np.full_like(xs, gy)]))
    return lines


# ===========================================================================
# Moving-camera support: keyframed pose, angle-aware interpolation
# ===========================================================================
def _lerp_angle(a, b, t):
    """Shortest-path interpolation between two angles in degrees."""
    diff = ((b - a + 180) % 360) - 180
    return a + diff * t


def interpolate_camera(keyframes, frame_idx):
    """keyframes: {frame_idx: cam_dict}. Returns the camera at frame_idx,
    holding the nearest keyframe outside the covered range, interpolating
    between the two bracketing keyframes inside it. Empty/1-keyframe input
    behaves like a fixed camera."""
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
    out = {}
    for k in DEFAULT_CAM:
        if k in angle_keys:
            out[k] = _lerp_angle(c0[k], c1[k], t)
        else:
            out[k] = c0[k] + (c1[k] - c0[k]) * t
    return out


# ===========================================================================
# Sources
# ===========================================================================
class VideoFrameSource:
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
        self._last_frame = None

    def frame_at(self, idx):
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        if idx == self._cached_idx and self._last_frame is not None:
            return self._last_frame
        if idx != self._cached_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
        if not ok:
            return self._last_frame
        self._cached_idx = idx
        self._last_frame = frame
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class PathSource:
    """A ground-plane path: either time-varying (recorded rollout) or
    static (planned/raceline, no time axis). Used for both the trajectory
    and the reference slot."""

    def __init__(self, path):
        self.path = path
        self.kind = None      # 'timeseries' | 'static'
        self.xy = None        # (N, 2)
        self.time = None      # (N,) seconds, timeseries only, may be None
        self.dt = None        # seconds/sample, timeseries only
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
                self.xy = state[:, :2]
                self.kind = "timeseries"
                if "time" in keys:
                    t = np.asarray(raw["time"], dtype=float)
                    t = t - t[0]
                    if len(t) and np.nanmax(t) > 1e6:   # perf_counter_ns
                        t = t * 1e-9
                    self.time = t
                    self.dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 25.0
                else:
                    self.dt = 1 / 25.0
            elif "x" in keys and "y" in keys:
                x = np.asarray(raw["x"], dtype=float).reshape(-1)
                y = np.asarray(raw["y"], dtype=float).reshape(-1)
                self.xy = np.column_stack([x, y])
                self.kind = "static"
            else:
                raise RuntimeError(
                    "npz has neither a 'state' array nor 'x'/'y' arrays")

        elif ext in (".csv", ".txt"):
            rows = []
            with open(self.path, newline="") as f:
                sniff = f.read(2048)
                f.seek(0)
                has_header = any(c.isalpha() for c in sniff.split("\n")[0])
                reader = csv.reader(f)
                ix = iy = it = None
                if has_header:
                    header = [h.strip().lower() for h in next(reader)]
                    if "x" in header and "y" in header:
                        ix, iy = header.index("x"), header.index("y")
                        it = header.index("t") if "t" in header else None
                    else:
                        ix, iy, it = 0, 1, None
                else:
                    ix, iy, it = 0, 1, None
                for row in reader:
                    if not row:
                        continue
                    rows.append((float(row[ix]), float(row[iy]),
                                 float(row[it]) if it is not None else None))
            self.xy = np.array([(r[0], r[1]) for r in rows], dtype=float)
            ts = [r[2] for r in rows]
            if rows and all(t is not None for t in ts) and len(ts) > 1:
                t = np.asarray(ts, dtype=float)
                t = t - t[0]
                self.time = t
                self.dt = float(np.median(np.diff(t)))
                self.kind = "timeseries"
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

    def sample_at(self, t_seconds):
        """Nearest sample index for a path-relative time, timeseries only."""
        if not self.is_timeseries or self.n == 0:
            return None
        if self.time is not None:
            idx = int(np.searchsorted(self.time, t_seconds))
        else:
            idx = int(round(t_seconds / max(self.dt, 1e-9)))
        return int(np.clip(idx, 0, self.n - 1))

    def full_time_range(self):
        """(t_start, t_end) covering the whole path, timeseries only."""
        if not self.is_timeseries or self.n == 0:
            return 0.0, 0.0
        if self.time is not None:
            return float(self.time[0]), float(self.time[-1])
        return 0.0, float((self.n - 1) * self.dt)

    def index_range(self, t0, t1):
        """Inclusive sample-index range [i0, i1] covering path-relative
        time window [t0, t1] (same time base as `sample_at`/full_time_
        range). Timeseries only; on a static path this just returns the
        whole thing since there's no time axis to section by."""
        if not self.is_timeseries or self.n == 0:
            return 0, max(self.n - 1, 0)
        if t0 > t1:
            t0, t1 = t1, t0
        if self.time is not None:
            i0 = int(np.searchsorted(self.time, t0, side="left"))
            i1 = int(np.searchsorted(self.time, t1, side="right")) - 1
        else:
            dt = max(self.dt, 1e-9)
            i0 = int(np.floor(t0 / dt))
            i1 = int(np.ceil(t1 / dt))
        i0 = int(np.clip(i0, 0, self.n - 1))
        i1 = int(np.clip(i1, 0, self.n - 1))
        if i1 < i0:
            i1 = i0
        return i0, i1


def offset_from_trimspec(spec):
    """Best-effort sync offset (seconds) recovered from a trim_marker.py
    .trimspec.json: the video time and the npz/path time that both spec's
    in-points line up to."""
    master = spec.get("master") or {}
    npz = spec.get("npz") or {}
    vfps = master.get("fps")
    vtrim = master.get("trim_frames")
    ntrim = npz.get("trim_samples")
    ndt = npz.get("dt")
    if not (vfps and vtrim and ntrim is not None and ndt):
        return None
    video_t0 = vtrim[0] / float(vfps)
    path_t0 = ntrim[0] * float(ndt)
    return video_t0 - path_t0


# ===========================================================================
# Video display widget with overlay
# ===========================================================================
class OverlayView(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#101014; border:1px solid #333;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self._full_pixmap = None  # full-resolution frame + overlay, for saving

        # rectangle-selection mode (for e.g. picking a video layer's blend
        # region): armed by begin_rect_selection, drawn live in paintEvent,
        # and reported back in full-resolution frame pixel coordinates.
        self._selecting = False
        self._sel_start = None    # QPoint, widget coords
        self._sel_current = None  # QPoint, widget coords
        self._sel_callback = None

    def set_full_pixmap(self, pm):
        self._full_pixmap = pm
        self._rescale()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self):
        if self._full_pixmap is None:
            return
        self.setPixmap(self._full_pixmap.scaled(
            self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    # -- rectangle selection -------------------------------------------
    def begin_rect_selection(self, callback):
        """Arm click-drag rectangle selection; `callback(rect)` fires on
        release with (x0, y0, x1, y1) in full-resolution frame pixels, or
        `callback(None)` if the drag was too small to count (a plain
        click)."""
        self._selecting = True
        self._sel_start = None
        self._sel_current = None
        self._sel_callback = callback
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def cancel_rect_selection(self):
        if not self._selecting:
            return
        self._selecting = False
        self._sel_start = None
        self._sel_current = None
        self._sel_callback = None
        self.unsetCursor()
        self.update()

    def mousePressEvent(self, ev):
        if self._selecting and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._sel_start = ev.position().toPoint()
            self._sel_current = self._sel_start
            self.update()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._selecting and self._sel_start is not None:
            self._sel_current = ev.position().toPoint()
            self.update()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if (self._selecting and self._sel_start is not None
                and ev.button() == QtCore.Qt.MouseButton.LeftButton):
            rect_widget = QtCore.QRect(self._sel_start, self._sel_current).normalized()
            callback = self._sel_callback
            self._selecting = False
            self._sel_start = None
            self._sel_current = None
            self._sel_callback = None
            self.unsetCursor()
            self.update()
            if callback is not None:
                callback(self._widget_rect_to_frame_rect(rect_widget))
        else:
            super().mouseReleaseEvent(ev)

    def _widget_rect_to_frame_rect(self, rect_widget, min_px=4):
        """Map a widget-space QRect to full-resolution frame pixel coords,
        accounting for the displayed pixmap's aspect-fit scale and its
        centering within this label. None if the drag was smaller than
        `min_px` widget pixels in either axis (a click, not a drag)."""
        disp = self.pixmap()
        if (self._full_pixmap is None or disp is None or disp.isNull()
                or rect_widget.width() < min_px or rect_widget.height() < min_px):
            return None
        ox = (self.width() - disp.width()) / 2.0
        oy = (self.height() - disp.height()) / 2.0
        sx = self._full_pixmap.width() / max(disp.width(), 1)
        sy = self._full_pixmap.height() / max(disp.height(), 1)
        fw, fh = self._full_pixmap.width(), self._full_pixmap.height()

        def _map(px, py):
            return ((px - ox) * sx, (py - oy) * sy)

        x0, y0 = _map(rect_widget.left(), rect_widget.top())
        x1, y1 = _map(rect_widget.right(), rect_widget.bottom())
        x0, x1 = sorted((int(np.clip(x0, 0, fw)), int(np.clip(x1, 0, fw))))
        y0, y1 = sorted((int(np.clip(y0, 0, fh)), int(np.clip(y1, 0, fh))))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._selecting and self._sel_start is not None and self._sel_current is not None:
            painter = QtGui.QPainter(self)
            pen = QtGui.QPen(QtGui.QColor(255, 220, 0, 230))
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 220, 0, 40)))
            painter.drawRect(QtCore.QRect(self._sel_start, self._sel_current).normalized())
            painter.end()


class CollapsibleBox(QtWidgets.QWidget):
    """A titled section that expands/collapses like a dropdown, rather than
    permanently occupying vertical space. Used for the right-hand panel so
    stacking Camera + Display + Trajectories + Keyframes + Calibration
    doesn't squish every field down to fit - only the open section(s) take
    up room, and the panel scrolls if there's still more than fits."""

    def __init__(self, title, expanded=True, parent=None):
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
            "QToolButton { border:none; font-weight:600; padding:4px 2px; }")
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        self.toggle.clicked.connect(self._on_toggled)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color:#333;")

        self.body = QtWidgets.QWidget()
        self.body.setVisible(expanded)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(line)
        lay.addWidget(self.body)

    def setContentLayout(self, layout):
        self.body.setLayout(layout)

    def _on_toggled(self, checked):
        self.toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked
                                 else QtCore.Qt.ArrowType.RightArrow)
        self.body.setVisible(checked)


class SegmentDialog(QtWidgets.QDialog):
    """Add/edit one color+opacity override for a time stretch of a path."""

    def __init__(self, parent, start=0.0, end=1.0, color=None,
                 time_range=None):
        super().__init__(parent)
        self.setWindowTitle("Path color segment")
        self.color = (
            QtGui.QColor(color) if color is not None
            else QtGui.QColor(255, 255, 255, 255)
        )

        if time_range is None:
            t0, t1 = 0.0, 1.0
        else:
            t0, t1 = time_range

        lo = min(t0, t1)
        hi = max(t0, t1)
        pad = max((hi - lo) * 2.0, 1.0)

        lay = QtWidgets.QFormLayout(self)

        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_start.setRange(lo - pad, hi + pad)
        self.spin_start.setDecimals(3)
        self.spin_start.setSingleStep(0.1)
        self.spin_start.setValue(start)
        self.spin_start.setSuffix(" s")

        self.spin_end = QtWidgets.QDoubleSpinBox()
        self.spin_end.setRange(lo - pad, hi + pad)
        self.spin_end.setDecimals(3)
        self.spin_end.setSingleStep(0.1)
        self.spin_end.setValue(end)
        self.spin_end.setSuffix(" s")

        lay.addRow("Start time", self.spin_start)
        lay.addRow("End time", self.spin_end)

        self.lbl_range = QtWidgets.QLabel(
            f"path time range: {t0:.3f}s to {t1:.3f}s"
        )
        self.lbl_range.setStyleSheet("color:#666; font-size:11px;")
        lay.addRow("", self.lbl_range)

        self.btn_color = QtWidgets.QPushButton()
        self.btn_color.setFixedSize(50, 24)
        self.btn_color.clicked.connect(self._pick_color)
        self._refresh_swatch()
        lay.addRow("Color / opacity", self.btn_color)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(
            self.color, self, "Pick color",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid():
            self.color = c
            self._refresh_swatch()

    def _refresh_swatch(self):
        self.btn_color.setStyleSheet(
            f"background:{self.color.name(QtGui.QColor.NameFormat.HexArgb)}; "
            "border:1px solid #333;")

    def values(self):
        return (
            float(self.spin_start.value()),
            float(self.spin_end.value()),
            QtGui.QColor(self.color)
        )


class LegendEntryDialog(QtWidgets.QDialog):
    """Add/edit one legend row: a line swatch or a marker, plus its label.

    The label can be plain text or LaTeX - tick the LaTeX box either way.
    What that LaTeX is allowed to contain then depends on the Legend
    panel's "Use a real LaTeX install (usetex)" toggle: off, it's
    matplotlib's mathtext (math mode only, wrap it in dollar signs, e.g.
    "$v_{max}$" or "planned $\\theta$"); on, it's a real `latex` + `dvipng`
    pass, so text-mode commands work too and no $...$ is needed, e.g.
    "\\textbf{\\texttt{LLA-MPC}}".
    """

    def __init__(self, parent, label="", kind="line", color=None,
                 style="solid", width=8.0, latex=False):
        super().__init__(parent)
        self.setWindowTitle("Legend entry")
        self.color = (QtGui.QColor(color) if color is not None
                      else QtGui.QColor(255, 255, 255, 255))

        lay = QtWidgets.QFormLayout(self)

        self.edit_label = QtWidgets.QLineEdit(label)
        lay.addRow("Label", self.edit_label)

        self.chk_latex = QtWidgets.QCheckBox("Label is LaTeX")
        self.chk_latex.setChecked(bool(latex))
        self.chk_latex.setToolTip(
            "Must be ticked for this entry to use LaTeX at all - the "
            "Legend panel's \"Use a real LaTeX install (usetex)\" toggle "
            "only takes effect for entries with this box checked too.\n\n"
            "Without usetex (mathtext): wrap math in dollar signs, e.g. "
            "\"$v_{max}$\", \"$\\theta=30^\\circ$\", \"rollout $\\dot{x}$\" "
            "- text outside $...$ stays plain.\n"
            "With usetex: write plain LaTeX directly, no $...$ needed - "
            "e.g. \"\\textbf{\\texttt{LLA-MPC}}\".")
        lay.addRow("", self.chk_latex)

        self.lbl_latex_note = QtWidgets.QLabel("")
        self.lbl_latex_note.setWordWrap(True)
        self.lbl_latex_note.setStyleSheet("color:#a33; font-size:11px;")
        if not latex_available():
            self.chk_latex.setEnabled(False)
            self.lbl_latex_note.setText(
                "matplotlib isn't installed, so LaTeX labels are "
                "unavailable (pip install matplotlib).")
            lay.addRow("", self.lbl_latex_note)

        self.combo_kind = QtWidgets.QComboBox()
        self.combo_kind.addItems(LEGEND_KINDS)
        self.combo_kind.setCurrentText(kind if kind in LEGEND_KINDS else "line")
        self.combo_kind.currentTextChanged.connect(self._on_kind)
        lay.addRow("Kind", self.combo_kind)

        self.combo_style = QtWidgets.QComboBox()
        self.combo_style.addItems(list(PEN_STYLES.keys()))
        self.combo_style.setCurrentText(style if style in PEN_STYLES else "solid")
        lay.addRow("Line style", self.combo_style)

        self.btn_color = QtWidgets.QPushButton()
        self.btn_color.setFixedSize(50, 24)
        self.btn_color.clicked.connect(self._pick_color)
        self._refresh_swatch()
        lay.addRow("Color / opacity", self.btn_color)

        self.spin_width = QtWidgets.QDoubleSpinBox()
        self.spin_width.setRange(0.5, 400.0)
        self.spin_width.setDecimals(1)
        self.spin_width.setSingleStep(1.0)
        self.spin_width.setValue(width)
        self.spin_width.setSuffix(" px")
        lay.addRow("Line width / star radius", self.spin_width)

        note = QtWidgets.QLabel(
            "Sizes are in full-resolution video pixels. The preview is "
            "downscaled to fit the window, so these usually need to be much "
            "larger than they look like they should be.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:11px;")
        lay.addRow("", note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)
        self._on_kind(self.combo_kind.currentText())

    def _on_kind(self, kind):
        # line style only means anything for a line swatch
        self.combo_style.setEnabled(kind == "line")

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(
            self.color, self, "Pick color",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid():
            self.color = c
            self._refresh_swatch()

    def _refresh_swatch(self):
        self.btn_color.setStyleSheet(
            f"background:{_css_rgba(self.color)}; border:1px solid #333;")

    def values(self):
        return {
            "label": self.edit_label.text(),
            "kind": self.combo_kind.currentText(),
            "style": self.combo_style.currentText(),
            "color": QtGui.QColor(self.color),
            "width": float(self.spin_width.value()),
            "latex": bool(self.chk_latex.isChecked()),
        }


class VideoLayerFrameDialog(QtWidgets.QDialog):
    """A small non-modal scrubber window for picking the one fixed frame a
    video layer contributes to the composite. Kept open (hidden, not
    destroyed, on the window-close button) so its scrub position survives
    and several layers' pickers can sit open side by side for comparison.
    `on_change` is called after every frame change so the main window can
    live-update the composited preview."""

    def __init__(self, parent, layer, on_change):
        super().__init__(parent)
        self.layer = layer
        self.on_change = on_change
        self.setWindowTitle(f"Pick frame - {os.path.basename(layer['video'].path)}")
        self.setModal(False)

        lay = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(480, 270)
        self.view.setStyleSheet("background:#101014; border:1px solid #333;")
        lay.addWidget(self.view, 1)

        video = layer["video"]
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, video.n_frames - 1))
        self.slider.setValue(layer["frame_idx"])
        self.slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self.slider)

        row = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("◀ step")
        btn_prev.clicked.connect(lambda: self.slider.setValue(
            max(0, self.slider.value() - 1)))
        btn_next = QtWidgets.QPushButton("step ▶")
        btn_next.clicked.connect(lambda: self.slider.setValue(
            min(self.slider.maximum(), self.slider.value() + 1)))
        self.lbl_frame = QtWidgets.QLabel()
        row.addWidget(btn_prev)
        row.addWidget(btn_next)
        row.addWidget(self.lbl_frame)
        row.addStretch(1)
        lay.addLayout(row)

        self.resize(560, 380)
        self._refresh_preview()

    def _on_slider(self, v):
        self.layer["frame_idx"] = v
        self._refresh_preview()
        self.on_change()

    def _refresh_preview(self):
        video = self.layer["video"]
        self.lbl_frame.setText(
            f"frame {self.layer['frame_idx']}/{video.n_frames - 1}")
        frame = video.frame_at(self.layer["frame_idx"])
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w,
                           QtGui.QImage.Format.Format_RGB888).copy()
        pm = QtGui.QPixmap.fromImage(img)
        self.view.setPixmap(pm.scaled(
            self.view.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refresh_preview()


# ===========================================================================
# Main window
# ===========================================================================
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("camera_align - fit path(s) onto a video (fixed or moving camera)")
        self.resize(1560, 920)

        self.video = None    # VideoFrameSource
        # extra videos composited on top of the base video (above) but
        # underneath the grid/trajectory/legend/timestamp drawing (below) -
        # each contributes one fixed, independently-picked frame at its own
        # opacity: [{"video": VideoFrameSource, "frame_idx": int,
        # "opacity": float 0-1, "visible": bool, "_dialog": QDialog|None},
        # ...], composited in list order (later entries drawn on top).
        self.layers = []
        # style (frame_idx/opacity/visible) restored from a loaded
        # calibration, keyed by file basename - applied like
        # _pending_traj_styles.
        self._pending_layer_styles = {}
        self.ref = None      # PathSource - reference (dotted line, always full)
        # trajectories: all solid-line paths, managed as one list from a
        # single panel - [{"path": PathSource, "color": QColor,
        # "width": float, "visible": bool, "segments": [...],
        # "trim_enabled": bool, "trim_start": float, "trim_end": float},
        # ...]. Each is independently styled/segmented/trimmed; trim/
        # segments apply to whichever entry is currently selected in the
        # list (see _selected_traj_entry).
        self.trajs = []
        # style (color/width/visible/segments/trim) restored from a loaded
        # calibration, keyed by file basename - applied to entries that
        # already match and to new ones added afterward (mirrors how the
        # reference path's own style settings apply regardless of when the
        # file itself gets (re)loaded).
        self._pending_traj_styles = {}
        self.cam = dict(DEFAULT_CAM)
        self.keyframes = {}   # {frame_idx: cam_dict}
        self.offset_s = 0.0

        self.show_ref = True
        self.show_point = True
        self.show_grid = True
        self.grid_spacing = 5.0
        self.frame_idx = 0

        self.ref_color = QtGui.QColor(*DEFAULT_REF_COLOR)
        self.ref_width = 2.0

        # Optional per-stretch color/opacity overrides for the reference
        # path, as a list of {"start": t, "end": t, "color": QColor} in its
        # own time base. Empty list -> whole path drawn in ref_color above
        # (which already carries its own opacity via alpha). Trajectories
        # carry the equivalent list per-entry (entry["segments"]).
        self.ref_segments = []

        # start/end markers - now stars, with their own size and colors
        self.show_endpoints = True
        self.endpoint_size = 14.0     # star tip radius, full-res px
        self.start_color = QtGui.QColor(*DEFAULT_START_COLOR)
        self.end_color = QtGui.QColor(*DEFAULT_END_COLOR)

        # legend: a list of {"label","kind","style","color","width","latex"}
        # rows, independent of what's actually drawn - you can advertise a
        # line style that isn't in this particular shot. Rows are dealt out
        # into legend["columns"] columns, column by column.
        self.legend = dict(DEFAULT_LEGEND)
        self.legend_entries = []
        # LaTeX rendering mode, shared by legend labels and the timestamp:
        # False -> matplotlib mathtext (no external deps, math mode only),
        # True  -> a real LaTeX install (text-mode commands, \textcolor,
        # custom packages via the preamble).
        self.use_real_latex = False
        self.latex_preamble = DEFAULT_LATEX_PREAMBLE
        self.legend_text_color = QtGui.QColor(255, 255, 255, 255)
        self.legend_bg_color = QtGui.QColor(0, 0, 0)

        # "t=x.x" readout
        self.timestamp = dict(DEFAULT_TIMESTAMP)
        self.ts_text_color = QtGui.QColor(255, 255, 255, 255)
        self.ts_bg_color = QtGui.QColor(0, 0, 0)

        # last LaTeX render issue already shown in the status bar, so a
        # persistent problem is reported once rather than on every redraw
        self._shown_latex_issue = None

        self._cam_spins = {}
        self._build()
        self.setAcceptDrops(True)
        self.statusBar().showMessage(
            "Open a video, a trajectory (solid line, can be time-sectioned) "
            "and/or a reference (dotted line, always shown in full). Nudge "
            "azimuth / elevation / FOV / position until the overlay lands "
            "on the real path. If the camera moves, set keyframes as it "
            "drifts.")

    # -- layout -------------------------------------------------------
    def _build(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)

        # -- left: video + scrub ---------------------------------------
        left = QtWidgets.QVBoxLayout()

        top = QtWidgets.QHBoxLayout()
        btn_video = QtWidgets.QPushButton("Open video…")
        btn_video.clicked.connect(self.open_video)
        btn_ref = QtWidgets.QPushButton("Open reference…")
        btn_ref.setToolTip(
            "Planned/ideal path to compare against - dotted line, always "
            "drawn in full")
        btn_ref.clicked.connect(self.open_ref)
        self.lbl_files = QtWidgets.QLabel(
            "<no video>  ·  traj: <none>  ·  ref: <none>")
        self.lbl_files.setStyleSheet("color:#555;")
        top.addWidget(btn_video)
        top.addWidget(btn_ref)
        top.addWidget(self.lbl_files, 1)
        left.addLayout(top)

        self.view = OverlayView()
        left.addWidget(self.view, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_frame_changed)
        left.addWidget(self.slider)

        row = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        btn_prev = QtWidgets.QPushButton("◀ step")
        btn_next = QtWidgets.QPushButton("step ▶")
        btn_prev.clicked.connect(lambda: self._step(-1))
        btn_next.clicked.connect(lambda: self._step(+1))
        self.spin_offset = QtWidgets.QDoubleSpinBox()
        self.spin_offset.setRange(-100000.0, 100000.0)
        self.spin_offset.setDecimals(3)
        self.spin_offset.setSingleStep(0.01)
        self.spin_offset.setPrefix("path sync offset ")
        self.spin_offset.setSuffix(" s")
        self.spin_offset.valueChanged.connect(self._on_offset_changed)
        btn_from_trimspec = QtWidgets.QPushButton("Sync from trimspec…")
        btn_from_trimspec.setToolTip(
            "Read the in-point offset from a trim_marker.py .trimspec.json")
        btn_from_trimspec.clicked.connect(self._sync_from_trimspec)
        btn_save_frame = QtWidgets.QPushButton("Save frame…")
        btn_save_frame.setToolTip(
            "Save the currently shown frame, with the overlay baked in, "
            "as a PNG or JPG (shortcut: P)")
        btn_save_frame.clicked.connect(self.save_current_frame)
        for w in (self.btn_play, btn_prev, btn_next, self.spin_offset,
                  btn_from_trimspec, btn_save_frame):
            row.addWidget(w)
        row.addStretch(1)
        left.addLayout(row)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("font-family:monospace; color:#333;")
        left.addWidget(self.lbl_info)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)

        # -- right: calibration panel (scrollable, collapsible sections) --
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(8, 4, 8, 4)
        right_content = QtWidgets.QWidget()
        right_content.setLayout(right)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # AsNeeded rather than AlwaysOff: if this panel is ever forced
        # narrower than its contents need (e.g. a window manager snapping
        # the window to a smaller monitor/tile on a drag-move, which can
        # squeeze a widget below its layout's preferred size), a scrollbar
        # appears instead of button rows clipping/overlapping.
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(right_content)

        right_host = QtWidgets.QWidget()
        right_host_lay = QtWidgets.QVBoxLayout(right_host)
        right_host_lay.setContentsMargins(0, 0, 0, 0)
        right_host_lay.addWidget(scroll)
        # a floor (so it can't collapse to nothing) but no hard ceiling -
        # button rows have grown since this was first set, and a fixed max
        # here fought the splitter/window resizing instead of adapting to
        # it. Drag the splitter handle for more width if a row still wraps.
        right_host.setMinimumWidth(340)

        box_cam = CollapsibleBox("Camera", expanded=True)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for key, label, lo, hi, step, dec in CAM_FIELDS:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSingleStep(step)
            sb.setValue(self.cam[key])
            sb.valueChanged.connect(
                lambda v, k=key: self._on_cam_changed(k, v))
            form.addRow(label, sb)
            self._cam_spins[key] = sb
        cam_container = QtWidgets.QVBoxLayout()
        cam_container.addLayout(form)
        btn_flip180 = QtWidgets.QPushButton("Flip track 180° (keep camera)")
        btn_flip180.setToolTip(
            "Rotates the track 180° and automatically compensates "
            "path_offset_x / path_offset_y / path_z so its on-screen "
            "position doesn't jump - no need to re-tune the camera. Uses "
            "the combined centroid of whichever of trajectory/reference "
            "are loaded.")
        btn_flip180.clicked.connect(self._flip_track_180)
        cam_container.addWidget(btn_flip180)
        box_cam.setContentLayout(cam_container)
        right.addWidget(box_cam)

        # -- video layers -------------------------------------------------
        box_layers = CollapsibleBox("Video layers", expanded=False)
        ll = QtWidgets.QVBoxLayout()
        ll_hint = QtWidgets.QLabel(
            "Extra videos composited on top of the base video above (and "
            "underneath the grid/trajectories/legend/timestamp, which "
            "always draw on top of the final composite). Each layer "
            "contributes one fixed frame, picked in its own scrubber "
            "window, blended at its own opacity - handy for ghosting "
            "another run's footage over the current one. Draw order "
            "follows this list top to bottom - entries lower in the list "
            "are composited on top of ones above them.")
        ll_hint.setWordWrap(True)
        ll_hint.setStyleSheet("color:#666; font-size:11px;")
        ll.addWidget(ll_hint)

        self.list_layers = QtWidgets.QListWidget()
        self.list_layers.setMaximumHeight(120)
        self.list_layers.setToolTip(
            "Checkbox toggles visibility. Select a row to edit its opacity "
            "or pick its frame below.")
        self.list_layers.itemChanged.connect(self._on_layer_item_changed)
        self.list_layers.currentItemChanged.connect(
            lambda *_: self._refresh_layer_style_controls())
        ll.addWidget(self.list_layers)

        row_lay1 = QtWidgets.QHBoxLayout()
        btn_layer_add = QtWidgets.QPushButton("+ Add video layer…")
        btn_layer_add.setToolTip(
            "Add one or more videos as layers (multi-select supported)")
        btn_layer_add.clicked.connect(self.open_video_layers)
        btn_layer_del = QtWidgets.QPushButton("Remove")
        btn_layer_del.clicked.connect(self._remove_video_layer)
        row_lay1.addWidget(btn_layer_add)
        row_lay1.addWidget(btn_layer_del)
        ll.addLayout(row_lay1)

        row_lay2 = QtWidgets.QHBoxLayout()
        btn_layer_up = QtWidgets.QPushButton("▲")
        btn_layer_up.setToolTip("Move selected layer up (composited earlier)")
        btn_layer_up.clicked.connect(lambda: self._move_video_layer(-1))
        btn_layer_dn = QtWidgets.QPushButton("▼")
        btn_layer_dn.setToolTip(
            "Move selected layer down (composited later, on top)")
        btn_layer_dn.clicked.connect(lambda: self._move_video_layer(+1))
        row_lay2.addWidget(btn_layer_up)
        row_lay2.addWidget(btn_layer_dn)
        ll.addLayout(row_lay2)

        row_lay3 = QtWidgets.QHBoxLayout()
        self.btn_layer_pick_frame = QtWidgets.QPushButton("Pick frame…")
        self.btn_layer_pick_frame.setToolTip(
            "Open a scrubber window to choose which frame of this layer's "
            "video gets composited")
        self.btn_layer_pick_frame.clicked.connect(self._pick_layer_frame)
        self.spin_layer_opacity = QtWidgets.QDoubleSpinBox()
        self.spin_layer_opacity.setRange(0.0, 100.0)
        self.spin_layer_opacity.setDecimals(0)
        self.spin_layer_opacity.setSingleStep(5.0)
        self.spin_layer_opacity.setSuffix(" %")
        self.spin_layer_opacity.valueChanged.connect(
            self._on_layer_opacity_changed)
        row_lay3.addWidget(self.btn_layer_pick_frame)
        row_lay3.addWidget(QtWidgets.QLabel("  opacity"))
        row_lay3.addWidget(self.spin_layer_opacity)
        row_lay3.addStretch(1)
        ll.addLayout(row_lay3)

        row_lay4 = QtWidgets.QHBoxLayout()
        self.btn_layer_region = QtWidgets.QPushButton("Select region…")
        self.btn_layer_region.setToolTip(
            "Drag a rectangle on the video to composite this layer only "
            "within it (blend the rest of the frame not at all) - Esc "
            "cancels the drag")
        self.btn_layer_region.clicked.connect(self._start_layer_region_select)
        self.btn_layer_region_clear = QtWidgets.QPushButton("Clear region")
        self.btn_layer_region_clear.setToolTip(
            "Go back to compositing this layer over the whole frame")
        self.btn_layer_region_clear.clicked.connect(self._clear_layer_region)
        row_lay4.addWidget(self.btn_layer_region)
        row_lay4.addWidget(self.btn_layer_region_clear)
        row_lay4.addStretch(1)
        ll.addLayout(row_lay4)

        self.lbl_layer_region = QtWidgets.QLabel("")
        self.lbl_layer_region.setWordWrap(True)
        self.lbl_layer_region.setStyleSheet("color:#555; font-size:11px;")
        ll.addWidget(self.lbl_layer_region)

        box_layers.setContentLayout(ll)
        right.addWidget(box_layers)
        self._refresh_layer_style_controls()

        box_disp = CollapsibleBox("Display", expanded=True)
        disp_lay = QtWidgets.QVBoxLayout()
        disp_lay.setSpacing(4)

        self.chk_endpoints = QtWidgets.QCheckBox("Show start/end stars (green / red)")
        self.chk_endpoints.setChecked(self.show_endpoints)
        self.chk_endpoints.toggled.connect(self._on_toggle_endpoints)
        disp_lay.addWidget(self.chk_endpoints)

        row_ends = QtWidgets.QHBoxLayout()
        self.btn_start_color = QtWidgets.QPushButton()
        self.btn_start_color.setFixedSize(30, 22)
        self.btn_start_color.setToolTip("Start star color")
        self.btn_start_color.clicked.connect(lambda: self._pick_color("start"))
        self.btn_end_color = QtWidgets.QPushButton()
        self.btn_end_color.setFixedSize(30, 22)
        self.btn_end_color.setToolTip("End star color")
        self.btn_end_color.clicked.connect(lambda: self._pick_color("end"))
        self.spin_endpoint_size = QtWidgets.QDoubleSpinBox()
        self.spin_endpoint_size.setRange(1.0, 400.0)
        self.spin_endpoint_size.setDecimals(1)
        self.spin_endpoint_size.setSingleStep(1.0)
        self.spin_endpoint_size.setSuffix(" px")
        self.spin_endpoint_size.setToolTip(
            "Star tip radius, in full-resolution video pixels - the preview "
            "is downscaled, so this usually wants to be big")
        self.spin_endpoint_size.setValue(self.endpoint_size)
        self.spin_endpoint_size.valueChanged.connect(self._on_endpoint_size)
        row_ends.addWidget(QtWidgets.QLabel("   start / end / size"))
        row_ends.addWidget(self.btn_start_color)
        row_ends.addWidget(self.btn_end_color)
        row_ends.addWidget(self.spin_endpoint_size)
        row_ends.addStretch(1)
        disp_lay.addLayout(row_ends)

        self.chk_ref = QtWidgets.QCheckBox("Show reference (dotted)")
        self.chk_ref.setChecked(True)
        self.chk_ref.toggled.connect(self._on_toggle_ref)
        disp_lay.addWidget(self.chk_ref)

        row_ref_style = QtWidgets.QHBoxLayout()
        self.btn_ref_color = QtWidgets.QPushButton()
        self.btn_ref_color.setFixedSize(30, 22)
        self.btn_ref_color.setToolTip("Reference line color")
        self.btn_ref_color.clicked.connect(lambda: self._pick_color("ref"))
        self.spin_ref_width = QtWidgets.QDoubleSpinBox()
        self.spin_ref_width.setRange(0.5, 20.0)
        self.spin_ref_width.setSingleStep(0.5)
        self.spin_ref_width.setDecimals(1)
        self.spin_ref_width.setSuffix(" px")
        self.spin_ref_width.setValue(self.ref_width)
        self.spin_ref_width.valueChanged.connect(self._on_ref_width)
        row_ref_style.addWidget(QtWidgets.QLabel("   color / width"))
        row_ref_style.addWidget(self.btn_ref_color)
        row_ref_style.addWidget(self.spin_ref_width)
        row_ref_style.addStretch(1)
        disp_lay.addLayout(row_ref_style)
        disp_lay.addLayout(self._build_segment_controls("ref"))

        self.chk_point = QtWidgets.QCheckBox("Show current-position marker")
        self.chk_point.setChecked(True)
        self.chk_point.toggled.connect(self._on_toggle_point)
        disp_lay.addWidget(self.chk_point)

        self.chk_grid = QtWidgets.QCheckBox("Show ground grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self._on_toggle_grid)
        disp_lay.addWidget(self.chk_grid)

        row_grid = QtWidgets.QHBoxLayout()
        self.spin_grid = QtWidgets.QDoubleSpinBox()
        self.spin_grid.setRange(0.01, 10000.0)
        self.spin_grid.setValue(self.grid_spacing)
        self.spin_grid.setSuffix(" units")
        self.spin_grid.valueChanged.connect(self._on_grid_spacing)
        row_grid.addWidget(QtWidgets.QLabel("   grid spacing"))
        row_grid.addWidget(self.spin_grid)
        row_grid.addStretch(1)
        disp_lay.addLayout(row_grid)

        box_disp.setContentLayout(disp_lay)
        right.addWidget(box_disp)

        # -- trajectories -------------------------------------------------
        box_multi = CollapsibleBox("Trajectories", expanded=True)
        ml = QtWidgets.QVBoxLayout()
        ml_hint = QtWidgets.QLabel(
            "All trajectories (solid lines) live in this one list - add as "
            "many as you like to overlay several rollouts on the same "
            "video/track. Select a row to edit its color/width/segments/"
            "trim window below; all of it applies to whichever trajectory "
            "is selected. They share the sync offset above and the global "
            "current-position marker toggle.")
        ml_hint.setWordWrap(True)
        ml_hint.setStyleSheet("color:#666; font-size:11px;")
        ml.addWidget(ml_hint)

        self.list_trajs = QtWidgets.QListWidget()
        self.list_trajs.setMaximumHeight(120)
        self.list_trajs.setToolTip(
            "Checkbox toggles visibility. Select a row to edit it below. "
            "Draw order follows this list top to bottom - entries lower "
            "in the list are drawn on top of ones above them.")
        self.list_trajs.itemChanged.connect(self._on_traj_item_changed)
        self.list_trajs.currentItemChanged.connect(
            lambda *_: self._refresh_traj_style_controls())
        ml.addWidget(self.list_trajs)

        row_traj1 = QtWidgets.QHBoxLayout()
        btn_traj_add = QtWidgets.QPushButton("+ Add trajectory…")
        btn_traj_add.setToolTip(
            "Add one or more paths as trajectories (multi-select supported)")
        btn_traj_add.clicked.connect(self.open_trajectories)
        btn_traj_del = QtWidgets.QPushButton("Remove")
        btn_traj_del.clicked.connect(self._remove_trajectory)
        row_traj1.addWidget(btn_traj_add)
        row_traj1.addWidget(btn_traj_del)
        ml.addLayout(row_traj1)

        row_traj2 = QtWidgets.QHBoxLayout()
        btn_traj_up = QtWidgets.QPushButton("▲")
        btn_traj_up.setToolTip("Move selected trajectory up (drawn earlier)")
        btn_traj_up.clicked.connect(lambda: self._move_trajectory(-1))
        btn_traj_dn = QtWidgets.QPushButton("▼")
        btn_traj_dn.setToolTip("Move selected trajectory down (drawn later, on top)")
        btn_traj_dn.clicked.connect(lambda: self._move_trajectory(+1))
        self.btn_traj_color = QtWidgets.QPushButton()
        self.btn_traj_color.setFixedSize(30, 22)
        self.btn_traj_color.setToolTip("Selected trajectory's line color")
        self.btn_traj_color.clicked.connect(lambda: self._pick_traj_color("line"))
        self.spin_traj_width = QtWidgets.QDoubleSpinBox()
        self.spin_traj_width.setRange(0.5, 20.0)
        self.spin_traj_width.setSingleStep(0.5)
        self.spin_traj_width.setDecimals(1)
        self.spin_traj_width.setSuffix(" px")
        self.spin_traj_width.valueChanged.connect(self._on_traj_width_changed)
        row_traj2.addWidget(btn_traj_up)
        row_traj2.addWidget(btn_traj_dn)
        row_traj2.addWidget(QtWidgets.QLabel("  line color/width"))
        row_traj2.addWidget(self.btn_traj_color)
        row_traj2.addWidget(self.spin_traj_width)
        row_traj2.addStretch(1)
        ml.addLayout(row_traj2)

        row_traj3 = QtWidgets.QHBoxLayout()
        self.btn_traj_start_color = QtWidgets.QPushButton()
        self.btn_traj_start_color.setFixedSize(30, 22)
        self.btn_traj_start_color.setToolTip("Selected trajectory's start star color")
        self.btn_traj_start_color.clicked.connect(
            lambda: self._pick_traj_color("start"))
        self.btn_traj_end_color = QtWidgets.QPushButton()
        self.btn_traj_end_color.setFixedSize(30, 22)
        self.btn_traj_end_color.setToolTip("Selected trajectory's end star color")
        self.btn_traj_end_color.clicked.connect(
            lambda: self._pick_traj_color("end"))
        row_traj3.addWidget(QtWidgets.QLabel("  start / end star color"))
        row_traj3.addWidget(self.btn_traj_start_color)
        row_traj3.addWidget(self.btn_traj_end_color)
        row_traj3.addStretch(1)
        ml.addLayout(row_traj3)

        ml.addLayout(self._build_segment_controls("traj"))

        ml.addSpacing(6)
        self.chk_trim = QtWidgets.QCheckBox(
            "Limit selected trajectory to a time window")
        self.chk_trim.setEnabled(False)
        self.chk_trim.toggled.connect(self._on_trim_toggle)
        ml.addWidget(self.chk_trim)

        row_start = QtWidgets.QHBoxLayout()
        self.spin_trim_start = QtWidgets.QDoubleSpinBox()
        self.spin_trim_start.setDecimals(3)
        self.spin_trim_start.setRange(-1e6, 1e6)
        self.spin_trim_start.setSuffix(" s")
        self.spin_trim_start.valueChanged.connect(self._on_trim_start)
        btn_start_here = QtWidgets.QPushButton("= playhead")
        btn_start_here.setToolTip(
            "Set start to this trajectory's own time at the current video "
            "frame (video time minus the sync offset above)")
        btn_start_here.clicked.connect(self._set_trim_start_to_playhead)
        row_start.addWidget(QtWidgets.QLabel("   start"))
        row_start.addWidget(self.spin_trim_start)
        row_start.addWidget(btn_start_here)
        ml.addLayout(row_start)

        row_end = QtWidgets.QHBoxLayout()
        self.spin_trim_end = QtWidgets.QDoubleSpinBox()
        self.spin_trim_end.setDecimals(3)
        self.spin_trim_end.setRange(-1e6, 1e6)
        self.spin_trim_end.setSuffix(" s")
        self.spin_trim_end.valueChanged.connect(self._on_trim_end)
        btn_end_here = QtWidgets.QPushButton("= playhead")
        btn_end_here.setToolTip(
            "Set end to this trajectory's own time at the current video "
            "frame (video time minus the sync offset above)")
        btn_end_here.clicked.connect(self._set_trim_end_to_playhead)
        row_end.addWidget(QtWidgets.QLabel("   end"))
        row_end.addWidget(self.spin_trim_end)
        row_end.addWidget(btn_end_here)
        ml.addLayout(row_end)

        self.lbl_trim_range = QtWidgets.QLabel("select a trajectory above")
        self.lbl_trim_range.setWordWrap(True)
        self.lbl_trim_range.setStyleSheet("color:#555; font-size:11px;")
        ml.addWidget(self.lbl_trim_range)

        box_multi.setContentLayout(ml)
        right.addWidget(box_multi)
        self._refresh_traj_style_controls()

        # -- legend ----------------------------------------------------
        box_leg = CollapsibleBox("Legend", expanded=False)
        fl = QtWidgets.QVBoxLayout()

        self.chk_legend = QtWidgets.QCheckBox("Draw legend")
        self.chk_legend.setChecked(self.legend["enabled"])
        self.chk_legend.toggled.connect(self._on_legend_enabled)
        fl.addWidget(self.chk_legend)

        leg_form = QtWidgets.QFormLayout()
        leg_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.combo_leg_corner = QtWidgets.QComboBox()
        self.combo_leg_corner.addItems(LEGEND_CORNERS)
        self.combo_leg_corner.setCurrentText(self.legend["corner"])
        self.combo_leg_corner.currentTextChanged.connect(
            lambda v: self._on_legend_key("corner", v))
        leg_form.addRow("Corner", self.combo_leg_corner)

        self.spin_leg_cols = QtWidgets.QSpinBox()
        self.spin_leg_cols.setRange(1, 8)
        self.spin_leg_cols.setValue(int(self.legend["columns"]))
        self.spin_leg_cols.setToolTip(
            "Split the legend into this many columns. Entries are dealt out "
            "column by column, top to bottom, in list order - so with 6 "
            "entries and 2 columns, rows 1-3 go in the left column and rows "
            "4-6 in the right.")
        self.spin_leg_cols.valueChanged.connect(
            lambda v: self._on_legend_key("columns", int(v)))
        leg_form.addRow("Columns", self.spin_leg_cols)

        self._leg_spins = {}
        for key, label, lo, hi, step in (
            ("text_size",  "Text size",        1.0, 1000.0, 4.0),
            ("sample_len", "Swatch length",    1.0, 2000.0, 10.0),
            ("margin",     "Edge margin",      0.0, 2000.0, 5.0),
            ("padding",    "Inner padding",    0.0, 1000.0, 2.0),
            ("gap",        "Swatch-text gap",  0.0, 1000.0, 2.0),
            ("row_gap",    "Row gap",          0.0, 1000.0, 2.0),
            ("col_gap",    "Column gap",       0.0, 2000.0, 5.0),
            ("bg_alpha",   "Backdrop alpha",   0.0,  255.0, 10.0),
        ):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(1)
            sb.setSingleStep(step)
            sb.setValue(float(self.legend[key]))
            if key != "bg_alpha":
                sb.setSuffix(" px")
            sb.valueChanged.connect(lambda v, k=key: self._on_legend_key(k, v))
            leg_form.addRow(label, sb)
            self._leg_spins[key] = sb

        self.chk_leg_bold = QtWidgets.QCheckBox("Bold text")
        self.chk_leg_bold.setChecked(bool(self.legend["bold"]))
        self.chk_leg_bold.toggled.connect(
            lambda v: self._on_legend_key("bold", bool(v)))
        leg_form.addRow("", self.chk_leg_bold)

        self.chk_usetex = QtWidgets.QCheckBox("Use a real LaTeX install (usetex)")
        self.chk_usetex.setChecked(self.use_real_latex)
        self.chk_usetex.setEnabled(usetex_available())
        self.chk_usetex.setToolTip(
            "Off: labels marked LaTeX go through matplotlib's mathtext - no "
            "extra dependencies, but math mode only (\\mathbf, \\frac, "
            "\\theta).\n"
            "On: they're typeset by an actual LaTeX install, so text-mode "
            "commands work as written - \\textbf, \\texttt, \\textcolor, "
            "and anything your preamble's packages provide. Needs `latex` "
            "and `dvipng` on PATH; slower on first render per unique string.\n"
            "Applies to both legend labels and the timestamp.")
        self.chk_usetex.toggled.connect(self._on_usetex_toggled)
        leg_form.addRow("", self.chk_usetex)

        self.edit_preamble = QtWidgets.QLineEdit(self.latex_preamble)
        self.edit_preamble.setEnabled(usetex_available() and self.use_real_latex)
        self.edit_preamble.setToolTip(
            "LaTeX preamble used in usetex mode - add \\usepackage lines "
            "here for any packages your labels need.")
        self.edit_preamble.textChanged.connect(self._on_preamble_changed)
        leg_form.addRow("Preamble", self.edit_preamble)

        usetex_reason = usetex_missing_reason()
        if usetex_reason:
            lbl_tex = QtWidgets.QLabel(
                f"usetex unavailable: {usetex_reason} - mathtext only. "
                "Needs both `latex` and `dvipng` (not just any LaTeX "
                "install - dvipng is often a separate package), visible "
                "on the PATH this app itself sees - which can differ "
                "from a terminal's PATH depending on how it was "
                "launched.")
            lbl_tex.setWordWrap(True)
            lbl_tex.setStyleSheet("color:#a33; font-size:11px;")
            leg_form.addRow("", lbl_tex)

        self.btn_leg_text_color = QtWidgets.QPushButton()
        self.btn_leg_text_color.setFixedSize(30, 22)
        self.btn_leg_text_color.clicked.connect(lambda: self._pick_color("legend_text"))
        self.btn_leg_bg_color = QtWidgets.QPushButton()
        self.btn_leg_bg_color.setFixedSize(30, 22)
        self.btn_leg_bg_color.clicked.connect(lambda: self._pick_color("legend_bg"))
        row_leg_colors = QtWidgets.QHBoxLayout()
        row_leg_colors.addWidget(self.btn_leg_text_color)
        row_leg_colors.addWidget(self.btn_leg_bg_color)
        row_leg_colors.addStretch(1)
        leg_form.addRow("Text / backdrop", row_leg_colors)
        fl.addLayout(leg_form)

        self.list_legend = QtWidgets.QListWidget()
        self.list_legend.setMaximumHeight(120)
        self.list_legend.setToolTip(
            "Legend rows, drawn top to bottom (and left to right once you "
            "use more than one column). These are free-form - an entry "
            "doesn't have to correspond to a line that's actually on screen. "
            "Labels can be LaTeX. Double-click to edit.")
        self.list_legend.itemDoubleClicked.connect(
            lambda _i: self._edit_legend_entry())
        fl.addWidget(self.list_legend)

        row_leg = QtWidgets.QHBoxLayout()
        btn_leg_add = QtWidgets.QPushButton("+ entry")
        btn_leg_add.clicked.connect(self._add_legend_entry)
        btn_leg_edit = QtWidgets.QPushButton("Edit…")
        btn_leg_edit.clicked.connect(self._edit_legend_entry)
        btn_leg_del = QtWidgets.QPushButton("Remove")
        btn_leg_del.clicked.connect(self._remove_legend_entry)
        row_leg.addWidget(btn_leg_add)
        row_leg.addWidget(btn_leg_edit)
        row_leg.addWidget(btn_leg_del)
        fl.addLayout(row_leg)

        row_leg2 = QtWidgets.QHBoxLayout()
        btn_leg_up = QtWidgets.QPushButton("▲")
        btn_leg_up.setToolTip("Move selected entry up")
        btn_leg_up.clicked.connect(lambda: self._move_legend_entry(-1))
        btn_leg_dn = QtWidgets.QPushButton("▼")
        btn_leg_dn.setToolTip("Move selected entry down")
        btn_leg_dn.clicked.connect(lambda: self._move_legend_entry(+1))
        btn_leg_auto = QtWidgets.QPushButton("Add from current")
        btn_leg_auto.setToolTip(
            "Seed the legend with rows matching what's loaded right now: "
            "trajectory, reference, and the start/end stars")
        btn_leg_auto.clicked.connect(self._seed_legend_from_current)
        row_leg2.addWidget(btn_leg_up)
        row_leg2.addWidget(btn_leg_dn)
        row_leg2.addWidget(btn_leg_auto)
        row_leg2.addStretch(1)
        fl.addLayout(row_leg2)

        leg_hint = QtWidgets.QLabel(
            "All sizes are full-resolution video pixels. The preview is "
            "scaled down to fit the window, so on a 4K frame a readable "
            "legend usually needs text around 60-150px and swatch widths "
            "around 8-30px. Check with 'Save frame…' (P) if unsure. "
            "Labels can use LaTeX per entry: tick the LaTeX box in the "
            "entry dialog, then write mathtext (\"$v_{max}$\") or, with "
            "usetex on, plain LaTeX like \"\\textbf{\\texttt{...}}\"."
            + ("" if latex_available()
               else "  [matplotlib not installed - LaTeX unavailable]"))
        leg_hint.setWordWrap(True)
        leg_hint.setStyleSheet("color:#666; font-size:11px;")
        fl.addWidget(leg_hint)

        box_leg.setContentLayout(fl)
        right.addWidget(box_leg)

        # -- timestamp -------------------------------------------------
        box_ts = CollapsibleBox("Timestamp (t=x.x)", expanded=False)
        fts = QtWidgets.QVBoxLayout()

        self.chk_ts = QtWidgets.QCheckBox("Draw timestamp")
        self.chk_ts.setChecked(bool(self.timestamp["enabled"]))
        self.chk_ts.toggled.connect(
            lambda v: self._on_ts_key("enabled", bool(v)))
        fts.addWidget(self.chk_ts)

        ts_form = QtWidgets.QFormLayout()
        ts_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.combo_ts_corner = QtWidgets.QComboBox()
        self.combo_ts_corner.addItems(LEGEND_CORNERS)
        self.combo_ts_corner.setCurrentText(self.timestamp["corner"])
        self.combo_ts_corner.currentTextChanged.connect(
            lambda v: self._on_ts_key("corner", v))
        ts_form.addRow("Corner", self.combo_ts_corner)

        self.combo_ts_source = QtWidgets.QComboBox()
        self.combo_ts_source.addItems(TIMESTAMP_SOURCES)
        self.combo_ts_source.setCurrentText(self.timestamp["source"])
        self.combo_ts_source.setToolTip(
            "'video' shows the raw video clock; 'trajectory' shows the "
            "path's own clock, i.e. video time minus the sync offset - the "
            "same time base as the trim window and color segments.")
        self.combo_ts_source.currentTextChanged.connect(
            lambda v: self._on_ts_key("source", v))
        ts_form.addRow("Clock", self.combo_ts_source)

        self.edit_ts_prefix = QtWidgets.QLineEdit(self.timestamp["prefix"])
        self.edit_ts_prefix.textChanged.connect(
            lambda v: self._on_ts_key("prefix", v))
        ts_form.addRow("Prefix", self.edit_ts_prefix)

        self.edit_ts_suffix = QtWidgets.QLineEdit(self.timestamp["suffix"])
        self.edit_ts_suffix.textChanged.connect(
            lambda v: self._on_ts_key("suffix", v))
        ts_form.addRow("Suffix", self.edit_ts_suffix)

        self.spin_ts_dec = QtWidgets.QSpinBox()
        self.spin_ts_dec.setRange(0, 6)
        self.spin_ts_dec.setValue(int(self.timestamp["decimals"]))
        self.spin_ts_dec.valueChanged.connect(
            lambda v: self._on_ts_key("decimals", int(v)))
        ts_form.addRow("Decimals", self.spin_ts_dec)

        self._ts_spins = {}
        for key, label, lo, hi, step in (
            ("text_size", "Text size",      1.0, 1000.0, 4.0),
            ("margin",    "Edge margin",    0.0, 2000.0, 5.0),
            ("padding",   "Inner padding",  0.0, 1000.0, 2.0),
            ("bg_alpha",  "Backdrop alpha", 0.0,  255.0, 10.0),
        ):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(1)
            sb.setSingleStep(step)
            sb.setValue(float(self.timestamp[key]))
            if key != "bg_alpha":
                sb.setSuffix(" px")
            sb.valueChanged.connect(lambda v, k=key: self._on_ts_key(k, v))
            ts_form.addRow(label, sb)
            self._ts_spins[key] = sb

        self.chk_ts_bold = QtWidgets.QCheckBox("Bold text")
        self.chk_ts_bold.setChecked(bool(self.timestamp["bold"]))
        self.chk_ts_bold.toggled.connect(
            lambda v: self._on_ts_key("bold", bool(v)))
        ts_form.addRow("", self.chk_ts_bold)

        self.chk_ts_latex = QtWidgets.QCheckBox("Render as LaTeX")
        self.chk_ts_latex.setChecked(bool(self.timestamp["latex"]))
        self.chk_ts_latex.setEnabled(latex_available())
        self.chk_ts_latex.setToolTip(
            "Must be ticked for the timestamp to use LaTeX at all - the "
            "Legend panel's \"Use a real LaTeX install (usetex)\" toggle "
            "(shared with legend labels) only takes effect here too if "
            "this is checked.\n\n"
            "Without usetex (mathtext): the assembled string (prefix + "
            "number + suffix) is passed through mathtext, so a prefix "
            "like \"$t=$\" or \"$\\tau=$\" renders as math.\n"
            "With usetex: the assembled string is plain LaTeX directly, "
            "no $...$ needed.")
        self.chk_ts_latex.toggled.connect(
            lambda v: self._on_ts_key("latex", bool(v)))
        ts_form.addRow("", self.chk_ts_latex)

        self.btn_ts_text_color = QtWidgets.QPushButton()
        self.btn_ts_text_color.setFixedSize(30, 22)
        self.btn_ts_text_color.clicked.connect(lambda: self._pick_color("ts_text"))
        self.btn_ts_bg_color = QtWidgets.QPushButton()
        self.btn_ts_bg_color.setFixedSize(30, 22)
        self.btn_ts_bg_color.clicked.connect(lambda: self._pick_color("ts_bg"))
        row_ts_colors = QtWidgets.QHBoxLayout()
        row_ts_colors.addWidget(self.btn_ts_text_color)
        row_ts_colors.addWidget(self.btn_ts_bg_color)
        row_ts_colors.addStretch(1)
        ts_form.addRow("Text / backdrop", row_ts_colors)
        fts.addLayout(ts_form)

        ts_hint = QtWidgets.QLabel(
            "Defaults to the bottom-left corner. Sizes are full-resolution "
            "video pixels, same as the legend - on a 4K frame, ~60-120px "
            "text reads well.")
        ts_hint.setWordWrap(True)
        ts_hint.setStyleSheet("color:#666; font-size:11px;")
        fts.addWidget(ts_hint)

        box_ts.setContentLayout(fts)
        right.addWidget(box_ts)

        self._refresh_line_style_swatches()


        box_kf = CollapsibleBox("Keyframes (camera motion)", expanded=False)
        fk = QtWidgets.QVBoxLayout()
        self.lbl_kf_status = QtWidgets.QLabel("no keyframes - camera fixed for whole video")
        self.lbl_kf_status.setWordWrap(True)
        self.lbl_kf_status.setStyleSheet("color:#555; font-size:11px;")
        self.list_kf = QtWidgets.QListWidget()
        self.list_kf.setMaximumHeight(110)
        self.list_kf.itemDoubleClicked.connect(self._goto_keyframe_item)
        row_kf = QtWidgets.QHBoxLayout()
        btn_kf_set = QtWidgets.QPushButton("Set keyframe here")
        btn_kf_set.clicked.connect(self._set_keyframe_here)
        btn_kf_del = QtWidgets.QPushButton("Delete selected")
        btn_kf_del.clicked.connect(self._delete_selected_keyframe)
        row_kf.addWidget(btn_kf_set)
        row_kf.addWidget(btn_kf_del)
        fk.addWidget(self.lbl_kf_status)
        fk.addWidget(self.list_kf)
        fk.addLayout(row_kf)
        box_kf.setContentLayout(fk)
        right.addWidget(box_kf)

        box_cal = CollapsibleBox("Calibration file", expanded=False)
        f3 = QtWidgets.QVBoxLayout()
        btn_save = QtWidgets.QPushButton("Save calibration…")
        btn_save.setStyleSheet("font-weight:bold;")
        btn_save.setToolTip(
            "Pose/keyframes/sync/display/trim only - portable across a "
            "re-encoded video or a renamed path file. Never reopens "
            "video/reference/trajectory/layer files on load; applies on "
            "top of whatever's already loaded.")
        btn_save.clicked.connect(self.save_calibration)
        btn_save_all = QtWidgets.QPushButton("Save all (incl. videos/paths)…")
        btn_save_all.setToolTip(
            "Everything Save calibration saves, plus the actual video/"
            "reference/trajectory/video-layer files themselves - loading "
            "this reopens them all (replacing whatever's currently "
            "loaded), recreating this exact session elsewhere.")
        btn_save_all.clicked.connect(self.save_project)
        btn_load = QtWidgets.QPushButton("Load calibration…")
        btn_load.setToolTip(
            "Works for either kind of saved file above - a \"Save all\" "
            "file additionally reopens its video/reference/trajectory/"
            "layer files; a plain calibration never does.")
        btn_load.clicked.connect(self.load_calibration_dialog)
        btn_reset = QtWidgets.QPushButton("Reset camera to defaults")
        btn_reset.clicked.connect(self.reset_camera)
        f3.addWidget(btn_save)
        f3.addWidget(btn_save_all)
        f3.addWidget(btn_load)
        f3.addWidget(btn_reset)
        box_cal.setContentLayout(f3)
        right.addWidget(box_cal)

        box_hint = CollapsibleBox("Keyboard shortcuts", expanded=False)
        fh = QtWidgets.QVBoxLayout()
        hint = QtWidgets.QLabel(
            "Nudge keys (view focused):\n"
            "  A / D   azimuth -/+\n"
            "  W / S   elevation +/-\n"
            "  Q / E   zoom out/in (FOV)\n"
            "  Z / C   roll -/+\n"
            "  hold Shift for a bigger step\n"
            "  Left/Right   step video frame\n"
            "  K   set keyframe at current frame\n"
            "  P   save current frame (with overlay) as an image\n"
            "  Esc   cancel an in-progress layer region drag")
        hint.setStyleSheet("color:#666; font-size:11px;")
        fh.addWidget(hint)
        box_hint.setContentLayout(fh)
        right.addWidget(box_hint)

        right.addStretch(1)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left_host = QtWidgets.QWidget()
        left_host.setLayout(left)
        split.addWidget(left_host)
        split.addWidget(right_host)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1160, 400])
        root.addWidget(split)
        self.setCentralWidget(central)
        self._shortcuts()

    def _shortcuts(self):
        def sc(key, fn):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
        sc("Left", lambda: self._step(-1))
        sc("Right", lambda: self._step(+1))
        sc("Space", lambda: self.btn_play.toggle())
        sc("A", lambda: self._nudge("azimuth", -1))
        sc("D", lambda: self._nudge("azimuth", +1))
        sc("Shift+A", lambda: self._nudge("azimuth", -1, big=True))
        sc("Shift+D", lambda: self._nudge("azimuth", +1, big=True))
        sc("W", lambda: self._nudge("elevation", +1))
        sc("S", lambda: self._nudge("elevation", -1))
        sc("Shift+W", lambda: self._nudge("elevation", +1, big=True))
        sc("Shift+S", lambda: self._nudge("elevation", -1, big=True))
        sc("Q", lambda: self._nudge("fov", +1))
        sc("E", lambda: self._nudge("fov", -1))
        sc("Shift+Q", lambda: self._nudge("fov", +1, big=True))
        sc("Shift+E", lambda: self._nudge("fov", -1, big=True))
        sc("Z", lambda: self._nudge("roll", -1))
        sc("C", lambda: self._nudge("roll", +1))
        sc("K", self._set_keyframe_here)
        sc("P", self.save_current_frame)
        sc("Escape", self._cancel_region_select)

    def _nudge(self, key, direction, fine=False, big=False):
        step = NUDGE_STEP[key] * (3.0 if big else (0.2 if fine else 1.0))
        self._cam_spins[key].setValue(self.cam[key] + direction * step)

    # -- drag & drop --------------------------------------------------
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if not os.path.exists(p):
                continue
            low = p.lower()
            if low.endswith(SUFFIX):
                self.load_calibration(p)
            elif low.endswith((".npz", ".csv", ".txt")):
                # path drops are added to the trajectory list; use the
                # dedicated "Open reference..." button for the reference
                # slot instead.
                self._add_trajectory(p)
            elif self.video is None:
                self._load_video(p)
            else:
                # the base video slot is already filled; further video
                # drops become extra composited layers instead.
                self._add_video_layer(p)

    # -- open/load ------------------------------------------------------
    def open_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)")
        if p:
            self._load_video(p)

    def open_ref(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open reference", "",
            "Path (*.npz *.csv *.txt);;All files (*)")
        if p:
            self._load_ref(p)

    def _load_video(self, path):
        try:
            src = VideoFrameSource(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Video", str(exc))
            return
        if self.video is not None:
            self.video.release()
        self.video = src
        self.slider.setRange(0, max(0, src.n_frames - 1))
        self.slider.setValue(0)
        self.frame_idx = 0
        self._refresh_title()
        self._apply_interpolated_camera()

    # -- video layers -------------------------------------------------------
    def open_video_layers(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add video layer(s)", "",
            "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)")
        for p in paths:
            self._add_video_layer(p)

    def _add_video_layer(self, path):
        try:
            vid = VideoFrameSource(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Video layer", str(exc))
            return
        layer = {"video": vid, "frame_idx": 0, "opacity": 0.5,
                 "visible": True, "rect": None, "_dialog": None}
        style = self._pending_layer_styles.get(os.path.basename(path))
        if style:
            if "frame_idx" in style:
                layer["frame_idx"] = int(np.clip(
                    style["frame_idx"], 0, vid.n_frames - 1))
            if "opacity" in style:
                layer["opacity"] = float(style["opacity"])
            if "visible" in style:
                layer["visible"] = bool(style["visible"])
            if style.get("rect"):
                layer["rect"] = tuple(int(v) for v in style["rect"])
        self.layers.append(layer)
        self._refresh_layer_list()
        self.list_layers.setCurrentRow(len(self.layers) - 1)
        self._refresh_title()
        self._redraw()

    def _refresh_layer_list(self):
        self.list_layers.blockSignals(True)
        self.list_layers.clear()
        for i, l in enumerate(self.layers):
            name = os.path.basename(l["video"].path)
            region = "" if l["rect"] is None else ", region"
            item = QtWidgets.QListWidgetItem(
                f"{name}  (frame {l['frame_idx']}, "
                f"{round(l['opacity'] * 100)}%{region})")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if l["visible"]
                               else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            self.list_layers.addItem(item)
        self.list_layers.blockSignals(False)
        self._refresh_layer_style_controls()

    def _on_layer_item_changed(self, item):
        i = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if i is None or i >= len(self.layers):
            return
        self.layers[i]["visible"] = (
            item.checkState() == QtCore.Qt.CheckState.Checked)
        self._redraw()

    def _selected_layer_index(self):
        item = self.list_layers.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _selected_layer(self):
        i = self._selected_layer_index()
        if i is None or i >= len(self.layers):
            return None
        return self.layers[i]

    def _refresh_layer_style_controls(self):
        layer = self._selected_layer()
        enabled = layer is not None
        self.spin_layer_opacity.setEnabled(enabled)
        self.btn_layer_pick_frame.setEnabled(enabled)
        self.btn_layer_region.setEnabled(enabled)
        self.btn_layer_region_clear.setEnabled(enabled and layer is not None
                                               and layer.get("rect") is not None)
        if enabled:
            self.spin_layer_opacity.blockSignals(True)
            self.spin_layer_opacity.setValue(layer["opacity"] * 100.0)
            self.spin_layer_opacity.blockSignals(False)
            if layer["rect"] is None:
                self.lbl_layer_region.setText(
                    "blend region: whole frame")
            else:
                x0, y0, x1, y1 = layer["rect"]
                self.lbl_layer_region.setText(
                    f"blend region: ({x0}, {y0}) to ({x1}, {y1}) "
                    "in full-resolution video pixels")
        else:
            self.lbl_layer_region.setText("")

    def _on_layer_opacity_changed(self, v):
        layer = self._selected_layer()
        if layer is None:
            return
        layer["opacity"] = v / 100.0
        i = self._selected_layer_index()
        self._refresh_layer_list()
        self.list_layers.setCurrentRow(i)
        self._redraw()

    def _pick_layer_frame(self):
        layer = self._selected_layer()
        if layer is None:
            return
        if layer["_dialog"] is None:
            layer["_dialog"] = VideoLayerFrameDialog(
                self, layer, self._on_layer_frame_picked)
        layer["_dialog"].show()
        layer["_dialog"].raise_()
        layer["_dialog"].activateWindow()

    def _on_layer_frame_picked(self):
        i = self._selected_layer_index()
        self._refresh_layer_list()
        if i is not None:
            self.list_layers.setCurrentRow(i)
        self._redraw()

    def _start_layer_region_select(self):
        layer = self._selected_layer()
        if layer is None:
            return
        if self.video is None:
            QtWidgets.QMessageBox.information(
                self, "Select region",
                "Open the base video first - the region is drawn on it.")
            return
        i = self._selected_layer_index()
        self.statusBar().showMessage(
            "Drag a rectangle on the video to set this layer's blend "
            "region (Esc to cancel)...", 0)
        self.view.begin_rect_selection(
            lambda rect: self._on_layer_region_selected(i, rect))

    def _on_layer_region_selected(self, layer_index, rect):
        self.statusBar().clearMessage()
        if layer_index is None or layer_index >= len(self.layers):
            return
        if rect is not None:
            self.layers[layer_index]["rect"] = rect
        self._refresh_layer_list()
        self.list_layers.setCurrentRow(layer_index)
        self._redraw()

    def _cancel_region_select(self):
        if self.view._selecting:
            self.view.cancel_rect_selection()
            self.statusBar().clearMessage()

    def _clear_layer_region(self):
        layer = self._selected_layer()
        if layer is None:
            return
        layer["rect"] = None
        i = self._selected_layer_index()
        self._refresh_layer_list()
        self.list_layers.setCurrentRow(i)
        self._redraw()

    def _remove_video_layer(self):
        i = self._selected_layer_index()
        if i is None:
            return
        layer = self.layers.pop(i)
        if layer["_dialog"] is not None:
            layer["_dialog"].setParent(None)
            layer["_dialog"].deleteLater()
        layer["video"].release()
        self._refresh_layer_list()
        self._refresh_title()
        self._redraw()

    def _move_video_layer(self, d):
        i = self._selected_layer_index()
        if i is None:
            return
        j = i + d
        if not (0 <= j < len(self.layers)):
            return
        self.layers[i], self.layers[j] = self.layers[j], self.layers[i]
        self._refresh_layer_list()
        self.list_layers.setCurrentRow(j)
        self._redraw()

    def _load_ref(self, path):
        try:
            src = PathSource(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Reference", str(exc))
            return
        self.ref = src
        self._auto_frame_camera()
        self._refresh_title()
        self._redraw()

    # -- trajectories -----------------------------------------------------
    def open_trajectories(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add trajectory(ies)", "",
            "Path (*.npz *.csv *.txt);;All files (*)")
        for p in paths:
            self._add_trajectory(p)

    def _add_trajectory(self, path):
        try:
            src = PathSource(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Trajectory", str(exc))
            return
        t0, t1 = src.full_time_range() if src.is_timeseries else (0.0, 0.0)
        entry = {
            "path": src,
            "color": QtGui.QColor(*DEFAULT_TRAJ_PALETTE[
                len(self.trajs) % len(DEFAULT_TRAJ_PALETTE)]),
            "width": 2.2,
            "visible": True,
            # per-stretch color/opacity overrides, same format as ref_segments
            "segments": [],
            # start/end star colors - independent per trajectory so several
            # overlaid runs can be told apart at a glance
            "start_color": QtGui.QColor(self.start_color),
            "end_color": QtGui.QColor(self.end_color),
            "trim_enabled": False,
            "trim_start": t0,
            "trim_end": t1,
        }
        style = self._pending_traj_styles.get(os.path.basename(path))
        if style:
            if style.get("color"):
                entry["color"] = QtGui.QColor(style["color"])
            if style.get("width"):
                entry["width"] = float(style["width"])
            if "visible" in style:
                entry["visible"] = bool(style["visible"])
            if style.get("start_color"):
                entry["start_color"] = QtGui.QColor(style["start_color"])
            if style.get("end_color"):
                entry["end_color"] = QtGui.QColor(style["end_color"])
            if src.is_timeseries:
                segs = []
                for s in style.get("segments") or []:
                    try:
                        segs.append({"start": float(s["start"]),
                                     "end": float(s["end"]),
                                     "color": QtGui.QColor(s["color"])})
                    except (KeyError, TypeError, ValueError):
                        continue
                entry["segments"] = segs
                trim = style.get("trim") or {}
                entry["trim_enabled"] = bool(trim.get("enabled", False))
                if "start_s" in trim:
                    entry["trim_start"] = float(trim["start_s"])
                if "end_s" in trim:
                    entry["trim_end"] = float(trim["end_s"])
        self.trajs.append(entry)
        self._refresh_traj_list()
        self.list_trajs.setCurrentRow(len(self.trajs) - 1)
        self._auto_frame_camera()
        self._refresh_title()
        self._redraw()

    def _refresh_traj_list(self):
        self.list_trajs.blockSignals(True)
        self.list_trajs.clear()
        for i, t in enumerate(self.trajs):
            name = os.path.basename(t["path"].path)
            item = QtWidgets.QListWidgetItem(f"{name}  [{t['path'].kind}]")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if t["visible"]
                               else QtCore.Qt.CheckState.Unchecked)
            item.setForeground(QtGui.QBrush(t["color"]))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            self.list_trajs.addItem(item)
        self.list_trajs.blockSignals(False)
        self._refresh_traj_style_controls()

    def _on_traj_item_changed(self, item):
        i = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if i is None or i >= len(self.trajs):
            return
        self.trajs[i]["visible"] = (
            item.checkState() == QtCore.Qt.CheckState.Checked)
        self._redraw()

    def _selected_traj_index(self):
        item = self.list_trajs.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _selected_traj_entry(self):
        i = self._selected_traj_index()
        if i is None or i >= len(self.trajs):
            return None
        return self.trajs[i]

    def _refresh_traj_style_controls(self):
        entry = self._selected_traj_entry()
        enabled = entry is not None
        for w in (self.btn_traj_color, self.spin_traj_width,
                 self.btn_traj_start_color, self.btn_traj_end_color):
            w.setEnabled(enabled)
        if enabled:
            self.btn_traj_color.setStyleSheet(
                f"background:{_css_rgba(entry['color'])}; border:1px solid #333;")
            self.btn_traj_start_color.setStyleSheet(
                f"background:{_css_rgba(entry['start_color'])}; "
                "border:1px solid #333;")
            self.btn_traj_end_color.setStyleSheet(
                f"background:{_css_rgba(entry['end_color'])}; "
                "border:1px solid #333;")
            self.spin_traj_width.blockSignals(True)
            self.spin_traj_width.setValue(entry["width"])
            self.spin_traj_width.blockSignals(False)
        else:
            for btn in (self.btn_traj_color, self.btn_traj_start_color,
                       self.btn_traj_end_color):
                btn.setStyleSheet("border:1px solid #333;")
        self._refresh_segment_list("traj")
        self._update_trim_spin_ranges()

    def _pick_traj_color(self, which="line"):
        entry = self._selected_traj_entry()
        if entry is None:
            return
        key = {"line": "color", "start": "start_color",
              "end": "end_color"}[which]
        c = QtWidgets.QColorDialog.getColor(
            entry[key], self, "Pick color",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if not c.isValid():
            return
        entry[key] = c
        i = self._selected_traj_index()
        self._refresh_traj_list()
        self.list_trajs.setCurrentRow(i)
        self._redraw()

    def _on_traj_width_changed(self, v):
        entry = self._selected_traj_entry()
        if entry is None:
            return
        entry["width"] = v
        self._redraw()

    def _remove_trajectory(self):
        i = self._selected_traj_index()
        if i is None:
            return
        self.trajs.pop(i)
        self._refresh_traj_list()
        self._refresh_title()
        self._redraw()

    def _move_trajectory(self, d):
        i = self._selected_traj_index()
        if i is None:
            return
        j = i + d
        if not (0 <= j < len(self.trajs)):
            return
        self.trajs[i], self.trajs[j] = self.trajs[j], self.trajs[i]
        self._refresh_traj_list()
        self.list_trajs.setCurrentRow(j)
        self._redraw()

    def _combined_xy_bounds(self):
        mins, maxs = [], []
        sources = [self.ref] + [t["path"] for t in self.trajs]
        for src in sources:
            if src is not None and src.n > 0:
                (xmin, ymin), (xmax, ymax) = src.bounds()
                mins.append((xmin, ymin))
                maxs.append((xmax, ymax))
        if not mins:
            return None
        xmin = min(m[0] for m in mins)
        ymin = min(m[1] for m in mins)
        xmax = max(m[0] for m in maxs)
        ymax = max(m[1] for m in maxs)
        return (xmin, ymin), (xmax, ymax)

    def _auto_frame_camera(self):
        """Point a fresh default camera roughly at the loaded path(s')
        combined centroid, so it isn't projecting into empty space before
        the user starts tuning."""
        bounds = self._combined_xy_bounds()
        if bounds is None:
            return
        (xmin, ymin), (xmax, ymax) = bounds
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        span = max(xmax - xmin, ymax - ymin, 1.0)
        # only move the camera if it's still sitting at the untouched default
        # and no keyframes have been set yet
        if self.cam == dict(DEFAULT_CAM) and not self.keyframes:
            self.cam["cam_x"] = cx
            self.cam["cam_y"] = cy - span
            self.cam["cam_z"] = max(span * 0.3, 2.0)
            self.grid_spacing = max(round(span / 20.0, 2), 0.01)
            self.spin_grid.blockSignals(True)
            self.spin_grid.setValue(self.grid_spacing)
            self.spin_grid.blockSignals(False)
            for k in ("cam_x", "cam_y", "cam_z"):
                self._cam_spins[k].blockSignals(True)
                self._cam_spins[k].setValue(self.cam[k])
                self._cam_spins[k].blockSignals(False)

    def _refresh_title(self):
        v = os.path.basename(self.video.path) if self.video else "<no video>"
        r = os.path.basename(self.ref.path) if self.ref else "<none>"
        rk = f" [{self.ref.kind}]" if self.ref else ""
        if self.trajs:
            names = ", ".join(os.path.basename(t["path"].path)
                              for t in self.trajs[:3])
            if len(self.trajs) > 3:
                names += f", +{len(self.trajs) - 3} more"
            t = f"{len(self.trajs)} [{names}]"
        else:
            t = "<none>"
        layers = f"  ·  +{len(self.layers)} layer(s)" if self.layers else ""
        self.lbl_files.setText(f"{v}  ·  traj: {t}  ·  ref: {r}{rk}{layers}")

    # -- transport ------------------------------------------------------
    def _on_frame_changed(self, idx):
        self.frame_idx = idx
        self._apply_interpolated_camera()

    def _step(self, d):
        if self.video is None:
            return
        self.slider.setValue(int(np.clip(self.slider.value() + d, 0,
                                         self.video.n_frames - 1)))

    def _toggle_play(self, on):
        self.btn_play.setText("⏸ Pause" if on else "▶ Play")
        self.timer.start() if on else self.timer.stop()

    def _tick(self):
        if self.video is None:
            return
        self.timer.setInterval(max(10, int(1000.0 / max(self.video.fps, 1.0))))
        nxt = self.slider.value() + 1
        if nxt > self.slider.maximum():
            nxt = 0
        self.slider.setValue(nxt)

    def _on_offset_changed(self, v):
        self.offset_s = v
        self._redraw()

    def _sync_from_trimspec(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load trimspec for sync", "",
            "Trim spec (*.trimspec.json);;JSON (*.json)")
        if not p:
            return
        try:
            with open(p) as f:
                spec = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "trimspec", str(exc))
            return
        off = offset_from_trimspec(spec)
        if off is None:
            QtWidgets.QMessageBox.warning(
                self, "trimspec",
                "That spec doesn't have both a video master trim and an "
                "npz trim with dt - can't derive an offset from it.")
            return
        self.spin_offset.setValue(off)
        self.statusBar().showMessage(f"Sync offset set from {os.path.basename(p)}: "
                                     f"{off:+.3f}s", 6000)

    # -- calibration params ----------------------------------------------
    def _on_cam_changed(self, key, value):
        self.cam[key] = value
        self._redraw()

    def _on_toggle_ref(self, on):
        self.show_ref = on
        self._redraw()

    def _on_toggle_point(self, on):
        self.show_point = on
        self._redraw()

    def _on_toggle_grid(self, on):
        self.show_grid = on
        self._redraw()

    def _on_grid_spacing(self, v):
        self.grid_spacing = v
        self._redraw()

    def _pick_color(self, which):
        attr = {"ref": "ref_color",
                "start": "start_color", "end": "end_color",
                "legend_text": "legend_text_color",
                "legend_bg": "legend_bg_color",
                "ts_text": "ts_text_color",
                "ts_bg": "ts_bg_color"}[which]
        c = QtWidgets.QColorDialog.getColor(
            getattr(self, attr), self, "Pick color",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if not c.isValid():
            return
        setattr(self, attr, c)
        self._refresh_line_style_swatches()
        self._redraw()

    def _on_toggle_endpoints(self, on):
        self.show_endpoints = on
        self._redraw()

    def _on_endpoint_size(self, v):
        self.endpoint_size = v
        self._redraw()

    # -- legend ------------------------------------------------------------
    def _on_legend_enabled(self, on):
        self.legend["enabled"] = bool(on)
        self._redraw()

    def _on_legend_key(self, key, value):
        self.legend[key] = value
        self._redraw()

    def _on_usetex_toggled(self, on):
        self.use_real_latex = bool(on)
        self.edit_preamble.setEnabled(bool(on) and usetex_available())
        _LATEX_CACHE.clear()
        self._redraw()

    def _on_preamble_changed(self, text):
        self.latex_preamble = text
        _LATEX_CACHE.clear()
        self._redraw()

    def _add_legend_entry(self):
        entry = self._selected_traj_entry()
        color = entry["color"] if entry else QtGui.QColor(*DEFAULT_TRAJ_COLOR)
        width = max(entry["width"] if entry else 2.2, 6.0)
        dlg = LegendEntryDialog(self, label="", kind="line",
                                color=color, style="solid",
                                width=width, latex=False)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.legend_entries.append(dlg.values())
            self._refresh_legend_list()
            self._redraw()

    def _edit_legend_entry(self):
        item = self.list_legend.currentItem()
        if item is None:
            return
        i = item.data(QtCore.Qt.ItemDataRole.UserRole)
        e = self.legend_entries[i]
        dlg = LegendEntryDialog(self, label=e["label"], kind=e["kind"],
                                color=e["color"], style=e["style"],
                                width=e["width"], latex=e.get("latex", False))
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.legend_entries[i] = dlg.values()
            self._refresh_legend_list()
            self._redraw()

    def _remove_legend_entry(self):
        item = self.list_legend.currentItem()
        if item is None:
            return
        self.legend_entries.pop(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self._refresh_legend_list()
        self._redraw()

    def _move_legend_entry(self, d):
        item = self.list_legend.currentItem()
        if item is None:
            return
        i = item.data(QtCore.Qt.ItemDataRole.UserRole)
        j = i + d
        if not (0 <= j < len(self.legend_entries)):
            return
        self.legend_entries[i], self.legend_entries[j] = \
            self.legend_entries[j], self.legend_entries[i]
        self._refresh_legend_list()
        self.list_legend.setCurrentRow(j)
        self._redraw()

    def _seed_legend_from_current(self):
        """Prefill rows for whatever's loaded, as a starting point - the
        entries are plain data afterward and can be edited or deleted."""
        if self.ref is not None:
            self.legend_entries.append(
                {"label": os.path.splitext(os.path.basename(self.ref.path))[0],
                 "kind": "line", "style": "dot", "latex": False,
                 "color": QtGui.QColor(self.ref_color),
                 "width": max(self.ref_width, 6.0)})
        for t in self.trajs:
            name = os.path.splitext(os.path.basename(t["path"].path))[0]
            self.legend_entries.append(
                {"label": name, "kind": "line", "style": "solid",
                 "latex": False, "color": QtGui.QColor(t["color"]),
                 "width": max(t["width"], 6.0)})
            if self.show_endpoints:
                self.legend_entries.append(
                    {"label": f"{name} start", "kind": "star", "style": "solid",
                     "latex": False, "color": QtGui.QColor(t["start_color"]),
                     "width": self.endpoint_size})
                self.legend_entries.append(
                    {"label": f"{name} end", "kind": "star", "style": "solid",
                     "latex": False, "color": QtGui.QColor(t["end_color"]),
                     "width": self.endpoint_size})
        self._refresh_legend_list()
        if not self.legend["enabled"]:
            self.chk_legend.setChecked(True)   # this redraws
        else:
            self._redraw()

    def _refresh_legend_list(self):
        self.list_legend.clear()
        for i, e in enumerate(self.legend_entries):
            kind = e["kind"]
            desc = e["style"] if kind == "line" else kind
            tex = "  TeX" if e.get("latex") else ""
            item = QtWidgets.QListWidgetItem(
                f"{e['label'] or '(no label)'}   "
                f"[{kind}/{desc}  w={e['width']:g}{tex}]")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            item.setForeground(QtGui.QBrush(e["color"]))
            self.list_legend.addItem(item)

    # -- timestamp ---------------------------------------------------------
    def _on_ts_key(self, key, value):
        self.timestamp[key] = value
        self._redraw()

    def _timestamp_text(self):
        """The assembled "t=x.x" string for the current frame, or None."""
        if self.video is None:
            return None
        video_t = self.frame_idx / max(self.video.fps, 1e-9)
        if self.timestamp.get("source") == "trajectory":
            t = video_t - self.offset_s
        else:
            t = video_t
        dec = int(np.clip(int(self.timestamp.get("decimals", 1)), 0, 6))
        return (f"{self.timestamp.get('prefix', '')}"
                f"{t:.{dec}f}"
                f"{self.timestamp.get('suffix', '')}")

    def _on_ref_width(self, v):
        self.ref_width = v
        self._redraw()

    def _refresh_line_style_swatches(self):
        self.btn_ref_color.setStyleSheet(
            f"background:{_css_rgba(self.ref_color)}; border:1px solid #333;")
        self.spin_ref_width.blockSignals(True)
        self.spin_ref_width.setValue(self.ref_width)
        self.spin_ref_width.blockSignals(False)

        for btn, col in ((self.btn_start_color, self.start_color),
                         (self.btn_end_color, self.end_color),
                         (self.btn_leg_text_color, self.legend_text_color),
                         (self.btn_leg_bg_color, self.legend_bg_color),
                         (self.btn_ts_text_color, self.ts_text_color),
                         (self.btn_ts_bg_color, self.ts_bg_color)):
            btn.setStyleSheet(
                f"background:{_css_rgba(col)}; border:1px solid #333;")

    # -- per-stretch color/opacity segments (trajectory and reference) -----
    def _build_segment_controls(self, which):
        """Build the 'color/opacity segments' mini-list + add/edit/remove
        row used for both the trajectory and reference paths. `which` is
        "traj" or "ref"; the resulting QListWidget is stashed as
        self.list_traj_segments / self.list_ref_segments."""
        lay = QtWidgets.QVBoxLayout()
        lay.setSpacing(2)
        lbl = QtWidgets.QLabel("   color/opacity segments (optional)")
        lbl.setToolTip(
            "Override the color and/or opacity of a time interval of this path. "
            "Start and end are seconds in the path's own time base. "
            "For a timeseries without an explicit time array, dt is used. "
            "Static paths without a time axis cannot use timing-based segments. "
            "Double-click a row to edit it."
        )
        lay.addWidget(lbl)

        list_widget = QtWidgets.QListWidget()
        list_widget.setMaximumHeight(70)
        list_widget.setToolTip(lbl.toolTip())
        list_widget.itemDoubleClicked.connect(
            lambda _item, w=which: self._edit_segment(w))
        lay.addWidget(list_widget)

        row = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("+ segment")
        btn_add.clicked.connect(lambda: self._add_segment(which))
        btn_edit = QtWidgets.QPushButton("Edit…")
        btn_edit.clicked.connect(lambda: self._edit_segment(which))
        btn_del = QtWidgets.QPushButton("Remove")
        btn_del.clicked.connect(lambda: self._remove_segment(which))
        row.addWidget(btn_add)
        row.addWidget(btn_edit)
        row.addWidget(btn_del)
        lay.addLayout(row)

        if which == "traj":
            self.list_traj_segments = list_widget
        else:
            self.list_ref_segments = list_widget
        return lay

    def _segments_for(self, which):
        if which == "ref":
            return self.ref_segments
        entry = self._selected_traj_entry()
        return entry["segments"] if entry else []

    def _segment_list_widget(self, which):
        return self.list_traj_segments if which == "traj" else self.list_ref_segments

    def _segment_src(self, which):
        if which == "ref":
            return self.ref
        entry = self._selected_traj_entry()
        return entry["path"] if entry else None

    def _segment_base_color(self, which):
        if which == "ref":
            return self.ref_color
        entry = self._selected_traj_entry()
        return entry["color"] if entry else QtGui.QColor(*DEFAULT_TRAJ_COLOR)

    def _add_segment(self, which):
        if which == "traj" and self._selected_traj_entry() is None:
            QtWidgets.QMessageBox.information(
                self, "Color segments",
                "Select a trajectory in the list above first.")
            return

        segs = self._segments_for(which)
        src = self._segment_src(which)

        if src is None or not src.is_timeseries:
            QtWidgets.QMessageBox.information(
                self,
                "Timing-based segments",
                "This path has no time axis, so timing-based color segments "
                "cannot be used.\n\n"
                "Load a timeseries path with a 'time'/'t' column to use them."
            )
            return

        t0, t1 = src.full_time_range()
        base_color = self._segment_base_color(which)

        start = segs[-1]["end"] if segs else t0
        start = float(np.clip(start, t0, t1))

        dlg = SegmentDialog(
            self,
            start=start,
            end=t1,
            color=base_color,
            time_range=(t0, t1),
        )

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            s, e, c = dlg.values()
            segs.append({"start": s, "end": e, "color": c})
            self._refresh_segment_list(which)
        self._redraw()

    def _edit_segment(self, which):
        list_widget = self._segment_list_widget(which)
        item = list_widget.currentItem()
        if item is None:
            return

        src = self._segment_src(which)
        if src is None or not src.is_timeseries:
            return

        segs = self._segments_for(which)
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        seg = segs[idx]

        dlg = SegmentDialog(
            self,
            start=seg["start"],
            end=seg["end"],
            color=seg["color"],
            time_range=src.full_time_range(),
        )

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            s, e, c = dlg.values()
            segs[idx] = {
                "start": s,
                "end": e,
                "color": c,
            }
            self._refresh_segment_list(which)
            self._redraw()

    def _remove_segment(self, which):
        list_widget = self._segment_list_widget(which)
        item = list_widget.currentItem()
        if item is None:
            return
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._segments_for(which).pop(idx)
        self._refresh_segment_list(which)
        self._redraw()

    def _refresh_segment_list(self, which):
        list_widget = self._segment_list_widget(which)
        segs = self._segments_for(which)
        list_widget.clear()
        for i, seg in enumerate(segs):
            lo, hi = sorted((seg["start"], seg["end"]))
            item = QtWidgets.QListWidgetItem(
                f"{lo:.3f}s-{hi:.3f}s   "
                f"{seg['color'].name(QtGui.QColor.NameFormat.HexArgb)}"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            item.setForeground(QtGui.QBrush(seg["color"]))
            list_widget.addItem(item)

    # -- trajectory time-window trim (of whichever entry is selected) ------
    def _update_trim_spin_ranges(self):
        entry = self._selected_traj_entry()
        if entry is None:
            self.lbl_trim_range.setText("select a trajectory above")
            self.chk_trim.setEnabled(False)
            return
        src = entry["path"]
        if not src.is_timeseries:
            self.lbl_trim_range.setText(
                "this trajectory has no time axis, so it can't be trimmed")
            self.chk_trim.setEnabled(False)
            return
        self.chk_trim.setEnabled(True)
        t0, t1 = src.full_time_range()
        pad = max((t1 - t0) * 2.0, 5.0)
        for sb in (self.spin_trim_start, self.spin_trim_end):
            sb.blockSignals(True)
            sb.setRange(t0 - pad, t1 + pad)
            sb.blockSignals(False)
        self.spin_trim_start.blockSignals(True)
        self.spin_trim_start.setValue(entry["trim_start"])
        self.spin_trim_start.blockSignals(False)
        self.spin_trim_end.blockSignals(True)
        self.spin_trim_end.setValue(entry["trim_end"])
        self.spin_trim_end.blockSignals(False)
        self.chk_trim.blockSignals(True)
        self.chk_trim.setChecked(entry["trim_enabled"])
        self.chk_trim.blockSignals(False)
        self.lbl_trim_range.setText(f"full trajectory range: {t0:.3f}s to {t1:.3f}s")

    def _on_trim_toggle(self, on):
        entry = self._selected_traj_entry()
        if entry is None:
            return
        entry["trim_enabled"] = on
        self._redraw()

    def _on_trim_start(self, v):
        entry = self._selected_traj_entry()
        if entry is None:
            return
        entry["trim_start"] = v
        self._redraw()

    def _on_trim_end(self, v):
        entry = self._selected_traj_entry()
        if entry is None:
            return
        entry["trim_end"] = v
        self._redraw()

    def _set_trim_start_to_playhead(self):
        if self._selected_traj_entry() is None or self.video is None:
            return
        self.spin_trim_start.setValue(self.frame_idx / self.video.fps - self.offset_s)

    def _set_trim_end_to_playhead(self):
        if self._selected_traj_entry() is None or self.video is None:
            return
        self.spin_trim_end.setValue(self.frame_idx / self.video.fps - self.offset_s)

    def _entry_window(self, entry):
        """(i0, i1) inclusive sample-index window for `entry`, honoring its
        own trim controls. None if trim is off (or it isn't time-varying),
        meaning the full path is drawn."""
        src = entry["path"]
        if not src.is_timeseries or not entry["trim_enabled"]:
            return None
        return src.index_range(entry["trim_start"], entry["trim_end"])

    def _flip_track_180(self):
        """Rotate the track 180° (path_yaw += 180) while shifting
        path_offset_x/y and path_z so the combined centroid of whichever
        path(s) are loaded stays exactly where it is on screen. Plain
        path_yaw edits rotate around the raw data's own (0,0), which -
        unless the path happens to be centered there - also flings the
        whole footprint sideways, forcing a full camera re-tune. This
        keeps the camera pose valid across the flip."""
        loaded = [s for s in ([self.ref] + [t["path"] for t in self.trajs])
                 if s is not None and s.n > 0]
        if not loaded:
            QtWidgets.QMessageBox.warning(
                self, "Flip track", "Open a trajectory or reference path first.")
            return
        scale = self.cam["path_scale"]
        all_xy = np.concatenate([s.xy for s in loaded], axis=0)
        cx = float(np.mean(all_xy[:, 0])) * scale
        cy = float(np.mean(all_xy[:, 1])) * scale
        local_centroid = np.array([cx, cy, 0.0])

        yaw0, pitch0, roll0 = (self.cam["path_yaw"], self.cam["path_pitch"],
                               self.cam["path_roll"])
        yaw1 = (yaw0 + 180.0) % 360.0

        world_before = _track_rotation_matrix(yaw0, pitch0, roll0) @ local_centroid
        world_after = _track_rotation_matrix(yaw1, pitch0, roll0) @ local_centroid
        delta = world_before - world_after  # add to offset to hold position

        self.cam["path_yaw"] = yaw1
        self.cam["path_offset_x"] += float(delta[0])
        self.cam["path_offset_y"] += float(delta[1])
        self.cam["path_z"] += float(delta[2])

        for k in ("path_yaw", "path_offset_x", "path_offset_y", "path_z"):
            sb = self._cam_spins[k]
            sb.blockSignals(True)
            sb.setValue(self.cam[k])
            sb.blockSignals(False)
        self._redraw()
        self.statusBar().showMessage(
            "Flipped track 180° - camera position/pose unchanged", 5000)

    def reset_camera(self):
        self.cam = dict(DEFAULT_CAM)
        for k, sb in self._cam_spins.items():
            sb.blockSignals(True)
            sb.setValue(self.cam[k])
            sb.blockSignals(False)
        self._redraw()

    # -- keyframes (moving camera) -----------------------------------------
    def _refresh_keyframe_list(self):
        self.list_kf.clear()
        for f in sorted(self.keyframes.keys()):
            item = QtWidgets.QListWidgetItem(f"frame {f}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, f)
            self.list_kf.addItem(item)
        n = len(self.keyframes)
        if n == 0:
            self.lbl_kf_status.setText("no keyframes - camera fixed for whole video")
        elif n == 1:
            self.lbl_kf_status.setText("1 keyframe - camera still fixed (add a second "
                                       "one where it moves)")
        else:
            self.lbl_kf_status.setText(f"{n} keyframes - interpolating pose between them")

    def _set_keyframe_here(self):
        if self.video is None:
            QtWidgets.QMessageBox.warning(self, "Keyframe", "Open a video first.")
            return
        self.keyframes[self.frame_idx] = dict(self.cam)
        self._refresh_keyframe_list()
        self.statusBar().showMessage(f"Keyframe set at frame {self.frame_idx}", 4000)

    def _delete_selected_keyframe(self):
        item = self.list_kf.currentItem()
        if item is None:
            return
        f = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.keyframes.pop(f, None)
        self._refresh_keyframe_list()
        self._apply_interpolated_camera()

    def _goto_keyframe_item(self, item):
        f = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.slider.setValue(int(f))

    def _apply_interpolated_camera(self):
        """Recompute self.cam from keyframes for the current frame and push
        it into the spinboxes without re-triggering their change handlers.
        With 0 keyframes this leaves self.cam (the manually-tuned fixed
        pose) untouched."""
        if not self.keyframes:
            self._redraw()
            return
        self.cam = interpolate_camera(self.keyframes, self.frame_idx)
        for k, sb in self._cam_spins.items():
            sb.blockSignals(True)
            sb.setValue(self.cam[k])
            sb.blockSignals(False)
        self._redraw()

    # -- rendering --------------------------------------------------------
    def _composite_layers(self, base_frame):
        """Alpha-blend each visible layer's picked frame over the base
        video frame, in list order (later entries drawn on top of earlier
        ones and of the base) - producing the "final composition" that the
        grid/trajectory/legend/timestamp drawing then happens on top of.
        Layer frames are resized to the base frame's resolution if they
        don't already match. If a layer has a "rect" (x0, y0, x1, y1) in
        base-frame pixel coords, the blend (the linear interpolation
        between base and layer pixels) is restricted to that rectangle -
        everywhere else stays the untouched base."""
        if not self.layers:
            return base_frame
        composite = base_frame.astype(np.float32)
        bh, bw = composite.shape[:2]
        for layer in self.layers:
            if not layer["visible"] or layer["opacity"] <= 0.0:
                continue
            lframe = layer["video"].frame_at(layer["frame_idx"])
            if lframe is None:
                continue
            if lframe.shape[:2] != (bh, bw):
                lframe = cv2.resize(lframe, (bw, bh))
            a = float(np.clip(layer["opacity"], 0.0, 1.0))
            rect = layer.get("rect")
            if rect is None:
                composite = composite * (1.0 - a) + lframe.astype(np.float32) * a
                continue
            x0, y0, x1, y1 = rect
            x0, x1 = int(np.clip(x0, 0, bw)), int(np.clip(x1, 0, bw))
            y0, y1 = int(np.clip(y0, 0, bh)), int(np.clip(y1, 0, bh))
            if x1 <= x0 or y1 <= y0:
                continue
            region = composite[y0:y1, x0:x1]
            lregion = lframe[y0:y1, x0:x1].astype(np.float32)
            composite[y0:y1, x0:x1] = region * (1.0 - a) + lregion * a
        return np.clip(composite, 0, 255).astype(np.uint8)

    def _redraw(self):
        if self.video is None:
            self.lbl_info.setText("")
            return
        frame = self.video.frame_at(self.frame_idx)
        if frame is None:
            return
        frame = self._composite_layers(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w,
                           QtGui.QImage.Format.Format_RGB888).copy()
        pm = QtGui.QPixmap.fromImage(img)

        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        if self.show_grid:
            self._draw_grid(painter, w, h)

        cur_samples = []   # [(entry, idx), ...] for the info bar
        for t in self.trajs:
            src = t["path"]
            if not t["visible"] or src.n == 0:
                continue
            win = self._entry_window(t)
            rng = win if win is not None else (0, src.n - 1)
            self._draw_path(painter, w, h, src, rng, t["segments"],
                            t["color"], t["width"], dotted=False)
            self._draw_endpoint_dots(painter, w, h, src, rng,
                                     t["start_color"], t["end_color"])
            if src.is_timeseries and self.show_point:
                video_t = self.frame_idx / self.video.fps
                path_t = video_t - self.offset_s
                idx = src.sample_at(path_t)
                if idx is not None and (win is None or win[0] <= idx <= win[1]):
                    cur_samples.append((t, idx))
                    self._draw_point(painter, w, h, src, idx, t["color"])

        # the reference goes on top of every trajectory (only the legend and
        # timestamp overlays sit above it)
        if self.ref is not None and self.ref.n > 0 and self.show_ref:
            ref_range = (0, self.ref.n - 1)
            self._draw_path(painter, w, h, self.ref, ref_range,
                            self.ref_segments, self.ref_color, self.ref_width,
                            dotted=True)

        self._draw_legend(painter, w, h)
        self._draw_timestamp(painter, w, h)

        painter.end()
        self.view.set_full_pixmap(pm)
        self._update_info(cur_samples)
        self._report_latex_issue()

    def _report_latex_issue(self):
        """Surface the most recent LaTeX-labeled entry that fell back to
        plain text, and why - once per distinct issue, in the status bar,
        since the matching console log line is easy to miss when the app
        isn't run from a visible terminal."""
        issue = _LATEX_LAST_ISSUE[0]
        if issue is None or issue == self._shown_latex_issue:
            return
        self._shown_latex_issue = issue
        text, reason = issue
        snippet = text if len(text) <= 50 else text[:47] + "..."
        self.statusBar().showMessage(
            f'LaTeX label "{snippet}" fell back to plain text - {reason}',
            10000)

    def _draw_grid(self, painter, w, h):
        bounds = self._combined_xy_bounds()
        if bounds is None:
            return
        lines = ground_grid_lines(bounds, self.grid_spacing)
        pen = QtGui.QPen(QtGui.QColor(90, 200, 255, 90))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        for line in lines:
            u, v, valid = project_points(line, self.cam, w, h)
            self._draw_polyline(painter, u, v, valid)

    def _draw_line(self, painter, w, h, xy, color, width, dotted=False):
        """Project and draw one path (or one segment of one) as a polyline
        in `color` at `width` pixels; `dotted` picks a dotted pen style
        (used for the reference line)."""
        u, v, valid = project_points(xy, self.cam, w, h)
        pen = QtGui.QPen(color)
        pen.setWidthF(width)
        if dotted:
            pen.setStyle(QtCore.Qt.PenStyle.DotLine)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        self._draw_polyline(painter, u, v, valid)

    @staticmethod
    def _segment_index_bounds(src, start_s, end_s):
        """Convert a path-relative time interval into inclusive sample indices."""
        if src is None or src.n == 0:
            return 0, -1

        if not src.is_timeseries:
            return 0, src.n - 1

        start_s, end_s = sorted((float(start_s), float(end_s)))

        if src.time is not None:
            i0 = int(np.searchsorted(src.time, start_s, side="left"))
            i1 = int(np.searchsorted(src.time, end_s, side="right")) - 1
        else:
            dt = max(src.dt or 0.0, 1e-9)
            i0 = int(np.ceil(start_s / dt))
            i1 = int(np.floor(end_s / dt))

        i0 = int(np.clip(i0, 0, src.n - 1))
        i1 = int(np.clip(i1, 0, src.n - 1))

        return i0, i1

    def _draw_path(self, painter, w, h, src, index_range, segments,
                   base_color, base_width, dotted=False):
        """Draw `src` between `index_range` (i0, i1) inclusive sample
        indices. With no `segments`, that's one polyline in base_color/
        base_width - the old single-color behavior (which already gets
        its opacity from base_color's alpha). With `segments` - a list of
        {"start","end","color"} time overrides - draw each segment's time
        range, clipped to index_range, in its own color, so different
        stretches of the same path can have different colors and/or
        opacities."""
        i0, i1 = index_range
        if src is None or src.n == 0 or i1 < i0:
            return
        if not segments:
            self._draw_line(painter, w, h, src.xy[i0:i1 + 1], base_color,
                            base_width, dotted=dotted)
            return
        for seg in segments:
            s0, s1 = self._segment_index_bounds(
                src, seg["start"], seg["end"]
            )
            lo, hi = max(s0, i0), min(s1, i1)
            if hi < lo:
                continue
            self._draw_line(painter, w, h, src.xy[lo:hi + 1], seg["color"],
                            base_width, dotted=dotted)

    def _draw_endpoint_dots(self, painter, w, h, src, index_range,
                           start_color=None, end_color=None):
        """Green/red (by default) stars at the start/end of the currently-
        visible extent of `src` (honoring any trim window) - drawn once
        regardless of how many colored segments were used to render the
        line itself, so the current section's ends stay easy to spot. Size
        is in full-res video pixels; the preview downscales, the saved
        frame doesn't. `start_color`/`end_color` let each trajectory use
        its own star colors instead of the global defaults."""
        if not self.show_endpoints:
            return
        i0, i1 = index_range
        if src is None or src.n == 0 or i1 < i0:
            return
        r = self.endpoint_size
        outline = QtGui.QPen(QtGui.QColor(0, 0, 0), max(1.0, r * 0.12))
        start_color = start_color or self.start_color
        end_color = end_color or self.end_color
        for idx, c in ((i0, start_color), (i1, end_color)):
            u, v, valid = project_points(src.xy[idx:idx + 1], self.cam, w, h)
            if len(u) and valid[0]:
                painter.setBrush(QtGui.QBrush(c))
                painter.setPen(outline)
                painter.drawPolygon(_star_polygon(u[0], v[0], r))

    def _draw_point(self, painter, w, h, src, idx, color=None):
        color = color or QtGui.QColor(255, 60, 60)
        u, v, valid = project_points(src.xy[idx:idx + 1], self.cam, w, h)
        if len(u) and valid[0]:
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1.5))
            painter.drawEllipse(QtCore.QPointF(u[0], v[0]), 7, 7)

    # -- legend drawing ----------------------------------------------------
    def _legend_row_metrics(self, entry, fm):
        """(latex_image_or_None, label_w, row_h) for one legend entry."""
        img = None
        if entry.get("latex"):
            img = render_latex(entry["label"], self.legend["text_size"],
                               self.legend_text_color,
                               bool(self.legend["bold"]),
                               usetex=self.use_real_latex,
                               preamble=self.latex_preamble)
        if img is not None:
            label_w, label_h = float(img.width()), float(img.height())
        else:
            label_w = fm.horizontalAdvance(entry["label"])
            label_h = fm.height()
        marker_h = (entry["width"] if entry["kind"] == "line"
                    else entry["width"] * 2.0)
        return img, label_w, max(label_h, marker_h)

    def _draw_legend(self, painter, w, h):
        """Draw the legend box in full-resolution frame pixels.

        Entries are dealt out into `columns` columns, filled column by
        column in list order (so 6 entries / 2 columns puts 1-3 left and
        4-6 right). Each column is sized to its own widest label; the box
        height is the tallest column."""
        entries = self.legend_entries
        if not self.legend["enabled"] or not entries:
            return

        painter.save()
        font = QtGui.QFont()
        font.setPixelSize(max(1, int(round(self.legend["text_size"]))))
        font.setBold(bool(self.legend["bold"]))
        painter.setFont(font)
        fm = QtGui.QFontMetricsF(font)

        pad = float(self.legend["padding"])
        gap = float(self.legend["gap"])
        sample = float(self.legend["sample_len"])
        row_gap = float(self.legend["row_gap"])
        col_gap = float(self.legend.get("col_gap", DEFAULT_LEGEND["col_gap"]))

        n = len(entries)
        ncols = int(np.clip(int(self.legend.get("columns", 1) or 1), 1, 8))
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / float(ncols)))

        # rows[c] = list of (entry, latex_img, label_w, row_h) for column c
        columns = []
        for c in range(ncols):
            chunk = entries[c * nrows:(c + 1) * nrows]
            rows = []
            for e in chunk:
                img, tw, rh = self._legend_row_metrics(e, fm)
                rows.append((e, img, tw, rh))
            if rows:
                columns.append(rows)

        col_widths = [pad * 0 + sample + gap + max(r[2] for r in rows)
                      for rows in columns]
        col_heights = [sum(r[3] for r in rows) + row_gap * (len(rows) - 1)
                       for rows in columns]

        box_w = pad * 2 + sum(col_widths) + col_gap * (len(columns) - 1)
        box_h = pad * 2 + (max(col_heights) if col_heights else 0.0)

        x, y = _corner_origin(self.legend["corner"], w, h, box_w, box_h,
                              float(self.legend["margin"]))

        if self.legend["bg_alpha"] > 0:
            bg = QtGui.QColor(self.legend_bg_color)
            bg.setAlpha(int(np.clip(self.legend["bg_alpha"], 0, 255)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(bg))
            painter.drawRoundedRect(QtCore.QRectF(x, y, box_w, box_h),
                                    pad * 0.5, pad * 0.5)

        cx0 = x + pad
        for rows, colw in zip(columns, col_widths):
            cy = y + pad
            text_w = colw - sample - gap
            for e, img, tw, rh in rows:
                mid = cy + rh / 2.0
                if e["kind"] == "line":
                    pen = QtGui.QPen(e["color"])
                    pen.setWidthF(e["width"])
                    pen.setStyle(PEN_STYLES.get(e["style"],
                                                QtCore.Qt.PenStyle.SolidLine))
                    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawLine(QtCore.QPointF(cx0, mid),
                                     QtCore.QPointF(cx0 + sample, mid))
                else:
                    r = e["width"]
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
                    painter.drawImage(
                        QtCore.QPointF(tx, mid - img.height() / 2.0), img)
                else:
                    painter.setPen(QtGui.QPen(self.legend_text_color))
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawText(
                        QtCore.QRectF(tx, cy, max(text_w, tw), rh),
                        int(QtCore.Qt.AlignmentFlag.AlignVCenter |
                            QtCore.Qt.AlignmentFlag.AlignLeft),
                        e["label"])
                cy += rh + row_gap
            cx0 += colw + col_gap
        painter.restore()

    # -- timestamp drawing -------------------------------------------------
    def _draw_timestamp(self, painter, w, h):
        """Draw the "t=x.x" readout, in full-resolution frame pixels."""
        if not self.timestamp.get("enabled"):
            return
        text = self._timestamp_text()
        if not text:
            return

        painter.save()
        font = QtGui.QFont()
        font.setPixelSize(max(1, int(round(self.timestamp["text_size"]))))
        font.setBold(bool(self.timestamp["bold"]))
        painter.setFont(font)
        fm = QtGui.QFontMetricsF(font)

        img = None
        if self.timestamp.get("latex"):
            img = render_latex(text, self.timestamp["text_size"],
                               self.ts_text_color,
                               bool(self.timestamp["bold"]),
                               usetex=self.use_real_latex,
                               preamble=self.latex_preamble)
        if img is not None:
            tw, th = float(img.width()), float(img.height())
        else:
            tw, th = fm.horizontalAdvance(text), fm.height()

        pad = float(self.timestamp["padding"])
        box_w = tw + pad * 2
        box_h = th + pad * 2
        x, y = _corner_origin(self.timestamp["corner"], w, h, box_w, box_h,
                              float(self.timestamp["margin"]))

        if self.timestamp["bg_alpha"] > 0:
            bg = QtGui.QColor(self.ts_bg_color)
            bg.setAlpha(int(np.clip(self.timestamp["bg_alpha"], 0, 255)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(bg))
            painter.drawRoundedRect(QtCore.QRectF(x, y, box_w, box_h),
                                    pad * 0.5, pad * 0.5)

        if img is not None:
            painter.drawImage(QtCore.QPointF(x + pad, y + pad), img)
        else:
            painter.setPen(QtGui.QPen(self.ts_text_color))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawText(
                QtCore.QRectF(x + pad, y + pad, tw, th),
                int(QtCore.Qt.AlignmentFlag.AlignVCenter |
                    QtCore.Qt.AlignmentFlag.AlignLeft),
                text)
        painter.restore()

    @staticmethod
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
        # drawPath both strokes AND fills with whatever brush is currently
        # set - and by the time later trajectories/grid lines are drawn,
        # an earlier star/point marker may have left a solid brush active.
        # Force it off so a polyline is always just a stroked line, never
        # filled.
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _update_info(self, cur_samples):
        bits = [f"video frame {self.frame_idx}/{self.video.n_frames - 1}   "
                f"t={self.frame_idx / self.video.fps:.3f}s"]
        if self.keyframes:
            bits.append(f"{len(self.keyframes)} keyframe(s)")
        if self.trajs:
            n = len(self.trajs)
            bits.append(f"{n} trajector{'y' if n == 1 else 'ies'}"
                        f"  (offset {self.offset_s:+.3f}s)")
        for entry, idx in cur_samples:
            name = os.path.splitext(os.path.basename(entry["path"].path))[0]
            bits.append(f"{name} sample {idx}/{entry['path'].n - 1}")
            win = self._entry_window(entry)
            if win is not None:
                bits.append(f"{name} section [{win[0]}:{win[1]}] of {entry['path'].n}")
        self.lbl_info.setText("   ".join(bits))

    def save_current_frame(self):
        """Save the currently shown frame - the video frame plus whatever
        overlay is currently toggled on (grid/trajectory/reference/point/
        legend/timestamp), exactly as displayed - to an image file. Full
        video resolution, not the (possibly downscaled) on-screen widget
        size."""
        if self.video is None or self.view._full_pixmap is None:
            QtWidgets.QMessageBox.warning(self, "Save frame", "Open a video first.")
            return
        base = os.path.splitext(self.video.path)[0]
        default = f"{base}_frame{self.frame_idx:06d}.png"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save current frame", default,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)")
        if not p:
            return
        if not os.path.splitext(p)[1]:
            p += ".png"
        if not self.view._full_pixmap.save(p):
            QtWidgets.QMessageBox.critical(
                self, "Save frame", f"Could not write {p}")
            return
        log(f"wrote {p}")
        self.statusBar().showMessage(f"Saved frame to {os.path.basename(p)}", 6000)

    # -- calibration file --------------------------------------------------
    def build_calibration(self, spec_path, reload_files=False):
        spec_dir = os.path.dirname(os.path.abspath(spec_path))
        return {
            "schema": SCHEMA,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # if true ("Save all"), load_calibration actually reopens the
            # video/trajectory/reference/layer files below (replacing
            # whatever's currently loaded); if false/absent (plain "Save
            # calibration"), they're provenance only and never read back -
            # settings apply to whatever's already open (or gets opened
            # afterward), not to these specific files.
            "reload_files": bool(reload_files),
            "video": _rel(self.video.path, spec_dir) if self.video else None,
            "video_abs": os.path.abspath(self.video.path) if self.video else None,
            "image_w": self.video.w if self.video else None,
            "image_h": self.video.h if self.video else None,
            "reference_file": _rel(self.ref.path, spec_dir) if self.ref else None,
            "reference_abs": os.path.abspath(self.ref.path) if self.ref else None,
            "camera": dict(self.cam),
            "keyframes": {str(f): c for f, c in self.keyframes.items()},
            "sync": {"offset_s": self.offset_s},
            "display": {
                "show_ref": self.show_ref,
                "show_point": self.show_point,
                "show_grid": self.show_grid,
                "grid_spacing": self.grid_spacing,
                "ref_color": self.ref_color.name(QtGui.QColor.NameFormat.HexArgb),
                "ref_width": self.ref_width,
                "ref_segments": [
                    {"start": s["start"], "end": s["end"],
                     "color": s["color"].name(QtGui.QColor.NameFormat.HexArgb)}
                    for s in self.ref_segments],

                "show_endpoints": self.show_endpoints,
                "endpoint_size": self.endpoint_size,
                "start_color": self.start_color.name(QtGui.QColor.NameFormat.HexArgb),
                "end_color": self.end_color.name(QtGui.QColor.NameFormat.HexArgb),
                "legend": dict(self.legend),
                "legend_text_color": self.legend_text_color.name(
                    QtGui.QColor.NameFormat.HexArgb),
                "legend_bg_color": self.legend_bg_color.name(
                    QtGui.QColor.NameFormat.HexArgb),
                "legend_entries": [
                    {"label": e["label"], "kind": e["kind"],
                     "style": e["style"], "width": e["width"],
                     "latex": bool(e.get("latex", False)),
                     "color": e["color"].name(QtGui.QColor.NameFormat.HexArgb)}
                    for e in self.legend_entries],
                "usetex": self.use_real_latex,
                "latex_preamble": self.latex_preamble,
                "timestamp": dict(self.timestamp),
                "timestamp_text_color": self.ts_text_color.name(
                    QtGui.QColor.NameFormat.HexArgb),
                "timestamp_bg_color": self.ts_bg_color.name(
                    QtGui.QColor.NameFormat.HexArgb),
                # provenance + per-file style (color/width/visibility/
                # segments/trim/star colors), keyed by basename on load -
                # like the reference path, the files themselves are never
                # auto-reloaded from this.
                "trajectories": [
                    {"file": _rel(t["path"].path, spec_dir),
                     "file_abs": os.path.abspath(t["path"].path),
                     "color": t["color"].name(QtGui.QColor.NameFormat.HexArgb),
                     "width": t["width"], "visible": t["visible"],
                     "start_color": t["start_color"].name(
                         QtGui.QColor.NameFormat.HexArgb),
                     "end_color": t["end_color"].name(
                         QtGui.QColor.NameFormat.HexArgb),
                     "segments": [
                         {"start": s["start"], "end": s["end"],
                          "color": s["color"].name(QtGui.QColor.NameFormat.HexArgb)}
                         for s in t["segments"]],
                     "trim": {"enabled": t["trim_enabled"],
                              "start_s": t["trim_start"],
                              "end_s": t["trim_end"]}}
                    for t in self.trajs],
                # same provenance-only-file / restore-by-basename-style
                # convention as trajectories above.
                "video_layers": [
                    {"file": _rel(l["video"].path, spec_dir),
                     "file_abs": os.path.abspath(l["video"].path),
                     "frame_idx": l["frame_idx"], "opacity": l["opacity"],
                     "visible": l["visible"],
                     "rect": list(l["rect"]) if l["rect"] else None}
                    for l in self.layers],
            },
        }

    def save_calibration(self):
        default = (os.path.splitext(self.video.path)[0] + SUFFIX if self.video
                   else "calibration" + SUFFIX)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration", default,
            f"Camera alignment (*{SUFFIX});;JSON (*.json)")
        if not p:
            return
        if not p.endswith(".json"):
            p += SUFFIX
        try:
            with open(p, "w") as f:
                json.dump(self.build_calibration(p), f, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save", str(exc))
            return
        log(f"wrote {p}")
        self.statusBar().showMessage(f"Saved {os.path.basename(p)}", 6000)

    def save_project(self):
        """Like save_calibration, but also marks the file so load_
        calibration reopens the video/reference/trajectory/layer files
        themselves - a full, self-contained snapshot of this session."""
        if self.video:
            base, _ = os.path.splitext(self.video.path)
            default = f"{base}.project{SUFFIX}"
        else:
            default = "project" + SUFFIX
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save all (settings + videos/paths)", default,
            f"Camera alignment project (*{SUFFIX});;JSON (*.json)")
        if not p:
            return
        if not p.endswith(".json"):
            p += SUFFIX
        try:
            with open(p, "w") as f:
                json.dump(self.build_calibration(p, reload_files=True), f,
                          indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save", str(exc))
            return
        log(f"wrote {p}")
        self.statusBar().showMessage(
            f"Saved {os.path.basename(p)} (settings + video/reference/"
            "trajectory/layer files - reopens them all on load)", 6000)

    def load_calibration_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration", "", f"Camera alignment (*{SUFFIX});;JSON (*.json)")
        if p:
            self.load_calibration(p)

    def _reload_files_from_spec(self, spec, spec_dir):
        """For a "Save all" project: actually (re)open the video/reference/
        trajectory/layer files it recorded, replacing whatever trajectories/
        layers are currently loaded (video/reference are only replaced if
        the project actually had one). Tries the absolute path first, then
        the path relative to the json's own directory (so a session moved
        or copied elsewhere alongside its files still resolves) - anything
        found neither way is skipped and reported rather than aborting the
        whole load. Returns a list of the file paths that couldn't be
        found."""
        missing = []

        def _find(rel, abs_):
            for cand in (abs_, _resolve(rel, spec_dir)):
                if cand and os.path.exists(cand):
                    return cand
            return None

        video_ref = spec.get("video") or spec.get("video_abs")
        if video_ref:
            p = _find(spec.get("video"), spec.get("video_abs"))
            if p:
                self._load_video(p)
            else:
                missing.append(video_ref)

        ref_ref = spec.get("reference_file") or spec.get("reference_abs")
        if ref_ref:
            p = _find(spec.get("reference_file"), spec.get("reference_abs"))
            if p:
                self._load_ref(p)
            else:
                missing.append(ref_ref)

        for layer in self.layers:
            if layer["_dialog"] is not None:
                layer["_dialog"].setParent(None)
                layer["_dialog"].deleteLater()
            layer["video"].release()
        self.layers = []
        self.trajs = []

        disp = spec.get("display") or {}
        for o in disp.get("trajectories") or []:
            ref = o.get("file") or o.get("file_abs")
            p = _find(o.get("file"), o.get("file_abs"))
            if p:
                self._add_trajectory(p)
            elif ref:
                missing.append(ref)

        for o in disp.get("video_layers") or []:
            ref = o.get("file") or o.get("file_abs")
            p = _find(o.get("file"), o.get("file_abs"))
            if p:
                self._add_video_layer(p)
            elif ref:
                missing.append(ref)

        return missing

    def load_calibration(self, path):
        """Apply a saved camera/track pose, keyframes, sync offset, display
        settings, and trajectory trim window. For a plain "Save
        calibration" file, this is deliberately agnostic of which video/
        trajectory/reference are involved: it never opens, requires, or
        even looks at whichever video/path files were open when it was
        saved (those are stored in the file only as provenance) - it just
        applies pose/keyframe/sync/display/trim settings on top of
        whatever is already loaded in this session, so the same
        calibration can be reused for a re-encoded video, a renamed path
        file, or a different rollout off the same rig. A "Save all" file
        (spec["reload_files"] true) is the opposite: it actually reopens
        its video/reference/trajectory/layer files too, replacing
        whatever's currently loaded, to recreate that exact session."""
        try:
            with open(path) as f:
                spec = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load", str(exc))
            return

        reload_files = bool(spec.get("reload_files"))
        missing_files = []
        if reload_files:
            missing_files = self._reload_files_from_spec(
                spec, os.path.dirname(os.path.abspath(path)))

        # older files (schema camalign/1) saved "path_rotation" (yaw-only,
        # no pitch/roll) instead of "path_yaw" - migrate it on load.
        def _migrate(c):
            c = dict(c)
            if "path_yaw" not in c and "path_rotation" in c:
                c["path_yaw"] = c["path_rotation"]
            return c

        cam = _migrate(spec.get("camera") or {})
        self.cam = {k: float(cam.get(k, DEFAULT_CAM[k])) for k in DEFAULT_CAM}
        for k, sb in self._cam_spins.items():
            sb.blockSignals(True)
            sb.setValue(self.cam[k])
            sb.blockSignals(False)

        kf = spec.get("keyframes") or {}
        self.keyframes = {int(f): {k: float(_migrate(c).get(k, DEFAULT_CAM[k]))
                                   for k in DEFAULT_CAM}
                          for f, c in kf.items()}
        self._refresh_keyframe_list()

        sync = spec.get("sync") or {}
        self.spin_offset.setValue(float(sync.get("offset_s", 0.0)))

        disp = spec.get("display") or {}
        self.chk_ref.setChecked(bool(disp.get("show_ref", True)))
        self.chk_point.setChecked(bool(disp.get("show_point", True)))
        self.chk_grid.setChecked(bool(disp.get("show_grid", True)))
        if disp.get("grid_spacing"):
            self.spin_grid.setValue(float(disp["grid_spacing"]))
        if disp.get("ref_color"):
            self.ref_color = QtGui.QColor(disp["ref_color"])
        if disp.get("ref_width"):
            self.ref_width = float(disp["ref_width"])
        self._refresh_line_style_swatches()

        def _load_segments(raw):
            out = []
            for s in raw or []:
                try:
                    out.append({"start": float(s["start"]), "end": float(s["end"]),
                                "color": QtGui.QColor(s["color"])})
                except (KeyError, TypeError, ValueError):
                    continue
            return out

        self.ref_segments = _load_segments(disp.get("ref_segments"))
        self._refresh_segment_list("ref")

        # trajectory style (color/width/visibility/segments/trim/star
        # colors), keyed by file basename: applied to whatever trajectories
        # are already loaded, and remembered so it also applies to matching
        # files added afterward. Understands both the current "trajectories"
        # key and pre-multi-trajectory calibrations' single-slot
        # "trajectory_file"/traj_*/trajectory_trim fields.
        self._pending_traj_styles = {}
        for o in disp.get("trajectories") or disp.get("overlay_trajectories") or []:
            name = os.path.basename(o.get("file") or o.get("file_abs") or "")
            if name:
                self._pending_traj_styles[name] = o
        legacy_name = os.path.basename(
            spec.get("trajectory_file") or spec.get("trajectory_abs") or "")
        if legacy_name and legacy_name not in self._pending_traj_styles:
            legacy_style = {}
            if disp.get("traj_color"):
                legacy_style["color"] = disp["traj_color"]
            if disp.get("traj_width"):
                legacy_style["width"] = disp["traj_width"]
            if "show_traj" in disp:
                legacy_style["visible"] = bool(disp["show_traj"])
            if disp.get("traj_segments"):
                legacy_style["segments"] = disp["traj_segments"]
            if spec.get("trajectory_trim"):
                legacy_style["trim"] = spec["trajectory_trim"]
            if legacy_style:
                self._pending_traj_styles[legacy_name] = legacy_style

        for t in self.trajs:
            o = self._pending_traj_styles.get(os.path.basename(t["path"].path))
            if not o:
                continue
            if o.get("color"):
                t["color"] = QtGui.QColor(o["color"])
            if o.get("width"):
                t["width"] = float(o["width"])
            if "visible" in o:
                t["visible"] = bool(o["visible"])
            if o.get("start_color"):
                t["start_color"] = QtGui.QColor(o["start_color"])
            if o.get("end_color"):
                t["end_color"] = QtGui.QColor(o["end_color"])
            if o.get("segments"):
                t["segments"] = _load_segments(o["segments"])
            trim = o.get("trim") or {}
            if trim:
                t["trim_enabled"] = bool(trim.get("enabled", t["trim_enabled"]))
                if "start_s" in trim:
                    t["trim_start"] = float(trim["start_s"])
                if "end_s" in trim:
                    t["trim_end"] = float(trim["end_s"])
        self._refresh_traj_list()

        # video layer style (frame_idx/opacity/visibility), keyed by file
        # basename - same restore-by-basename convention as trajectories;
        # the layer videos themselves are never auto-reloaded from this.
        self._pending_layer_styles = {}
        for o in disp.get("video_layers") or []:
            name = os.path.basename(o.get("file") or o.get("file_abs") or "")
            if name:
                self._pending_layer_styles[name] = o
        for l in self.layers:
            o = self._pending_layer_styles.get(os.path.basename(l["video"].path))
            if not o:
                continue
            if "frame_idx" in o:
                l["frame_idx"] = int(np.clip(
                    o["frame_idx"], 0, l["video"].n_frames - 1))
            if "opacity" in o:
                l["opacity"] = float(o["opacity"])
            if "visible" in o:
                l["visible"] = bool(o["visible"])
            if o.get("rect"):
                l["rect"] = tuple(int(v) for v in o["rect"])
        self._refresh_layer_list()

        if "show_endpoints" in disp:
            self.show_endpoints = bool(disp["show_endpoints"])
            self.chk_endpoints.blockSignals(True)
            self.chk_endpoints.setChecked(self.show_endpoints)
            self.chk_endpoints.blockSignals(False)
        if disp.get("endpoint_size"):
            self.endpoint_size = float(disp["endpoint_size"])
            self.spin_endpoint_size.blockSignals(True)
            self.spin_endpoint_size.setValue(self.endpoint_size)
            self.spin_endpoint_size.blockSignals(False)
        if disp.get("start_color"):
            self.start_color = QtGui.QColor(disp["start_color"])
        if disp.get("end_color"):
            self.end_color = QtGui.QColor(disp["end_color"])
        if disp.get("legend_text_color"):
            self.legend_text_color = QtGui.QColor(disp["legend_text_color"])
        if disp.get("legend_bg_color"):
            self.legend_bg_color = QtGui.QColor(disp["legend_bg_color"])

        leg = disp.get("legend") or {}
        self.legend = {k: leg.get(k, DEFAULT_LEGEND[k]) for k in DEFAULT_LEGEND}
        self.chk_legend.blockSignals(True)
        self.chk_legend.setChecked(bool(self.legend["enabled"]))
        self.chk_legend.blockSignals(False)
        self.combo_leg_corner.blockSignals(True)
        self.combo_leg_corner.setCurrentText(
            self.legend["corner"] if self.legend["corner"] in LEGEND_CORNERS
            else "top-right")
        self.combo_leg_corner.blockSignals(False)
        self.chk_leg_bold.blockSignals(True)
        self.chk_leg_bold.setChecked(bool(self.legend["bold"]))
        self.chk_leg_bold.blockSignals(False)
        self.spin_leg_cols.blockSignals(True)
        self.spin_leg_cols.setValue(int(np.clip(int(self.legend["columns"] or 1),
                                                1, 8)))
        self.spin_leg_cols.blockSignals(False)
        self.legend["columns"] = int(self.spin_leg_cols.value())
        for k, sb in self._leg_spins.items():
            sb.blockSignals(True)
            sb.setValue(float(self.legend[k]))
            sb.blockSignals(False)

        self.legend_entries = []
        for e in disp.get("legend_entries") or []:
            try:
                self.legend_entries.append({
                    "label": str(e.get("label", "")),
                    "kind": e.get("kind", "line"),
                    "style": e.get("style", "solid"),
                    "width": float(e.get("width", 8.0)),
                    "latex": bool(e.get("latex", False)),
                    "color": QtGui.QColor(e["color"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        self._refresh_legend_list()

        self.use_real_latex = bool(disp.get("usetex", False))
        self.latex_preamble = str(
            disp.get("latex_preamble", DEFAULT_LATEX_PREAMBLE))
        self.chk_usetex.blockSignals(True)
        self.chk_usetex.setChecked(self.use_real_latex and usetex_available())
        self.chk_usetex.blockSignals(False)
        self.edit_preamble.blockSignals(True)
        self.edit_preamble.setText(self.latex_preamble)
        self.edit_preamble.setEnabled(self.use_real_latex and usetex_available())
        self.edit_preamble.blockSignals(False)
        _LATEX_CACHE.clear()

        # -- timestamp --
        ts = disp.get("timestamp") or {}
        self.timestamp = {k: ts.get(k, DEFAULT_TIMESTAMP[k])
                          for k in DEFAULT_TIMESTAMP}
        if self.timestamp["corner"] not in LEGEND_CORNERS:
            self.timestamp["corner"] = DEFAULT_TIMESTAMP["corner"]
        if self.timestamp["source"] not in TIMESTAMP_SOURCES:
            self.timestamp["source"] = DEFAULT_TIMESTAMP["source"]
        if disp.get("timestamp_text_color"):
            self.ts_text_color = QtGui.QColor(disp["timestamp_text_color"])
        if disp.get("timestamp_bg_color"):
            self.ts_bg_color = QtGui.QColor(disp["timestamp_bg_color"])

        for widget, setter, value in (
            (self.chk_ts, "setChecked", bool(self.timestamp["enabled"])),
            (self.combo_ts_corner, "setCurrentText", self.timestamp["corner"]),
            (self.combo_ts_source, "setCurrentText", self.timestamp["source"]),
            (self.edit_ts_prefix, "setText", str(self.timestamp["prefix"])),
            (self.edit_ts_suffix, "setText", str(self.timestamp["suffix"])),
            (self.spin_ts_dec, "setValue", int(self.timestamp["decimals"])),
            (self.chk_ts_bold, "setChecked", bool(self.timestamp["bold"])),
            (self.chk_ts_latex, "setChecked",
             bool(self.timestamp["latex"]) and latex_available()),
        ):
            widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(False)
        for k, sb in self._ts_spins.items():
            sb.blockSignals(True)
            sb.setValue(float(self.timestamp[k]))
            sb.blockSignals(False)

        self._refresh_line_style_swatches()

        self._apply_interpolated_camera()
        if reload_files:
            msg = (f"Loaded {os.path.basename(path)} - reopened its video/"
                   "reference/trajectory/layer files")
            if missing_files:
                msg += f" ({len(missing_files)} not found, skipped)"
                QtWidgets.QMessageBox.warning(
                    self, "Load project",
                    "Some files this project referenced could not be found "
                    "and were skipped:\n\n" + "\n".join(missing_files))
        else:
            msg = (f"Loaded {os.path.basename(path)} (pose/keyframes/sync/"
                   "display/trim only - video and path files, if any, are "
                   "unaffected)")
        self.statusBar().showMessage(msg, 6000)

    def closeEvent(self, ev):
        if self.video is not None:
            self.video.release()
        for layer in self.layers:
            layer["video"].release()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()

    for a in sys.argv[1:]:
        if not os.path.exists(a):
            continue
        low = a.lower()
        if low.endswith(SUFFIX):
            win.load_calibration(a)
        elif low.endswith((".npz", ".csv", ".txt")):
            # all bare path args are added as trajectories; use the GUI's
            # "Open reference..." button to load a reference path.
            win._add_trajectory(a)
        elif win.video is None:
            win._load_video(a)
        else:
            # the base video slot is already filled; further video args
            # become extra composited layers instead.
            win._add_video_layer(a)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()