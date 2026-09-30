"""trim_marker.py - mark the cuts, write a spec, convert nothing.

Half of a two-part tool:

  trim_marker.py   (this file)  GUI. Load one or more videos of the same run
                                plus a rollout .npz, set in/out points and
                                per-camera frame offsets, then save a
                                "<name>.trimspec.json" describing the cuts.

  trim_runner.py                Headless. Walk a folder tree, execute every
                                spec it finds, skip anything already done,
                                survive being killed and resume.

Marking is interactive and cheap; encoding is slow and mechanical. Keeping
them apart means you can sit and mark twenty runs in ten minutes, then leave
the runner grinding overnight - and restart it as often as you like.

Requires: PyQt6, opencv-python, numpy, matplotlib.
    pip install PyQt6 opencv-python numpy matplotlib
"""
from __future__ import annotations

import os
import sys
import json
import math
import time

import numpy as np
import cv2

from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


SPEC_SCHEMA = "trimspec/1"
SPEC_SUFFIX = ".trimspec.json"


# ===========================================================================
# Helpers
# ===========================================================================
def _as_points(arr):
    """Coerce a ragged rollout/reference entry into an (N, D) point array."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.zeros((0, 2))
    rows, cols = arr.shape
    state_widths = (2, 3, 4, 5, 6)
    if cols in state_widths:
        return arr
    if rows in state_widths:
        return arr.T
    return arr


def _fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--.---"
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
    return f"{int(m):02d}:{s:06.3f}"


def _safe_stem(path):
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    keep = ("-_.() abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(c for c in stem if c in keep).strip().replace(" ", "_") or "src"


def _rel(path, base):
    """Path relative to the spec folder, so a marked tree can be moved."""
    if not path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(base))
    except ValueError:          # different drive on Windows
        return os.path.abspath(path).replace("\\", "/")
    return rel.replace("\\", "/")


def _resolve(path, base):
    """Inverse of _rel: absolute path from a spec-relative one."""
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Where each file dialog opens
# ---------------------------------------------------------------------------
# Videos and rollouts usually live in different places, so each kind of dialog
# keeps its own folder. Resolution order, first hit wins:
#
#   1. an environment variable   (TRIM_VIDEO_DIR / TRIM_NPZ_DIR / TRIM_SPEC_DIR)
#   2. a folder pinned from the Folders menu (persists between sessions)
#   3. the last folder you browsed to for that kind
#   4. the folder of whatever is already loaded
#   5. your home folder
#
# Pinning wins over history, so a pinned capture folder stays put even after
# you go rummaging somewhere else once.
ENV_DIR_VARS = {"video": "TRIM_VIDEO_DIR",
                "npz": "TRIM_NPZ_DIR",
                "spec": "TRIM_SPEC_DIR"}


class DirMemory:
    """Per-kind start folders for the file dialogs, stored in QSettings."""

    def __init__(self):
        self._s = QtCore.QSettings("trim_tools", "trim_marker")

    def _get(self, group, kind):
        v = self._s.value(f"{group}/{kind}", "", type=str)
        return v if v and os.path.isdir(v) else None

    def pinned(self, kind):
        env = os.environ.get(ENV_DIR_VARS.get(kind, ""), "")
        if env and os.path.isdir(env):
            return env
        return self._get("pinned", kind)

    def pin(self, kind, path):
        self._s.setValue(f"pinned/{kind}", path or "")

    def unpin(self, kind):
        self._s.remove(f"pinned/{kind}")

    def start(self, kind, fallback=None):
        for cand in (self.pinned(kind), self._get("last", kind), fallback,
                     os.path.expanduser("~")):
            if cand and os.path.isdir(cand):
                return cand
        return ""

    def remember(self, kind, path):
        """Record the folder a file came from (ignored while that kind is
        pinned - a pin is a deliberate choice, not a suggestion)."""
        if not path or self.pinned(kind):
            return
        d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        if os.path.isdir(d):
            self._s.setValue(f"last/{kind}", d)


_DIRS = None


def dirs():
    """Lazy singleton - QSettings wants a QApplication to exist first."""
    global _DIRS
    if _DIRS is None:
        _DIRS = DirMemory()
    return _DIRS


# ===========================================================================
# Trim bar : a slider plus draggable start/end handles
# ===========================================================================
class TrimBar(QtWidgets.QWidget):
    """Scrub slider with in/out markers drawn over the groove."""

    positionChanged = QtCore.pyqtSignal(int)
    trimChanged = QtCore.pyqtSignal(int, int)

    HANDLE_W = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(46)
        self.setMouseTracking(True)
        self._n = 1
        self._pos = 0
        self._start = 0
        self._end = 0
        self._drag = None  # None | 'pos' | 'start' | 'end'

    # -- state ------------------------------------------------------------
    def set_length(self, n, keep_trim=False):
        """Resize the timeline; keep_trim rescales in/out to the new length."""
        old_n = self._n
        self._n = max(1, int(n))
        if keep_trim and old_n > 1 and self._n > 1:
            f = (self._n - 1) / (old_n - 1)
            self._start = int(round(self._start * f))
            self._end = int(round(self._end * f))
            self._pos = int(round(self._pos * f))
        else:
            self._start = 0
            self._end = self._n - 1
            self._pos = 0
        self._start = int(np.clip(self._start, 0, self._n - 1))
        self._end = int(np.clip(self._end, 0, self._n - 1))
        self._pos = int(np.clip(self._pos, 0, self._n - 1))
        self.update()
        self.trimChanged.emit(self._start, self._end)

    def length(self):
        return self._n

    def position(self):
        return self._pos

    def trim(self):
        return self._start, self._end

    def set_position(self, i, emit=True):
        i = int(np.clip(i, 0, self._n - 1))
        if i == self._pos:
            return
        self._pos = i
        self.update()
        if emit:
            self.positionChanged.emit(self._pos)

    def set_trim(self, start, end, emit=True):
        start = int(np.clip(start, 0, self._n - 1))
        end = int(np.clip(end, 0, self._n - 1))
        if start > end:
            start, end = end, start
        self._start, self._end = start, end
        self.update()
        if emit:
            self.trimChanged.emit(self._start, self._end)

    # -- geometry ---------------------------------------------------------
    def _track_rect(self):
        m = self.HANDLE_W + 2
        return QtCore.QRect(m, 12, max(1, self.width() - 2 * m), 18)

    def _x_of(self, idx):
        r = self._track_rect()
        if self._n <= 1:
            return r.left()
        return r.left() + int(round(r.width() * idx / (self._n - 1)))

    def _idx_of(self, x):
        r = self._track_rect()
        if r.width() <= 0:
            return 0
        f = (x - r.left()) / r.width()
        return int(round(np.clip(f, 0.0, 1.0) * (self._n - 1)))

    # -- painting ---------------------------------------------------------
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = self._track_rect()

        p.fillRect(r, QtGui.QColor(226, 228, 233))

        xs, xe = self._x_of(self._start), self._x_of(self._end)
        sel = QtCore.QRect(xs, r.top(), max(1, xe - xs), r.height())
        p.fillRect(sel, QtGui.QColor(120, 175, 255, 170))

        p.setPen(QtGui.QPen(QtGui.QColor(150, 154, 160)))
        p.drawRect(r)

        for x, col in ((xs, QtGui.QColor(20, 110, 40)),
                       (xe, QtGui.QColor(160, 30, 30))):
            h = QtCore.QRect(x - self.HANDLE_W // 2, r.top() - 4,
                             self.HANDLE_W, r.height() + 8)
            p.fillRect(h, col)

        xp = self._x_of(self._pos)
        p.setPen(QtGui.QPen(QtGui.QColor(15, 15, 15), 2))
        p.drawLine(xp, r.top() - 6, xp, r.bottom() + 6)

        p.setPen(QtGui.QColor(70, 70, 70))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        band = QtCore.QRect(0, r.bottom() + 6, self.width(), 14)
        p.drawText(band, QtCore.Qt.AlignmentFlag.AlignLeft, f"in {self._start}")
        p.drawText(band, QtCore.Qt.AlignmentFlag.AlignHCenter, f"{self._pos}")
        p.drawText(band, QtCore.Qt.AlignmentFlag.AlignRight, f"out {self._end}")

    # -- interaction ------------------------------------------------------
    def _nearest(self, x):
        d_s = abs(x - self._x_of(self._start))
        d_e = abs(x - self._x_of(self._end))
        if min(d_s, d_e) <= 8:
            return "start" if d_s <= d_e else "end"
        return "pos"

    def mousePressEvent(self, ev):
        x = ev.position().x()
        self._drag = self._nearest(x)
        self._apply_drag(x)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            self.setCursor(QtGui.QCursor(
                QtCore.Qt.CursorShape.SizeHorCursor
                if self._nearest(ev.position().x()) != "pos"
                else QtCore.Qt.CursorShape.ArrowCursor))
            return
        self._apply_drag(ev.position().x())

    def mouseReleaseEvent(self, _):
        self._drag = None

    def _apply_drag(self, x):
        i = self._idx_of(x)
        if self._drag == "start":
            self._start = min(i, self._end)
            self.update()
            self.trimChanged.emit(self._start, self._end)
        elif self._drag == "end":
            self._end = max(i, self._start)
            self.update()
            self.trimChanged.emit(self._start, self._end)
        else:
            self.set_position(i)


# ===========================================================================
# One video source
# ===========================================================================
class VideoSource:
    """A single opened video file plus its alignment offset."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {path}")

        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:  # some containers lie; count the hard way
            n = 0
            while self.cap.grab():
                n += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.n_frames = max(1, n)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 1e-3 else 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.offset = 0          # frames, added on top of the time-mapped index
        self._cached_idx = -1
        self._last_frame = None

    @property
    def duration(self):
        return self.n_frames / max(self.fps, 1e-6)

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

    def invalidate(self):
        self._cached_idx = -1

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ===========================================================================
# One tile in the video grid
# ===========================================================================
class SourceTile(QtWidgets.QFrame):

    offsetChanged = QtCore.pyqtSignal(object)       # source
    removeRequested = QtCore.pyqtSignal(object)     # source
    moveRequested = QtCore.pyqtSignal(object, int)  # source, delta

    def __init__(self, source: VideoSource, parent=None):
        super().__init__(parent)
        self.source = source
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        self.lbl_title = QtWidgets.QLabel()
        self.lbl_title.setStyleSheet("font-size:11px; color:#333;")
        self.lbl_title.setToolTip(source.path)
        lay.addWidget(self.lbl_title)

        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(200, 140)
        self.view.setStyleSheet("background:#101014; border:1px solid #333;")
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Expanding)
        lay.addWidget(self.view, 1)

        foot = QtWidgets.QHBoxLayout()
        foot.setSpacing(3)
        self.btn_left = QtWidgets.QToolButton()
        self.btn_left.setText("◀")
        self.btn_left.setToolTip("Move earlier in export order")
        self.btn_right = QtWidgets.QToolButton()
        self.btn_right.setText("▶")
        self.btn_right.setToolTip("Move later in export order")
        self.spin_off = QtWidgets.QSpinBox()
        self.spin_off.setRange(-100000, 100000)
        self.spin_off.setPrefix("off ")
        self.spin_off.setToolTip(
            "Frame offset for this source relative to the master timeline")
        self.spin_off.setMaximumWidth(110)
        self.btn_del = QtWidgets.QToolButton()
        self.btn_del.setText("✕")
        self.btn_del.setToolTip("Remove this source")

        self.btn_left.clicked.connect(
            lambda: self.moveRequested.emit(self.source, -1))
        self.btn_right.clicked.connect(
            lambda: self.moveRequested.emit(self.source, +1))
        self.btn_del.clicked.connect(
            lambda: self.removeRequested.emit(self.source))
        self.spin_off.valueChanged.connect(self._on_offset)

        foot.addWidget(self.btn_left)
        foot.addWidget(self.btn_right)
        foot.addWidget(self.spin_off, 1)
        foot.addWidget(self.btn_del)
        lay.addLayout(foot)

        self.lbl_stat = QtWidgets.QLabel("")
        self.lbl_stat.setStyleSheet(
            "font-family:monospace; font-size:10px; color:#555;")
        lay.addWidget(self.lbl_stat)

    def _on_offset(self, v):
        self.source.offset = int(v)
        self.source.invalidate()
        self.offsetChanged.emit(self.source)

    def set_index(self, i, is_master):
        s = self.source
        tag = "  [MASTER]" if is_master else ""
        self.lbl_title.setText(f"<b>{i + 1}.</b> {os.path.basename(s.path)}{tag}")
        self.spin_off.setEnabled(not is_master)
        if is_master:
            self.spin_off.blockSignals(True)
            self.spin_off.setValue(0)
            self.spin_off.blockSignals(False)
            s.offset = 0

    def paint(self, frame, src_idx=None):
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w,
                           QtGui.QImage.Format.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.view.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        if src_idx is not None:
            s = self.source
            self.lbl_stat.setText(
                f"f {src_idx}/{s.n_frames - 1}  {s.fps:.2f}fps  {s.duration:.2f}s")

    def repaint_last(self):
        self.paint(self.source._last_frame)


# ===========================================================================
# Video panel : many sources on one shared timeline
# ===========================================================================
class MultiVideoPanel(QtWidgets.QWidget):

    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sources: list[VideoSource] = []
        self.tiles: list[SourceTile] = []
        self._build()

    # -- convenience ------------------------------------------------------
    def has_media(self):
        return bool(self.sources)

    @property
    def master(self):
        return self.sources[0] if self.sources else None

    @property
    def fps(self):
        return self.master.fps if self.sources else 30.0

    @property
    def n_frames(self):
        return self.master.n_frames if self.sources else 0

    @property
    def path(self):
        return self.master.path if self.sources else None

    # -- ui ---------------------------------------------------------------
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        top = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add video(s)…")
        self.btn_add.clicked.connect(self.open_dialog)
        self.btn_clear = QtWidgets.QPushButton("Clear all")
        self.btn_clear.clicked.connect(self.clear)
        self.lbl_count = QtWidgets.QLabel("<no videos>")
        self.lbl_count.setStyleSheet("color:#555;")
        top.addWidget(self.btn_add)
        top.addWidget(self.btn_clear)
        top.addWidget(self.lbl_count, 1)
        lay.addLayout(top)

        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        self.grid_host.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                     QtWidgets.QSizePolicy.Policy.Expanding)

        self.placeholder = QtWidgets.QLabel(
            "Add one or more videos of the same run\n(drag & drop works too)")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(
            "background:#101014; color:#888; border:1px solid #333;")
        self.placeholder.setMinimumHeight(240)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.grid_host)
        lay.addWidget(self.stack, 1)

        self.bar = TrimBar()
        self.bar.positionChanged.connect(self.show_frame)
        self.bar.trimChanged.connect(lambda *_: self._update_info())
        lay.addWidget(self.bar)

        row = QtWidgets.QHBoxLayout()
        self.btn_set_in = QtWidgets.QPushButton("Set in")
        self.btn_set_out = QtWidgets.QPushButton("Set out")
        self.btn_go_in = QtWidgets.QPushButton("⇤ in")
        self.btn_go_out = QtWidgets.QPushButton("out ⇥")
        self.btn_reset = QtWidgets.QPushButton("Reset trim")
        self.btn_set_in.clicked.connect(
            lambda: self.bar.set_trim(self.bar.position(), self.bar.trim()[1]))
        self.btn_set_out.clicked.connect(
            lambda: self.bar.set_trim(self.bar.trim()[0], self.bar.position()))
        self.btn_go_in.clicked.connect(
            lambda: self.bar.set_position(self.bar.trim()[0]))
        self.btn_go_out.clicked.connect(
            lambda: self.bar.set_position(self.bar.trim()[1]))
        self.btn_reset.clicked.connect(
            lambda: self.bar.set_trim(0, max(0, self.n_frames - 1)))
        for b in (self.btn_set_in, self.btn_set_out, self.btn_go_in,
                  self.btn_go_out, self.btn_reset):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("font-family:monospace; color:#333;")
        lay.addWidget(self.lbl_info)

    # -- io ---------------------------------------------------------------
    def open_dialog(self):
        start = dirs().start("video", os.path.dirname(self.path) if self.path else None)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add video sources", start,
            "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)")
        if not paths:
            return
        dirs().remember("video", paths[0])
        had = bool(self.sources)
        for p in paths:
            self.add(p, refresh=False)
        self._rebuild_grid()
        self._resync_timeline(keep_trim=had)

    def add(self, path, refresh=True):
        try:
            src = VideoSource(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Video", str(exc))
            return None
        had = bool(self.sources)
        self.sources.append(src)
        if refresh:
            self._rebuild_grid()
            self._resync_timeline(keep_trim=had)
        return src

    def clear(self):
        for s in self.sources:
            s.release()
        self.sources.clear()
        self._rebuild_grid()
        self.bar.set_length(1)
        self.lbl_count.setText("<no videos>")
        self.lbl_info.setText("")

    def _remove(self, src):
        was_master = bool(self.sources) and self.sources[0] is src
        src.release()
        self.sources = [s for s in self.sources if s is not src]
        self._rebuild_grid()
        if self.sources:
            self._resync_timeline(keep_trim=not was_master)
        else:
            self.bar.set_length(1)
            self.lbl_count.setText("<no videos>")
            self.lbl_info.setText("")

    def _move(self, src, delta):
        i = self.sources.index(src)
        j = int(np.clip(i + delta, 0, len(self.sources) - 1))
        if i == j:
            return
        self.sources.insert(j, self.sources.pop(i))
        self._rebuild_grid()
        if 0 in (i, j):          # master changed -> remap the timeline
            self._resync_timeline(keep_trim=True)
        else:
            self.show_frame(self.bar.position())

    # -- grid -------------------------------------------------------------
    def _rebuild_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.tiles = []

        if not self.sources:
            self.stack.setCurrentIndex(0)
            return
        self.stack.setCurrentIndex(1)

        n = len(self.sources)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        for k, src in enumerate(self.sources):
            tile = SourceTile(src)
            tile.spin_off.blockSignals(True)
            tile.spin_off.setValue(src.offset)
            tile.spin_off.blockSignals(False)
            tile.set_index(k, k == 0)
            tile.offsetChanged.connect(
                lambda *_: self.show_frame(self.bar.position()))
            tile.removeRequested.connect(self._remove)
            tile.moveRequested.connect(self._move)
            self.grid.addWidget(tile, k // cols, k % cols)
            self.tiles.append(tile)

        self.lbl_count.setText(
            f"{n} source{'s' if n != 1 else ''}  ·  master: "
            f"{os.path.basename(self.sources[0].path)}")

    def _resync_timeline(self, keep_trim):
        if not self.sources:
            return
        self.bar.set_length(self.master.n_frames, keep_trim=keep_trim)
        self.show_frame(self.bar.position())

    def set_offset(self, src, frames):
        """Programmatic offset change (used when restoring a spec)."""
        src.offset = int(frames)
        src.invalidate()
        for tile in self.tiles:
            if tile.source is src:
                tile.spin_off.blockSignals(True)
                tile.spin_off.setValue(int(frames))
                tile.spin_off.blockSignals(False)

    # -- index mapping ----------------------------------------------------
    def map_index(self, master_idx, src):
        """Master-timeline frame -> this source's frame (clipped)."""
        if src is self.master:
            j = master_idx
        else:
            t = master_idx / max(self.master.fps, 1e-6)
            j = int(round(t * src.fps)) + src.offset
        return int(np.clip(j, 0, src.n_frames - 1))

    def map_trim(self, src):
        s, e = self._clamped_trim()
        return self.map_index(s, src), self.map_index(e, src)

    # -- display ----------------------------------------------------------
    def show_frame(self, idx):
        if not self.sources:
            return
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        for tile in self.tiles:
            j = self.map_index(idx, tile.source)
            tile.paint(tile.source.frame_at(j), j)
        self.bar.set_position(idx, emit=False)
        self._update_info()
        self.frameChanged.emit(idx)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        for tile in self.tiles:
            tile.repaint_last()

    def _update_info(self):
        if not self.sources:
            self.lbl_info.setText("")
            return
        i = self._current_idx()
        s, e = self._clamped_trim()
        fps = self.fps
        line1 = (f"master frame {i}/{self.n_frames - 1}   "
                 f"t={_fmt_time(i / fps)}   fps={fps:.3f}")
        line2 = (f"trim [{s}..{e}]  ({e - s + 1} frames, "
                 f"{(e - s + 1) / fps:.3f}s)")
        if len(self.sources) > 1:
            durs = [x.duration for x in self.sources]
            line2 += ("   lengths: "
                      + " / ".join(f"{d:.2f}s" for d in durs)
                      + f"  (spread {max(durs) - min(durs):.3f}s)")
        self.lbl_info.setText(line1 + "\n" + line2)

    # -- spec -------------------------------------------------------------
    def spec_entry(self, src, number, out_rel, spec_dir):
        """One video clip, described well enough for the runner to encode it."""
        s, e = self.map_trim(src)
        return {
            "number": number,
            "source": _rel(src.path, spec_dir),
            "source_abs": os.path.abspath(src.path).replace("\\", "/"),
            "output": out_rel,
            "fps": src.fps,
            "width": src.w,
            "height": src.h,
            "n_frames": src.n_frames,
            "offset_frames": int(src.offset),
            "trim_frames": [int(s), int(e)],
            "duration_s": round((e - s + 1) / max(src.fps, 1e-6), 4),
            "is_master": src is self.master,
        }

    def _current_idx(self):
        if not self.sources:
            return 0
        return int(np.clip(self.bar.position(), 0, self.n_frames - 1))

    def _clamped_trim(self):
        if not self.sources:
            return 0, 0
        hi = self.n_frames - 1
        s, e = self.bar.trim()
        return int(np.clip(s, 0, hi)), int(np.clip(e, 0, hi))


# ===========================================================================
# NPZ rollout panel
# ===========================================================================
class NpzPanel(QtWidgets.QWidget):

    frameChanged = QtCore.pyqtSignal(int)

    PARAM_NAMES = {0: "Cf", 1: "Cr", 2: "muf", 3: "mur", 4: "Cro"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = None
        self.data = None
        self.n_frames = 0
        self.dt = 1.0 / 25.0
        self._build()

    def has_media(self):
        return self.data is not None

    # -- ui ---------------------------------------------------------------
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        top = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Load .npz…")
        self.btn_open.clicked.connect(self.open_dialog)
        self.lbl_path = QtWidgets.QLabel("<no npz>")
        self.lbl_path.setStyleSheet("color:#555;")
        self.chk_rollout = QtWidgets.QCheckBox("MPC rollout")
        self.chk_rollout.setChecked(True)
        self.chk_ref = QtWidgets.QCheckBox("Reference")
        self.chk_ref.setChecked(True)
        self.chk_rollout.toggled.connect(lambda *_: self.redraw())
        self.chk_ref.toggled.connect(lambda *_: self.redraw())
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_path, 1)
        top.addWidget(self.chk_rollout)
        top.addWidget(self.chk_ref)
        lay.addLayout(top)

        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Expanding)
        lay.addWidget(self.canvas, 1)

        self.bar = TrimBar()
        self.bar.positionChanged.connect(self.show_frame)
        self.bar.trimChanged.connect(lambda *_: self._update_info())
        lay.addWidget(self.bar)

        row = QtWidgets.QHBoxLayout()
        self.btn_set_in = QtWidgets.QPushButton("Set in")
        self.btn_set_out = QtWidgets.QPushButton("Set out")
        self.btn_go_in = QtWidgets.QPushButton("⇤ in")
        self.btn_go_out = QtWidgets.QPushButton("out ⇥")
        self.btn_reset = QtWidgets.QPushButton("Reset trim")
        self.btn_set_in.clicked.connect(
            lambda: self.bar.set_trim(self.bar.position(), self.bar.trim()[1]))
        self.btn_set_out.clicked.connect(
            lambda: self.bar.set_trim(self.bar.trim()[0], self.bar.position()))
        self.btn_go_in.clicked.connect(
            lambda: self.bar.set_position(self.bar.trim()[0]))
        self.btn_go_out.clicked.connect(
            lambda: self.bar.set_position(self.bar.trim()[1]))
        self.btn_reset.clicked.connect(
            lambda: self.bar.set_trim(0, max(0, self.n_frames - 1)))
        for b in (self.btn_set_in, self.btn_set_out, self.btn_go_in,
                  self.btn_go_out, self.btn_reset):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("font-family:monospace; color:#333;")
        lay.addWidget(self.lbl_info)

    # -- io ---------------------------------------------------------------
    def open_dialog(self):
        start = dirs().start("npz", os.path.dirname(self.path) if self.path else None)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open rollout npz", start,
            "NumPy archive (*.npz);;All files (*)")
        if path:
            dirs().remember("npz", path)
            self.load(path)

    def load(self, path):
        try:
            raw = np.load(path, allow_pickle=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "npz", f"{exc}")
            return

        self._idx = 0

        self.path = path
        self.keys = list(raw.files)
        self.data = {k: raw[k] for k in self.keys}

        state = np.asarray(self.data["state"], dtype=float)
        self.state = state
        self.n_frames = len(state)

        t = np.asarray(self.data["time"], dtype=float)
        if t.size:
            t = t - t[0]
            if np.nanmax(t) > 1e6:   # logger stores perf_counter_ns
                t = t * 1e-9
        self.time = t
        if len(t) > 1:
            self.dt = float(np.median(np.diff(t)))

        try:
            pr = np.asarray(self.data["params"], dtype=float)
            if pr.ndim == 2 and pr.shape[0] == self.n_frames:
                self.params = [pr[:, i] for i in range(pr.shape[1])]
            elif pr.ndim == 2:
                self.params = [pr[i, :] for i in range(pr.shape[0])]
            else:
                self.params = []
        except Exception:
            self.params = []

        self.ctrl = np.asarray(self.data.get("ctrl", np.zeros((self.n_frames, 2))),
                               dtype=float)
        self.model_idx = self.data.get("model_index", np.zeros(self.n_frames))
        self.mpc_rollout = self.data.get("mpc_rollout")
        self.ref_trajectory = self.data.get("ref_trajectory")
        self.solve_time = self.data.get("solve_time")

        mu = self.data.get("mu_est")
        if mu is not None:
            try:
                mu = np.array([np.nan if v is None else v for v in mu], dtype=float)
                mu = mu if np.any(np.isfinite(mu)) else None
            except Exception:
                mu = None
        self.mu_est = mu

        self.lbl_path.setText(os.path.basename(path))
        self.lbl_path.setToolTip(path)
        self._build_axes()
        self.bar.set_length(self.n_frames)
        self.show_frame(0)

    # -- plotting ---------------------------------------------------------
    def _build_axes(self):
        self.fig.clear()
        n_par = min(len(self.params), 5)
        gs = self.fig.add_gridspec(max(1, n_par), 2, width_ratios=[1.7, 1.0])

        self.ax = self.fig.add_subplot(gs[:, 0])
        x, y = self.state[:, 0], self.state[:, 1]
        self.ax.plot(x, y, "-", color="#9fb6d6", lw=1.0, zorder=1)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X", fontsize=8)
        self.ax.set_ylabel("Y", fontsize=8)
        self.ax.tick_params(labelsize=7)
        m = max(0.5, 0.08 * max(np.ptp(x), np.ptp(y), 1.0))
        self.ax.set_xlim(x.min() - m, x.max() + m)
        self.ax.set_ylim(y.min() - m, y.max() + m)

        self.trail, = self.ax.plot([], [], "-", color="#1f4e9c", lw=1.8, zorder=3)
        self.point, = self.ax.plot([], [], "o", color="k", ms=8, zorder=6)
        self.heading, = self.ax.plot([], [], "-", color="k", lw=2, zorder=6)
        self.roll_line, = self.ax.plot([], [], "m--", lw=1.4, alpha=0.8, zorder=4)
        self.roll_pts, = self.ax.plot([], [], "mo", ms=4, alpha=0.9, zorder=5)
        self.ref_line, = self.ax.plot([], [], "g--", lw=1.4, alpha=0.8, zorder=4)
        self.ref_pts, = self.ax.plot([], [], "g^", ms=4, alpha=0.9, zorder=5)
        self.span_in = self.ax.plot([], [], "s", color="#146e28", ms=7, zorder=7)[0]
        self.span_out = self.ax.plot([], [], "s", color="#a01e1e", ms=7, zorder=7)[0]

        self.par_axes, self.par_vlines, self.par_dots = [], [], []
        for i in range(n_par):
            axp = self.fig.add_subplot(gs[i, 1])
            axp.plot(self.time[:len(self.params[i])], self.params[i],
                     "-", color="#2b6cb0", lw=1.2)
            axp.set_ylabel(self.PARAM_NAMES.get(i, f"p{i}"), fontsize=8)
            axp.grid(True, alpha=0.3)
            axp.tick_params(labelsize=6)
            if len(self.time) > 1:
                axp.set_xlim(self.time[0], self.time[-1])
            if i < n_par - 1:
                axp.set_xticklabels([])
            else:
                axp.set_xlabel("t (s)", fontsize=7)
            self.par_vlines.append(axp.axvline(0, color="r", ls="--", lw=1.0))
            self.par_dots.append(axp.plot([], [], "ro", ms=4)[0])
            self.par_axes.append(axp)

        self.canvas.draw_idle()

    def show_frame(self, idx):
        if self.data is None:
            return
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        self._idx = idx
        self.bar.set_position(idx, emit=False)
        self.redraw()
        self._update_info()
        self.frameChanged.emit(idx)

    def redraw(self):
        if self.data is None:
            return
        i = self._current_idx()
        s, e = self._clamped_trim()

        self.trail.set_data(self.state[s:i + 1, 0], self.state[s:i + 1, 1])
        self.point.set_data([self.state[i, 0]], [self.state[i, 1]])
        th = self.state[i, 2]
        L = 0.35
        self.heading.set_data([self.state[i, 0], self.state[i, 0] + L * np.cos(th)],
                              [self.state[i, 1], self.state[i, 1] + L * np.sin(th)])
        self.span_in.set_data([self.state[s, 0]], [self.state[s, 1]])
        self.span_out.set_data([self.state[e, 0]], [self.state[e, 1]])

        def _seg(container, want):
            if not want or container is None or i >= len(container):
                return None
            seg = container[i]
            if seg is None or len(seg) == 0:
                return None
            pts = _as_points(seg)
            return (pts[:, 0], pts[:, 1]) if pts.shape[1] >= 2 else None

        r = _seg(self.mpc_rollout, self.chk_rollout.isChecked())
        self.roll_line.set_data(*(r or ([], [])))
        self.roll_pts.set_data(*(r or ([], [])))
        rf = _seg(self.ref_trajectory, self.chk_ref.isChecked())
        self.ref_line.set_data(*(rf or ([], [])))
        self.ref_pts.set_data(*(rf or ([], [])))

        t = self.time[i] if i < len(self.time) else 0.0
        for k, (vl, dot) in enumerate(zip(self.par_vlines, self.par_dots)):
            vl.set_xdata([t, t])
            if i < len(self.params[k]):
                dot.set_data([t], [self.params[k][i]])

        self.canvas.draw_idle()

    def _update_info(self):
        if self.data is None:
            self.lbl_info.setText("")
            return
        i = self._current_idx()
        s, e = self._clamped_trim()
        st = self.state[i]
        solve = "n/a"
        if self.solve_time is not None and i < len(self.solve_time):
            try:
                solve = f"{float(self.solve_time[i]):.2f} ms"
            except Exception:
                pass
        mu = ""
        if self.mu_est is not None and i < len(self.mu_est):
            mu = f"  mu={self.mu_est[i]:.3f}"
        t = self.time[i] if i < len(self.time) else 0.0
        self.lbl_info.setText(
            f"frame {i}/{self.n_frames - 1}   t={_fmt_time(t)}   "
            f"dt≈{self.dt * 1e3:.1f} ms\n"
            f"θ={st[2]:.3f}  vx={st[3]:.3f}  vy={st[4]:.3f}  ω={st[5]:.3f}  "
            f"solve={solve}{mu}\n"
            f"trim [{s}..{e}]  ({e - s + 1} samples, {(e - s + 1) * self.dt:.3f}s)")

    # -- spec -------------------------------------------------------------
    def spec_entry(self, out_rel, spec_dir, rezero, offset_frames):
        s, e = self.bar.trim()
        return {
            "source": _rel(self.path, spec_dir),
            "source_abs": os.path.abspath(self.path).replace("\\", "/"),
            "output": out_rel,
            "trim_samples": [int(s), int(e)],
            "n_samples": int(self.n_frames),
            "dt": float(self.dt),
            "duration_s": round((e - s + 1) * self.dt, 4),
            "rezero_time": bool(rezero),
            "offset_frames": int(offset_frames),
        }

    def _current_idx(self):
        """Playhead, clamped to the data actually loaded right now."""
        if self.data is None or self.n_frames == 0:
            return 0
        return int(np.clip(getattr(self, "_idx", 0), 0, self.n_frames - 1))

    def _clamped_trim(self):
        if self.n_frames == 0:
            return 0, 0
        s, e = self.bar.trim()
        hi = self.n_frames - 1
        return int(np.clip(s, 0, hi)), int(np.clip(e, 0, hi))


# ===========================================================================
# Main window
# ===========================================================================
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trim marker - video ↔ npz")
        self.resize(1700, 980)

        self.video = MultiVideoPanel()
        self.npz = NpzPanel()

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for w, title in ((self.video, "Video sources"), (self.npz, "NPZ rollout")):
            box = QtWidgets.QGroupBox(title)
            l = QtWidgets.QVBoxLayout(box)
            l.setContentsMargins(4, 14, 4, 4)
            l.addWidget(w)
            split.addWidget(box)
        split.setSizes([880, 820])

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addWidget(split, 1)
        lay.addWidget(self._transport())
        lay.addWidget(self._spec_bar())
        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)

        self._syncing = False
        self.video.frameChanged.connect(self._video_moved)
        self.npz.frameChanged.connect(self._npz_moved)

        self.statusBar().showMessage(
            "Mark the cuts here, then run trim_runner.py over the tree to "
            "encode them. Nothing is converted by this window.")
        self._folder_menu()
        self._shortcuts()
        self.setAcceptDrops(True)

    # -- start folders ----------------------------------------------------
    def _folder_menu(self):
        menu = self.menuBar().addMenu("&Folders")
        self._folder_actions = {}
        for kind, label in (("video", "Videos"), ("npz", "Rollout .npz"),
                            ("spec", "Specs")):
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, k=kind: self._pin_dir(k))
            self._folder_actions[kind] = act
        menu.addSeparator()
        act_clear = menu.addAction("Unpin all (use last-used folders)")
        act_clear.triggered.connect(self._unpin_all)
        self._refresh_folder_menu()

    def _refresh_folder_menu(self):
        for kind, act in self._folder_actions.items():
            base = {"video": "Videos", "npz": "Rollout .npz",
                    "spec": "Specs"}[kind]
            pin = dirs().pinned(kind)
            act.setText(f"{base}…   {pin}" if pin else f"{base}…   (last used)")
            env = os.environ.get(ENV_DIR_VARS[kind], "")
            frozen = bool(env and os.path.isdir(env))
            act.setEnabled(not frozen)
            act.setToolTip(f"Set by {ENV_DIR_VARS[kind]}" if frozen
                           else "Pick a folder these dialogs always open in")

    def _pin_dir(self, kind):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, f"Always start here for {kind}", dirs().start(kind))
        if d:
            dirs().pin(kind, d)
            self._refresh_folder_menu()
            self.statusBar().showMessage(f"{kind} dialogs now open in {d}", 6000)

    def _unpin_all(self):
        for kind in self._folder_actions:
            dirs().unpin(kind)
        self._refresh_folder_menu()
        self.statusBar().showMessage(
            "Unpinned - dialogs will follow the last folder you used", 6000)

    # -- drag & drop ------------------------------------------------------
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        had = self.video.has_media()
        added = False
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if not os.path.exists(p):
                continue
            low = p.lower()
            if low.endswith(SPEC_SUFFIX):
                self.load_spec(p)
                return
            if low.endswith(".npz"):
                dirs().remember("npz", p)
                self.npz.load(p)
            else:
                dirs().remember("video", p)
                self.video.add(p, refresh=False)
                added = True
        if added:
            self.video._rebuild_grid()
            self.video._resync_timeline(keep_trim=had)
        self._autofill_spec_path()

    # -- transport --------------------------------------------------------
    def _transport(self):
        w = QtWidgets.QGroupBox("Transport")
        lay = QtWidgets.QHBoxLayout(w)

        self.chk_lock = QtWidgets.QCheckBox(
            "Lock playheads (time-aligned from in-points)")
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)

        self.spin_rate = QtWidgets.QDoubleSpinBox()
        self.spin_rate.setRange(0.05, 8.0)
        self.spin_rate.setSingleStep(0.25)
        self.spin_rate.setValue(1.0)
        self.spin_rate.setPrefix("speed ×")

        self.cmb_master = QtWidgets.QComboBox()
        self.cmb_master.addItems(["Master: video", "Master: npz"])

        self.btn_prev = QtWidgets.QPushButton("◀ step")
        self.btn_next = QtWidgets.QPushButton("step ▶")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))

        self.spin_offset = QtWidgets.QSpinBox()
        self.spin_offset.setRange(-100000, 100000)
        self.spin_offset.setPrefix("npz offset ")
        self.spin_offset.setSuffix(" frames")
        self.spin_offset.setToolTip(
            "Extra frame offset applied to the npz playhead when locked.")

        for x in (self.btn_play, self.spin_rate, self.cmb_master,
                  self.btn_prev, self.btn_next, self.chk_lock, self.spin_offset):
            lay.addWidget(x)
        lay.addStretch(1)

        self.lbl_align = QtWidgets.QLabel("")
        self.lbl_align.setStyleSheet("font-family:monospace; color:#444;")
        lay.addWidget(self.lbl_align)
        return w

    # -- spec bar ---------------------------------------------------------
    def _spec_bar(self):
        w = QtWidgets.QGroupBox("Spec")
        g = QtWidgets.QGridLayout(w)

        self.ed_spec = QtWidgets.QLineEdit("")
        self.ed_spec.setPlaceholderText(
            "…/run_042/run_042" + SPEC_SUFFIX + "  (load media to autofill)")
        btn_spec = QtWidgets.QPushButton("Browse…")
        btn_spec.clicked.connect(self._pick_spec_path)
        btn_load = QtWidgets.QPushButton("Load spec…")
        btn_load.clicked.connect(self._pick_spec_to_load)

        self.ed_outdir = QtWidgets.QLineEdit("trimmed")
        self.ed_outdir.setToolTip(
            "Where the runner writes. Relative paths resolve against the spec\n"
            "folder, so the whole tree stays movable.")

        self.ed_video_name = QtWidgets.QLineEdit("clip_trimmed")
        self.ed_video_name.setToolTip(
            "Base name; each source gets its number appended (name_1, name_2, …)")
        self.ed_npz_name = QtWidgets.QLineEdit("rollout_trimmed")
        self.cmb_vid_ext = QtWidgets.QComboBox()
        self.cmb_vid_ext.addItems([".mp4", ".mov", ".avi", ".mkv"])

        self.spin_start_num = QtWidgets.QSpinBox()
        self.spin_start_num.setRange(0, 9999)
        self.spin_start_num.setValue(1)
        self.spin_start_num.setPrefix("start at ")
        self.spin_pad = QtWidgets.QSpinBox()
        self.spin_pad.setRange(1, 5)
        self.spin_pad.setValue(1)
        self.spin_pad.setPrefix("digits ")
        self.chk_stem = QtWidgets.QCheckBox("Append source name too")
        self.chk_stem.setToolTip("name_1_frontcam.mp4 instead of name_1.mp4")

        self.chk_ffmpeg = QtWidgets.QCheckBox("Use ffmpeg (keeps audio/quality)")
        self.chk_ffmpeg.setChecked(True)
        self.chk_rezero = QtWidgets.QCheckBox("Re-zero npz time")
        self.chk_rezero.setChecked(True)
        self.chk_sidecar = QtWidgets.QCheckBox("Write pairing .json")
        self.chk_sidecar.setChecked(True)
        self.chk_clear = QtWidgets.QCheckBox("Clear after save")
        self.chk_clear.setToolTip(
            "Empty both panels once the spec is written, ready for the next run.")

        self.spin_crf = QtWidgets.QSpinBox()
        self.spin_crf.setRange(0, 51)
        self.spin_crf.setValue(18)
        self.spin_crf.setPrefix("crf ")
        self.cmb_preset = QtWidgets.QComboBox()
        self.cmb_preset.addItems(["ultrafast", "veryfast", "fast", "medium", "slow"])
        self.cmb_preset.setCurrentText("veryfast")

        btn_save = QtWidgets.QPushButton("Save spec")
        btn_save.clicked.connect(self._save_spec)
        btn_save.setStyleSheet("font-weight:bold;")

        g.addWidget(QtWidgets.QLabel("Spec file"), 0, 0)
        g.addWidget(self.ed_spec, 0, 1, 1, 3)
        g.addWidget(btn_spec, 0, 4)
        g.addWidget(btn_load, 0, 5)

        g.addWidget(QtWidgets.QLabel("Output dir"), 1, 0)
        g.addWidget(self.ed_outdir, 1, 1)
        g.addWidget(QtWidgets.QLabel("Video base"), 1, 2)
        g.addWidget(self.ed_video_name, 1, 3)
        g.addWidget(self.cmb_vid_ext, 1, 4)
        g.addWidget(self.chk_stem, 1, 5)

        g.addWidget(QtWidgets.QLabel("NPZ name"), 2, 0)
        g.addWidget(self.ed_npz_name, 2, 1)
        g.addWidget(self.spin_start_num, 2, 2)
        g.addWidget(self.spin_pad, 2, 3)
        g.addWidget(self.spin_crf, 2, 4)
        g.addWidget(self.cmb_preset, 2, 5)

        g.addWidget(self.chk_ffmpeg, 3, 1)
        g.addWidget(self.chk_rezero, 3, 2)
        g.addWidget(self.chk_sidecar, 3, 3)
        g.addWidget(self.chk_clear, 3, 4)
        g.addWidget(btn_save, 3, 5)

        self.lbl_preview = QtWidgets.QLabel("")
        self.lbl_preview.setStyleSheet(
            "font-family:monospace; color:#555; font-size:11px;")
        g.addWidget(self.lbl_preview, 4, 0, 1, 6)

        for widget, sig in ((self.ed_video_name, "textChanged"),
                            (self.ed_npz_name, "textChanged"),
                            (self.ed_outdir, "textChanged"),
                            (self.spin_start_num, "valueChanged"),
                            (self.spin_pad, "valueChanged"),
                            (self.chk_stem, "toggled"),
                            (self.cmb_vid_ext, "currentIndexChanged")):
            getattr(widget, sig).connect(lambda *_: self._update_preview())
        return w

    def _shortcuts(self):
        def sc(key, fn):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
        sc("Space", lambda: self.btn_play.toggle())
        sc("Left", lambda: self._step(-1))
        sc("Right", lambda: self._step(+1))
        sc("Shift+Left", lambda: self._step(-10))
        sc("Shift+Right", lambda: self._step(+10))
        sc("I", self._set_in)
        sc("O", self._set_out)
        sc("Ctrl+S", self._save_spec)

    # -- master helpers ---------------------------------------------------
    def _master(self):
        return self.video if self.cmb_master.currentIndex() == 0 else self.npz

    def _set_in(self):
        m = self._master()
        m.bar.set_trim(m.bar.position(), m.bar.trim()[1])

    def _set_out(self):
        m = self._master()
        m.bar.set_trim(m.bar.trim()[0], m.bar.position())

    def _step(self, d):
        m = self._master()
        m.bar.set_position(m.bar.position() + d)

    def _toggle_play(self, on):
        self.btn_play.setText("⏸ Pause" if on else "▶ Play")
        self.timer.start() if on else self.timer.stop()

    def _tick(self):
        m = self._master()
        if not m.has_media():
            return
        rate = self.spin_rate.value()
        fps = self.video.fps if m is self.video else (1.0 / max(self.npz.dt, 1e-6))
        self.timer.setInterval(max(10, int(1000.0 / max(fps * rate, 1e-3))))
        s, e = m.bar.trim()
        nxt = m.bar.position() + 1
        if nxt > e:
            nxt = s
        m.bar.set_position(nxt)

    # -- sync -------------------------------------------------------------
    def _video_moved(self, idx):
        if (self._syncing or not self.chk_lock.isChecked()
                or self.cmb_master.currentIndex() != 0
                or not self.npz.has_media()):
            self._update_align()
            return
        self._syncing = True
        try:
            vs = self.video.bar.trim()[0]
            ns = self.npz.bar.trim()[0]
            t = (idx - vs) / max(self.video.fps, 1e-6)
            j = ns + int(round(t / max(self.npz.dt, 1e-9))) + self.spin_offset.value()
            self.npz.show_frame(int(np.clip(j, 0, self.npz.n_frames - 1)))
        finally:
            self._syncing = False
        self._update_align()

    def _npz_moved(self, idx):
        if (self._syncing or not self.chk_lock.isChecked()
                or self.cmb_master.currentIndex() != 1
                or not self.video.has_media()):
            self._update_align()
            return
        self._syncing = True
        try:
            vs = self.video.bar.trim()[0]
            ns = self.npz.bar.trim()[0]
            t = (idx - ns - self.spin_offset.value()) * self.npz.dt
            j = vs + int(round(t * self.video.fps))
            self.video.show_frame(int(np.clip(j, 0, self.video.n_frames - 1)))
        finally:
            self._syncing = False
        self._update_align()

    def _update_align(self):
        parts = []
        vd = nd = None
        if self.video.has_media():
            vs, ve = self.video.bar.trim()
            vd = (ve - vs + 1) / self.video.fps
            parts.append(f"video {vd:7.3f}s ×{len(self.video.sources)}")
        if self.npz.has_media():
            ns, ne = self.npz.bar.trim()
            nd = (ne - ns + 1) * self.npz.dt
            parts.append(f"npz {nd:7.3f}s")
        if vd is not None and nd is not None:
            parts.append(f"Δ {vd - nd:+.3f}s")
        self.lbl_align.setText("   ".join(parts))
        self._update_preview()

    # -- naming -----------------------------------------------------------
    def _anchor_path(self):
        """The file whose folder the spec lands in by default."""
        return self.npz.path or self.video.path

    def _autofill_spec_path(self):
        if self.ed_spec.text().strip():
            return
        anchor = self._anchor_path()
        if not anchor:
            return
        folder = os.path.dirname(os.path.abspath(anchor))
        self.ed_spec.setText(
            os.path.join(folder, _safe_stem(anchor) + SPEC_SUFFIX))
        self._update_preview()

    def _spec_dir(self):
        p = self.ed_spec.text().strip()
        return os.path.dirname(os.path.abspath(p)) if p else ""

    def _out_rel(self, filename):
        sub = self.ed_outdir.text().strip().replace("\\", "/").strip("/")
        return f"{sub}/{filename}" if sub else filename

    def _video_out_names(self):
        base = self.ed_video_name.text().strip() or "clip_trimmed"
        ext = self.cmb_vid_ext.currentText()
        pad = self.spin_pad.value()
        n0 = self.spin_start_num.value()
        names = []
        for k, src in enumerate(self.video.sources):
            num = str(n0 + k).zfill(pad)
            stem = f"_{_safe_stem(src.path)}" if self.chk_stem.isChecked() else ""
            names.append(f"{base}_{num}{stem}{ext}")
        return names

    def _update_preview(self):
        bits = []
        if self.video.has_media():
            names = [self._out_rel(n) for n in self._video_out_names()]
            shown = (names if len(names) <= 3
                     else names[:2] + [f"… (+{len(names) - 2} more)"])
            bits += shown
        if self.npz.has_media():
            bits.append(self._out_rel(
                (self.ed_npz_name.text().strip() or "rollout_trimmed") + ".npz"))
        self.lbl_preview.setText(
            ("runner will write:  " + "   ".join(bits)) if bits else "")

    # -- spec io ----------------------------------------------------------
    def _pick_spec_path(self):
        # A path already in the box beats folder memory - it was derived from
        # the loaded media, which is where the spec usually belongs.
        start = self.ed_spec.text().strip() or dirs().start("spec", self._spec_dir())
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save spec as", start, f"Trim spec (*{SPEC_SUFFIX});;JSON (*.json)")
        if p:
            if not p.endswith(".json"):
                p += SPEC_SUFFIX
            dirs().remember("spec", p)
            self.ed_spec.setText(p)
            self._update_preview()

    def _pick_spec_to_load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load spec", dirs().start("spec", self._spec_dir()),
            f"Trim spec (*{SPEC_SUFFIX});;JSON (*.json)")
        if p:
            dirs().remember("spec", p)
            self.load_spec(p)

    def build_spec(self):
        spec_path = self.ed_spec.text().strip()
        if not spec_path:
            raise RuntimeError("pick a spec file path first")
        spec_dir = os.path.dirname(os.path.abspath(spec_path))

        videos = []
        if self.video.has_media():
            n0 = self.spin_start_num.value()
            for k, (src, name) in enumerate(
                    zip(self.video.sources, self._video_out_names())):
                videos.append(self.video.spec_entry(
                    src, n0 + k, self._out_rel(name), spec_dir))

        npz_entry = None
        if self.npz.has_media():
            name = self.ed_npz_name.text().strip() or "rollout_trimmed"
            npz_entry = self.npz.spec_entry(
                self._out_rel(name + ".npz"), spec_dir,
                self.chk_rezero.isChecked(), self.spin_offset.value())

        if not videos and npz_entry is None:
            raise RuntimeError("nothing loaded - no clips and no npz to describe")

        return {
            "schema": SPEC_SCHEMA,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "created_by": "trim_marker.py",
            "output_dir": self.ed_outdir.text().strip().replace("\\", "/"),
            "encode": {
                "use_ffmpeg": self.chk_ffmpeg.isChecked(),
                "crf": self.spin_crf.value(),
                "preset": self.cmb_preset.currentText(),
                "video_ext": self.cmb_vid_ext.currentText(),
            },
            "naming": {
                "video_base": self.ed_video_name.text().strip(),
                "npz_base": self.ed_npz_name.text().strip(),
                "start_numbe_r": self.spin_start_num.value(),
                "digits": self.spin_pad.value(),
                "append_source_stem": self.chk_stem.isChecked(),
            },
            "master": {
                "source": (_rel(self.video.path, spec_dir)
                           if self.video.has_media() else None),
                "fps": self.video.fps if self.video.has_media() else None,
                "trim_frames": (list(self.video.bar.trim())
                                if self.video.has_media() else None),
            },
            "videos": videos,
            "npz": npz_entry,
            "pairing_json": (self.chk_sidecar.isChecked()
                             and self._out_rel(
                                 (self.ed_npz_name.text().strip()
                                  or self.ed_video_name.text().strip()
                                  or "pair") + "_pairing.json")),
        }

    def _save_spec(self):
        try:
            spec = self.build_spec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save spec", str(exc))
            return
        path = self.ed_spec.text().strip()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(spec, f, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save spec", str(exc))
            return

        n_v = len(spec["videos"])
        log(f"wrote {path}  ({n_v} clip(s)"
            f"{', npz' if spec['npz'] else ''})")
        self.statusBar().showMessage(
            f"Saved {os.path.basename(path)} - {n_v} clip(s)"
            f"{' + npz' if spec['npz'] else ''}. "
            f"Run trim_runner.py to encode.", 8000)

        if self.chk_clear.isChecked():
            self.video.clear()
            self.npz.data = None
            self.npz.path = None
            self.npz.n_frames = 0
            self.npz.lbl_path.setText("<no npz>")
            self.npz.fig.clear()
            self.npz.canvas.draw_idle()
            self.npz.bar.set_length(1)
            self.npz.lbl_info.setText("")
            self.ed_spec.clear()
            self._update_preview()

    def load_spec(self, path):
        """Restore a saved spec so the marks can be adjusted and re-saved."""
        try:
            with open(path) as f:
                spec = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load spec", str(exc))
            return
        if spec.get("schema") != SPEC_SCHEMA:
            QtWidgets.QMessageBox.warning(
                self, "Load spec",
                f"Unexpected schema {spec.get('schema')!r}; trying anyway.")

        spec_dir = os.path.dirname(os.path.abspath(path))
        self.ed_spec.setText(os.path.abspath(path))

        enc = spec.get("encode", {})
        nam = spec.get("naming", {})
        self.ed_outdir.setText(spec.get("output_dir", "trimmed"))
        self.chk_ffmpeg.setChecked(bool(enc.get("use_ffmpeg", True)))
        self.spin_crf.setValue(int(enc.get("crf", 18)))
        if enc.get("preset"):
            self.cmb_preset.setCurrentText(enc["preset"])
        if enc.get("video_ext"):
            self.cmb_vid_ext.setCurrentText(enc["video_ext"])
        self.ed_video_name.setText(nam.get("video_base", "clip_trimmed"))
        self.ed_npz_name.setText(nam.get("npz_base", "rollout_trimmed"))
        self.spin_start_num.setValue(int(nam.get("start_number", 1)))
        self.spin_pad.setValue(int(nam.get("digits", 1)))
        self.chk_stem.setChecked(bool(nam.get("append_source_stem", False)))
        self.chk_sidecar.setChecked(bool(spec.get("pairing_json")))

        missing = []
        self.video.clear()
        for entry in spec.get("videos", []):
            p = _resolve(entry.get("source"), spec_dir)
            if not (p and os.path.exists(p)):
                p2 = entry.get("source_abs")
                p = p2 if p2 and os.path.exists(p2) else p
            if not (p and os.path.exists(p)):
                missing.append(entry.get("source"))
                continue
            src = self.video.add(p, refresh=False)
            if src is not None:
                src.offset = int(entry.get("offset_frames", 0))
        self.video._rebuild_grid()
        self.video._resync_timeline(keep_trim=False)

        master = spec.get("master") or {}
        if self.video.has_media() and master.get("trim_frames"):
            s, e = master["trim_frames"]
            self.video.bar.set_trim(int(s), int(e))

        npz = spec.get("npz")
        if npz:
            p = _resolve(npz.get("source"), spec_dir)
            if not (p and os.path.exists(p)):
                p2 = npz.get("source_abs")
                p = p2 if p2 and os.path.exists(p2) else p
            if p and os.path.exists(p):
                self.npz.load(p)
                s, e = npz.get("trim_samples", [0, self.npz.n_frames - 1])
                self.npz.bar.set_trim(int(s), int(e))
                self.chk_rezero.setChecked(bool(npz.get("rezero_time", True)))
                self.spin_offset.setValue(int(npz.get("offset_frames", 0)))
            else:
                missing.append(npz.get("source"))

        self._update_preview()
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Load spec",
                "These sources were not found:\n  "
                + "\n  ".join(str(m) for m in missing))
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}", 6000)

    def closeEvent(self, ev):
        self.video.clear()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()

    added = False
    for a in sys.argv[1:]:
        if not os.path.exists(a):
            continue
        low = a.lower()
        if low.endswith(SPEC_SUFFIX):
            win.load_spec(a)
            added = False
            break
        if low.endswith(".npz"):
            win.npz.load(a)
        else:
            win.video.add(a, refresh=False)
            added = True
    if added:
        win.video._rebuild_grid()
        win.video._resync_timeline(keep_trim=False)
    win._autofill_spec_path()

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()