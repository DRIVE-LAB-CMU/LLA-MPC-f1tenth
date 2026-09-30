"""trim_runner.py - execute every *.trimspec.json under a folder tree.

Companion to trim_marker.py, which writes the specs. This half never opens a
window: point it at a root, walk away, come back to finished clips.

    python trim_runner.py C:/Users/henry/Videos/ws/runs
    python trim_runner.py runs -w 4 --log runs/encode.log
    python trim_runner.py runs --dry-run
    python trim_runner.py runs --watch 60          # keep polling for new specs

Designed to be interrupted. Every finished output is recorded in a state file
beside its spec, each clip is encoded to a temporary file and renamed into
place only on success, and Ctrl-C tears down cleanly. Start it again and it
picks up exactly where it stopped - already-encoded clips cost one stat call.

Re-marking a run in the marker changes that clip's fingerprint (source file,
trim points, encoder settings), so the runner notices and redoes just that
clip, leaving its siblings alone.

Requires: numpy, opencv-python, and ffmpeg on PATH (OpenCV is the fallback).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2


SPEC_SCHEMA = "trimspec/1"
STATE_SCHEMA = "trimstate/1"
DEFAULT_PATTERN = "*.trimspec.json"
LOCK_STALE_S = 180.0        # a lock untouched this long belongs to a dead run
HEARTBEAT_S = 30.0

CANCEL = threading.Event()
_LIVE_PROCS: set[subprocess.Popen] = set()
_PROC_LOCK = threading.Lock()


# ===========================================================================
# Small utilities
# ===========================================================================
def human_size(n_bytes):
    n = float(n_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def human_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class Log:
    """Thread-safe console + optional file logging. No cursor tricks, so it
    reads fine when redirected into a file or a background job."""

    def __init__(self, path=None, quiet=False):
        self._lock = threading.Lock()
        self._quiet = quiet
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                        exist_ok=True)
            self._fh = open(path, "a", buffering=1, encoding="utf-8")

    def __call__(self, msg="", important=True):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            if important or not self._quiet:
                print(line, flush=True)
            if self._fh:
                self._fh.write(line + "\n")

    def close(self):
        if self._fh:
            self._fh.close()


def file_sig(path):
    """Cheap content fingerprint: size + mtime. Re-recording a source with the
    same name changes this, which is exactly when the clip should be redone."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


def job_hash(payload):
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def resolve(path, base):
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def pick_source(entry, spec_dir):
    """Relative path first (tree may have moved), absolute hint as fallback."""
    p = resolve(entry.get("source"), spec_dir)
    if p and os.path.exists(p):
        return p
    alt = entry.get("source_abs")
    if alt and os.path.exists(alt):
        return alt
    return p


def _pack(vals):
    """Stack if rectangular, otherwise build an object array (ragged keys)."""
    try:
        out = np.array(vals)
        if out.dtype == object:
            raise ValueError
        return out
    except ValueError:
        out = np.empty(len(vals), dtype=object)
        out[:] = list(vals)
        return out


def temp_path(out_path):
    root, ext = os.path.splitext(out_path)
    return f"{root}.part{ext}"          # keep the extension so ffmpeg can infer


def discard(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ===========================================================================
# State file : what has already been produced
# ===========================================================================
def state_path_for(spec_path):
    root = spec_path[:-5] if spec_path.lower().endswith(".json") else spec_path
    return root + ".state.json"


def load_state(path):
    try:
        with open(path) as f:
            st = json.load(f)
        if st.get("schema") == STATE_SCHEMA and isinstance(st.get("entries"), dict):
            return st
    except Exception:
        pass
    return {"schema": STATE_SCHEMA, "entries": {}}


def save_state(path, state):
    """Atomic-ish: write beside, then replace. A kill mid-write can't corrupt
    the record of work already finished."""
    tmp = path + ".tmp"
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


# ===========================================================================
# Spec locking : two runners over one tree stay out of each other's way
# ===========================================================================
class SpecLock:
    def __init__(self, spec_path):
        root = (spec_path[:-5] if spec_path.lower().endswith(".json")
                else spec_path)
        self.path = root + ".lock"
        self.held = False

    def acquire(self):
        if os.path.exists(self.path):
            age = time.time() - os.path.getmtime(self.path)
            if age < LOCK_STALE_S:
                return False
        host = (os.environ.get("COMPUTERNAME")
                or (os.uname().nodename if hasattr(os, "uname") else None)
                or "?")
        try:
            with open(self.path, "w") as f:
                json.dump({"pid": os.getpid(), "host": host,
                           "since": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
            self.held = True
            return True
        except OSError:
            return False

    def touch(self):
        if self.held:
            try:
                os.utime(self.path, None)
            except OSError:
                pass

    def release(self):
        if self.held:
            discard(self.path)
            self.held = False


# ===========================================================================
# The actual conversions
# ===========================================================================
def encode_clip(job, log):
    """Trim one video. Writes to a .part file and renames on success."""
    src, out = job["src"], job["out"]
    s, e, fps = job["s"], job["e"], job["fps"]
    n_out = e - s + 1
    dur = n_out / max(fps, 1e-6)
    t0 = s / max(fps, 1e-6)
    tmp = temp_path(out)
    started = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    discard(tmp)

    ff = shutil.which("ffmpeg")
    if job.get("use_ffmpeg", True) and ff:
        cmd = [ff, "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1",
               "-ss", f"{t0:.6f}", "-i", src,
               "-t", f"{dur:.6f}",
               "-map", "0:v:0",          # first video stream only
               "-map", "0:a?",           # audio if present, don't fail if not
               "-dn", "-sn",             # drop data and subtitle streams
               "-ignore_unknown",
               "-c:v", "libx264",
               "-crf", str(job.get("crf", 18)),
               "-preset", job.get("preset", "veryfast"),
               "-pix_fmt", "yuv420p",
               # x264 rejects odd dimensions; round down to even
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        if job.get("threads"):
            cmd += ["-threads", str(job["threads"])]
        cmd += ["-c:a", "aac", tmp]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        with _PROC_LOCK:
            _LIVE_PROCS.add(proc)
        noise = []
        try:
            for line in proc.stdout:
                if CANCEL.is_set():
                    proc.terminate()
                    break
                line = line.strip()
                if line and "=" not in line:
                    noise.append(line)
            rc = proc.wait()
        finally:
            with _PROC_LOCK:
                _LIVE_PROCS.discard(proc)

        if CANCEL.is_set():
            discard(tmp)
            return {"ok": False, "canceled": True, "error": "canceled"}
        if rc == 0:
            os.replace(tmp, out)
            return {"ok": True, "canceled": False, "backend": "ffmpeg",
                    "frames": n_out, "elapsed": time.time() - started,
                    "bytes": os.path.getsize(out)}
        discard(tmp)
        log(f"    ffmpeg rc={rc} on {os.path.basename(out)}; trying OpenCV")
        for ln in noise[-3:]:
            log(f"      ffmpeg: {ln}")
    elif job.get("use_ffmpeg", True):
        log("    ffmpeg not on PATH; using OpenCV (no audio)")

    # --- OpenCV fallback ---------------------------------------------------
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return {"ok": False, "canceled": False, "error": f"cannot open {src}"}
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (job["w"], job["h"]))
    if not vw.isOpened():
        cap.release()
        return {"ok": False, "canceled": False,
                "error": f"cannot open writer for {tmp}"}
    written = 0
    for _ in range(n_out):
        if CANCEL.is_set():
            vw.release()
            cap.release()
            discard(tmp)
            return {"ok": False, "canceled": True, "error": "canceled"}
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
        written += 1
    vw.release()
    cap.release()
    if written == 0:
        discard(tmp)
        return {"ok": False, "canceled": False, "error": "no frames read"}
    os.replace(tmp, out)
    return {"ok": True, "canceled": False, "backend": "opencv",
            "frames": written, "elapsed": time.time() - started,
            "bytes": os.path.getsize(out)}


def slice_npz(job, log):
    """Trim one rollout archive. Keys as long as the state array get sliced;
    everything else is copied through untouched."""
    src, out = job["src"], job["out"]
    s, e = job["s"], job["e"]
    tmp = temp_path(out)
    started = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    discard(tmp)

    raw = np.load(src, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n_frames = len(np.asarray(data["state"]))
    s = int(np.clip(s, 0, n_frames - 1))
    e = int(np.clip(e, s, n_frames - 1))

    result, sliced, passthrough = {}, [], []
    for k, arr in data.items():
        if CANCEL.is_set():
            discard(tmp)
            return {"ok": False, "canceled": True, "error": "canceled"}
        try:
            n = len(arr)
        except TypeError:
            result[k] = arr
            passthrough.append(k)
            continue
        if n == n_frames:
            chunk = arr[s:e + 1]
            result[k] = (_pack(list(chunk)) if getattr(arr, "dtype", None) == object
                         else np.array(chunk))
            sliced.append(k)
        else:
            result[k] = arr
            passthrough.append(f"{k}[{n}]")

    if job.get("rezero", True) and "time" in result:
        try:
            tt = np.asarray(result["time"], dtype=float)
            result["time"] = tt - tt[0]
        except Exception:
            pass
    result["_trim_start"] = np.array(s)
    result["_trim_end"] = np.array(e)
    result["_source_file"] = np.array(os.path.basename(src))

    np.savez(tmp, **result)
    # np.savez appends .npz when the name lacks it; tmp already ends in .npz
    if not os.path.exists(tmp) and os.path.exists(tmp + ".npz"):
        os.replace(tmp + ".npz", tmp)
    os.replace(tmp, out)
    log(f"    sliced {len(sliced)} key(s), {len(passthrough)} passed through "
        f"({e - s + 1} of {n_frames} samples)", important=False)
    return {"ok": True, "canceled": False, "backend": "numpy",
            "samples": e - s + 1, "elapsed": time.time() - started,
            "bytes": os.path.getsize(out)}


def write_pairing(spec, spec_dir, log):
    """Sidecar describing what lines up with what, for downstream tooling."""
    rel = spec.get("pairing_json")
    if not rel:
        return None
    out = resolve(rel, spec_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    meta = {
        "spec": os.path.basename(spec.get("_path", "")),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "master": spec.get("master"),
        "videos": [{k: v for k, v in e.items() if k != "source_abs"}
                   for e in spec.get("videos", [])],
        "npz": ({k: v for k, v in spec["npz"].items() if k != "source_abs"}
                if spec.get("npz") else None),
    }
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    return out


# ===========================================================================
# Planning : spec -> list of pending jobs
# ===========================================================================
def discover_specs(roots, pattern):
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(os.path.abspath(root))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in sorted(filenames):
                if fnmatch.fnmatch(fn, pattern):
                    found.append(os.path.abspath(os.path.join(dirpath, fn)))
    return sorted(set(found))


def plan_spec(spec_path, args, log):
    """Read one spec and return (spec, pending_jobs, skipped_count)."""
    try:
        with open(spec_path) as f:
            spec = json.load(f)
    except Exception as exc:
        log(f"  ! unreadable spec {spec_path}: {exc}")
        return None, [], 0
    if spec.get("schema") != SPEC_SCHEMA:
        log(f"  ! skipping {os.path.basename(spec_path)}: schema "
            f"{spec.get('schema')!r}")
        return None, [], 0

    spec["_path"] = spec_path
    spec_dir = os.path.dirname(spec_path)
    enc = spec.get("encode", {})
    state = load_state(state_path_for(spec_path))
    jobs, skipped = [], 0

    def consider(key, out_path, fingerprint, make_job):
        nonlocal skipped
        rec = state["entries"].get(key)
        if (not args.force and rec and rec.get("status") == "done"
                and rec.get("hash") == fingerprint
                and os.path.exists(out_path)
                and os.path.getsize(out_path) > 0):
            skipped += 1
            return
        jobs.append(make_job(fingerprint))

    if args.only in ("all", "videos"):
        for entry in spec.get("videos", []):
            src = pick_source(entry, spec_dir)
            out = resolve(entry["output"], spec_dir)
            key = entry["output"]
            if not (src and os.path.exists(src)):
                log(f"  ! missing source for {key}: {entry.get('source')}")
                continue
            s, e = entry["trim_frames"]
            fp = job_hash({"src": file_sig(src), "s": s, "e": e,
                           "fps": entry["fps"], "enc": enc, "kind": "video"})
            consider(key, out, fp, lambda fp_, entry=entry, src=src,
                     out=out, key=key, s=s, e=e: {
                         "kind": "video", "spec_path": spec_path, "key": key,
                         "src": src, "out": out, "hash": fp_,
                         "fps": entry["fps"], "w": entry["width"],
                         "h": entry["height"], "s": int(s), "e": int(e),
                         "use_ffmpeg": enc.get("use_ffmpeg", True),
                         "crf": enc.get("crf", 18),
                         "preset": enc.get("preset", "veryfast"),
                     })

    npz = spec.get("npz")
    if npz and args.only in ("all", "npz"):
        src = pick_source(npz, spec_dir)
        out = resolve(npz["output"], spec_dir)
        key = npz["output"]
        if src and os.path.exists(src):
            s, e = npz["trim_samples"]
            fp = job_hash({"src": file_sig(src), "s": s, "e": e,
                           "rezero": npz.get("rezero_time", True), "kind": "npz"})
            consider(key, out, fp, lambda fp_, src=src, out=out, key=key,
                     s=s, e=e, npz=npz: {
                         "kind": "npz", "spec_path": spec_path, "key": key,
                         "src": src, "out": out, "hash": fp_,
                         "s": int(s), "e": int(e),
                         "rezero": npz.get("rezero_time", True),
                     })
        else:
            log(f"  ! missing npz for {key}: {npz.get('source')}")

    return spec, jobs, skipped


# ===========================================================================
# Execution
# ===========================================================================
class Runner:
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.state_locks: dict[str, threading.Lock] = {}
        self.states: dict[str, dict] = {}
        self.counter_lock = threading.Lock()
        self.done = 0
        self.failed = 0
        self.total = 0
        self.bytes_out = 0

    def _record(self, job, result):
        sp = job["spec_path"]
        with self.state_locks[sp]:
            st = self.states[sp]
            st["entries"][job["key"]] = {
                "status": "done" if result.get("ok") else "failed",
                "hash": job["hash"],
                "kind": job["kind"],
                "output": os.path.abspath(job["out"]),
                "source": os.path.abspath(job["src"]),
                "bytes": result.get("bytes", 0),
                "backend": result.get("backend"),
                "elapsed_s": round(result.get("elapsed", 0.0), 2),
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": result.get("error"),
            }
            save_state(state_path_for(sp), st)

    def run_job(self, job):
        if CANCEL.is_set():
            return
        # Every run writes into its own "trimmed/", so name the run folder
        # instead - otherwise half the tree logs the same line.
        label = os.path.basename(job["out"])
        folder = os.path.basename(os.path.dirname(job["spec_path"])) or "."
        try:
            if job["kind"] == "video":
                res = encode_clip(job, self.log)
            else:
                res = slice_npz(job, self.log)
        except Exception as exc:
            res = {"ok": False, "canceled": False,
                   "error": f"{type(exc).__name__}: {exc}"}

        if res.get("canceled"):
            return
        with self.counter_lock:
            if res.get("ok"):
                self.done += 1
                self.bytes_out += res.get("bytes", 0)
                n = self.done + self.failed
            else:
                self.failed += 1
                n = self.done + self.failed
        if res.get("ok"):
            self.log(f"  [{n}/{self.total}] {folder}/{label}  "
                     f"{human_size(res.get('bytes', 0))} via "
                     f"{res.get('backend')} in {res.get('elapsed', 0):.1f}s")
        else:
            self.log(f"  [{n}/{self.total}] FAILED {folder}/{label}: "
                     f"{res.get('error')}")
        self._record(job, res)

    def execute(self, jobs, specs_by_path, locks):
        self.total = len(jobs)
        for sp in {j["spec_path"] for j in jobs}:
            self.state_locks.setdefault(sp, threading.Lock())
            self.states.setdefault(sp, load_state(state_path_for(sp)))

        stop_beat = threading.Event()

        def heartbeat():
            while not stop_beat.wait(HEARTBEAT_S):
                for lk in locks.values():
                    lk.touch()

        beat = threading.Thread(target=heartbeat, daemon=True)
        beat.start()

        workers = max(1, min(self.args.workers, len(jobs)))
        cores = os.cpu_count() or 1
        threads = 0 if workers == 1 else max(1, cores // workers)
        for j in jobs:
            if j["kind"] == "video":
                j["threads"] = self.args.threads_per_job or threads

        self.log(f"running {len(jobs)} job(s) on {workers} worker(s)"
                 + (f" x {threads} ffmpeg threads" if threads else "")
                 + f"  ({cores} cores)")
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self.run_job, jobs))
        finally:
            stop_beat.set()
        return time.time() - t0


def finalize_specs(specs_by_path, log, args):
    """Write the pairing sidecar for any spec whose outputs are all present."""
    for sp, spec in specs_by_path.items():
        if not spec.get("pairing_json"):
            continue
        spec_dir = os.path.dirname(sp)
        outs = [resolve(e["output"], spec_dir) for e in spec.get("videos", [])]
        if spec.get("npz"):
            outs.append(resolve(spec["npz"]["output"], spec_dir))
        if outs and all(os.path.exists(o) for o in outs):
            try:
                p = write_pairing(spec, spec_dir, log)
                if p:
                    log(f"  pairing: {os.path.basename(spec_dir)}/"
                        f"{os.path.relpath(p, spec_dir)}", important=False)
            except Exception as exc:
                log(f"  ! pairing json failed for {os.path.basename(sp)}: {exc}")


# ===========================================================================
# One full pass over the tree
# ===========================================================================
def one_pass(args, log):
    specs = discover_specs(args.roots, args.pattern)
    if not specs:
        log(f"no {args.pattern} found under: {', '.join(args.roots)}")
        return 0

    log(f"found {len(specs)} spec(s)")
    all_jobs, specs_by_path, locks = [], {}, {}
    total_skipped = 0

    for sp in specs:
        if CANCEL.is_set():
            break
        lock = SpecLock(sp)
        if not args.no_lock and not lock.acquire():
            log(f"  locked by another runner, skipping: "
                f"{os.path.relpath(sp, os.path.curdir)}")
            continue
        spec, jobs, skipped = plan_spec(sp, args, log)
        total_skipped += skipped
        if spec is None:
            lock.release()
            continue
        specs_by_path[sp] = spec
        if jobs:
            locks[sp] = lock
            all_jobs.extend(jobs)
        else:
            lock.release()

    log(f"pending: {len(all_jobs)} job(s)   already done: {total_skipped}")

    if args.dry_run:
        for j in all_jobs:
            span = (f"frames {j['s']}..{j['e']}" if j["kind"] == "video"
                    else f"samples {j['s']}..{j['e']}")
            log(f"  would write {j['out']}  ({j['kind']}, {span})")
        for lk in locks.values():
            lk.release()
        return 0

    runner = Runner(args, log)
    try:
        if all_jobs:
            wall = runner.execute(all_jobs, specs_by_path, locks)
            log("-" * 70)
            log(f"{'canceled - ' if CANCEL.is_set() else ''}"
                f"{runner.done} done, {runner.failed} failed, "
                f"{len(all_jobs) - runner.done - runner.failed} not started, "
                f"{human_size(runner.bytes_out)} written in {human_time(wall)}")
        if not CANCEL.is_set():
            finalize_specs(specs_by_path, log, args)
    finally:
        for lk in locks.values():
            lk.release()
    return runner.failed


# ===========================================================================
# Entry point
# ===========================================================================
def install_signals(log):
    def handler(signum, _frame):
        if CANCEL.is_set():
            log("second interrupt - exiting now")
            os._exit(130)
        CANCEL.set()
        log("interrupt: finishing current teardown, partial files removed. "
            "Re-run to resume.")
        with _PROC_LOCK:
            for p in list(_LIVE_PROCS):
                try:
                    p.terminate()
                except Exception:
                    pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, AttributeError, OSError):
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Execute trim specs written by trim_marker.py.")
    p.add_argument("roots", nargs="*", default=["."],
                   help="folders (or single spec files) to search; default .")
    p.add_argument("-w", "--workers", type=int,
                   default=min(4, max(1, os.cpu_count() or 4)),
                   help="clips to encode at once (default %(default)s)")
    p.add_argument("--threads-per-job", type=int, default=0,
                   help="ffmpeg -threads per worker; 0 splits cores evenly")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help="spec filename glob (default %(default)s)")
    p.add_argument("--only", choices=["all", "videos", "npz"], default="all",
                   help="restrict to one kind of output")
    p.add_argument("--force", action="store_true",
                   help="redo everything, ignoring recorded state")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be written and stop")
    p.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                   help="keep polling for new or changed specs")
    p.add_argument("--log", default=None, help="also append output to this file")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="console gets milestones only (the log file gets all)")
    p.add_argument("--no-lock", action="store_true",
                   help="skip per-spec lock files")
    args = p.parse_args(argv)
    args.roots = [os.path.abspath(r) for r in (args.roots or ["."])]
    return args


def main(argv=None):
    args = parse_args(argv)
    log = Log(args.log, args.quiet)
    install_signals(log)

    log("=" * 70)
    log(f"trim_runner  roots: {', '.join(args.roots)}")
    log("=" * 70)

    failures = 0
    try:
        while True:
            failures = one_pass(args, log)
            if not args.watch or CANCEL.is_set():
                break
            log(f"watching - next sweep in {args.watch:.0f}s", important=False)
            waited = 0.0
            while waited < args.watch and not CANCEL.is_set():
                time.sleep(min(1.0, args.watch - waited))
                waited += 1.0
    finally:
        log("=" * 70)
        log.close()

    if CANCEL.is_set():
        return 130
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())