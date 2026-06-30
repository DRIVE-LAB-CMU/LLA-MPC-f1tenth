"""Interactive state-trajectory visualizer.

Two windows:
  * main      - trajectory plot + per-parameter time series + playback controls
  * diagnostics - sliding-window cost, one-step difference, and (optional)
                  M-step lookahead error, sharing cost-component checkboxes

Run as a script. Rollout backend lives in rollouts.py; if it is unavailable the
visualizer falls back to precomputed npz arrays or the recorded run only.
"""
import os
# Force Linux window backends to scale UI elements correctly.
# NOTE: GDK_SCALE only accepts integers (GTK silently ignores/rejects
# fractional values like "1.5"), so it was previously a no-op. Qt's
# AUTO_SCREEN_SCALE_FACTOR is the one safe "just do the right thing" knob;
# we compute an explicit fractional factor for Qt/GTK below instead of
# guessing a hard-coded value.
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.pop("GDK_SCALE", None)


def _detect_dpi_scale(base_dpi=96.0):
    """Best-effort screen-DPI probe so fonts/figures scale on HiDPI displays.

    Falls back to 1.0 (no scaling) if no display toolkit is available, e.g.
    headless/CI environments, rather than silently doing nothing like the
    previous env-var-only approach.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        dpi = root.winfo_fpixels('1i')
        root.destroy()
        if dpi and dpi > 0:
            return max(1.0, dpi / base_dpi)
    except Exception:
        pass
    return 1.0


_DPI_SCALE = _detect_dpi_scale()
# Qt reads fractional scaling from this var (GTK has no equivalent fractional
# knob, so on GTK backends we rely on the rcParams scaling below instead).
os.environ.setdefault("QT_SCALE_FACTOR", f"{_DPI_SCALE:.2f}")

import matplotlib.pyplot as plt
# Set readable base sizes for high-res monitors; scaled by the detected DPI
# factor instead of a fixed guess, so they actually grow on HiDPI screens.
plt.rcParams.update({
    'font.size': 13 * _DPI_SCALE,
    'axes.labelsize': 14 * _DPI_SCALE,
    'axes.titlesize': 14 * _DPI_SCALE,
    'xtick.labelsize': 11 * _DPI_SCALE,
    'ytick.labelsize': 11 * _DPI_SCALE,
    'legend.fontsize': 10 * _DPI_SCALE,
    'lines.linewidth': 2.0
})

import os
import sys

# Ensure rollouts.py is imported from THIS file's directory, regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import FancyArrow, Circle, Patch

from rollouts import (
    _ROLLOUT_OK, _ROLLOUT_ERR,
    BANK_ORDER, DEFAULT_LOG_ORDER, COST_DIM_LABELS,
    dict_to_bank_vec, remap_to_bank_order,
    simulate_general_models, simulate_lla_rollout, simulate_lla_one_step,
    simulate_general_m_step, simulate_lla_m_step,
)

if _ROLLOUT_OK:
    from rollouts import F110
else:
    F110 = None


# ===========================================================================
# Visualizer
# ===========================================================================
class StateVisualizer:
    """Interactive visualizer for state trajectory data."""

    def __init__(self, filepath, ref_filepath=None, n_params_to_show=None, params_per_column=3,
                 param_names=None, obstacles=None, r_car=0.04,
                 general_models=None, compute_rollouts=True,
                 dt=1.0 / 40.0, ol_reset_interval=40, cost_weights=None,
                 full_open_loop=False, log_order=None, window_P=20, cost_form=None,
                 compute_m_step=False, m_step_M=10):
        self.n_params_to_show = n_params_to_show
        self.params_per_column = params_per_column
        self.param_names = param_names
        self.ref_filepath = ref_filepath

        self.r_car = r_car
        self.obstacles = []
        if obstacles:
            for p_obs, r_obs in obstacles:
                self.obstacles.append((np.asarray(p_obs, dtype=float), float(r_obs)))

        # --- Rollout config ---
        self.compute_rollouts = compute_rollouts
        self.dt = dt
        self.ol_reset_interval = ol_reset_interval
        self.cost_weights = (np.array([1.0, 1.0, 0.0, 0.0, 0, 0])
                             if cost_weights is None else np.asarray(cost_weights))
        # Weighting for the sliding-window cost graph (yaw wrapped at index 2),
        # analogous to grid search's cost_form. Defaults to cost_weights.
        self.cost_form = (self.cost_weights if cost_form is None
                          else np.asarray(cost_form, dtype=float))
        # Which state-dimension cost components are summed into the displayed
        # total cost. Each dim's cost is always computed across the full run;
        # this mask only controls what gets added together for display.
        # Defaults to whichever dims have nonzero weight in cost_form.
        self.active_cost_dims = [bool(w != 0) for w in self.cost_form]
        self.full_open_loop = full_open_loop
        self.general_models = general_models or {}
        self.log_order = log_order or DEFAULT_LOG_ORDER
        self.params_car = F110() if _ROLLOUT_OK else None

        # M-step lookahead config (expensive: ~total * M integrator calls).
        self.compute_m_step = bool(compute_m_step)
        self.m_step_M = int(m_step_M)
        self.show_m_step = True
        self._has_m_step = False

        # Rollout state (filled by prepare_rollouts)
        self.lla_traj = None           # open-loop (or periodically-reset) dynamic traj
        self.lla_one_step_traj = None  # one-step traj for LLA
        self.lla_params_over_time = None
        self.general_trajs = {}     # name -> {'open_loop': arr, 'one_step': arr}
        self.general_order = []
        self.rollout_len = 0

        # Display toggles (defaults so early update_frame calls are safe)
        self.show_lla = True            # LLA dynamic rollout on/off
        self.show_general = True        # general fixed-param models on/off
        self.show_velocity = True       # velocity/accel state-indicator arrows on/off
        self.model_mode = "open_loop"   # 'open_loop' or 'one_step' (applies to LLA + general)
        self.window_P = int(window_P)   # length of the model evaluation window

        self.load_data(filepath)
        self.load_ref_data()
        self.prepare_rollouts()

        self.setup_figure()
        self.setup_cost_figure()
        self.current_frame = 0
        self.playing = False
        self.setup_artists()
        self.setup_controls()

    def load_data(self, filepath):
        """Load trajectory data from npz file."""
        data = np.load(filepath, allow_pickle=True)
        self._raw = data
        self.time = data["time"]
        state = data["state"]
        self.state = state
        self.x = state[:, 0]
        self.y = state[:, 1]
        self.theta = state[:, 2]
        self.dx = state[:, 3]
        self.dy = state[:, 4]
        self.omega = state[:, 5]
        self.n_frames = len(self.x)

        params_raw = data["params"]

        if isinstance(params_raw, np.ndarray) and params_raw.ndim == 2:
            if params_raw.shape[0] == len(self.time):
                self.params = [params_raw[:, i] for i in range(params_raw.shape[1])]
            else:
                self.params = [params_raw[i, :] for i in range(params_raw.shape[0])]
        else:
            self.params = list(params_raw)

        self.model_idx = data["model_index"]
        ctrl = data["ctrl"]
        self.ctrl = ctrl
        self.accel = ctrl[:, 0]
        self.steer = ctrl[:, 1]

        # Recording view used by the rollout simulators.
        self.recording = {"state": state, "ctrl": ctrl, "time": self.time}
        self.rollout_total = max(0, len(self.time) - 1)

        self.mpc_rollout = None
        if "mpc_rollout" in data:
            rollout_data = data["mpc_rollout"]
            if len(rollout_data) > 0 and len(rollout_data[0]) > 0:
                self.mpc_rollout = rollout_data

        self.ref_trajectory = None
        if "ref_trajectory" in data:
            ref_traj_data = data["ref_trajectory"]
            print(ref_traj_data.shape)
            if len(ref_traj_data) > 0 and len(ref_traj_data[0]) > 0:
                self.ref_trajectory = ref_traj_data

        if self.n_params_to_show is None:
            self.n_params_to_show = [x for x in range(len(self.params))]

    def load_ref_data(self):
        """Load reference raceline data if provided."""
        self.ref_x = None
        self.ref_y = None

        if self.ref_filepath and os.path.exists(self.ref_filepath):
            try:
                ref_data = np.load(self.ref_filepath, allow_pickle=True)
                self.ref_x = ref_data['x']
                self.ref_y = ref_data['y']
            except Exception as e:
                print(f"Warning: Failed to load reference trajectory from {self.ref_filepath}: {e}")

    def prepare_rollouts(self):
        """Build the LLA dynamic/one-step rollouts and the general fixed-model rollouts.

        Priority: (1) use precomputed arrays already in the npz (e.g. from
        replay_lla.py); (2) otherwise compute from the log if the backend is
        importable. Degrades to "recorded run only" if neither is possible.
        """
        raw = self._raw
        files = set(getattr(raw, "files", []))

        # ---------------- LLA trajectories ----------------
        if "lla_dynamic_trajectory" in files:
            self.lla_traj = np.asarray(raw["lla_dynamic_trajectory"], dtype=float)
            if "lla_optimal_params" in files:
                self.lla_params_over_time = np.asarray(raw["lla_optimal_params"], dtype=float)
            # One-step may also be precomputed.
            if "lla_one_step_trajectory" in files:
                self.lla_one_step_traj = np.asarray(raw["lla_one_step_trajectory"], dtype=float)
        elif self.compute_rollouts and _ROLLOUT_OK and "params" in files and self.rollout_total > 0:
            logged = np.asarray(raw["params"])
            lla_total = min(self.rollout_total, len(logged))
            lla_params = remap_to_bank_order(logged[:lla_total], self.log_order)
            self.lla_params_over_time = lla_params
            self.lla_traj = simulate_lla_rollout(
                lla_total, self.recording, lla_params, self.params_car,
                self.dt, self.ol_reset_interval, full_open_loop=self.full_open_loop
            )
            self.lla_one_step_traj = simulate_lla_one_step(
                lla_total, self.recording, lla_params, self.params_car, self.dt
            )

        # If we have the dynamic LLA traj but no one-step yet (e.g. loaded from npz
        # without the precomputed field), try to compute it now.
        if (self.lla_traj is not None and self.lla_one_step_traj is None
                and self.lla_params_over_time is not None
                and self.compute_rollouts and _ROLLOUT_OK):
            lla_total = len(self.lla_traj)
            self.lla_one_step_traj = simulate_lla_one_step(
                lla_total, self.recording, self.lla_params_over_time,
                self.params_car, self.dt
            )

        # ---------------- General fixed-param models ----------------
        if "traj_open_loop" in files and "model_names" in files:
            names = [str(n) for n in raw["model_names"]]
            ol = np.asarray(raw["traj_open_loop"], dtype=float)     # (M, total, state)
            os_arr = (np.asarray(raw["traj_one_step"], dtype=float)
                      if "traj_one_step" in files else ol)
            for i, name in enumerate(names):
                self.general_trajs[name] = {"open_loop": ol[i], "one_step": os_arr[i]}
                self.general_order.append(name)
        elif self.compute_rollouts and _ROLLOUT_OK and self.general_models and self.rollout_total > 0:
            names = list(self.general_models.keys())
            bank = np.stack([dict_to_bank_vec(self.general_models[n]) for n in names])
            ol, os_arr = simulate_general_models(
                self.rollout_total, self.recording, bank, self.params_car,
                self.dt, self.ol_reset_interval, self.cost_weights,
                full_open_loop=self.full_open_loop
            )
            # ol, os_arr: (total, M, state) -> store per model as (total, state)
            for i, name in enumerate(names):
                self.general_trajs[name] = {"open_loop": ol[:, i, :], "one_step": os_arr[:, i, :]}
                self.general_order.append(name)

        # Common rollout length for safe frame clamping.
        lens = []
        if self.lla_traj is not None:
            lens.append(len(self.lla_traj))
        for name in self.general_order:
            lens.append(len(self.general_trajs[name]["open_loop"]))
        self.rollout_len = min(lens) if lens else 0

        if (self.compute_rollouts and not _ROLLOUT_OK
                and self.lla_traj is None and not self.general_trajs):
            print(f"[visualizer] Rollout backend unavailable ({_ROLLOUT_ERR}); "
                  f"showing recorded trajectory only.")

        # Sliding-window cost curves for the cost subplot.
        self._compute_cost_curves()
        # M-step lookahead error (gated by compute_m_step; expensive).
        self._compute_m_step_curves()

    def _step_cost_components(self, traj):
        """Per-timestep, per-state-dimension weighted squared error vs truth,
        yaw wrapped at index 2 (matches grid search get_lookback_error).
        traj: (T, state). Returns (T, state) - one column per cost component,
        UNSUMMED, so each dimension's contribution stays separable.
        """
        T = min(len(traj), self.rollout_len)
        tr = np.asarray(traj[:T], dtype=float)
        truth = np.asarray(self.state[:T], dtype=float)
        w = np.asarray(self.cost_form, dtype=float)
        err = truth - tr
        err[:, 2] = (err[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return (err ** 2) * w[None, :]

    def _sliding_window(self, step_cost):
        """Running sum of step_cost over the trailing window_P samples
        (the cost_history/running_cost queue, length P).

        step_cost: (T,) for a single series, or (T, D) for D components at
        once (each column windowed independently). Returns same shape.
        """
        P = max(1, int(self.window_P))
        step_cost = np.asarray(step_cost, dtype=float)
        csum = np.concatenate([np.zeros((1,) + step_cost.shape[1:]),
                                np.cumsum(step_cost, axis=0)], axis=0)
        idx = np.arange(len(step_cost))
        lo = np.maximum(0, idx - P + 1)
        return csum[idx + 1] - csum[lo]

    def _combine_active(self, component_curves):
        """Sum the windowed per-component curves for only the currently-active
        cost dimensions. component_curves: (T, D). Returns (T,)."""
        mask = np.asarray(self.active_cost_dims, dtype=float)
        if not np.any(mask):
            return np.zeros(component_curves.shape[0])
        return component_curves @ mask

    def _compute_cost_curves(self):
        self.cost_time = None
        self.lla_cost_components = {}      # mode -> (T, D) windowed
        self.gen_cost_components = {}      # name -> {mode: (T, D) windowed}
        self.lla_onestep_components = None # (T, D) un-windowed, one-step traj
        self.gen_onestep_components = {}   # name -> (T, D) un-windowed
        if self.rollout_len == 0:
            return

        T = self.rollout_len
        self.cost_time = np.asarray(self.time[:T], dtype=float)

        if self.lla_traj is not None:
            self.lla_cost_components['open_loop'] = self._sliding_window(
                self._step_cost_components(self.lla_traj))
        if self.lla_one_step_traj is not None:
            self.lla_cost_components['one_step'] = self._sliding_window(
                self._step_cost_components(self.lla_one_step_traj))
            self.lla_onestep_components = self._step_cost_components(self.lla_one_step_traj)

        for name in self.general_order:
            self.gen_cost_components[name] = {
                mode: self._sliding_window(
                    self._step_cost_components(self.general_trajs[name][mode]))
                for mode in ("open_loop", "one_step")
            }
            self.gen_onestep_components[name] = self._step_cost_components(
                self.general_trajs[name]['one_step'])

    def _compute_m_step_curves(self):
        """Per-dim, per-start-time M-step lookahead error for LLA + general
        models, for BOTH modes (so the mode toggle just switches arrays).
        Gated by compute_m_step; requires the rollout backend."""
        self.lla_m_step_components = {}     # mode -> (T, D)
        self.gen_m_step_components = {}     # name -> {mode: (T, D)}
        if not self.compute_m_step or self.rollout_len == 0:
            return
        if not _ROLLOUT_OK:
            print("[m-step] rollout backend unavailable; skipping M-step lookahead.")
            return

        T = self.rollout_len
        M = self.m_step_M
        modes = ("open_loop", "one_step")

        if self.lla_params_over_time is not None:
            for mode in modes:
                comp = simulate_lla_m_step(
                    T, self.recording, self.lla_params_over_time, self.params_car,
                    self.dt, M, self.cost_form, mode, self.ol_reset_interval,
                    full_open_loop=self.full_open_loop)
                self.lla_m_step_components[mode] = comp[:T]

        names = [n for n in self.general_order if n in self.general_models]
        if names:
            bank = np.stack([dict_to_bank_vec(self.general_models[n]) for n in names])
            for mode in modes:
                comp = simulate_general_m_step(
                    T, self.recording, bank, self.params_car,
                    self.dt, M, self.cost_form, mode, self.ol_reset_interval,
                    full_open_loop=self.full_open_loop)
                for i, name in enumerate(names):
                    self.gen_m_step_components.setdefault(name, {})[mode] = comp[:T, i, :]

        self._has_m_step = bool(self.lla_m_step_components or self.gen_m_step_components)

    def _iter_limit_trajs(self):
        """Trajectories used to size the axes (the well-behaved ones)."""
        if self.lla_traj is not None:
            yield self.lla_traj
        for name in self.general_order:
            yield self.general_trajs[name]["one_step"]

    def _active_lla_traj(self):
        """Return the LLA trajectory array for the current model_mode."""
        if self.model_mode == 'one_step' and self.lla_one_step_traj is not None:
            return self.lla_one_step_traj
        return self.lla_traj

    def _lla_cost_curve(self):
        """Return the LLA total cost curve (sum of active components) for the
        current model_mode, or None if that mode's components aren't available."""
        comp = self.lla_cost_components.get(self.model_mode)
        if comp is None:
            return None
        return self._combine_active(comp)

    def _gen_cost_curve(self, name):
        """Return a general model's total cost curve (sum of active
        components) for the current model_mode."""
        comp = self.gen_cost_components[name][self.model_mode]
        return self._combine_active(comp)

    def _lla_onestep_curve(self):
        if self.lla_onestep_components is None:
            return None
        return self._combine_active(self.lla_onestep_components)

    def _gen_onestep_curve(self, name):
        return self._combine_active(self.gen_onestep_components[name])

    def _lla_m_step_curve(self):
        comp = self.lla_m_step_components.get(self.model_mode)
        if comp is None:
            return None
        return self._combine_active(comp)

    def _gen_m_step_curve(self, name):
        modes = self.gen_m_step_components.get(name)
        if not modes or self.model_mode not in modes:
            return None
        return self._combine_active(modes[self.model_mode])

    def setup_figure(self):
        """Create figure with appropriate layout."""
        n_param_cols = int(np.ceil(len(self.n_params_to_show) / self.params_per_column))
        n_param_rows = min(self.params_per_column, len(self.n_params_to_show))

        self.fig = plt.figure(figsize=(8 + n_param_cols * 4, max(9, 4 + n_param_rows * 1.8)))

        gs = self.fig.add_gridspec(
            n_param_rows, 1 + n_param_cols,
            left=0.08, right=0.96, bottom=0.30, top=0.95,
            wspace=0.35, hspace=0.4,
            width_ratios=[1.8] + [1] * n_param_cols
        )

        self.ax = self.fig.add_subplot(gs[:, 0])

        min_x, max_x = self.x.min(), self.x.max()
        min_y, max_y = self.y.min(), self.y.max()

        if self.ref_x is not None and self.ref_y is not None:
            min_x = min(min_x, self.ref_x.min())
            max_x = max(max_x, self.ref_x.max())
            min_y = min(min_y, self.ref_y.min())
            max_y = max(max_y, self.ref_y.max())

        for p_obs, r_obs in self.obstacles:
            reach = r_obs + self.r_car
            min_x = min(min_x, p_obs[0] - reach)
            max_x = max(max_x, p_obs[0] + reach)
            min_y = min(min_y, p_obs[1] - reach)
            max_y = max(max_y, p_obs[1] + reach)

        # Include (well-behaved) rollout trajectories so they stay in view.
        for traj in self._iter_limit_trajs():
            if traj is None or len(traj) == 0:
                continue
            min_x = min(min_x, float(np.min(traj[:, 0])))
            max_x = max(max_x, float(np.max(traj[:, 0])))
            min_y = min(min_y, float(np.min(traj[:, 1])))
            max_y = max(max_y, float(np.max(traj[:, 1])))

        margin = max(1.0, 0.1 * max(max_x - min_x, max_y - min_y))
        self.ax.set_xlim(min_x - margin, max_x + margin)
        self.ax.set_ylim(min_y - margin, max_y + margin)
        self.ax.set_xlabel('X Position', fontsize=10)
        self.ax.set_ylabel('Y Position', fontsize=10)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Trajectory', fontsize=12, fontweight='bold')

        self.ax_params = []
        for count in range(len(self.n_params_to_show)):
            col = count // self.params_per_column
            row = count % self.params_per_column
            ax_p = self.fig.add_subplot(gs[row, col + 1])
            self.ax_params.append(ax_p)

            idx = self.n_params_to_show[count]
            if self.param_names and idx in self.param_names:
                param_label = self.param_names[idx]
            else:
                param_label = f'Param {idx}'

            ax_p.set_xlabel('Time (s)', fontsize=9)
            ax_p.set_ylabel(param_label, fontsize=9)
            ax_p.grid(True, alpha=0.3)
            ax_p.tick_params(labelsize=8)

            param_data = self.params[idx]
            ax_p.plot(self.time, param_data, 'b-', linewidth=1.5, alpha=0.8)
            ax_p.set_xlim(self.time[0], self.time[-1])

            param_range = np.ptp(param_data)
            if param_range > 0:
                margin = 0.1 * param_range
                ax_p.set_ylim(param_data.min() - margin, param_data.max() + margin)
            else:
                ax_p.set_ylim(param_data.min() - 0.1, param_data.max() + 0.1)

    def setup_cost_figure(self):
        """Diagnostics window:
        row 0 = sliding-window cost (mode-aware)
        row 1 = per-step one-step difference (always one-step)
        row 2 = M-step lookahead error (mode-aware)  [only if computed]
        right = cost-component checkboxes, shared by all subplots.
        """
        have_m = self._has_m_step
        n_rows = 3 if have_m else 2

        self.fig_diag = plt.figure(figsize=(11, 9 if have_m else 7))
        try:
            self.fig_diag.canvas.manager.set_window_title('Cost / lookahead diagnostics')
        except Exception:
            pass

        gs = self.fig_diag.add_gridspec(
            n_rows, 1, left=0.08, right=0.80, top=0.95, bottom=0.07, hspace=0.40
        )
        self.ax_cost = self.fig_diag.add_subplot(gs[0])
        self.ax_onestep = self.fig_diag.add_subplot(gs[1], sharex=self.ax_cost)
        self.ax_mstep = (self.fig_diag.add_subplot(gs[2], sharex=self.ax_cost)
                         if have_m else None)

        self.ax_cost.set_title(f'Sliding-window model cost (P={self.window_P})',
                               fontsize=10, fontweight='bold', loc='left')
        self.ax_cost.set_ylabel('Windowed cost', fontsize=9)
        self.ax_cost.grid(True, alpha=0.3)
        self.ax_cost.tick_params(labelsize=8)

        self.ax_onestep.set_title('One-step prediction difference (per step)',
                                  fontsize=10, fontweight='bold', loc='left')
        self.ax_onestep.set_ylabel('Per-step cost', fontsize=9)
        self.ax_onestep.grid(True, alpha=0.3)
        self.ax_onestep.tick_params(labelsize=8)

        if self.ax_mstep is not None:
            self.ax_mstep.set_title(
                f'M-step lookahead error (M={self.m_step_M}, mode-aware)',
                fontsize=10, fontweight='bold', loc='left')
            self.ax_mstep.set_ylabel('Horizon cost', fontsize=9)
            self.ax_mstep.grid(True, alpha=0.3)
            self.ax_mstep.tick_params(labelsize=8)

        # X label only on the bottom-most subplot.
        bottom_ax = self.ax_mstep if self.ax_mstep is not None else self.ax_onestep
        bottom_ax.set_xlabel('Time (s)', fontsize=9)

        if self.cost_time is not None and len(self.cost_time) > 1:
            self.ax_cost.set_xlim(self.cost_time[0], self.cost_time[-1])

        self.ax_cost_dims = self.fig_diag.add_axes([0.82, 0.40, 0.16, 0.30])
        self.ax_cost_dims.set_title('Cost terms', fontsize=8, fontweight='bold')
        self.ax_cost_dims.axis('off')

    def setup_artists(self):
        """Initialize plot elements."""
        if self.ref_x is not None and self.ref_y is not None:
            self.ax.plot(self.ref_x, self.ref_y, 'k--', alpha=0.4, linewidth=1.5, label='Raceline', zorder=1)

        for p_obs, r_obs in self.obstacles:
            obs_circle = Circle(
                (p_obs[0], p_obs[1]), r_obs,
                facecolor='red', edgecolor='darkred',
                alpha=0.25, linewidth=1.5, zorder=1
            )
            self.ax.add_patch(obs_circle)

            keepout_circle = Circle(
                (p_obs[0], p_obs[1]), r_obs + self.r_car,
                facecolor='none', edgecolor='red',
                linestyle='--', alpha=0.6, linewidth=1.2, zorder=1
            )
            self.ax.add_patch(keepout_circle)

        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label='Trajectory', zorder=2)

        self.rollout_line,   = self.ax.plot([], [], 'm--', alpha=0.5, linewidth=1.5, zorder=3)
        self.rollout_points, = self.ax.plot([], [], 'mo',  markersize=5, alpha=0.9,  zorder=4)

        self.ref_traj_line,   = self.ax.plot([], [], 'g--', alpha=0.5, linewidth=1.5, zorder=3)
        self.ref_traj_points, = self.ax.plot([], [], 'g^',  markersize=5, alpha=0.9,  zorder=4)

        self.point, = self.ax.plot([], [], 'ko', markersize=10, label='Position', zorder=5)
        self.heading_line, = self.ax.plot([], [], 'k-', linewidth=2, zorder=4)

        self.x_vel_arrow = None
        self.y_vel_arrow = None
        self.accel_arrow = None

        self.rollout_yaw_arrows = []
        self.ref_traj_yaw_arrows = []

        # ---- Model-rollout artists (LLA + general fixed models) ----
        self.lla_trail = None
        self.lla_point = None
        self.lla_heading = None
        self.model_artists = {}   # name -> {'trail':.., 'point':.., 'color':.., 'heading':..}

        legend_elements = []

        if self.ref_x is not None:
            legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=1.5, alpha=0.4, label='Reference'))

        if self.obstacles:
            legend_elements.append(Patch(facecolor='red', edgecolor='darkred', alpha=0.25, label='Obstacle'))
            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', lw=1.2, alpha=0.6,
                                          label='Keep-out (r_obs+r_car)'))

        if self.ref_trajectory is not None:
            legend_elements.append(Line2D([0], [0], color='g', linestyle='--', marker='^',
                                          markersize=5, lw=1.5, label='Ref Trajectory'))

        if self.mpc_rollout is not None:
            legend_elements.append(Line2D([0], [0], color='m', linestyle='--', marker='o',
                                          markersize=5, lw=1.5, label='MPC Rollout'))

        # LLA dynamic rollout (per-timestep optimal params).
        if self.lla_traj is not None:
            self.lla_trail, = self.ax.plot([], [], '-', color='orange', alpha=0.55,
                                           linewidth=1.8, zorder=3)
            self.lla_point, = self.ax.plot([], [], 'o', color='orange', mec='k',
                                           markersize=9, zorder=6)
            # Predicted-orientation bar (like the black heading bar, in orange).
            self.lla_heading, = self.ax.plot([], [], '-', color='orange',
                                             linewidth=2, zorder=6)
            legend_elements.append(Line2D([0], [0], color='orange', marker='o', lw=1.8,
                                          label='LLA (per-step params)'))

        # General fixed-parameter models.
        cmap = plt.get_cmap('tab10')
        for i, name in enumerate(self.general_order):
            color = cmap(i % 10)
            trail, = self.ax.plot([], [], '-', color=color, alpha=0.45, linewidth=1.3, zorder=2)
            point, = self.ax.plot([], [], 's', color=color, mec='k', markersize=7, zorder=5)
            # Predicted-orientation bar (like the black heading bar, in model color).
            heading, = self.ax.plot([], [], '-', color=color, linewidth=2, zorder=5)
            self.model_artists[name] = {"trail": trail, "point": point,
                                        "color": color, "heading": heading}
            legend_elements.append(Line2D([0], [0], color=color, marker='s', lw=1.3, label=str(name)))

        # ---- "Last P points" evaluation-window artists ----
        self.window_true, = self.ax.plot([], [], 'o-', color='navy', markersize=4,
                                          linewidth=2.0, alpha=0.9, zorder=4)
        self.window_lla = None
        if self.lla_traj is not None:
            self.window_lla, = self.ax.plot([], [], 'o-', color='orange', markersize=4,
                                            linewidth=2.2, alpha=1.0, zorder=7)
        self.window_models = {}
        for name in self.general_order:
            color = self.model_artists[name]["color"]
            art, = self.ax.plot([], [], 'o-', color=color, markersize=4,
                                linewidth=2.0, alpha=1.0, zorder=6)
            self.window_models[name] = art

        legend_elements.append(Line2D([0], [0], color='navy', marker='o', lw=2.0,
                                      label=f'Eval window (last {self.window_P})'))

        legend_elements.extend([
            Line2D([0], [0], color='black', lw=2, label='Heading (Yaw)'),
            Patch(facecolor='blue', alpha=0.7, label='X Velocity'),
            Patch(facecolor='green', alpha=0.7, label='Y Velocity'),
            Patch(facecolor='red', alpha=0.7, label='Acceleration')
        ])

        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        self.info_text = self.fig.text(
            0.01, 0.99, '', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=9, family='monospace'
        )

        self.param_vlines = []
        self.param_points = []
        for idx, ax_p in enumerate(self.ax_params):
            vline = ax_p.axvline(x=0, color='red', linestyle='--',
                                 linewidth=1.5, alpha=0.7, zorder=3)
            self.param_vlines.append(vline)

            point, = ax_p.plot([], [], 'ro', markersize=6, zorder=4)
            self.param_points.append(point)

        self._setup_cost_artists()

    def _setup_cost_artists(self):
        self.cost_line_lla = None
        self.cost_dot_lla = None
        self.cost_lines_gen = {}
        self.cost_dots_gen = {}
        self.os_line_lla = None
        self.os_dot_lla = None
        self.os_lines_gen = {}
        self.os_dots_gen = {}
        self.ms_line_lla = None
        self.ms_dot_lla = None
        self.ms_lines_gen = {}
        self.ms_dots_gen = {}

        if self.lla_traj is not None:
            self.cost_line_lla, = self.ax_cost.plot([], [], '-', color='orange',
                                                    linewidth=1.6, label='LLA')
            self.cost_dot_lla, = self.ax_cost.plot([], [], 'o', color='orange',
                                                   mec='k', markersize=6, zorder=5)
        if self.lla_one_step_traj is not None:
            self.os_line_lla, = self.ax_onestep.plot([], [], '-', color='orange',
                                                     linewidth=1.6, label='LLA')
            self.os_dot_lla, = self.ax_onestep.plot([], [], 'o', color='orange',
                                                    mec='k', markersize=6, zorder=5)

        for name in self.general_order:
            color = self.model_artists[name]["color"]
            line, = self.ax_cost.plot([], [], '-', color=color, linewidth=1.4, label=str(name))
            dot, = self.ax_cost.plot([], [], 's', color=color, mec='k', markersize=5, zorder=5)
            self.cost_lines_gen[name] = line
            self.cost_dots_gen[name] = dot

            oline, = self.ax_onestep.plot([], [], '-', color=color, linewidth=1.4, label=str(name))
            odot, = self.ax_onestep.plot([], [], 's', color=color, mec='k', markersize=5, zorder=5)
            self.os_lines_gen[name] = oline
            self.os_dots_gen[name] = odot

        # ---- M-step lookahead lines/dots (only if computed) ----
        if getattr(self, "ax_mstep", None) is not None:
            if self.lla_m_step_components:
                self.ms_line_lla, = self.ax_mstep.plot([], [], '-', color='orange',
                                                       linewidth=1.6, label='LLA')
                self.ms_dot_lla, = self.ax_mstep.plot([], [], 'o', color='orange',
                                                      mec='k', markersize=6, zorder=5)
            for name in self.general_order:
                if name not in self.gen_m_step_components:
                    continue
                color = self.model_artists[name]["color"]
                mline, = self.ax_mstep.plot([], [], '-', color=color,
                                            linewidth=1.4, label=str(name))
                mdot, = self.ax_mstep.plot([], [], 's', color=color, mec='k',
                                           markersize=5, zorder=5)
                self.ms_lines_gen[name] = mline
                self.ms_dots_gen[name] = mdot

        # Red shaded trailing-window span on the one-step subplot, marking the
        # same trailing P-sample window under the cursor.
        self.cost_window_span = self.ax_onestep.axvspan(
            0, 0, color='red', alpha=0.12, zorder=1, linewidth=0)

        self.cost_cursor = self.ax_cost.axvline(x=0, color='red', linestyle='--',
                                                linewidth=1.2, alpha=0.7, zorder=3)
        self.onestep_cursor = self.ax_onestep.axvline(x=0, color='red', linestyle='--',
                                                      linewidth=1.2, alpha=0.7, zorder=3)
        self.mstep_cursor = None
        if getattr(self, "ax_mstep", None) is not None:
            self.mstep_cursor = self.ax_mstep.axvline(
                x=0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, zorder=3)

        if self.cost_line_lla is not None or self.cost_lines_gen:
            ncol = min(4, 1 + len(self.general_order))
            self.ax_cost.legend(loc='upper left', fontsize=7, ncol=ncol)
            self.ax_onestep.legend(loc='upper left', fontsize=7, ncol=ncol)
            if (getattr(self, "ax_mstep", None) is not None
                    and (self.ms_line_lla is not None or self.ms_lines_gen)):
                self.ax_mstep.legend(loc='upper left', fontsize=7, ncol=ncol)

        self._setup_cost_dim_checkboxes()
        self._refresh_cost_lines()

    def _setup_cost_dim_checkboxes(self):
        """Checkboxes to toggle which state-dimension cost components are
        summed into the displayed total cost. Each dimension's windowed cost
        is already computed across the full run (see _compute_cost_curves);
        toggling here only changes the on-the-fly recombination, no recompute."""
        if getattr(self, "ax_cost_dims", None) is None:
            self.cost_dim_checks = None
            return
        self.cost_dim_checks = CheckButtons(
            self.ax_cost_dims, COST_DIM_LABELS, list(self.active_cost_dims)
        )
        for label in self.cost_dim_checks.labels:
            label.set_fontsize(8)
        self.cost_dim_checks.on_clicked(self.on_cost_dim_toggle)

    def on_cost_dim_toggle(self, label):
        """Flip one cost-component's active flag and redraw the recombined
        total cost curves (LLA + general models) and the live cursor dots."""
        i = COST_DIM_LABELS.index(label)
        self.active_cost_dims[i] = not self.active_cost_dims[i]
        self._refresh_cost_lines()
        self._update_cost_cursor(self.current_frame)
        if getattr(self, "fig_diag", None) is not None:
            self.fig_diag.canvas.draw_idle()

    @staticmethod
    def _autoscale_axis_y(ax, lines, pad_frac=0.08, floor_zero=True):
        """Manually rescale an axis's y-limits from a set of Line2D artists.

        ax.relim()/autoscale_view() can silently fail to grow the view when
        a line's data contains NaNs, or when it's transiently empty (e.g.
        right after a model is toggled off), leaving stale/cramped limits.
        This walks the actual y-data, ignoring NaNs and empty series, and
        sets explicit limits with a fractional margin so curves never get
        clipped at the top after new data comes in.
        """
        vals = []
        for line in lines:
            if line is None:
                continue
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size == 0:
                continue
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            vals.append(y)

        if not vals:
            return

        all_y = np.concatenate(vals)
        ymin, ymax = float(all_y.min()), float(all_y.max())
        if floor_zero:
            ymin = min(ymin, 0.0)

        span = ymax - ymin
        if span <= 0:
            pad = max(abs(ymax), 1.0) * pad_frac
        else:
            pad = span * pad_frac

        ax.set_ylim(ymin - (0 if floor_zero else pad), ymax + pad)

    def _refresh_cost_lines(self):
        if getattr(self, "ax_cost", None) is None or self.cost_time is None:
            return

        # --- windowed cost (mode-aware) ---
        if self.cost_line_lla is not None:
            curve = self._lla_cost_curve()
            if self.show_lla and curve is not None:
                self.cost_line_lla.set_data(self.cost_time, curve)
            else:
                self.cost_line_lla.set_data([], [])
        for name, line in self.cost_lines_gen.items():
            if self.show_general:
                line.set_data(self.cost_time, self._gen_cost_curve(name))
            else:
                line.set_data([], [])
        self._autoscale_axis_y(
            self.ax_cost,
            [self.cost_line_lla, *self.cost_lines_gen.values()]
        )

        # --- one-step difference (always one-step traj) ---
        if self.os_line_lla is not None:
            curve = self._lla_onestep_curve()
            if self.show_lla and curve is not None:
                self.os_line_lla.set_data(self.cost_time, curve)
            else:
                self.os_line_lla.set_data([], [])
        for name, line in self.os_lines_gen.items():
            if self.show_general:
                line.set_data(self.cost_time, self._gen_onestep_curve(name))
            else:
                line.set_data([], [])
        self._autoscale_axis_y(
            self.ax_onestep,
            [self.os_line_lla, *self.os_lines_gen.values()]
        )

        # --- M-step lookahead (mode-aware) ---
        if getattr(self, "ax_mstep", None) is not None:
            if self.ms_line_lla is not None:
                curve = self._lla_m_step_curve()
                if self.show_m_step and self.show_lla and curve is not None:
                    self.ms_line_lla.set_data(self.cost_time, curve)
                else:
                    self.ms_line_lla.set_data([], [])
            for name, line in self.ms_lines_gen.items():
                curve = self._gen_m_step_curve(name)
                if self.show_m_step and self.show_general and curve is not None:
                    line.set_data(self.cost_time, curve)
                else:
                    line.set_data([], [])
            self._autoscale_axis_y(
                self.ax_mstep,
                [self.ms_line_lla, *self.ms_lines_gen.values()]
            )

    def _update_cost_cursor(self, frame_idx):
        if getattr(self, "ax_cost", None) is None or self.cost_time is None:
            return
        ridx = min(frame_idx, self.rollout_len - 1)
        ct = float(self.time[ridx])

        self.cost_cursor.set_xdata([ct, ct])

        # Shade [t - P + 1, t] in time on the one-step subplot, matching the
        # sliding window that produced the windowed-cost value under the cursor.
        rs = max(0, ridx - self.window_P + 1)
        t_start = float(self.time[rs])
        self.cost_window_span.set_x(t_start)
        self.cost_window_span.set_width(max(ct - t_start, 1e-9))

        if self.cost_dot_lla is not None:
            curve = self._lla_cost_curve()
            if self.show_lla and curve is not None:
                self.cost_dot_lla.set_data([ct], [curve[ridx]])
            else:
                self.cost_dot_lla.set_data([], [])
        for name, dot in self.cost_dots_gen.items():
            if self.show_general:
                dot.set_data([ct], [self._gen_cost_curve(name)[ridx]])
            else:
                dot.set_data([], [])

        # one-step subplot
        self.onestep_cursor.set_xdata([ct, ct])
        if self.os_dot_lla is not None:
            curve = self._lla_onestep_curve()
            if self.show_lla and curve is not None:
                self.os_dot_lla.set_data([ct], [curve[ridx]])
            else:
                self.os_dot_lla.set_data([], [])
        for name, dot in self.os_dots_gen.items():
            if self.show_general:
                dot.set_data([ct], [self._gen_onestep_curve(name)[ridx]])
            else:
                dot.set_data([], [])

        # M-step subplot
        if getattr(self, "ax_mstep", None) is not None and self.mstep_cursor is not None:
            self.mstep_cursor.set_xdata([ct, ct])
            if self.ms_dot_lla is not None:
                curve = self._lla_m_step_curve()
                if self.show_m_step and self.show_lla and curve is not None:
                    self.ms_dot_lla.set_data([ct], [curve[ridx]])
                else:
                    self.ms_dot_lla.set_data([], [])
            for name, dot in self.ms_dots_gen.items():
                curve = self._gen_m_step_curve(name)
                if self.show_m_step and self.show_general and curve is not None:
                    dot.set_data([ct], [curve[ridx]])
                else:
                    dot.set_data([], [])

    @staticmethod
    def _as_points(arr):
        """Normalize ref/rollout data to an (N, state) array with N >= 1."""
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            return arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError(f"unexpected ref/rollout ndim={arr.ndim}")

        rows, cols = arr.shape
        state_widths = (2, 3)

        if cols in state_widths:
            return arr        # already (N, state)
        if rows in state_widths:
            return arr.T      # (state, N) -> (N, state)

        return arr

    def _clear_yaw_arrows(self, arrow_list):
        """Remove all arrows in a list from the axes."""
        for arrow in arrow_list:
            if arrow in self.ax.patches:
                arrow.remove()
        arrow_list.clear()

    def _draw_yaw_arrows(self, arr, color, arrow_list, yaw_len=0.15):
        """Draw yaw arrows at each node of a trajectory array."""
        pts = self._as_points(arr)
        if pts.shape[1] < 3:
            return

        xs, ys, yaws = pts[:, 0], pts[:, 1], pts[:, 2]

        for x, y, yaw in zip(xs, ys, yaws):
            if yaw == 0.0:
                continue
            arrow = FancyArrow(
                x, y,
                yaw_len * np.cos(yaw),
                yaw_len * np.sin(yaw),
                head_width=0.06, head_length=0.05,
                fc=color, ec=color, alpha=0.8, zorder=5
            )
            self.ax.add_patch(arrow)
            arrow_list.append(arrow)

    def _extract_xy(self, arr):
        """Extract x, y from a single point or a list of points (any layout)."""
        pts = self._as_points(arr)
        return pts[:, 0], pts[:, 1]

    @staticmethod
    def _break_segments(pts, idx_start, interval):
        """Return x, y for pts (absolute indices idx_start..), inserting a NaN
        break before every re-anchor index so the polyline is not drawn across a
        reset jump. interval None/<=0 -> no breaks (single connected line)."""
        if len(pts) == 0:
            return np.array([]), np.array([])
        xs = np.asarray(pts[:, 0], dtype=float)
        ys = np.asarray(pts[:, 1], dtype=float)
        if not interval or interval <= 0:
            return xs, ys
        out_x, out_y = [], []
        for i in range(len(xs)):
            if i > 0 and (idx_start + i) % interval == 0:
                out_x.append(np.nan)
                out_y.append(np.nan)
            out_x.append(xs[i])
            out_y.append(ys[i])
        return np.asarray(out_x), np.asarray(out_y)

    def _lla_interval(self):
        """Reset cadence for the LLA trajectory.

        One-step always re-anchors to truth each step (continuous, no jumps);
        dynamic open-loop resets every ol_reset_interval unless full_open_loop.
        """
        if self.model_mode == 'one_step':
            return None
        return None if self.full_open_loop else self.ol_reset_interval

    def _gen_interval(self):
        """Reset cadence of the general models. One-step re-anchors every step,
        producing a near-truth continuous path (no jumps), so it is not broken;
        open-loop jumps back to truth every OL_reset_interval."""
        if self.model_mode == "one_step":
            return None
        return None if self.full_open_loop else self.ol_reset_interval

    def _set_heading_bar(self, art, x, y, theta, length=0.5):
        """Draw a short orientation bar from (x, y) along theta, mirroring the
        black current-position heading bar but for a predicted model state."""
        art.set_data([x, x + length * np.cos(theta)],
                     [y, y + length * np.sin(theta)])

    def _update_model_overlays(self, frame_idx):
        """Update LLA + general model rollout markers/trails for this frame.
        Both LLA and general models respect self.model_mode."""
        if self.rollout_len == 0:
            if self.lla_point is not None:
                self.lla_trail.set_data([], [])
                self.lla_point.set_data([], [])
                if self.lla_heading is not None:
                    self.lla_heading.set_data([], [])
            for a in self.model_artists.values():
                a["trail"].set_data([], [])
                a["point"].set_data([], [])
                a["heading"].set_data([], [])
            return

        ridx = min(frame_idx, self.rollout_len - 1)

        # ---- LLA rollout (mode-aware) ----
        if self.lla_traj is not None:
            traj = self._active_lla_traj()
            if self.show_lla and traj is not None:
                r = min(ridx, len(traj) - 1)
                xs, ys = self._break_segments(traj[:r + 1], 0, self._lla_interval())
                self.lla_trail.set_data(xs, ys)
                self.lla_point.set_data([traj[r, 0]], [traj[r, 1]])
                if self.lla_heading is not None:
                    self._set_heading_bar(self.lla_heading,
                                          traj[r, 0], traj[r, 1], traj[r, 2])
            else:
                self.lla_trail.set_data([], [])
                self.lla_point.set_data([], [])
                if self.lla_heading is not None:
                    self.lla_heading.set_data([], [])

        # ---- General fixed-parameter models ----
        for name, a in self.model_artists.items():
            if self.show_general:
                traj = self.general_trajs[name][self.model_mode]
                r = min(ridx, len(traj) - 1)
                xs, ys = self._break_segments(traj[:r + 1], 0, self._gen_interval())
                a["trail"].set_data(xs, ys)
                a["point"].set_data([traj[r, 0]], [traj[r, 1]])
                self._set_heading_bar(a["heading"], traj[r, 0], traj[r, 1], traj[r, 2])
            else:
                a["trail"].set_data([], [])
                a["point"].set_data([], [])
                a["heading"].set_data([], [])

    def _update_window_overlays(self, frame_idx):
        """Highlight the last P samples (the model evaluation window).

        The recorded trajectory's window is always drawn as the comparison
        baseline. Both LLA and general model windows follow model_mode and
        their own on/off toggles.
        """
        P = self.window_P

        # Recorded trajectory: always show its last-P window (continuous truth).
        start = max(0, frame_idx - P + 1)
        self.window_true.set_data(self.x[start:frame_idx + 1], self.y[start:frame_idx + 1])

        if self.rollout_len == 0:
            if self.window_lla is not None:
                self.window_lla.set_data([], [])
            for art in self.window_models.values():
                art.set_data([], [])
            return

        ridx = min(frame_idx, self.rollout_len - 1)

        # ---- LLA window (mode-aware) ----
        if self.lla_traj is not None and self.window_lla is not None:
            traj = self._active_lla_traj()
            if self.show_lla and traj is not None:
                r = min(ridx, len(traj) - 1)
                rs = max(0, r - P + 1)
                xs, ys = self._break_segments(traj[rs:r + 1], rs, self._lla_interval())
                self.window_lla.set_data(xs, ys)
            else:
                self.window_lla.set_data([], [])

        # ---- General model windows ----
        for name, art in self.window_models.items():
            if self.show_general:
                traj = self.general_trajs[name][self.model_mode]
                r = min(ridx, len(traj) - 1)
                rs = max(0, r - P + 1)
                xs, ys = self._break_segments(traj[rs:r + 1], rs, self._gen_interval())
                art.set_data(xs, ys)
            else:
                art.set_data([], [])

    def _maybe_expand_trajectory_view(self):
        """Grow (never shrink) the main trajectory axis limits to fit whatever
        model overlay is currently visible.

        setup_figure() only sizes the initial view from the LLA open-loop
        trajectory and each general model's one-step trajectory
        (_iter_limit_trajs). It does not account for the LLA one-step
        trajectory or the general models' open-loop trajectories, so
        switching model_mode (or toggling LLA's one-step view) can push
        points outside the fixed initial bounds with no way back into view.
        This checks the actually-visible trail/point artists each frame and
        expands the view (only ever outward) to keep them on screen.
        """
        xs_all, ys_all = [], []

        for art in (self.lla_trail, self.lla_point, self.window_lla):
            if art is None:
                continue
            xd = np.asarray(art.get_xdata(), dtype=float)
            yd = np.asarray(art.get_ydata(), dtype=float)
            mask = np.isfinite(xd) & np.isfinite(yd)
            if mask.any():
                xs_all.append(xd[mask])
                ys_all.append(yd[mask])

        for a in self.model_artists.values():
            for art in (a["trail"], a["point"]):
                xd = np.asarray(art.get_xdata(), dtype=float)
                yd = np.asarray(art.get_ydata(), dtype=float)
                mask = np.isfinite(xd) & np.isfinite(yd)
                if mask.any():
                    xs_all.append(xd[mask])
                    ys_all.append(yd[mask])

        if not xs_all:
            return

        data_xmin = float(np.min(np.concatenate(xs_all)))
        data_xmax = float(np.max(np.concatenate(xs_all)))
        data_ymin = float(np.min(np.concatenate(ys_all)))
        data_ymax = float(np.max(np.concatenate(ys_all)))

        cur_xmin, cur_xmax = self.ax.get_xlim()
        cur_ymin, cur_ymax = self.ax.get_ylim()

        pad = max(0.5, 0.1 * max(data_xmax - data_xmin, data_ymax - data_ymin, 1.0))

        new_xmin = min(cur_xmin, data_xmin - pad)
        new_xmax = max(cur_xmax, data_xmax + pad)
        new_ymin = min(cur_ymin, data_ymin - pad)
        new_ymax = max(cur_ymax, data_ymax + pad)

        if (new_xmin, new_xmax, new_ymin, new_ymax) != (cur_xmin, cur_xmax, cur_ymin, cur_ymax):
            self.ax.set_xlim(new_xmin, new_xmax)
            self.ax.set_ylim(new_ymin, new_ymax)

    def update_frame(self, frame_idx):
        """Update visualization for given frame."""
        frame_idx = int(frame_idx)
        self.current_frame = frame_idx

        self.trail.set_data(self.x[:frame_idx + 1], self.y[:frame_idx + 1])
        self.point.set_data([self.x[frame_idx]], [self.y[frame_idx]])

        # Model rollouts (LLA + general fixed models)
        self._update_model_overlays(frame_idx)
        # Grow (never shrink) the view if the active overlay falls outside
        # the initial fixed bounds (e.g. after switching model_mode).
        self._maybe_expand_trajectory_view()
        # Last-P evaluation window on the true path + model rollouts
        self._update_window_overlays(frame_idx)

        self._clear_yaw_arrows(self.rollout_yaw_arrows)
        if self.show_rollout and self.mpc_rollout is not None and frame_idx < len(self.mpc_rollout):
            current_rollout = self.mpc_rollout[frame_idx]
            if len(current_rollout) > 0:
                xs, ys = self._extract_xy(current_rollout)
                self.rollout_line.set_data(xs, ys)
                self.rollout_points.set_data(xs, ys)
                self._draw_yaw_arrows(current_rollout, 'magenta', self.rollout_yaw_arrows)
            else:
                self.rollout_line.set_data([], [])
                self.rollout_points.set_data([], [])
        else:
            self.rollout_line.set_data([], [])
            self.rollout_points.set_data([], [])

        self._clear_yaw_arrows(self.ref_traj_yaw_arrows)
        if self.ref_trajectory is not None and frame_idx < len(self.ref_trajectory):
            current_ref_traj = self.ref_trajectory[frame_idx]
            if len(current_ref_traj) > 0:
                xs, ys = self._extract_xy(current_ref_traj)
                self.ref_traj_line.set_data(xs, ys)
                self.ref_traj_points.set_data(xs, ys)
                self._draw_yaw_arrows(current_ref_traj, 'lime', self.ref_traj_yaw_arrows)
            else:
                self.ref_traj_line.set_data([], [])
                self.ref_traj_points.set_data([], [])
        else:
            self.ref_traj_line.set_data([], [])
            self.ref_traj_points.set_data([], [])

        line_length = 0.5
        end_x = self.x[frame_idx] + line_length * np.cos(self.theta[frame_idx])
        end_y = self.y[frame_idx] + line_length * np.sin(self.theta[frame_idx])
        self.heading_line.set_data([self.x[frame_idx], end_x], [self.y[frame_idx], end_y])

        if self.x_vel_arrow and self.x_vel_arrow in self.ax.patches:
            self.x_vel_arrow.remove()
        if self.y_vel_arrow and self.y_vel_arrow in self.ax.patches:
            self.y_vel_arrow.remove()
        if self.accel_arrow and self.accel_arrow in self.ax.patches:
            self.accel_arrow.remove()

        dx_arrow = self.dx[frame_idx] * np.cos(self.theta[frame_idx])
        dy_arrow = self.dx[frame_idx] * np.sin(self.theta[frame_idx])
        if self.show_velocity and abs(self.dx[frame_idx]) > 0.01:
            self.x_vel_arrow = FancyArrow(
                self.x[frame_idx], self.y[frame_idx],
                dx_arrow, dy_arrow,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='blue', alpha=0.7, zorder=3
            )
            self.ax.add_patch(self.x_vel_arrow)

        if self.show_velocity and abs(self.dy[frame_idx]) > 0.01:
            self.y_vel_arrow = FancyArrow(
                self.x[frame_idx], self.y[frame_idx],
                self.dy[frame_idx] * -np.sin(self.theta[frame_idx]),
                self.dy[frame_idx] * np.cos(self.theta[frame_idx]),
                head_width=0.1, head_length=0.08,
                fc='green', ec='green', alpha=0.7, zorder=3
            )
            self.ax.add_patch(self.y_vel_arrow)

        if self.show_velocity and abs(self.accel[frame_idx]) > 0.01:
            self.accel_arrow = FancyArrow(
                self.x[frame_idx], self.y[frame_idx],
                self.accel[frame_idx] * np.cos(self.theta[frame_idx]),
                self.accel[frame_idx] * np.sin(self.theta[frame_idx]),
                head_width=0.1, head_length=0.08,
                fc='red', ec='red', alpha=0.7, zorder=3
            )
            self.ax.add_patch(self.accel_arrow)

        time_str = f"Time: {self.time[frame_idx]:.3f}s"
        info = (f"Frame: {frame_idx}/{self.n_frames-1}\n{time_str}\n"
                f"\u03b8: {self.theta[frame_idx]:.3f} rad\n"
                f"vx: {self.dx[frame_idx]:.3f} m/s\n"
                f"vy: {self.dy[frame_idx]:.3f} m/s\n"
                f"\u03c9: {self.omega[frame_idx]:.3f} rad/s\n"
                f"accel: {self.accel[frame_idx]:.3f}\n"
                f"steer: {self.steer[frame_idx]:.3f}\n"
                f"Model: {self.model_idx[frame_idx]}")
        self.info_text.set_text(info)

        current_time = self.time[frame_idx]

        for vline in self.param_vlines:
            if vline in vline.axes.lines:
                vline.remove()
        self.param_vlines = []

        for idx, (ax_p, point) in enumerate(zip(self.ax_params, self.param_points)):
            vline = ax_p.axvline(x=current_time, color='red', linestyle='--',
                                 linewidth=1.5, alpha=0.7, zorder=3)
            self.param_vlines.append(vline)

            param_val = self.params[idx][frame_idx]
            point.set_data([current_time], [param_val])

        # Sliding-window cost cursor + value dots
        self._update_cost_cursor(frame_idx)

        self.fig.canvas.draw_idle()
        if getattr(self, "fig_diag", None) is not None:
            self.fig_diag.canvas.draw_idle()

    def setup_controls(self):
        """Create interactive controls."""
        plt.figure(self.fig.number)      # ensure controls land on the main window
        bottom_margin = 0.08

        ax_slider = plt.axes([0.2, bottom_margin + 0.02, 0.6, 0.02])
        self.slider = Slider(
            ax_slider, 'Frame', 0, self.n_frames - 1,
            valinit=0, valstep=1, valfmt='%d'
        )
        self.slider.on_changed(self.on_slider_change)

        ax_play = plt.axes([0.25, bottom_margin - 0.03, 0.08, 0.03])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)

        ax_reset = plt.axes([0.35, bottom_margin - 0.03, 0.08, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

        ax_rollout = plt.axes([0.69, bottom_margin - 0.03, 0.12, 0.03])
        self.btn_rollout = Button(ax_rollout, 'Rollout: ON')
        self.btn_rollout.on_clicked(self.toggle_rollout)
        self.show_rollout = True

        # State-indicator (velocity/accel arrows) show/hide. Independent of the
        # MPC-rollout toggle above.
        ax_vel = plt.axes([0.52, bottom_margin - 0.03, 0.13, 0.03])
        self.btn_velocity = Button(ax_vel, 'Velocity: ON')
        self.btn_velocity.on_clicked(self.toggle_velocity)

        # --- Model-rollout controls ---
        ax_lla = plt.axes([0.83, bottom_margin - 0.03, 0.13, 0.03])
        self.btn_lla = Button(ax_lla, 'LLA: ON')
        self.btn_lla.on_clicked(self.toggle_lla)

        ax_general = plt.axes([0.69, bottom_margin - 0.07, 0.12, 0.03])
        self.btn_general = Button(ax_general, 'Others: ON')
        self.btn_general.on_clicked(self.toggle_general)

        ax_mode = plt.axes([0.83, bottom_margin - 0.07, 0.13, 0.03])
        mode_label = 'Gen: OL' if self.model_mode == 'open_loop' else 'Gen: 1-step'
        self.btn_mode = Button(ax_mode, mode_label)
        self.btn_mode.on_clicked(self.toggle_mode)

        # M-step show/hide (only if computed).
        if self._has_m_step:
            ax_mstep_btn = plt.axes([0.55, bottom_margin - 0.07, 0.12, 0.03])
            self.btn_m_step = Button(ax_mstep_btn, 'M-step: ON')
            self.btn_m_step.on_clicked(self.toggle_m_step)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.animate_step)

    def on_slider_change(self, val):
        self.update_frame(val)

    def toggle_velocity(self, event):
        """Show/hide the velocity/accel state-indicator arrows (independent of
        the MPC-rollout toggle). Arrows are cleared each frame and simply not
        re-added while off, so this just redraws the current frame."""
        self.show_velocity = not self.show_velocity
        self.btn_velocity.label.set_text(
            'Velocity: ON' if self.show_velocity else 'Velocity: OFF')
        self.update_frame(self.current_frame)

    def toggle_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text('Pause')
            self.timer.start()
        else:
            self.btn_play.label.set_text('Play')
            self.timer.stop()

    def toggle_rollout(self, event):
        self.show_rollout = not self.show_rollout
        label = 'Rollout: ON' if self.show_rollout else 'Rollout: OFF'
        self.btn_rollout.label.set_text(label)
        if not self.show_rollout:
            self.rollout_line.set_data([], [])
            self.rollout_points.set_data([], [])
            self._clear_yaw_arrows(self.rollout_yaw_arrows)
        self.update_frame(self.current_frame)

    def toggle_lla(self, event):
        """Show/hide the LLA rollout (independent of general models)."""
        self.show_lla = not self.show_lla
        self.btn_lla.label.set_text('LLA: ON' if self.show_lla else 'LLA: OFF')
        self._refresh_cost_lines()
        self.update_frame(self.current_frame)

    def toggle_general(self, event):
        """Show/hide the general fixed-param models (independent of LLA)."""
        self.show_general = not self.show_general
        self.btn_general.label.set_text('Others: ON' if self.show_general else 'Others: OFF')
        self._refresh_cost_lines()
        self.update_frame(self.current_frame)

    def toggle_mode(self, event):
        """Switch ALL models (LLA + general) between open-loop and one-step."""
        self.model_mode = 'one_step' if self.model_mode == 'open_loop' else 'open_loop'
        label = 'Mode: 1-step' if self.model_mode == 'one_step' else 'Mode: OL'
        self.btn_mode.label.set_text(label)
        self._refresh_cost_lines()
        self.update_frame(self.current_frame)

    def toggle_m_step(self, event):
        """Show/hide the M-step lookahead curves (computation already done)."""
        self.show_m_step = not self.show_m_step
        self.btn_m_step.label.set_text(
            'M-step: ON' if self.show_m_step else 'M-step: OFF')
        self._refresh_cost_lines()
        self.update_frame(self.current_frame)

    def animate_step(self):
        if self.playing:
            next_frame = self.current_frame + 1
            if next_frame >= self.n_frames:
                next_frame = 0
            self.slider.set_val(next_frame)

    def reset(self, event):
        self.playing = False
        self.btn_play.label.set_text('Play')
        self.timer.stop()
        self.slider.set_val(0)

    def show(self):
        self.update_frame(0)
        plt.show()


def main():
    """Main entry point."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'cbfr.npz')

    ref_filepath = os.path.join(os.path.dirname(dir_path), 'tracks', 'mocap_square2slow.npz')

    param_names = {
        0: 'Bf',
        1: 'Br',
        2: 'Cf',
        3: 'Cr',
        4: 'Df',
        5: 'Dr',
        6: 'Cro',
        7: 'Cd',
        8: 'Ce',
        9: 'Cm',
        10: 'Roll',
        11: 'Pitch'
    }

    obstacles = [
        (np.array([1.75, -1.0]), 0.75),
        # (np.array([-1, 1.7]), 0.5),
        (np.array([0.5, 0.5]), 0.5)
    ]
    r_car = 0.04

    general_models = {
        "nominal": {
            'Bf': 6.5, 'Br': 6.5, 'Cf': 1.4, 'Cr': 1.4,
            'Df': 17.0, 'Dr': 17.0, 'Cro': 0.0, 'Cd': 0.0,
            'Ce': 10.0, 'Cm': 0.0,
        },
    }

    visualizer = StateVisualizer(
        filepath,
        ref_filepath=ref_filepath,
        n_params_to_show=range(12),
        params_per_column=6,
        param_names=param_names,
        obstacles=obstacles,
        r_car=r_car,
        general_models=general_models,
        compute_rollouts=True,
        dt=1.0 / 25.0,
        ol_reset_interval=5,
        full_open_loop=False,
        window_P=20,
        cost_form=np.array([0.0, 0.0, 20.0, 5.0, 10.0, 0.01]),
        compute_m_step=False,    # set False to skip the slow M-step computation
        m_step_M=10,
    )
    visualizer.show()


if __name__ == "__main__":
    main()