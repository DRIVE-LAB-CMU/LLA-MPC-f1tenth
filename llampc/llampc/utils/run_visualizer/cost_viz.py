import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrow
import os


class LLAMPCVisualizer:
    """Interactive visualizer for LLA-MPC predicted trajectories and costs."""

    def __init__(self, filepath, show_all_models=False):
        self.show_all_models = show_all_models
        self.load_data(filepath)
        self.build_frame_cache()
        self.setup_figure()
        self.current_frame = 0
        self.playing = False
        self.setup_artists()
        self.setup_controls()

    def load_data(self, filepath):
        data = np.load(filepath, allow_pickle=True)

        time_ns = data["time"]
        self.time = (time_ns - time_ns[0]) * 1e-9

        self.actual_state = data["state"]
        predicted_states_raw = data["states"]

        self.n_frames = len(self.time)
        self.num_models = predicted_states_raw[0].shape[0]
        self.state_dim = predicted_states_raw[0].shape[1]
        self.predicted_states = np.stack(predicted_states_raw, axis=0)

        self.one_step_cost = data["one_step_cost"]
        self.running_cost = data["running_cost"]
        self.ctrl = data["ctrl"]
        self.params = data["params"]

        self.actual_x = self.actual_state[:, 0]
        self.actual_y = self.actual_state[:, 1]
        self.actual_theta = self.actual_state[:, 2]

        self.best_model_idx = np.argmin(self.running_cost, axis=1)

        print(f"Loaded data:")
        print(f"  Timesteps: {self.n_frames}")
        print(f"  Models: {self.num_models}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Time range: {self.time[0]:.2f}s to {self.time[-1]:.2f}s")

    def build_frame_cache(self):
        """Precompute all per-frame data to avoid heavy work during rendering."""
        self.frame_cache = []
        for i in range(self.n_frames):
            best = self.best_model_idx[i]
            curr = self.actual_state[i]
            pred = self.predicted_states[i, best]
            err = np.linalg.norm(pred[:2] - self.actual_state[i + 1, :2]) if i < self.n_frames - 1 else 0.0
            self.frame_cache.append({
                "x": curr[0],
                "y": curr[1],
                "theta": curr[2],
                "pred_x": pred[0],
                "pred_y": pred[1],
                "best": best,
                "costs": self.running_cost[i],
                "one_step_cost": self.one_step_cost[i, best],
                "running_cost": self.running_cost[i, best],
                "error": err,
                "time": self.time[i],
                "ctrl": self.ctrl[i],
                # x,y positions of ALL models — shape (num_models, 2)
                "all_preds": self.predicted_states[i, :, :2],
            })

    def setup_figure(self):
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(
            3, 3,
            left=0.08, right=0.96, bottom=0.15, top=0.95,
            wspace=0.35, hspace=0.4,
            width_ratios=[2, 1, 1],
            height_ratios=[2, 1, 1]
        )

        self.ax_traj = self.fig.add_subplot(gs[0:2, 0])
        margin = max(1.0, 0.1 * max(self.actual_x.ptp(), self.actual_y.ptp()))
        self.ax_traj.set_xlim(self.actual_x.min() - margin, self.actual_x.max() + margin)
        self.ax_traj.set_ylim(self.actual_y.min() - margin, self.actual_y.max() + margin)
        self.ax_traj.set_xlabel('X Position (m)', fontsize=10)
        self.ax_traj.set_ylabel('Y Position (m)', fontsize=10)
        self.ax_traj.set_aspect('equal', adjustable='box')
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.set_title('LLA-MPC: One-Step Predictions vs Actual', fontsize=12, fontweight='bold')

        self.ax_models = self.fig.add_subplot(gs[0, 1:])
        self.ax_models.set_xlabel('Model Index', fontsize=9)
        self.ax_models.set_ylabel('Running Cost', fontsize=9)
        self.ax_models.set_title('Model Cost Distribution', fontsize=10, fontweight='bold')
        self.ax_models.grid(True, alpha=0.3)

        self.ax_cost_time = self.fig.add_subplot(gs[1, 1:])
        self.ax_cost_time.set_xlabel('Time (s)', fontsize=9)
        self.ax_cost_time.set_ylabel('Best Model Running Cost', fontsize=9)
        self.ax_cost_time.set_title('Cost Evolution', fontsize=10, fontweight='bold')
        self.ax_cost_time.grid(True, alpha=0.3)
        self.ax_cost_time.set_xlim(self.time[0], self.time[-1])

        self.ax_control = self.fig.add_subplot(gs[2, 0])
        self.ax_control.set_xlabel('Time (s)', fontsize=9)
        self.ax_control.set_ylabel('Control', fontsize=9)
        self.ax_control.set_title('Control Inputs', fontsize=10, fontweight='bold')
        self.ax_control.grid(True, alpha=0.3)
        self.ax_control.set_xlim(self.time[0], self.time[-1])

        self.ax_pred_error = self.fig.add_subplot(gs[2, 1:])
        self.ax_pred_error.set_xlabel('Time (s)', fontsize=9)
        self.ax_pred_error.set_ylabel('Prediction Error (m)', fontsize=9)
        self.ax_pred_error.set_title('One-Step Prediction Error', fontsize=10, fontweight='bold')
        self.ax_pred_error.grid(True, alpha=0.3)
        self.ax_pred_error.set_xlim(self.time[0], self.time[-1])

    def setup_artists(self):
        # Actual trajectory trail
        self.actual_trail, = self.ax_traj.plot(
            [], [], 'b-', alpha=0.4, linewidth=2, label='Actual Trajectory'
        )

        # Current actual position
        self.actual_point, = self.ax_traj.plot(
            [], [], 'bo', markersize=10, label='Current Position', zorder=5
        )

        # --- Non-optimal model predictions (green dots, capped at 10) ---
        # One Line2D artist per shown model; best model's dot is hidden each frame.
        self.max_other_models = min(10, self.num_models)
        self.all_pred_points = []
        for _ in range(self.max_other_models):
            point, = self.ax_traj.plot(
                [], [], 'o',
                color='green',
                alpha=0.4,
                markersize=5,
                zorder=7,
                markeredgewidth=0,
                label='_nolegend_',
            )
            self.all_pred_points.append(point)

        # Single legend entry for all non-optimal dots
        self.ax_traj.plot([], [], 'o', color='green', alpha=0.5,
                          markersize=5, label='Other Predictions')

        # Best model one-step prediction (red square on top)
        self.best_pred_point, = self.ax_traj.plot(
            [], [], 'rs', markersize=10, label='Best Model Prediction', zorder=6
        )

        self.prediction_arrow = None
        self.direction_arrow = None

        self.ax_traj.legend(loc='upper right', fontsize=9)

        self.info_text = self.fig.text(
            0.01, 0.99, '', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=9, family='monospace'
        )

        # Model distribution bar chart
        self.model_bars = self.ax_models.bar(
            np.arange(self.num_models), self.running_cost[0], color='blue', alpha=0.6
        )
        self.ax_models.set_xlim(-0.5, self.num_models - 0.5)
        self.ax_models.set_ylim(0, np.max(self.running_cost) * 1.1)

        # Cost over time
        best_costs = self.running_cost[np.arange(len(self.best_model_idx)), self.best_model_idx]
        self.cost_line, = self.ax_cost_time.plot(self.time, best_costs, 'b-', linewidth=2)
        self.cost_vline = self.ax_cost_time.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Control inputs
        self.control_accel, = self.ax_control.plot(self.time, self.ctrl[:, 0], 'r-', label='Acceleration', linewidth=1.5)
        self.control_steer, = self.ax_control.plot(self.time, self.ctrl[:, 1], 'b-', label='Steering', linewidth=1.5)
        self.ax_control.legend(fontsize=8)
        self.control_vline = self.ax_control.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Prediction error over time
        self.pred_errors = np.zeros(self.n_frames - 1)
        for i in range(self.n_frames - 1):
            best_idx = self.best_model_idx[i]
            pred = self.predicted_states[i, best_idx, :2]
            actual_next = self.actual_state[i + 1, :2]
            self.pred_errors[i] = np.linalg.norm(pred - actual_next)

        self.pred_error_line, = self.ax_pred_error.plot(
            self.time[:-1], self.pred_errors, 'g-', linewidth=1.5, label='Prediction Error'
        )
        self.pred_error_vline = self.ax_pred_error.axvline(
            x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7
        )
        if len(self.pred_errors) > 0:
            self.ax_pred_error.set_ylim(0, max(self.pred_errors) * 1.2)

    def setup_controls(self):
        bottom_margin = 0.08

        ax_slider = plt.axes([0.2, bottom_margin + 0.02, 0.6, 0.02])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1,
                             valinit=0, valstep=1, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        ax_play = plt.axes([0.25, bottom_margin - 0.03, 0.08, 0.03])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)

        ax_reset = plt.axes([0.35, bottom_margin - 0.03, 0.08, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

        ax_speed = plt.axes([0.46, bottom_margin - 0.03, 0.2, 0.02])
        self.speed_slider = Slider(ax_speed, 'Speed', 50, 1000, valinit=200, valstep=50)
        self.speed_slider.on_changed(self.on_speed_change)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.animate_step)

    def on_speed_change(self, val):
        if self.playing:
            self.timer.stop()
            self.timer = self.fig.canvas.new_timer(interval=int(1000 / val))
            self.timer.add_callback(self.animate_step)
            self.timer.start()

    def update_frame(self, frame_idx):
        frame_idx = int(frame_idx)
        self.current_frame = frame_idx
        frame = self.frame_cache[frame_idx]

        curr_x      = frame["x"]
        curr_y      = frame["y"]
        curr_theta  = frame["theta"]
        pred_x      = frame["pred_x"]
        pred_y      = frame["pred_y"]
        best_idx    = frame["best"]
        model_costs = frame["costs"]
        pred_error  = frame["error"]
        current_time = frame["time"]
        ctrl        = frame["ctrl"]
        all_preds   = frame["all_preds"]   # shape (num_models, 2)

        # Actual trajectory trail
        self.actual_trail.set_data(self.actual_x[:frame_idx + 1],
                                   self.actual_y[:frame_idx + 1])
        self.actual_point.set_data([curr_x], [curr_y])

        # Direction arrow
        if self.direction_arrow and self.direction_arrow in self.ax_traj.patches:
            self.direction_arrow.remove()
        arrow_length = 0.5
        self.direction_arrow = FancyArrow(
            curr_x, curr_y,
            arrow_length * np.cos(curr_theta),
            arrow_length * np.sin(curr_theta),
            head_width=0.2, head_length=0.15,
            fc='blue', ec='blue', alpha=0.6, zorder=4
        )
        self.ax_traj.add_patch(self.direction_arrow)

        # === Non-optimal predictions — green dots (capped at 10) ===
        # Pick the self.max_other_models non-best models with the lowest running cost.
        other_indices = [m for m in range(self.num_models) if m != best_idx]
        other_indices.sort(key=lambda m: model_costs[m])
        shown = other_indices[:self.max_other_models]

        for slot, point in enumerate(self.all_pred_points):
            if slot < len(shown):
                m = shown[slot]
                point.set_data([all_preds[m, 0]], [all_preds[m, 1]])
            else:
                point.set_data([], [])

        # Best model prediction (red square)
        self.best_pred_point.set_data([pred_x], [pred_y])

        # Prediction arrow (best model)
        if self.prediction_arrow and self.prediction_arrow in self.ax_traj.patches:
            self.prediction_arrow.remove()
        pred_dx = pred_x - curr_x
        pred_dy = pred_y - curr_y
        if pred_dx ** 2 + pred_dy ** 2 > 1e-4:
            self.prediction_arrow = FancyArrow(
                curr_x, curr_y, pred_dx, pred_dy,
                head_width=0.15, head_length=0.1,
                fc='red', ec='red', alpha=0.5, zorder=3, linestyle='--'
            )
            self.ax_traj.add_patch(self.prediction_arrow)

        # Model distribution bars
        for i, bar in enumerate(self.model_bars):
            bar.set_height(model_costs[i])
            bar.set_color('red' if i == best_idx else 'blue')

        # Vertical time markers
        self.cost_vline.set_xdata([current_time, current_time])
        self.control_vline.set_xdata([current_time, current_time])
        self.pred_error_vline.set_xdata([current_time, current_time])

        # Info text
        self.info_text.set_text(
            f"Frame: {frame_idx}/{self.n_frames-1}\n"
            f"Time: {current_time:.3f}s\n"
            f"Position: ({curr_x:.2f}, {curr_y:.2f})\n"
            f"Heading: {curr_theta:.3f} rad\n"
            f"Predicted: ({pred_x:.2f}, {pred_y:.2f})\n"
            f"Best Model: {best_idx}/{self.num_models-1}\n"
            f"Running Cost: {frame['running_cost']:.4f}\n"
            f"One-step Cost: {frame['one_step_cost']:.4f}\n"
            f"Prediction Error: {pred_error:.4f} m\n"
            f"Control: a={ctrl[0]:.3f}, δ={ctrl[1]:.3f}"
        )

        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        self.update_frame(val)

    def toggle_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text('Pause')
            self.timer.start()
        else:
            self.btn_play.label.set_text('Play')
            self.timer.stop()

    def animate_step(self):
        if self.playing:
            next_frame = self.current_frame + 1
            if next_frame >= self.n_frames:
                next_frame = 0
            self.slider.set_val(next_frame)
            self.timer.interval = int(1000 / self.speed_slider.val)

    def reset(self, event):
        self.playing = False
        self.btn_play.label.set_text('Play')
        self.timer.stop()
        self.slider.set_val(0)

    def show(self):
        self.update_frame(0)
        plt.show()


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'hall.npz')
    # filepath = os.path.join(dir_path, 'spinout.npz')

    visualizer = LLAMPCVisualizer(filepath, show_all_models=False)
    visualizer.show()


if __name__ == "__main__":
    main()
