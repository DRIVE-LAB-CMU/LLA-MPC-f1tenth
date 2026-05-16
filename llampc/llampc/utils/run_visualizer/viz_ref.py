from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrow
import os


class StateVisualizer:
    """Interactive visualizer for state trajectory data."""
    
    def __init__(self, filepath, ref_filepath=None, n_params_to_show=None, params_per_column=3, param_names=None):
        self.n_params_to_show = n_params_to_show
        self.params_per_column = params_per_column
        self.param_names = param_names
        self.ref_filepath = ref_filepath
        
        self.load_data(filepath)
        self.load_ref_data()
        self.setup_figure()
        self.current_frame = 0
        self.playing = False
        self.setup_artists()
        self.setup_controls()
        
    def load_data(self, filepath):
        """Load trajectory data from npz file."""
        data = np.load(filepath, allow_pickle=True)
        self.time = data["time"]
        state = data["state"]
        self.x = state[:, 0]
        self.y = state[:, 1]
        self.theta = state[:, 2]
        self.dx = state[:, 3]
        self.dy = state[:, 4]
        self.omega = state[:, 5]
        self.n_frames = len(self.x)

        # params could be either:
        # - shape (n_timesteps, n_params) - need to transpose
        # - list of arrays where each is length n_timesteps
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
        self.accel = ctrl[:, 0]
        self.steer = ctrl[:, 1]

        # --- NEW: Load MPC Rollout ---
        self.mpc_rollout = None
        if "mpc_rollout" in data:
            rollout_data = data["mpc_rollout"]
            # Check if it actually contains populated arrays
            if len(rollout_data) > 0 and len(rollout_data[0]) > 0:
                self.mpc_rollout = rollout_data
        
        # Determine how many parameters to show
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
        
    def setup_figure(self):
        """Create figure with appropriate layout."""
        n_param_cols = int(np.ceil(len(self.n_params_to_show)/ self.params_per_column))
        n_param_rows = min(self.params_per_column, len(self.n_params_to_show))
        
        self.fig = plt.figure(figsize=(8 + n_param_cols * 4, max(8, 3 + n_param_rows * 1.8)))
        
        gs = self.fig.add_gridspec(
            n_param_rows, 1 + n_param_cols, 
            left=0.08, right=0.96, bottom=0.15, top=0.95,
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
        
    def setup_artists(self):
        """Initialize plot elements."""
        if self.ref_x is not None and self.ref_y is not None:
            self.ax.plot(self.ref_x, self.ref_y, 'k--', alpha=0.4, linewidth=1.5, label='Raceline', zorder=1)

        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label='Trajectory', zorder=2)
        
        # --- NEW: MPC Rollout Line ---
        self.rollout_line, = self.ax.plot([], [], 'm--', alpha=0.8, linewidth=2, zorder=3)
        
        self.point, = self.ax.plot([], [], 'ko', markersize=10, label='Position', zorder=5)
        self.heading_line, = self.ax.plot([], [], 'k-', linewidth=2, zorder=4)

        self.x_vel_arrow = None
        self.y_vel_arrow = None
        self.accel_arrow = None
        
        from matplotlib.patches import Patch
        legend_elements = []
        
        if self.ref_x is not None:
            legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=1.5, alpha=0.4, label='Reference'))
            
        legend_elements.extend([
            Line2D([0], [0], color='black', lw=2, label='Heading (Yaw)'),
            Patch(facecolor='blue', alpha=0.7, label='X Velocity'),
            Patch(facecolor='green', alpha=0.7, label='Y Velocity'),
            Patch(facecolor='red', alpha=0.7, label='Acceleration')
        ])

        # Add MPC Rollout to legend if data exists
        if self.mpc_rollout is not None:
            legend_elements.insert(0, Line2D([0], [0], color='m', linestyle='--', lw=2, label='MPC Rollout'))
            
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
            
    def setup_controls(self):
        """Create interactive controls."""
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
        
        ax_speed = plt.axes([0.46, bottom_margin - 0.03, 0.2, 0.02])
        self.speed_slider = Slider(
            ax_speed, 'Speed', 50, 1000,
            valinit=200, valstep=50
        )

        self.speed_slider.on_changed(self.on_speed_change)
        
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.animate_step)

    def on_speed_change(self, val):
        """Update playback speed dynamically."""
        if self.playing:
            interval = int(1000 / val)
            self.timer.stop()
            self.timer = self.fig.canvas.new_timer(interval=interval)
            self.timer.add_callback(self.animate_step)
            self.timer.start()

    def update_frame(self, frame_idx):
        """Update visualization for given frame."""
        frame_idx = int(frame_idx)
        self.current_frame = frame_idx
        
        self.trail.set_data(self.x[:frame_idx+1], self.y[:frame_idx+1])
        
        self.point.set_data([self.x[frame_idx]], [self.y[frame_idx]])

        # --- NEW: Update MPC Rollout ---
        if self.mpc_rollout is not None and frame_idx < len(self.mpc_rollout):
            current_rollout = self.mpc_rollout[frame_idx]
            if len(current_rollout) > 0:
                # Assuming state format [x, y, theta, dx, dy, omega]
                current_rollout = np.array(current_rollout)
                self.rollout_line.set_data(current_rollout[:, 0], current_rollout[:, 1])
            else:
                self.rollout_line.set_data([], [])
        else:
            self.rollout_line.set_data([], [])

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
        if abs(self.dx[frame_idx]) > 0.01:
            self.x_vel_arrow = FancyArrow(
                self.x[frame_idx], self.y[frame_idx],
                dx_arrow, dy_arrow,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='blue', alpha=0.7, zorder=3
            )
            self.ax.add_patch(self.x_vel_arrow)
        
        if abs(self.dy[frame_idx]) > 0.01:
            self.y_vel_arrow = FancyArrow(
                self.x[frame_idx], self.y[frame_idx],
                self.dy[frame_idx] * -np.sin(self.theta[frame_idx]),
                self.dy[frame_idx] * np.cos(self.theta[frame_idx]),
                head_width=0.1, head_length=0.08,
                fc='green', ec='green', alpha=0.7, zorder=3
            )
            self.ax.add_patch(self.y_vel_arrow)

        if abs(self.accel[frame_idx]) > 0.01:
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
                f"θ: {self.theta[frame_idx]:.3f} rad\n"
                f"vx: {self.dx[frame_idx]:.3f} m/s\n"
                f"vy: {self.dy[frame_idx]:.3f} m/s\n"
                f"ω: {self.omega[frame_idx]:.3f} rad/s\n"
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
            
            interval = int(1000 / self.speed_slider.val)
            self.timer.interval = interval
            
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
    # filepath = os.path.join(dir_path, 'simlhnoterm.npz')
    # filepath = os.path.join(dir_path, 'simlhmultimm.npz')
    # filepath = os.path.join(dir_path, 'sim8oz.npz')
    # filepath = os.path.join(dir_path, 'sim8term.npz')
    # filepath = os.path.join(dir_path, 'blevel_oval_noadapt.npz')
    # filepath = os.path.join(dir_path, 'blevel_oval_os.npz')
    # filepath = os.path.join(dir_path, 'blevel_fig8_noadapt.npz')
    # filepath = os.path.join(dir_path, 'blevel_fig8_multi.npz')
    # filepath = os.path.join(dir_path, 'sim_os.npz')
    # filepath = os.path.join(dir_path, 'sim_multi.npz')
    # filepath = os.path.join(dir_path, 'nsimos.npz')
    # filepath = os.path.join(dir_path, 'nsimmulti.npz')
    # filepath = os.path.join(dir_path, 'sim_multi.npz')
    # filepath = os.path.join(dir_path, 'sim_multi.npz')
    # filepath = os.path.join(dir_path, 'sim_noterm.npz')
    # filepath = os.path.join(dir_path, 'blevel_circle_sim.npz')
    # filepath = os.path.join(dir_path, 'blevel_circle_mllampcgood.npz')
    # filepath = os.path.join(dir_path, 'blevel_circle_mnollampcbad.npz')
    # filepath = os.path.join(dir_path, 'sim_8c.npz')
    # filepath = os.path.join(dir_path, 'sim_oval.npz')
    # filepath = os.path.join(dir_path, 'sim_ovala.npz')
    filepath = os.path.join(dir_path, 'sysid.npz')

    ref_filepath = os.path.join(os.path.dirname(dir_path), 'tracks', 'mocap_square1.npz') 
    
    # Optional: Define parameter names
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
    
    # Create visualizer
    visualizer = StateVisualizer(
        filepath, 
        ref_filepath=ref_filepath, # Pass the raceline file path here
        n_params_to_show=range(12), 
        params_per_column=6,
        param_names=param_names
    )
    visualizer.show()

if __name__ == "__main__":
    main()