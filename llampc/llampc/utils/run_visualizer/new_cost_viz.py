import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

class GridSearchVisualizer:
    def __init__(self, recording_filepath, batch_filepath):
        self.load_data(recording_filepath, batch_filepath)
        self.setup_figure()
        self.setup_artists()
        self.setup_controls()
        self.current_frame = 0
        self.playing = False

    def load_data(self, rec_path, batch_path):
        print("Loading data...")
        rec = np.load(rec_path, allow_pickle=True)
        sim = np.load(batch_path, allow_pickle=True)

        self.actual_state = rec["state"]
        self.actual_x = self.actual_state[:, 0]
        self.actual_y = self.actual_state[:, 1]
        
        # Grid search predictions
        self.traj_open_loop = sim["traj_open_loop"] 
        self.traj_one_step = sim["traj_one_step"]
        self.params = sim["params"]
        self.global_best_cost = sim["global_best_cost"]
        self.global_best_params = sim["global_best_params"]

        # CRITICAL: Ensure shape is (models, timesteps, states)
        if self.traj_open_loop.shape[0] != len(self.params):
            self.traj_open_loop = np.transpose(self.traj_open_loop, (1, 0, 2))
            self.traj_one_step = np.transpose(self.traj_one_step, (1, 0, 2))
            
        self.num_models = self.traj_open_loop.shape[0]
        self.n_frames = min(len(self.actual_x), self.traj_open_loop.shape[1])
        
        # Check if trajectories are identical (Data problem, not plot problem)
        if self.num_models > 1:
            diff = np.linalg.norm(self.traj_open_loop[0] - self.traj_open_loop[1])
            if diff < 1e-6:
                print("⚠️ WARNING: Model 0 and Model 1 trajectories are MATHEMATICALLY IDENTICAL.")
            else:
                print(f"Success: Models differ by {diff:.4f} units.")

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.75)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

    def setup_artists(self):
        # 1. Info Box
        self.info_text = self.fig.text(0.76, 0.4, '', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9), family='monospace')

        # 2. Ground Truth
        self.ax.plot(self.actual_x, self.actual_y, 'k--', alpha=0.5, label='Actual Path')
        self.actual_point, = self.ax.plot([], [], 'k*', markersize=12, label='Current Pos', zorder=10)

        # 3. Trajectories
        self.os_points = []
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i in range(self.num_models):
            color = colors[i % len(colors)]
            # Draw the second model (blue) with a higher zorder and thicker line to ensure visibility
            z = 5 if i == 0 else 6 
            
            # Plot the Open Loop Path
            self.ax.plot(
                self.traj_open_loop[i, :, 0], 
                self.traj_open_loop[i, :, 1], 
                color=color, linewidth=2, alpha=0.7, 
                label=f'Model {i} (OL)', zorder=z
            )

            # Setup the One-Step Prediction Dot
            pt, = self.ax.plot([], [], 'o', color=color, markersize=8, zorder=z+1)
            self.os_points.append(pt)

        self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    def setup_controls(self):
        ax_slider = plt.axes([0.15, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_frame)
        
        ax_play = plt.axes([0.15, 0.04, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.animate_step)

    def update_frame(self, frame_idx):
        frame_idx = int(frame_idx)
        self.current_frame = frame_idx

        self.actual_point.set_data([self.actual_x[frame_idx]], [self.actual_y[frame_idx]])

        for i, pt in enumerate(self.os_points):
            pt.set_data([self.traj_one_step[i, frame_idx, 0]], 
                        [self.traj_one_step[i, frame_idx, 1]])

        best_p = np.round(self.global_best_params, 3)
        self.info_text.set_text(f"FRAME: {frame_idx}\nCost: {self.global_best_cost:.4f}")
        self.fig.canvas.draw_idle()

    def toggle_play(self, event):
        self.playing = not self.playing
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')
        if self.playing: self.timer.start()
        else: self.timer.stop()

    def animate_step(self):
        if self.playing:
            next_frame = (self.current_frame + 1) % self.n_frames
            self.slider.set_val(next_frame)

    def show(self):
        self.update_frame(0)
        plt.show()

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    visualizer = GridSearchVisualizer(os.path.join(dir_path, 'hall.npz'), os.path.join(dir_path, 'traj_3.npz'))
    visualizer.show()

if __name__ == "__main__":
    main()