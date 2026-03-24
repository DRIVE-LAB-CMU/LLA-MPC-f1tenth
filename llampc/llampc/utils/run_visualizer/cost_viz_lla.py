import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import os

class GridSearchVisualizer:
    def __init__(self, recording_filepath, batch_filepath, reset_interval=None):
        self.reset_interval = reset_interval
        
        self.load_data(recording_filepath, batch_filepath)
        
        self.show_ol = True
        self.show_os = True
        self.show_lla = True
        self.follow_camera = False 
        self.show_model = [True] * self.num_models
        
        self.setup_figure()
        self.setup_artists()
        self.setup_controls()
        
        self.current_frame = 0
        self.playing = False
        
        self.refresh_visibility()

    def load_data(self, rec_path, batch_path):
        print("Loading data...")
        rec = np.load(rec_path, allow_pickle=True)
        sim = np.load(batch_path, allow_pickle=True)

        self.actual_state = rec["state"]
        self.actual_x = self.actual_state[:, 0]
        self.actual_y = self.actual_state[:, 1]
        
        self.traj_open_loop = sim["traj_open_loop"] 
        self.traj_one_step = sim["traj_one_step"]
        self.params = sim["params"]
        
        # Accommodate the naming conventions from the grid search saving logic
        self.global_best_cost = sim.get("global_best_static_cost", sim.get("global_best_cost", 0.0))
        self.global_best_params = sim.get("global_best_static_params", sim.get("global_best_params", self.params[0]))

        # --- NEW: Load LLA dynamic rollout data ---
        self.has_lla = "lla_dynamic_trajectory" in sim
        if self.has_lla:
            self.lla_traj = sim["lla_dynamic_trajectory"]
            self.lla_params = sim["lla_optimal_params"]
            self.lla_costs = sim["lla_optimal_costs"]
            print("LLA dynamic trajectory and parameters loaded.")
        else:
            print("No LLA data found in simulation file.")

        if self.traj_open_loop.shape[0] != len(self.params):
            self.traj_open_loop = np.transpose(self.traj_open_loop, (1, 0, 2))
            self.traj_one_step = np.transpose(self.traj_one_step, (1, 0, 2))
            
        self.num_models = self.traj_open_loop.shape[0]
        self.n_frames = min(len(self.actual_x), self.traj_open_loop.shape[1])
        print(f"Loaded {self.num_models} static models with {self.n_frames} frames.")

        self.optimal_idx = 0
        for i, p in enumerate(self.params):
            if np.allclose(p, self.global_best_params):
                self.optimal_idx = i
                break
        print(f"Optimal static model identified at index {self.optimal_idx}.")

    def filter_interval(self, x, y):
        if not self.reset_interval or self.reset_interval <= 0:
            return x, y
            
        insert_indices = np.arange(self.reset_interval, len(x), self.reset_interval)
        
        if len(insert_indices) == 0:
            return x, y
        
        x_filtered = np.insert(x, insert_indices, np.nan)
        y_filtered = np.insert(y, insert_indices, np.nan)
        
        return x_filtered, y_filtered

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.28, right=0.70)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        x_min, x_max = np.min(self.actual_x), np.max(self.actual_x)
        y_min, y_max = np.min(self.actual_y), np.max(self.actual_y)
        margin_x = max((x_max - x_min) * 0.1, 5.0)
        margin_y = max((y_max - y_min) * 0.1, 5.0)
        
        self.global_xlim = (x_min - margin_x, x_max + margin_x)
        self.global_ylim = (y_min - margin_y, y_max + margin_y)
        
        self.ax.set_xlim(self.global_xlim)
        self.ax.set_ylim(self.global_ylim)

    def setup_artists(self):
        self.info_text = self.fig.text(0.72, 0.60, '', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9), family='monospace')

        plot_act_x, plot_act_y = self.filter_interval(self.actual_x, self.actual_y)
        self.ax.plot(plot_act_x, plot_act_y, 'k--', linewidth=2, alpha=0.6, label='Actual Path', zorder=2)
        
        self.actual_step_line, = self.ax.plot([], [], 'k-', linewidth=2, alpha=0.8, zorder=2)
        self.actual_point, = self.ax.plot([], [], 'k*', markersize=14, label='Current Pos', zorder=3)
        self.next_point, = self.ax.plot([], [], 'ko', markerfacecolor='none', markersize=10, 
                                        markeredgewidth=2, label='Next Pos', zorder=3)

        # --- NEW: Setup LLA dynamic artists ---
        if self.has_lla:
            plot_lla_x, plot_lla_y = self.filter_interval(self.lla_traj[:, 0], self.lla_traj[:, 1])
            self.lla_line, = self.ax.plot(plot_lla_x, plot_lla_y, 'm-', linewidth=3, alpha=0.9, label='LLA Dynamic', zorder=15)
            self.lla_point, = self.ax.plot([], [], 'mo', markersize=8, zorder=16)

        self.ol_lines = []
        self.os_points = []
        
        for i in range(self.num_models):
            if i == self.optimal_idx:
                color = 'red'
                lw = 3          
                ms = 10         
                alpha = 1.0     
                z_line = 10     
                z_dot = 11
                label_str = f'Model {i} (Optimal)'
            else:
                color = 'blue'  
                lw = 1          
                ms = 4          
                alpha = 0.6     
                z_line = 20     
                z_dot = 21      
                label_str = f'Model {i}'

            plot_ol_x, plot_ol_y = self.filter_interval(self.traj_open_loop[i, :, 0], self.traj_open_loop[i, :, 1])
            line, = self.ax.plot(
                plot_ol_x, 
                plot_ol_y, 
                color=color, linewidth=lw, alpha=alpha, 
                label=label_str, zorder=z_line
            )
            self.ol_lines.append(line)

            pt, = self.ax.plot([], [], 'o', color=color, markersize=ms, alpha=alpha, zorder=z_dot)
            self.os_points.append(pt)

        handles, labels = self.ax.get_legend_handles_labels()
        desired_labels = ['Actual Path', 'Current Pos', 'Next Pos', f'Model {self.optimal_idx} (Optimal)']
        if self.has_lla: desired_labels.append('LLA Dynamic')
        
        filtered_handles = [h for h, l in zip(handles, labels) if l in desired_labels]
        filtered_labels = [l for l in labels if l in desired_labels]
                
        self.ax.legend(filtered_handles, filtered_labels, loc='upper left', bbox_to_anchor=(1.02, 1))

        self.annot = self.ax.annotate(
            "", xy=(0,0), xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            zorder=30
        )
        self.annot.set_visible(False)

    def setup_controls(self):
        ax_slider = plt.axes([0.15, 0.15, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_frame)
        
        ax_play = plt.axes([0.15, 0.05, 0.1, 0.05])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)

        ax_check_type = plt.axes([0.28, 0.01, 0.25, 0.12])
        ax_check_type.axis('off') 
        
        # Inject LLA toggle if data exists
        labels = ['Show Open-Loop', 'Show One-Step', 'Follow Camera']
        actives = [True, True, False]
        if self.has_lla:
            labels.insert(2, 'Show LLA Dynamic')
            actives.insert(2, True)

        self.check_type = CheckButtons(ax_check_type, tuple(labels), tuple(actives))
        self.check_type.on_clicked(self.toggle_type_visibility)

        ax_check_models = plt.axes([0.55, 0.01, 0.3, 0.12])
        ax_check_models.axis('off')
        
        model_labels = tuple([f'Model {i}' for i in range(self.num_models)])
        self.check_models = CheckButtons(ax_check_models, model_labels, tuple(self.show_model))
        self.cb_cid = self.check_models.on_clicked(self.toggle_model_visibility)

        ax_btn_all_on = plt.axes([0.86, 0.07, 0.12, 0.04])
        self.btn_all_on = Button(ax_btn_all_on, 'All OL On')
        self.btn_all_on.on_clicked(self.turn_all_models_on)

        ax_btn_all_off = plt.axes([0.86, 0.02, 0.12, 0.04])
        self.btn_all_off = Button(ax_btn_all_off, 'All OL Off')
        self.btn_all_off.on_clicked(self.turn_all_models_off)
        
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.animate_step)

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)


    def toggle_model_visibility(self, label):
        # Extract the model index from the label string (e.g., "Model 0" -> 0)
        try:
            model_idx = int(label.split()[1])
            self.show_model[model_idx] = not self.show_model[model_idx]
            self.refresh_visibility()
        except (ValueError, IndexError):
            pass
        
    def turn_all_models_on(self, event):
        self.check_models.disconnect(self.cb_cid)
        for i in range(self.num_models):
            if not self.check_models.get_status()[i]:
                self.check_models.set_active(i)  
            self.show_model[i] = True            
        self.cb_cid = self.check_models.on_clicked(self.toggle_model_visibility)
        self.refresh_visibility()

    def turn_all_models_off(self, event):
        self.check_models.disconnect(self.cb_cid)
        for i in range(self.num_models):
            if self.check_models.get_status()[i]:
                self.check_models.set_active(i)  
            self.show_model[i] = False           
        self.cb_cid = self.check_models.on_clicked(self.toggle_model_visibility)
        self.refresh_visibility()

    def on_hover(self, event):
        if event.inaxes == self.ax and self.show_os:
            for i, pt in enumerate(self.os_points):
                cont, ind = pt.contains(event)
                if cont:
                    x_data, y_data = pt.get_data()
                    self.annot.xy = (x_data[0], y_data[0])
                    
                    param_str = np.array2string(self.params[i], precision=3, separator=', ')
                    text = f"Model {i}\nParams: {param_str}"
                    if i == self.optimal_idx:
                        text += "\n(Optimal)"
                        
                    self.annot.set_text(text)
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                    return 
            
        if self.annot.get_visible():
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()

    def refresh_visibility(self):
        for i in range(self.num_models):
            self.ol_lines[i].set_visible(self.show_ol and self.show_model[i])
            self.os_points[i].set_visible(self.show_os)
            
        if self.has_lla:
            self.lla_line.set_visible(self.show_lla)
            self.lla_point.set_visible(self.show_lla)
            
        self.fig.canvas.draw_idle()

    def toggle_type_visibility(self, label):
        if label == 'Show Open-Loop':
            self.show_ol = not self.show_ol
        elif label == 'Show One-Step':
            self.show_os = not self.show_os
        elif label == 'Show LLA Dynamic':
            self.show_lla = not self.show_lla
        elif label == 'Follow Camera':
            self.follow_camera = not self.follow_camera
            if not self.follow_camera:
                self.ax.set_xlim(self.global_xlim)
                self.ax.set_ylim(self.global_ylim)
            else:
                self.update_frame(self.current_frame)
                
        self.refresh_visibility()

    def update_frame(self, frame_idx):
        frame_idx = int(frame_idx)
        self.current_frame = frame_idx

        curr_x = self.actual_x[frame_idx]
        curr_y = self.actual_y[frame_idx]
        
        if frame_idx < self.n_frames - 1:
            next_x = self.actual_x[frame_idx + 1]
            next_y = self.actual_y[frame_idx + 1]
        else:
            next_x = curr_x
            next_y = curr_y

        self.actual_point.set_data([curr_x], [curr_y])
        self.next_point.set_data([next_x], [next_y])
        
        if self.reset_interval and (frame_idx + 1) % self.reset_interval == 0:
            self.actual_step_line.set_data([], [])
        else:
            self.actual_step_line.set_data([curr_x, next_x], [curr_y, next_y])

        for i, pt in enumerate(self.os_points):
             pt.set_data([self.traj_one_step[i, frame_idx, 0]], 
                         [self.traj_one_step[i, frame_idx, 1]])

        # Update LLA position dot
        if self.has_lla:
            lla_idx = min(frame_idx, len(self.lla_traj) - 1)
            self.lla_point.set_data([self.lla_traj[lla_idx, 0]], [self.lla_traj[lla_idx, 1]])

        if self.follow_camera:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            width = xlim[1] - xlim[0]
            height = ylim[1] - ylim[0]
            
            self.ax.set_xlim(curr_x - width / 2, curr_x + width / 2)
            self.ax.set_ylim(curr_y - height / 2, curr_y + height / 2)

        # --- UPDATE INFO TEXT ---
        best_p_str = np.array2string(self.global_best_params, precision=3, separator=',\n')
        
        lla_str = ""
        if self.has_lla:
            lla_p_idx = min(frame_idx, len(self.lla_params) - 1)
            cur_lla_p = np.array2string(self.lla_params[lla_p_idx], precision=3, separator=',\n')
            cur_lla_cost = self.lla_costs[lla_p_idx]
            lla_str = (
                f"--- LLA DYNAMIC ---\n"
                f"Cost at t={lla_p_idx}: {cur_lla_cost:.4f}\n"
                f"Params at t={lla_p_idx}:\n{cur_lla_p}\n"
                f"-------------------\n"
            )

        info_str = (
            f"FRAME: {frame_idx}\n"
            f"Static Cost: {self.global_best_cost:.4f}\n"
            f"-------------------\n"
            f"{lla_str}"
            f"GLOBAL STATIC PARAMS:\n"
            f"{best_p_str}"
        )
        self.info_text.set_text(info_str)
        
        if self.annot.get_visible():
            self.annot.set_visible(False)
            
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
    # Assuming 'all_batch_best_trajectories.npz' or similar is the output
    visualizer = GridSearchVisualizer(os.path.join(dir_path, 'nshtrack.npz'), os.path.join(dir_path, 'lla_first.npz'), 20)
    visualizer.show()

if __name__ == "__main__":
    main()