import numpy as np
import matplotlib.pyplot as plt
import os

def plot_control_signals(rec_path):
    if not os.path.exists(rec_path):
        print(f"Error: File {rec_path} not found.")
        return

    # Load data
    rec = np.load(rec_path, allow_pickle=True)
    
    # Assuming 'ctrl' is a 2D array where col 0 is steer and col 1 is accel
    controls = rec["ctrl"]
    steer = controls[:, 1]
    accel = controls[:, 0]
    
    # Create time/frame axis
    frames = np.arange(len(steer))

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.canvas.manager.set_window_title('Vehicle Control Profile')

    # Subplot 1: Steering
    ax1.plot(frames, steer, color='tab:blue', linewidth=1.5)
    ax1.set_ylabel('Steer $\delta$ [rad]')
    ax1.set_title('Control Inputs Over Time')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Acceleration
    ax2.plot(frames, accel, color='tab:red', linewidth=1.5)
    ax2.set_ylabel('Accel $a$ [$m/s^2$]')
    ax2.set_xlabel('Frame Index')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # Adjust 'hall.npz' to your specific recording filename
    plot_control_signals(os.path.join(dir_path, 'nshtrack.npz'))