import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # Update this to match your exact batch filename if different
    batch_filepath = os.path.join(dir_path, 'traj_nsht_ol.npz')
    
    print("Loading data...")
    sim = np.load(batch_filepath, allow_pickle=True)
    
    traj_open_loop = sim["traj_open_loop"]
    params = sim["params"]
    
    # Ensure correct shape: (num_models, n_frames, 2)
    if traj_open_loop.shape[0] != len(params):
        traj_open_loop = np.transpose(traj_open_loop, (1, 0, 2))
        
    num_models = traj_open_loop.shape[0]
    n_frames = traj_open_loop.shape[1]
    
    print(f"Plotting {num_models} models with {n_frames} frames each...")

    # Set up a clean, static plot
    plt.figure(figsize=(12, 8))
    plt.title("All Open-Loop Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Plot every model's full trajectory
    for i in range(num_models):
        x_data = traj_open_loop[i, :, 0]
        y_data = traj_open_loop[i, :, 1]
        
        # Plotting with some transparency so overlapping lines are visible
        plt.plot(x_data, y_data, alpha=0.5, linewidth=1)

    print("Rendering plot...")
    plt.show()

if __name__ == "__main__":
    main()