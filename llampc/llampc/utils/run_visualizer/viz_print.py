import numpy as np
import os

def main():
    """Loads trajectory data and prints all states, controls, and parameters to the console."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'nshworks.npz')
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return

    print(f"Loading data from {filepath}...\n")
    data = np.load(filepath, allow_pickle=True)
    
    # Extract Base Arrays
    time = data["time"]
    n_frames = len(time)
    
    state = data["state"]
    x = state[:, 0]
    y = state[:, 1]
    theta = state[:, 2]
    dx = state[:, 3]
    dy = state[:, 4]
    omega = state[:, 5]

    ctrl = data["ctrl"]
    accel = ctrl[:, 0]
    steer = ctrl[:, 1]
    
    # Some older files might not have model_index, so we use .get() to be safe
    model_idx = data.get("model_index", np.zeros(n_frames, dtype=int))

    # Handle Parameter Transposing (matching your original visualizer logic)
    params_raw = data["params"]
    if isinstance(params_raw, np.ndarray) and params_raw.ndim == 2:
        if params_raw.shape[0] == len(time):
            params = [params_raw[:, i] for i in range(params_raw.shape[1])]
        else:
            params = [params_raw[i, :] for i in range(params_raw.shape[0])]
    else:
        params = list(params_raw)

    param_names = [
        'Bf', 'Br', 'Cf', 'Cr', 'Df', 'Dr', 
        'Cro', 'Cd', 'Ce', 'Cm', 'Roll', 'Pitch'
    ]

    print(f"Total frames to print: {n_frames}")
    print("=" * 60)

    # Loop through and print every frame
    for i in range(n_frames):
        print(f"--- Frame {i:04d} | Time: {time[i]:.3f}s ---")
        
        # Print States
        print(f"  State : X: {x[i]:8.3f} | Y: {y[i]:8.3f} | Theta: {theta[i]:6.3f} rad")
        print(f"  Vels  : vx: {dx[i]:7.3f} | vy: {dy[i]:7.3f} | omega: {omega[i]:6.3f} rad/s")
        
        # Print Controls and Model Index
        print(f"  Ctrl  : accel: {accel[i]:5.3f} | steer: {steer[i]:5.3f}")
        print(f"  Model : {model_idx[i]}")
        
        # Print Parameters
        param_strs = []
        for p_idx, p_name in enumerate(param_names):
            if p_idx < len(params):
                param_strs.append(f"{p_name}: {params[p_idx][i]:.3f}")
        
        # Group parameters into chunks of 6 for readable terminal printing
        print("  Params: " + ", ".join(param_strs[:6]))
        if len(param_strs) > 6:
            print("          " + ", ".join(param_strs[6:]))
            
        print("-" * 60)

if __name__ == "__main__":
    main()