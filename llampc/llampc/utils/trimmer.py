import numpy as np
import os
data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_visualizer','sysid.npz'), allow_pickle=True)

trimmed = {}
for key in data.files:
    arr = data[key]
    # only trim arrays that have n_frames as first dimension
    if isinstance(arr, np.ndarray) and arr.shape[0] > 900:
        trimmed[key] = arr[900:]
    else:
        trimmed[key] = arr

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),'run_visualizer','sysid_trimmed.npz'), **trimmed)