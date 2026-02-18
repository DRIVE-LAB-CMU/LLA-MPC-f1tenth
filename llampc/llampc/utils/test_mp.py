from ..rollout import history_no_record 
from ..rollout import dynamic as dynamics
from llampc.params import F110
from llampc.rollout.rk6 import rk6Factory
import numpy as np
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

def main():
    params_range = {
        'Bf': [0.1, 30],
        'Br': [0.1, 30],
        'Cf': [0.1, 2.0],
        'Cr': [0.1, 2.0],
        'Df': [0.1, 1],
        'Dr': [0.1, 1],
        'Ce': [0.1, 1],
        'Cm': [0, 0.1], 
    }

    fixed_params = {
        'Cro': 0.0,
        'Cd': 0.0
    }

    discretization = 5
    
    param_series = {
        param_key: np.linspace(value[0], value[1], discretization + 1, dtype=np.float32)
        for param_key, value in params_range.items()
    }

    vary_keys = list(params_range.keys())
    

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'hallway_recording.npz')

    recording = np.load(filepath, allow_pickle=True)
    for record in recording["ctrl"]:
        print(record)
        

    

    

    
if __name__ == '__main__':
    main()