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
    

    grids = np.meshgrid(*(param_series[k] for k in vary_keys), indexing='ij')

    param_bank = {}
    for i, key in enumerate(vary_keys):
        param_bank[key] = grids[i].flatten()

    num_models = len(param_bank[vary_keys[0]])
    print(f"Total models generated: {num_models}")

    for key, value in fixed_params.items():
        param_bank[key] = np.full(num_models, value, dtype=np.float32)

    params_car = F110()

    dynamics_bank = dynamics.DBMPacejkaBank(
        params_car['lf'], params_car['lr'], 
        params_car['mass'], params_car['Iz'], 
        param_bank['Bf'], param_bank['Br'],
        param_bank['Cf'], param_bank['Cr'],
        param_bank['Df'], param_bank['Dr'],
        param_bank['Cro'], param_bank['Cd'],
        param_bank['Ce'], param_bank['Cm'],
        0, 0,
        num_models
    )

    cost_weights = np.array([1.0, 1.0, 0.1, 0.01, 0, 0])

    lb_history = history_no_record.LBHistory(
        num_models,
        1/40, cost_weights,
        6, rk6Factory,
        dynamics_bank, dynamics.diffequation,
        buffer_size = [0, 0]
    )

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'hallway_recording.npz')

    recording = np.load(filepath, allow_pickle=True)
    total = len(recording["time"])-1
    
    for t in range(total):

        print(f"--- Processing Timestep {t} of {total} ---", flush=True)

        current_state =  recording["state"][t]
        u_opt = recording["ctrl"][t]
        lb_history.predict_states(
            current_state, u_opt
        )

        if(recording["ok_time"][t+1]):
            next_state =  recording["state"][t+1]

            diff, costs = lb_history.update_lookback_error(
                next_state
            )

            cur_best_model = lb_history.get_best_model()

            if(t %50== 0 or t == total):
                print("*********************************", flush=True)
                print(f"Timestep{t}: of {total}")
                print(f"Best Model: {cur_best_model}")
                print(f"Best params: {dynamics_bank.get_model_params_arr(cur_best_model)}")
                print(f"Diff: {diff[cur_best_model]}")
                print(f"Cost: {costs[cur_best_model]}")
        

    

    

    
if __name__ == '__main__':
    main()