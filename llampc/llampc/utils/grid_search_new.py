import itertools
from ..rollout import history_no_record 
from ..rollout import dynamic as dynamics
from llampc.params import F110
from llampc.rollout.rk6 import rk6Factory
import numpy as np
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
 
def main():
    # 1. Setup Parameters (Same as before)
    params_range = {
        'Bf': [0.1, 30], 'Br': [0.1, 30],
        'Cf': [0.1, 2.0], 'Cr': [0.1, 2.0],
        'Df': [0.1, 1], 'Dr': [0.1, 1],
        'Ce': [0.1, 1], 'Cm': [0, 0.1], 
    }
    fixed_params = {'Cro': 0.0, 'Cd': 0.0}
    discretization = 4
    
    param_series = {
        k: np.linspace(v[0], v[1], discretization + 1, dtype=np.float32)
        for k, v in params_range.items()
    }
    grid_iterator = itertools.product(*param_series)

    num_total_models = (discretization + 1) ** len(params_range)
    print(f"Total models to evaluate: {num_total_models}")

    batch_size = 1000000 
    num_batches = int(np.ceil(num_total_models / batch_size))
    params_car = F110()

    # 1. Setup Iterator (Memory Efficient)
    # We use .values() to ensure we pass the actual arrays to product
    grid_iterator = itertools.product(*param_series.values())

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'run_visualizer', 'hallway_recording.npz')

    recording = np.load(filepath, allow_pickle=True)
    total = len(recording["time"])-1

    global_best_cost= np.inf

    for b in range(num_batches):
        # 2. Extract the batch from the iterator
        # This is the "magic" line that prevents RAM explosion
        batch_params = np.array(list(itertools.islice(grid_iterator, batch_size)), dtype=np.float32)
        
        if len(batch_params) == 0:
            break
            
        current_batch_count = len(batch_params)
        start_idx = b * batch_size
        end_idx = start_idx + current_batch_count
        
        


        # 3. Initialize the Bank with this specific batch
        db_batch = dynamics.DBMPacejkaBank(
            params_car['lf'], params_car['lr'], params_car['mass'], params_car['Iz'],
            batch_params[:, 0], batch_params[:, 1], # Bf, Br
            batch_params[:, 2], batch_params[:, 3], # Cf, Cr
            batch_params[:, 4], batch_params[:, 5], # Df, Dr
            np.full(current_batch_count, fixed_params['Cro']), 
            np.full(current_batch_count, fixed_params['Cd']),
            batch_params[:, 6], batch_params[:, 7], # Ce, Cm
            0, 0, current_batch_count
        )

        lb_history = history_no_record.LBHistory(
            current_batch_count, 1/40, np.array([1.0, 1.0, 0.1, 0.01, 0, 0]),
            6, rk6Factory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
        )

        print("*********************************", flush=True)
        print(f"--- Processing Batch {b+1}/{num_batches} (Models {start_idx} to {end_idx}) ---", flush=True)
    
        for t in range(total):

            if(t %100== 0 or t == total-1):
                print(f"--- Processing Timestep {t} of {total} ---", flush=True)

            current_state =  recording["state"][t]
            u_opt = -recording["ctrl"][t]
            lb_history.predict_states(
                current_state, u_opt
            )

            if(recording["ok_time"][t+1]):
                next_state =  recording["state"][t+1]

                diff, costs = lb_history.update_lookback_error(
                    next_state
                )

                cur_best_model = lb_history.get_best_model()

                if(t %500== 0 or t == total-1):
                    print(f"Timestep{ t}: of {total}")
                    print(f"Best Model: {cur_best_model}")
                    print(f"Best params: {db_batch.get_model_params_arr(cur_best_model)}")
                    print(f"One step diff: {diff[cur_best_model]}")
                    print(f"One step cost: {costs[cur_best_model]}")

        best_idx_in_batch = lb_history.get_best_model()
        batch_min_cost = lb_history.running_cost[best_idx_in_batch]

        print(f"\n\n  Batch Min cost: {batch_min_cost}")
        if batch_min_cost < global_best_cost:
            global_best_cost = batch_min_cost
            global_best_params = batch_params[best_idx_in_batch]
            print(f"New Global Best Found! Cost: {global_best_cost}")
            print(f"Params: {global_best_params}")


    print("--- SEARCH COMPLETE ---")
    print(f"Final Best Params: {global_best_params}")
    print(f"Final Best Cost: {global_best_cost}")

if __name__ == '__main__':
    main()