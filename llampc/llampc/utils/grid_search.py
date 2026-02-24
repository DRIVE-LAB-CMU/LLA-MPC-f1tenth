import numpy as np
import os
import multiprocessing
import logging
import sys


num_cores = str(multiprocessing.cpu_count())
# Force threading backends to use all cores
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["OPENBLAS_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores


from ..rollout import history_no_record 
from ..rollout import dynamic as dynamics
from llampc.params import F110
from llampc.rollout.rk6 import rk6Factory, rk6MultiStepFactory

# Automatically detect your max CPU cores


# Create a custom logger
logger = logging.getLogger("BatchOptimization")
logger.setLevel(logging.INFO)

multi_step = True

# Prevent adding multiple handlers if the cell/script is run multiple times
if not logger.handlers:
    # 1. Terminal output handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 2. File output handler (creates/overwrites 'optimization.log')
    if(multi_step):
        file_handler = logging.FileHandler("grid_search_multi_step.log", mode="w")
    else:
        file_handler = logging.FileHandler("grid_search_one_step.log", mode="w")

    
    # Use a basic formatter so it looks exactly like your normal logger.info statements
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def one_step_grid_search(total, recording, lb_history, dynamics_bank):
    for t in range(total):
        

        logger.info(f"--- Processing Timestep {t} of {total} ---")

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
                logger.info("*********************************")
                logger.info(f"Timestep{t}: of {total}")
                logger.info(f"Best Model: {cur_best_model}")
                logger.info(f"Best params: {dynamics_bank.get_model_params_arr(cur_best_model)}")
                logger.info(f"Diff: {diff[cur_best_model]}")
                logger.info(f"Cost: {costs[cur_best_model]}")

def multi_step_grid_search(N, total, recording, lb_history, dynamics_bank):
    # Step by N to evaluate distinct, non-overlapping windows 
    # (Total is reduced by N to prevent index out of bounds)
    for t in range(0, total - N, N):
        logger.info(f"--- Processing Timestep {t} -> {t+N} of {total} ---")

        current_state = recording["state"][t]

        # Grab the sequence of N controls
        # Note: Apply any control inversions (like -recording) here if you normally do
        U_seq = recording["ctrl"][t : t+N] 

        # 1. Predict N steps into the future blindly
        lb_history.predict_multi_step(
            current_state, U_seq
        )

        # 2. Only grade the models if the final target state is marked as valid
        if recording["ok_time"][t+N]:
            true_future_state = recording["state"][t+N]

            # 3. Grade the models based on where they ended up at t+N
            diff, costs = lb_history.update_multi_step_error(
                true_future_state
            )

            cur_best_model = lb_history.get_best_model()

            # Adjust logger.infoing frequency since we are skipping N steps at a time
            if (t % (5 * N) == 0) or (t >= total - 2 * N):
                logger.info("*********************************")
                logger.info(f"Timestep Window: {t} to {t+N} (out of {total})")
                logger.info(f"Best Model ID: {cur_best_model}")
                logger.info(f"Best params: {dynamics_bank.get_model_params_arr(cur_best_model)}")
                logger.info(f"Diff at t+{N}: {diff[cur_best_model]}")
                logger.info(f"Cost of Best: {costs[cur_best_model]}")


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
    logger.info(f"Total models generated: {num_models}")

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

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'run_visualizer','hall.npz')

    recording = np.load(filepath, allow_pickle=True)
    total = len(recording["time"])-1

    if multi_step:
        N = 20
        lb_history = history_no_record.LBHistory(
            num_models,
            1/40, cost_weights,
            6, rk6MultiStepFactory,
            dynamics_bank, dynamics.diffequation,
            buffer_size = [0, 0]
        )
        multi_step_grid_search(N, total, recording, lb_history,dynamics_bank)
    else:
        lb_history = history_no_record.LBHistory(
            num_models,
            1/40, cost_weights,
            6, rk6Factory,
            dynamics_bank, dynamics.diffequation,
            buffer_size = [0, 0]
        )
        one_step_grid_search(total, recording, lb_history, dynamics_bank)

    
if __name__ == '__main__':
    main()