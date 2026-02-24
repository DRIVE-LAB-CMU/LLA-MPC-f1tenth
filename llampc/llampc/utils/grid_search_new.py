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

import itertools
from ..rollout import history_no_record 
from ..rollout import dynamic as dynamics
from llampc.params import F110
from llampc.rollout.rk6 import rk6Factory, rk6MultiStepFactory

multi_step = True
# Create a custom logger
logger = logging.getLogger("BatchOptimization")
logger.setLevel(logging.INFO)

if(multi_step):
    file_handler = logging.FileHandler("grid_search_batch_multi_step.log", mode="w")
else:
    file_handler = logging.FileHandler("grid_search_batch_one_step.log", mode="w")

# Prevent adding multiple handlers if the cell/script is run multiple times
if not logger.handlers:
    # 1. Terminal output handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 2. File output handler (creates/overwrites 'optimization.log')
    file_handler = logging.FileHandler("grid_search_batched.log", mode="w")
    
    # Use a basic formatter so it looks exactly like your normal logger.info statements
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def grid_search_one_step(total, recording, lb_history, db_batch):

    for t in range(total):

        if(t %100== 0 or t == total-1):
            logger.info(f"--- Processing Timestep {t} of {total} ---")

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

            if(t %100== 0 or t == total-1):
                logger.info(f"Timestep{ t}: of {total}")
                logger.info(f"Best Model: {cur_best_model}")
                logger.info(f"Best params: {db_batch.get_model_params_arr(cur_best_model)}")
                logger.info(f"One step diff: {diff[cur_best_model]}")
                logger.info(f"One step cost: {costs[cur_best_model]}")

        best_idx_in_batch = lb_history.get_best_model()
        batch_min_cost = lb_history.running_cost[best_idx_in_batch]

def grid_search_multi_step(N, total, recording, lb_history, db_batch):
    # Jump by N steps at a time
        for t in range(0, total - N, N):
            
            current_state = recording["state"][t]
            # Grab the next N controls
            U_seq = -recording["ctrl"][t : t+N] 
            
            # Predict N steps into the future blindly
            lb_history.predict_multi_step(
                current_state, U_seq
            )

            if recording["ok_time"][t+N]:
                true_future_state = recording["state"][t+N]
                diff, costs = lb_history.update_multi_step_error(
                    true_future_state
                )

                cur_best_model = lb_history.get_best_model()

                # logger.info every 5 jumps (100 timesteps total) or on the last valid jump
                logger.info(f"--- Timestep Window: {t:04d} -> {t+N:04d} (out of {total}) ---")
                if (t % (5 * N) == 0) or (t >= total - 2 * N):
                    logger.info(f"  Best Local Model : {cur_best_model}")
                    logger.info(f"  Best Parameters  : {np.round(db_batch.get_model_params_arr(cur_best_model), 4)}")
                    logger.info(f"  Multi-step Diff  : {np.round(diff[cur_best_model], 4)}")
                    logger.info(f"  Multi-step Cost  : {costs[cur_best_model]:.4f}\n")

def main():
    all_evaluated_params = []
    all_evaluated_costs = []
    global_best_cost = np.inf
    # 1. Setup Parameters (Same as before)
    params_car = F110()
    g = 9.81
    mass = params_car["mass"]
    params_range = {
        'Bf': [0.1, 30] ,# tire stiffness 
        'Br': [0.1, 30], # 30 like driving on glue, .1 like driving on ice
        'Cf': [1.0, 2.0], # curve shape param, multiplied by pi/2
        'Cr': [1.0, 2.0], # at 2, tire force drops to 0 (any further is negative), at 1 never reaches peak force
        'Df': [0.1, 2.0 * mass * g], # maximum lateral friction tires can provide
        'Dr': [0.1, 2.0 * mass * g], # pulling 2 gs
        'Ce': [0.1, 1.2], # should be maximum 1.1, i.e. how efficiently motor command turns a into F=ma
        'Cm': [0, 1.2/ 10] # maximum at Ce/vmax, 
    }

    
    fixed_params = {'Cro': 0.0, 'Cd': 0.0}
    discretization = 7
    
    param_series = {
        k: np.linspace(v[0], v[1], discretization + 1, dtype=np.float32)
        for k, v in params_range.items()
    }
    grid_iterator = itertools.product(*param_series)

    num_total_models = (discretization + 1) ** len(params_range)
    logger.info(f"Total models to evaluate: {num_total_models}")

    batch_size = 5000000 
    num_batches = int(np.ceil(num_total_models / batch_size))
    

    # 1. Setup Iterator (Memory Efficient)
    # We use .values() to ensure we pass the actual arrays to product
    grid_iterator = itertools.product(*param_series.values())

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'run_visualizer', 'hall.npz')

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

        if(multi_step):
            N = 20
            lb_history = history_no_record.LBHistory(
                current_batch_count, 1/40, np.array([1.0, 1.0, 0.1, 0.01, 0, 0]),
                6, rk6MultiStepFactory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
            )

            logger.info("\n" + "="*60)
            logger.info(f" PROCESSING BATCH {b+1}/{num_batches} | Models {start_idx} to {end_idx}")
            logger.info("="*60)
            grid_search_multi_step(N, total, recording, lb_history, db_batch)

        else:
            lb_history = history_no_record.LBHistory(
                current_batch_count, 1/40, np.array([1.0, 1.0, 0.1, 0.01, 0, 0]),
                6, rk6Factory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
            )

            logger.info("*********************************")
            logger.info(f"--- Processing Batch {b+1}/{num_batches} (Models {start_idx} to {end_idx}) ---")
            grid_search_one_step(total, recording, lb_history, db_batch)
    
        best_idx_in_batch = lb_history.get_best_model()
        batch_min_cost = lb_history.running_cost[best_idx_in_batch]

        logger.info("-"*60)
        logger.info(f" Batch {b+1} Summary")
        logger.info(f" Best Model Index  : {best_idx_in_batch}")
        logger.info(f" Batch Minimum Cost: {batch_min_cost:.4f}")
        
        if batch_min_cost < global_best_cost:
            global_best_cost = batch_min_cost
            global_best_params = batch_params[best_idx_in_batch]
            logger.info("\n  *** NEW GLOBAL BEST FOUND ***")
            logger.info(f"  Global Cost      : {global_best_cost:.4f}")
            logger.info(f"  Global Parameters: {np.round(global_best_params, 4)}")
        logger.info("-" * 60 + "\n")

if __name__ == '__main__':
    main()