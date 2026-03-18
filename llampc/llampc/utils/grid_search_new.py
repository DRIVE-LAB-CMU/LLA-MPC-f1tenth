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
import jax

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

def simulate_batched_trajectories(total, recording, all_best_params, params_car, fixed_params):
    """
    Simulates all best batch models in parallel to record trajectories.
    all_best_params should be a numpy array of shape (num_best_models, 8).
    """
    
    num_models = len(all_best_params)
    logger.info(f"Building parallel simulation bank for {num_models} models...")

    logger.info(f"{all_best_params}")

    # 1. Create a bank with a batch size equal to the number of best models
    best_db = dynamics.DBMPacejkaBank(
        params_car['lf'], params_car['lr'], params_car['mass'], params_car['Iz'],
        all_best_params[:, 0], all_best_params[:, 1], # Bf, Br
        all_best_params[:, 2], all_best_params[:, 3], # Cf, Cr
        all_best_params[:, 4], all_best_params[:, 5], # Df, Dr
        np.full(num_models, fixed_params['Cro']), 
        np.full(num_models, fixed_params['Cd']),
        all_best_params[:, 8], all_best_params[:, 9], # Ce, Cm
        0, 0, num_models # batch_size = num_models
    )

    # 2. Instantiate the lookback history for this specific batch
    lb_batched = history_no_record.LBHistory(
        num_models, 1/40, np.array([1.0, 1.0, 0.1, 0.01, 0, 0]),
        6, rk6Factory, best_db, dynamics.diffequation, buffer_size=[0, 0]
    )

    traj_open_loop = []
    traj_one_step = []

    # Initialize the open-loop state. 
    # Shape starts as (6,) but will naturally broadcast to (num_models, 6) after step 1.
    current_ol_state = recording["state"][0]

    for t in range(total):
        if t % 20 == 0 or t == total - 1:
            logger.info(f"  Parallel Sim Timestep {t}/{total}")

        u_opt = -recording["ctrl"][t]
        true_state = recording["state"][t]

        
        
        lb_batched.predict_states(true_state, u_opt)
        pred_one_step = np.array(lb_batched.last_predicted_states) # Shape: (num_models, 6)
        traj_one_step.append(pred_one_step)
        # print(t)
        # print(pred_one_step)

        # print(u_opt)
        # print("hello")
        # print(len(current_ol_state.shape))
        # print(best_db.param_bank[1])
        # if len((current_ol_state.shape) ) == 1:
        #     derivs = dynamics.diffequation(best_db.param_bank[1], best_db.get_known_params(), current_ol_state, u_opt)
        #     print(f"Derivatives (dx/dt): {derivs}")
        # else:
        #     derivs = dynamics.diffequation(best_db.param_bank[1], best_db.get_known_params(), current_ol_state[0], u_opt)
        #     print(f"Derivatives (dx/dt): {derivs}")

        if t % 20 == 0:
            current_ol_state = recording["state"][t]
        
        lb_batched.predict_states(current_ol_state, u_opt)
        pred_ol = np.array(lb_batched.last_predicted_states) # Shape: (num_models, 6)
        traj_open_loop.append(pred_ol)

        # print(pred_ol)
    
        current_ol_state = np.array(pred_ol) 

    return np.array(traj_open_loop), np.array(traj_one_step)

def grid_search_one_step(total, recording, lb_history, db_batch):

    for t in range(total):

        if(t %100== 0 or t == total-1):
            logger.info(f"--- Processing Timestep {t} of {total} ---")

        current_state =  recording["state"][t]
        u_opt = -recording["ctrl"][t]
        logger.info(type(current_state))
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

def grid_search_multi_step(reset_interval, total, recording, lb_history, db_batch):
    # e.g., reset_interval = 20
    for t in range(total - 1):
        
        u_t = -recording["ctrl"][t]
        
        # 1. Predict the next state
        if t % reset_interval == 0:
            # RESET: Anchor the start of this window to the true state
            current_state = recording["state"][t]
            lb_history.predict_states(current_state, u_t)
            # pass
        else:
            # CONTINUE: Predict from each model's individual last predicted state
            lb_history.predict_states(lb_history.last_predicted_states, u_t)

        # 2. Evaluate and accrue error against reality for THIS timestep
        if recording["ok_time"][t+1]:
            true_next_state = recording["state"][t+1]
            # This automatically adds the cost to self.running_cost
            diff, costs = lb_history.update_lookback_error(true_next_state)

        # 3. Log periodically (e.g., every 5 reset intervals)
        if (t + 1) % (5 * reset_interval) == 0 or (t == total - 2):
            cur_best_model = lb_history.get_best_model()
            logger.info(f"--- Timestep: {t+1:04d} / {total} ---")
            logger.info(f"  Best Local Model : {cur_best_model}")
            logger.info(f"  Best Parameters  : {np.round(db_batch.get_model_params_arr(cur_best_model), 4)}")
            logger.info(f"  Accrued Cost     : {costs[cur_best_model]:.4f}\n")

def main():
    all_evaluated_params = []
    all_evaluated_costs = []
    global_best_cost = np.inf
    # 1. Setup Parameters (Same as before)
    params_car = F110()
    g = 9.81
    mass = params_car["mass"]
    params_range = {
        'Bf': [0.1, 40] ,# tire stiffness 
        'Br': [0.1, 40], # 30 like driving on glue, .1 like driving on ice
        'Cf': [1.0, 2], # curve shape param, multiplied by pi/2
        'Cr': [1.0, 2], # at 2, tire force drops to 0 (any further is negative), at 1 never reaches peak force
        'Df': [0.1, 2.0 * mass * g], # maximum lateral friction tires can provide
        'Dr': [0.1, 2.0 * mass * g], # pulling 2 gs
        'Ce': [0.0, 0.1], # should be maximum 1.1, i.e. how efficiently motor command turns a into F=ma
        'Cm': [0, 1.2/10] # maximum at Ce/vmax, 
    }

    
    fixed_params = {'Cro': 0.0, 'Cd': 0.0}
    discretization = 4
    
    param_series = {
        k: np.linspace(v[0], v[1], discretization + 1, dtype=np.float32)
        for k, v in params_range.items()
    }
    grid_iterator = itertools.product(*param_series)

    num_total_models = (discretization + 1) ** len(params_range)
    logger.info(f"Total models to evaluate: {num_total_models}")

    batch_size = 50000
    num_batches = int(np.ceil(num_total_models / batch_size))
    

    # 1. Setup Iterator (Memory Efficient)
    # We use .values() to ensure we pass the actual arrays to product
    grid_iterator = itertools.product(*param_series.values())

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'run_visualizer', 'rec_circle.npz')

    recording = np.load(filepath, allow_pickle=True)
    total = len(recording["time"])-1

    global_best_cost= np.inf
    batch_best_trajectories = []

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
                6, rk6Factory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
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
            global_best_params = db_batch.get_model_params_arr(best_idx_in_batch)
            logger.info("\n  *** NEW GLOBAL BEST FOUND ***")
            logger.info(f"  Global Cost      : {global_best_cost:.4f}")
            logger.info(f"  Global Parameters: {np.round(global_best_params, 4)}")
        logger.info("-" * 60 + "\n")
        logger.info(f"  Global Cost      : {global_best_cost:.4f}")
        logger.info(f"  Global Parameters: {np.round(global_best_params, 4)}")


        # best_params_in_batch = np.array(db_batch.get_model_params_arr(best_idx_in_batch))
        best_params_in_batch = np.array(db_batch.get_model_params_arr(int(np.random.randint(current_batch_count))))

        batch_best_trajectories.append({
            "batch": b + 1,
            "params": best_params_in_batch,
            "traj_open_loop": None, # Will fill this in later
            "traj_one_step": None   # Will fill this in later
        })

        del lb_history
        del db_batch
        import gc
        gc.collect()

    logger.info("All batches complete! Simulating trajectories for all best models simultaneously...")

    # Extract all parameters into a single (num_batches, 8) array
    all_best_params_stacked = np.array([item["params"] for item in batch_best_trajectories])

    # Run the parallelized simulation once
    all_traj_open_loop, all_traj_one_step = simulate_batched_trajectories(
        total, recording, all_best_params_stacked, params_car, fixed_params
    )

    # all_traj_open_loop is shape: (total_timesteps, num_batches, state_dim)
    # If you want to put them back into your dictionary format:
    for i, item in enumerate(batch_best_trajectories):
        # Slice out the i-th model's entire trajectory across all timesteps
        item["traj_open_loop"] = all_traj_open_loop[:, i, :]
        item["traj_one_step"] = all_traj_one_step[:, i, :]

    logger.info("Parallel trajectory simulations complete. Ready to save!")

    for i, item in enumerate(batch_best_trajectories):
        # Slice out the i-th model's entire trajectory across all timesteps
        item["traj_open_loop"] = all_traj_open_loop[:, i, :]
        item["traj_one_step"] = all_traj_one_step[:, i, :]

    batches = [d["batch"] for d in batch_best_trajectories]
    all_params = [d["params"] for d in batch_best_trajectories]
    all_traj_open_loop = [d["traj_open_loop"] for d in batch_best_trajectories]
    all_traj_one_step = [d["traj_one_step"] for d in batch_best_trajectories]

    # Save to a single compressed .npz file
    save_path = "all_batch_best_trajectories.npz"
    np.savez_compressed(
        save_path,
        batch=batches,
        params=all_params,
        traj_open_loop=all_traj_open_loop,
        traj_one_step=all_traj_one_step,
        global_best_params=global_best_params, # Good idea to save the overall winner too!
        global_best_cost=global_best_cost
    )
    logger.info(f"Successfully saved all trajectories to {save_path}")

if __name__ == '__main__':
    main()