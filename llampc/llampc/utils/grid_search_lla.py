import numpy as np
import os
import multiprocessing
import logging
import sys
import itertools
import gc

num_cores = str(multiprocessing.cpu_count())
# Force threading backends to use all cores
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["OPENBLAS_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores

from ..rollout import history_no_record, history
from ..rollout import dynamic as dynamics
from llampc.params import F110
from llampc.rollout.rk6 import rk6Factory, rk6MultiStepFactory
import jax
import jax.numpy as jnp

multi_step = True

# Create a custom logger
logger = logging.getLogger("BatchOptimization")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler("grid_search_batch_multi_step.log" if multi_step else "grid_search_batch_one_step.log", mode="w")
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def simulate_batched_trajectories(total, recording, all_best_params, params_car, fixed_params):
    """
    Simulates all best batch models in parallel to record trajectories.
    all_best_params should be a numpy array of shape (num_best_models, 8).
    """
    num_models = len(all_best_params)
    logger.info(f"Building parallel simulation bank for {num_models} fixed models...")

    best_db = dynamics.DBMPacejkaBank(
        params_car['lf'], params_car['lr'], params_car['mass'], params_car['Iz'],
        all_best_params[:, 0], all_best_params[:, 1], # Bf, Br
        all_best_params[:, 2], all_best_params[:, 3], # Cf, Cr
        all_best_params[:, 4], all_best_params[:, 5], # Df, Dr
        np.full(num_models, fixed_params['Cro']), 
        np.full(num_models, fixed_params['Cd']),
        all_best_params[:, 6], all_best_params[:, 7], # Ce, Cm
        0, 0, num_models 
    )

    lb_batched = history_no_record.LBHistory(
        num_models, 1/40, np.array([1.0, 1.0, 1.0, 0.01, 0, 0]),
        6, rk6Factory, best_db, dynamics.diffequation, buffer_size=[0, 0]
    )

    traj_open_loop = []
    traj_one_step = []
    current_ol_state = recording["state"][0]

    for t in range(total - 1):
        if t % 100 == 0 or t == total - 2:
            logger.info(f"  Parallel Sim Timestep {t}/{total-1}")

        u_opt = -recording["ctrl"][t]
        true_state = recording["state"][t]
        
        # ONE STEP
        lb_batched.predict_states(true_state, u_opt)
        pred_one_step = np.array(lb_batched.last_predicted_states) 
        traj_one_step.append(pred_one_step)

        # OPEN LOOP (Anchor periodically)
        if t % 20 == 0:
            current_ol_state = recording["state"][t]
        
        lb_batched.predict_states(current_ol_state, u_opt)
        pred_ol = np.array(lb_batched.last_predicted_states) 
        traj_open_loop.append(pred_ol)
    
        current_ol_state = np.array(pred_ol) 

    return np.array(traj_open_loop), np.array(traj_one_step)

def simulate_dynamic_rollout(total, recording, optimal_params_over_time, params_car, fixed_params):
    """
    Rolls out a single trajectory where the Pacejka parameters change at EVERY timestep 
    based on the globally optimal sequence found across all batches.
    """
    logger.info("Simulating final global dynamic rollout (parameters changing per timestep)...")
    num_steps = len(optimal_params_over_time)
    
    # Create a giant bank containing the exact parameter sequence
    dynamic_db = dynamics.DBMPacejkaBank(
        params_car['lf'], params_car['lr'], params_car['mass'], params_car['Iz'],
        optimal_params_over_time[:, 0], optimal_params_over_time[:, 1],
        optimal_params_over_time[:, 2], optimal_params_over_time[:, 3],
        optimal_params_over_time[:, 4], optimal_params_over_time[:, 5],
        np.full(num_steps, fixed_params['Cro']), 
        np.full(num_steps, fixed_params['Cd']),
        optimal_params_over_time[:, 6], optimal_params_over_time[:, 7],
        0, 0, num_steps 
    )
    
    # Instantiate integrator
    integrator = rk6Factory(jax.device_put(dynamic_db.param_bank), dynamics.diffequation, 1/40)
    known_params = dynamic_db.get_known_params()
    
    traj_dynamic = [recording["state"][0]]
    current_state = recording["state"][0]
    
    for t in range(num_steps):
        u_t = -recording["ctrl"][t]
        
        # We broadcast the current state and control to match the batch size of the integrator,
        # but we specifically extract the t-th index since it contains the parameters for time t.
        batched_state = np.tile(current_state, (num_steps, 1))
        batched_ctrl = np.tile(u_t, (2, 1)).T # Assuming control is size 2
        
        # Step physics
        next_states = integrator(known_params, batched_state, batched_ctrl)
        
        # Extract purely the prediction mapped to the t-th parameter set
        current_state = np.array(next_states[t])
        traj_dynamic.append(current_state)
        
    return np.array(traj_dynamic)


def grid_search_multi_step_LLA(reset_interval, total, recording, lb_history, db_batch):
    batch_best_models_over_time = []
    batch_best_costs_over_time = []

    for t in range(total - 1):
        u_t = -recording["ctrl"][t]
        
        # 1. Predict Next State
        if t % reset_interval == 0:
            current_state = recording["state"][t]
            lb_history.predict_states(current_state, u_t)
        else:
            lb_history.predict_states(lb_history.last_predicted_states, u_t)

        # 2. Evaluate and accrue error
        if recording["ok_time"][t+1]:
            true_next_state = recording["state"][t+1]
            # Assuming update_lookback_error handles rolling cost appropriately
            lb_history.update_lookback_error(true_next_state)

        # 3. Log current best for this timestep
        cur_best_model = lb_history.get_best_model()
        cur_cost = lb_history.running_cost[cur_best_model]
        
        batch_best_models_over_time.append(int(cur_best_model))
        batch_best_costs_over_time.append(float(cur_cost))

        # 4. Console logging
        if (t + 1) % (5 * reset_interval) == 0 or (t == total - 2):
            logger.info(f"--- Timestep: {t+1:04d} / {total} ---")
            logger.info(f"  Best Local Model : {cur_best_model}")
            logger.info(f"  Accrued Cost     : {cur_cost:.4f}\n")
            
    return np.array(batch_best_models_over_time), np.array(batch_best_costs_over_time)

def main():
    params_car = F110()
    g = 9.81
    lla = True
    mass = params_car["mass"]
    params_range = {
        'Bf': [0.1, 40],
        'Br': [0.1, 40],
        'Cf': [1.0, 2],
        'Cr': [1.0, 2],
        'Df': [0.1, 2.0 * mass * g],
        'Dr': [0.1, 2.0 * mass * g],
        'Ce': [0.0, 0.1],
        'Cm': [0, 1.2/10]
    }

    fixed_params = {'Cro': 0.0, 'Cd': 0.0}
    discretization = 4
    
    param_series = {k: np.linspace(v[0], v[1], discretization + 1, dtype=np.float32) for k, v in params_range.items()}
    num_total_models = (discretization + 1) ** len(params_range)
    logger.info(f"Total models to evaluate: {num_total_models}")

    batch_size = 50000
    num_batches = int(np.ceil(num_total_models / batch_size))
    grid_iterator = itertools.product(*param_series.values())

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'run_visualizer', 'nshtrack.npz')

    recording = np.load(filepath, allow_pickle=True)
    total = len(recording["time"])

    batch_best_trajectories = []
    
    # Store dynamic data across all batches
    all_batches_costs = []
    all_batches_models = []
    all_batches_params = []

    for b in range(num_batches):
        batch_params = np.array(list(itertools.islice(grid_iterator, batch_size)), dtype=np.float32)
        if len(batch_params) == 0:
            break
            
        current_batch_count = len(batch_params)
        start_idx = b * batch_size
        end_idx = start_idx + current_batch_count
        
        db_batch = dynamics.DBMPacejkaBank(
            params_car['lf'], params_car['lr'], params_car['mass'], params_car['Iz'],
            batch_params[:, 0], batch_params[:, 1], batch_params[:, 2], batch_params[:, 3],
            batch_params[:, 4], batch_params[:, 5], 
            np.full(current_batch_count, fixed_params['Cro']), np.full(current_batch_count, fixed_params['Cd']),
            batch_params[:, 6], batch_params[:, 7], 
            0, 0, current_batch_count
        )

        logger.info("\n" + "="*60)
        logger.info(f" PROCESSING BATCH {b+1}/{num_batches} | Models {start_idx} to {end_idx}")
        logger.info("="*60)

        if lla:
            N = 20
            lb_history = history.LBHistory(
                current_batch_count, 20, 1/40, np.array([1.0, 1.0, 1.0, 0.01, 0, 0]),
                6, rk6Factory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
            )
            batch_models_over_time, batch_costs_over_time = grid_search_multi_step_LLA(N, total, recording, lb_history, db_batch)
        
            all_batches_costs.append(batch_costs_over_time)
            all_batches_models.append(batch_models_over_time)
            all_batches_params.append(batch_params)
            
            # For backward compatibility, record the overall batch winner based on final timestep
            best_idx_in_batch = lb_history.get_best_model()
            batch_min_cost = lb_history.running_cost[best_idx_in_batch]
            best_params_in_batch = np.array(db_batch.get_model_params_arr(best_idx_in_batch))
            
            batch_best_trajectories.append({
                "batch": b + 1,
                "params": best_params_in_batch,
                "traj_open_loop": None,
                "traj_one_step": None,
                "cost": batch_min_cost
            })

        del lb_history
        del db_batch
        gc.collect()

    logger.info("All batches complete!")
    
    # --------------------------------------------------------------------------------
    # GLOBAL TIMESTEP COMPARISON & DYNAMIC LLA ROLLOUT
    # --------------------------------------------------------------------------------
    
    logger.info("Comparing costs across all batches to build global optimal parameter sequence...")
    num_timesteps = total - 1
    global_optimal_params_over_time = np.zeros((num_timesteps, 8))
    global_optimal_costs_over_time = np.zeros(num_timesteps)
    
    # Convert list of arrays to a 2D array: shape (num_batches, num_timesteps)
    all_batches_costs_arr = np.array(all_batches_costs) 
    
    for t in range(num_timesteps):
        # Find which batch had the lowest cost at this specific timestep
        best_batch_idx = np.argmin(all_batches_costs_arr[:, t])
        
        # Retrieve the model index from that batch that yielded the lowest cost
        best_model_in_batch = all_batches_models[best_batch_idx][t]
        
        # Extract the actual physical parameters
        best_params = all_batches_params[best_batch_idx][best_model_in_batch]
        
        global_optimal_params_over_time[t] = best_params
        global_optimal_costs_over_time[t] = all_batches_costs_arr[best_batch_idx, t]
        
    # Run the rollout using the dynamically changing parameters
    global_dynamic_trajectory = simulate_dynamic_rollout(
        num_timesteps, recording, global_optimal_params_over_time, params_car, fixed_params
    )

    all_best_params_stacked = np.array([item["params"] for item in batch_best_trajectories])
    all_traj_open_loop, all_traj_one_step = simulate_batched_trajectories(
        total, recording, all_best_params_stacked, params_car, fixed_params
    )

    for i, item in enumerate(batch_best_trajectories):
        item["traj_open_loop"] = all_traj_open_loop[:, i, :]
        item["traj_one_step"] = all_traj_one_step[:, i, :]

    # Organize data for save
    batches = [d["batch"] for d in batch_best_trajectories]
    all_params = [d["params"] for d in batch_best_trajectories]
    all_traj_open_loop = [d["traj_open_loop"] for d in batch_best_trajectories]
    all_traj_one_step = [d["traj_one_step"] for d in batch_best_trajectories]
    overall_best_batch = np.argmin([d["cost"] for d in batch_best_trajectories])

    save_path = "all_batch_best_trajectories.npz"
    save_data = {
        "batch": batches,
        "params": all_params,
        "traj_open_loop": all_traj_open_loop,
        "traj_one_step": all_traj_one_step,
        "global_best_static_params": all_params[overall_best_batch],
        "global_best_static_cost": batch_best_trajectories[overall_best_batch]["cost"]
    }
    
    if lla:
        save_data["lla_optimal_params"] = global_optimal_params_over_time
        save_data["lla_optimal_costs"] = global_optimal_costs_over_time
        save_data["lla_dynamic_trajectory"] = global_dynamic_trajectory
        
    np.savez_compressed(save_path, **save_data)
    logger.info(f"Successfully saved all trajectories to {save_path}")

if __name__ == '__main__':
    main()