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
from llampc.rollout.rk6 import rk6Factory
import jax

OL_reset_interval = 10

# Create a custom logger
logger = logging.getLogger("BatchOptimization")
logger.setLevel(logging.INFO)

def setup_logger(lla, multi_step):
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if lla:
            log_name = "grid_search_batch_LLA.log"
        elif multi_step:
            log_name = "grid_search_batch_multi_step.log"
        else:
            log_name = "grid_search_batch_one_step.log"
            
        file_handler = logging.FileHandler(log_name, mode="w")
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Trajectory Simulation Functions
# ---------------------------------------------------------------------------

def simulate_batched_trajectories(total, recording, all_best_params, params_car, fixed_params, full_open_loop=False):
    """
    Simulates all best batch models in parallel to record trajectories.
    all_best_params should be a numpy array of shape (num_best_models, 8).
    If full_open_loop is True, the simulation never anchors back to the true state after t=0.
    """
    num_models = len(all_best_params)
    logger.info(f"Building parallel simulation bank for {num_models} fixed models...")

    # Create a bank with a batch size equal to the number of best models
    best_db = dynamics.DBMPacejkaBank(
        params_car['lf'], params_car['lr'], params_car['mass'], params_car['Iz'],
        all_best_params[:, 0], all_best_params[:, 1], # Bf, Br
        all_best_params[:, 2], all_best_params[:, 3], # Cf, Cr
        all_best_params[:, 4], all_best_params[:, 5], # Df, Dr
        np.full(num_models, fixed_params['Cro']), 
        np.full(num_models, fixed_params['Cd']),
        all_best_params[:, 6], np.full(num_models, fixed_params['Cm']), # Ce, Cm
        0, 0, num_models 
    )

    lb_batched = history_no_record.LBHistory(
        num_models, 1/40, np.array([1.0, 1.0, 1.0, 0.01, 0, 0]),
        6, rk6Factory, best_db, dynamics.diffequation, buffer_size=[0, 0]
    )

    traj_open_loop = []
    traj_one_step = []
    current_ol_state = recording["state"][0]

    for t in range(total):
        if t % 100 == 0 or t == total - 1:
            logger.info(f"  Parallel Sim Timestep {t}/{total}")

        u_opt = recording["ctrl"][t]
        true_state = recording["state"][t]
        
        # ONE STEP
        lb_batched.predict_states(true_state, u_opt)
        pred_one_step = np.array(lb_batched.last_predicted_states) 
        traj_one_step.append(pred_one_step)

        # OPEN LOOP (Anchor periodically or never if full_open_loop is True)
        if not full_open_loop and t % OL_reset_interval == 0:
            current_ol_state = recording["state"][t]
        
        lb_batched.predict_states(current_ol_state, u_opt)
        pred_ol = np.array(lb_batched.last_predicted_states) 
        traj_open_loop.append(pred_ol)
    
        current_ol_state = np.array(pred_ol) 

    return np.array(traj_open_loop), np.array(traj_one_step)


def simulate_dynamic_rollout(total, recording, optimal_params_over_time, params_car, fixed_params, full_open_loop=False):
    """
    Rolls out a single trajectory where the Pacejka parameters change at EVERY timestep 
    based on the globally optimal sequence found across all batches.
    If full_open_loop is True, the simulation never anchors back to the true state after t=0.
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
        optimal_params_over_time[:, 6], np.full(num_steps, fixed_params['Cm']),
        0, 0, num_steps 
    )
    
    # Instantiate integrator
    integrator = rk6Factory(jax.device_put(dynamic_db.param_bank), dynamics.diffequation, 1/40)
    known_params = dynamic_db.get_known_params()
    
    traj_dynamic = []
    current_state = recording["state"][0]

    for t in range(num_steps):
        u_t = recording["ctrl"][t]
        
        # 1. Anchor periodically unless full open loop
        if not full_open_loop and t % OL_reset_interval == 0:
            current_state = recording["state"][t]
            
        batched_state = np.tile(current_state, (num_steps, 1))
        next_states = integrator(known_params, batched_state, u_t)
        
        # 2. Extract prediction mapped to the t-th parameter set
        current_state = np.array(next_states[t])
        traj_dynamic.append(current_state)

        # Do NOT unconditionally reset current_state to reality here!

    return np.array(traj_dynamic)

# ---------------------------------------------------------------------------
# Grid Search Functions
# ---------------------------------------------------------------------------

def grid_search_one_step(total, recording, lb_history, db_batch):
    for t in range(total):
        if t % 100 == 0 or t == total - 1:
            logger.info(f"--- Processing Timestep {t} of {total} ---")

        current_state = recording["state"][t]
        u_opt = recording["ctrl"][t]
        lb_history.predict_states(current_state, u_opt)

        if recording["ok_time"][t+1]:
            next_state = recording["state"][t+1]
            diff, costs = lb_history.update_lookback_error(next_state)

            if t % 100 == 0 or t == total - 1:
                cur_best_model = lb_history.get_best_model()
                logger.info(f"  Best Model: {cur_best_model}")
                logger.info(f"  Best params: {np.round(db_batch.get_model_params_arr(cur_best_model), 4)}")
                logger.info(f"  One step cost: {costs[cur_best_model]:.4f}")


def grid_search_multi_step(reset_interval, total, recording, lb_history, db_batch):
    for t in range(total):
        u_t = recording["ctrl"][t]
        
        # 1. Predict the next state
        if t % reset_interval == 0:
            current_state = recording["state"][t]
            lb_history.predict_states(current_state, u_t)
        else:
            lb_history.predict_states(lb_history.last_predicted_states, u_t)

        # 2. Evaluate and accrue error against reality
        if recording["ok_time"][t+1]:
            true_next_state = recording["state"][t+1]
            diff, costs = lb_history.update_lookback_error(true_next_state)

        # 3. Log periodically
        if (t + 1) % (5 * reset_interval) == 0 or t == total - 1:
            cur_best_model = lb_history.get_best_model()
            logger.info(f"--- Timestep: {t+1:04d} / {total} ---")
            logger.info(f"  Best Local Model : {cur_best_model}")
            logger.info(f"  Best Parameters  : {np.round(db_batch.get_model_params_arr(cur_best_model), 4)}")
            logger.info(f"  Accrued Cost     : {costs[cur_best_model]:.4f}\n")

def grid_search_multi_step_LLA(reset_interval, total, recording, lb_history, db_batch, cost_form):
    batch_best_models_over_time = []
    batch_best_costs_over_time = []
    
    # Track cumulative cost forever for the global best static model
    total_static_costs = np.zeros(lb_history.num_models)
    
    # --- DEBUG: Track cumulative cost components for the LLA chosen models ---
    cumulative_cost_components = np.zeros(6) 

    for t in range(total):
        u_t = recording["ctrl"][t]
        
        if t % reset_interval == 0:
            current_state = recording["state"][t]
            lb_history.predict_states(current_state, u_t)
        else:
            lb_history.predict_states(lb_history.last_predicted_states, u_t)

        if recording["ok_time"][t+1]:
            true_next_state = recording["state"][t+1]
            
            # Catch the instantaneous step cost for all models
            step_costs = lb_history.update_lookback_error(true_next_state)
            total_static_costs += step_costs

            # --- DEBUG: Calculate component cost for the best model ---
            cur_best_model = lb_history.get_best_model()
            best_pred = lb_history.last_predicted_states[cur_best_model]
            
            # Weighted squared error for each of the 6 states [x, y, yaw, vx, vy, yaw_rate]
            step_component_cost = ((best_pred - true_next_state) ** 2) * cost_form
            cumulative_cost_components += step_component_cost

        cur_best_model = lb_history.get_best_model()
        cur_cost = lb_history.running_cost[cur_best_model]
        
        batch_best_models_over_time.append(int(cur_best_model))
        batch_best_costs_over_time.append(float(cur_cost))

        # Log periodically
        if (t + 1) % (5 * reset_interval) == 0 or t == total - 1:
            logger.info(f"--- Timestep: {t+1:04d} / {total} ---")
            logger.info(f"  Best Local Model : {cur_best_model}")
            logger.info(f"  Accrued Cost     : {cur_cost:.4f}")
            
            # --- DEBUG: Log the breakdown ---
            total_component_sum = np.sum(cumulative_cost_components)
            if total_component_sum > 0:
                pcts = (cumulative_cost_components / total_component_sum) * 100
                logger.info(f"  Cost Breakdown   :")
                logger.info(f"    X: {pcts[0]:.1f}% | Y: {pcts[1]:.1f}% | Yaw: {pcts[2]:.1f}%")
                logger.info(f"    Vx: {pcts[3]:.1f}% | Vy: {pcts[4]:.1f}% | Yaw Rate: {pcts[5]:.1f}%\n")
            else:
                logger.info("  Cost Breakdown   : 0.0 for all components\n")
            
    return np.array(batch_best_models_over_time), np.array(batch_best_costs_over_time), total_static_costs

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main():
    # --- CONFIGURATION FLAGS ---
    lla = True          # Enable dynamic parameters over time
    multi_step = True   # Enable multi-step window if LLA is False
    full_open_loop_sim = False # <-- NEW FLAG: Set to True for full uninterrupted open-loop simulation
    
    setup_logger(lla, multi_step)

    params_car = F110()
    g = 9.81
    mass = params_car["mass"]
    
    params_range = {
        'Bf': [0.1, 15],
        'Br': [0.1, 15],
        'Cf': [1.1, 1.9],
        'Cr': [1.1, 1.9],
        'Df': [0.1, 2.0 * mass * g],
        'Dr': [0.1, 2.0 * mass * g],
        'Ce': [0.0, 20.0],
        
    }

    fixed_params = {'Cro': 0.0, 'Cd': 0.0, 'Cm': 0}
    discretization = 5
    
    param_series = {k: np.linspace(v[0], v[1], discretization + 1, dtype=np.float32) for k, v in params_range.items()}
    num_total_models = (discretization + 1) ** len(params_range)
    logger.info(f"Total models to evaluate: {num_total_models}")

    batch_size = 100000
    num_batches = int(np.ceil(num_total_models / batch_size))
    grid_iterator = itertools.product(*param_series.values())

    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, 'run_visualizer', 'sysid_trimmed.npz')

    recording = np.load(filepath, allow_pickle=True)
    total = len(recording["time"]) - 1  # -1 to protect ok_time[t+1]

    # Containers for tracking
    global_best_cost = np.inf
    global_best_params = None
    batch_best_trajectories = []
    
    # Store dynamic data across all batches specifically for LLA
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
            batch_params[:, 6], np.full(current_batch_count, fixed_params['Cm']), 
            0, 0, current_batch_count
        )

        logger.info("\n" + "="*60)
        logger.info(f" PROCESSING BATCH {b+1}/{num_batches} | Models {start_idx} to {end_idx}")
        logger.info("="*60)

        # ROUTING LOGIC based on flags
        cost_form = np.array([10.0, 10.0, 10, 10.0, 0.0, 0.0])
        if lla:
            N = OL_reset_interval
            lb_history = history.LBHistory(
                current_batch_count, 80, 1/40, cost_form,
                6, rk6Factory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
            )
            
            batch_models_over_time, batch_costs_over_time, total_static_costs = grid_search_multi_step_LLA(N, total, recording, lb_history, db_batch, cost_form)
            
            all_batches_costs.append(batch_costs_over_time)
            all_batches_models.append(batch_models_over_time)
            all_batches_params.append(batch_params)

            best_idx_in_batch = np.argmin(total_static_costs)
            batch_min_cost = total_static_costs[best_idx_in_batch]
            
        else:
            lb_history = history_no_record.LBHistory(
                current_batch_count, 1/40, cost_form,
                6, rk6Factory, db_batch, dynamics.diffequation, buffer_size=[0, 0]
            )
            if multi_step:
                N = OL_reset_interval
                grid_search_multi_step(N, total, recording, lb_history, db_batch)
            else:
                grid_search_one_step(total, recording, lb_history, db_batch)
        
            # Get the standard static winner for the batch
            best_idx_in_batch = lb_history.get_best_model()
            batch_min_cost = lb_history.running_cost[best_idx_in_batch]
            
        best_params_in_batch = np.array(db_batch.get_model_params_arr(best_idx_in_batch))
        
        logger.info("-" * 60)
        logger.info(f" Batch {b+1} Summary")
        logger.info(f" Best Static Model Index : {best_idx_in_batch}")
        logger.info(f" Batch Minimum Cost      : {batch_min_cost:.4f}")

        # Keep track of the absolute global static best
        if batch_min_cost < global_best_cost:
            global_best_cost = batch_min_cost
            global_best_params = best_params_in_batch
            logger.info("\n  *** NEW GLOBAL STATIC BEST FOUND ***")
            logger.info(f"  Global Cost      : {global_best_cost:.4f}")
            logger.info(f"  Global Parameters: {np.round(global_best_params, 4)}")
        
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

    logger.info("\nAll batches complete!")
    
    # --------------------------------------------------------------------------------
    # LLA POST-PROCESSING: Global Dynamic Timestep Stitching
    # --------------------------------------------------------------------------------
    global_dynamic_trajectory = None
    global_optimal_params_over_time = None
    global_optimal_costs_over_time = None

    if lla:
        logger.info("Comparing costs across all batches to build global optimal parameter sequence...")
        
        global_optimal_params_over_time = np.zeros((total, 7))
        global_optimal_costs_over_time = np.zeros(total)
        all_batches_costs_arr = np.array(all_batches_costs) 
        
        for t in range(total):
            best_batch_idx = np.argmin(all_batches_costs_arr[:, t])
            best_model_in_batch = all_batches_models[best_batch_idx][t]
            best_params = all_batches_params[best_batch_idx][best_model_in_batch]
            
            global_optimal_params_over_time[t] = best_params
            global_optimal_costs_over_time[t] = all_batches_costs_arr[best_batch_idx, t]
            
        global_dynamic_trajectory = simulate_dynamic_rollout(
            total, recording, global_optimal_params_over_time, params_car, fixed_params, full_open_loop=full_open_loop_sim
        )

    # --------------------------------------------------------------------------------
    # PARALLEL TRAJECTORY SIMULATIONS (For standard batch bests)
    # --------------------------------------------------------------------------------
    all_best_params_stacked = np.array([item["params"] for item in batch_best_trajectories])
    
    all_traj_open_loop, all_traj_one_step = simulate_batched_trajectories(
        total, recording, all_best_params_stacked, params_car, fixed_params, full_open_loop=full_open_loop_sim
    )

    for i, item in enumerate(batch_best_trajectories):
        item["traj_open_loop"] = all_traj_open_loop[:, i, :]
        item["traj_one_step"] = all_traj_one_step[:, i, :]

    # --------------------------------------------------------------------------------
    # DATA SAVING
    # --------------------------------------------------------------------------------
    batches = [d["batch"] for d in batch_best_trajectories]
    all_params = [d["params"] for d in batch_best_trajectories]
    all_traj_open_loop_list = [d["traj_open_loop"] for d in batch_best_trajectories]
    all_traj_one_step_list = [d["traj_one_step"] for d in batch_best_trajectories]
    
    save_path = os.path.join("llampc", "utils", "run_visualizer", "sysid_ol2.npz")
    save_data = {
        "batch": batches,
        "params": all_params,
        "traj_open_loop": all_traj_open_loop_list,
        "traj_one_step": all_traj_one_step_list,
        "global_best_static_params": global_best_params,
        "global_best_static_cost": global_best_cost
    }
    
    if lla:
        save_data["lla_optimal_params"] = global_optimal_params_over_time
        save_data["lla_optimal_costs"] = global_optimal_costs_over_time
        save_data["lla_dynamic_trajectory"] = global_dynamic_trajectory
        
    np.savez_compressed(save_path, **save_data)
    logger.info(f"\nSuccessfully saved all trajectories and optimizations to {save_path}")

if __name__ == '__main__':
    main()