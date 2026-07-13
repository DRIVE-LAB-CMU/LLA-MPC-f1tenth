"""Rollout backend for the state visualizer.

Contains the optional llampc-backed dynamics, the bank-order helpers, and the
four trajectory/error simulators:

    simulate_general_models   - N fixed-param models, open-loop + one-step
    simulate_lla_rollout      - single dynamic (per-timestep params) trajectory
    simulate_lla_one_step     - one-step predictions for the LLA model
    simulate_general_m_step   - M-step lookahead error for fixed models
    simulate_lla_m_step       - M-step lookahead error for the LLA model

If the llampc backend is unavailable, _ROLLOUT_OK is False and the visualizer
falls back to precomputed npz arrays (or the recorded run only).
"""

import os
import multiprocessing

# Force threading backends to use all cores (must be set before jax import).
_num_cores = str(multiprocessing.cpu_count())
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
os.environ.setdefault("OMP_NUM_THREADS", _num_cores)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _num_cores)
os.environ.setdefault("MKL_NUM_THREADS", _num_cores)

import numpy as np

# ---------------------------------------------------------------------------
# Rollout backend (optional). Absolute imports so this file still runs as a
# plain script. If unavailable, the visualizer falls back to loading any
# precomputed rollout arrays already in the npz, or just the recorded run.
# ---------------------------------------------------------------------------
try:
    from llampc.params import F110
    from llampc.rollout import history_no_record
    from llampc.rollout import dynamic_fiala as dynamics
    from llampc.rollout.rk6 import rk6Factory
    import jax
    _ROLLOUT_OK = True
    _ROLLOUT_ERR = None
except Exception as e:  # noqa: BLE001
    _ROLLOUT_OK = False
    _ROLLOUT_ERR = e


# ===========================================================================
# Rollout configuration / helpers
# ===========================================================================
# Order the DBMPacejkaBank constructor expects:
#   lf, lr, mass, Iz, Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm, roll, pitch, n
BANK_ORDER = ['Cf', 'Cr', 'muf', 'mur', 'Cro']

# Order in which the logged `params` field is laid out by get_model_params_arr().
# >>> VERIFY against your dynamics.DBMPacejkaBank.get_model_params_arr(). <<<
DEFAULT_LOG_ORDER = ['Cf', 'Cr', 'muf', 'mur', 'Cro']

# State layout: [x, y, theta, dx, dy, omega]. Short labels used for the
# per-component cost checkboxes; index 2 (theta) is angle-wrapped in cost calc.
COST_DIM_LABELS = ['x', 'y', 'theta', 'vx', 'vy', 'omega']


def dict_to_bank_vec(d):
    """Build a single bank-order param vector from a {name: value} dict."""
    return np.array([d.get(k, 0.0) for k in BANK_ORDER], dtype=np.float32)


def remap_to_bank_order(params, source_order):
    """Reorder an (N, P) param array from `source_order` into BANK_ORDER.

    Any BANK_ORDER key missing from source_order is filled with 0.0.
    """
    params = np.asarray(params, dtype=np.float32)
    if params.ndim == 1:
        params = params[None, :]

    out = np.zeros((params.shape[0], len(BANK_ORDER)), dtype=np.float32)
    for i, key in enumerate(BANK_ORDER):
        if key in source_order:
            col = source_order.index(key)
            if col < params.shape[1]:
                out[:, i] = params[:, col]
    return out


def build_bank(params_bank, params_car, num_models):
    """params_bank: (num_models, 10) in BANK_ORDER."""
    p = params_bank
    return dynamics.DBMFialaBank(
        params_car['lf'], params_car['lr'], 
        params_car['mass'], params_car['Iz'],
        params_car['rw'],
        p[:, 0], p[:, 1],   # Cf, Cr,
        p[:, 2], p[:, 3],   # muf, mur
        p[:, 4], # Cro
        num_models     # roll, pitch, n
    )


def simulate_general_models(total, recording, general_params_bank, params_car,
                            dt, ol_reset_interval, cost_weights,
                            full_open_loop=False):
    """Simulate N fixed models in parallel; each keeps its OWN params all run.

    general_params_bank: (num_models, 10) in BANK_ORDER.
    Returns (traj_open_loop, traj_one_step), each (total, num_models, state_dim).
    """
    num_models = len(general_params_bank)
    print(f"[rollout] general models: {num_models} fixed banks, {total} steps")

    bank = build_bank(general_params_bank, params_car, num_models)
    lb = history_no_record.LBHistory(
        num_models, dt, np.asarray(cost_weights),
        6, rk6Factory, bank, dynamics.diffequation, buffer_size=[0, 0]
    )

    state0 = recording["state"][0]

    # ONE STEP: store the 1-step prediction of the state AT time t (made from
    # truth[t-1]); index 0 is truth, so it sits on the recorded path. This keeps
    # one_step[t] comparable to truth[t] at the same index (no off-by-one).
    one_step = [np.tile(state0, (num_models, 1))]
    for t in range(total):
        lb.predict_states(recording["state"][t], recording["ctrl"][t])
        one_step.append(np.array(lb.last_predicted_states))
    traj_one_step = np.array(one_step[:total])

    # OPEN LOOP: store the model state AT time t. At each reset the state is
    # snapped to truth[t] and that anchor is what gets stored, so the reset point
    # lands exactly on the recorded trajectory; we then integrate forward.
    open_loop = []
    current = np.tile(state0, (num_models, 1))
    for t in range(total):
        if not full_open_loop and t % ol_reset_interval == 0:
            current = np.tile(recording["state"][t], (num_models, 1))   # anchor = truth[t]
        open_loop.append(np.array(current))                             # state AT time t
        lb.predict_states(current, recording["ctrl"][t])                # advance t -> t+1
        current = np.array(lb.last_predicted_states)
    traj_open_loop = np.array(open_loop)

    return traj_open_loop, traj_one_step


def simulate_lla_rollout(total, recording, lla_params_over_time, params_car,
                         dt, ol_reset_interval, full_open_loop=False,
                         known_params_over_time=None):
    """Single trajectory whose Pacejka params change EVERY timestep, fed by the
    optimal-per-timestep sequence taken from the LLA log.

    lla_params_over_time: (total, 10) in BANK_ORDER. Returns (total, state_dim).
    traj[t] is the state AT time t: reset to truth[t] every ol_reset_interval
    (so the reset point sits exactly on the recorded trajectory), then
    integrated forward using the t-th parameter set.

    known_params_over_time: optional (total, ...) array of the EXACT
    known_params logged by the real-time node at each control tick (e.g. via
    dynamics_bank.update_known_params(vx) -> get_known_params()). If given,
    step t uses known_params_over_time[t] instead of a single static
    bank.get_known_params() computed once for the whole rollout - this matters
    whenever known_params is velocity- (or otherwise state-) dependent, since
    the real node's value tracked the actual current_state[3] at every tick
    rather than one fixed value for the entire replay.
    """
    print(f"[rollout] LLA dynamic: {total} steps (params change per timestep)")
    num_steps = len(lla_params_over_time)

    bank = build_bank(lla_params_over_time, params_car, num_steps)
    integrator = rk6Factory(jax.device_put(bank.param_bank), dynamics.diffequation, dt)
    static_known_params = bank.get_known_params()

    logged_kp = None
    if known_params_over_time is not None:
        logged_kp = np.asarray(known_params_over_time)
        print(f"[rollout] LLA dynamic: using logged known_params {logged_kp.shape} "
              f"in place of recomputed bank.get_known_params()")

    traj_dynamic = []
    current_state = recording["state"][0]

    for t in range(num_steps):
        u_t = recording["ctrl"][t]

        if not full_open_loop and t % ol_reset_interval == 0:
            current_state = recording["state"][t]          # anchor = truth[t]

        traj_dynamic.append(np.array(current_state))        # store the state AT time t

        kp_t = (logged_kp[t] if logged_kp is not None and t < len(logged_kp)
                else static_known_params)

        # Advance t -> t+1 using the t-th parameter set.
        batched_state = np.tile(current_state, (num_steps, 1))
        next_states = integrator(kp_t, batched_state, u_t)
        current_state = np.array(next_states[t])

    return np.array(traj_dynamic)


def simulate_lla_one_step(total, recording, lla_params_over_time, params_car, dt,
                          known_params_over_time=None):
    """One-step predictions for the LLA model: each step predicts from truth[t],
    using the t-th parameter set, producing the predicted state at t+1.

    Returns (total, state_dim) where traj[0] = truth[0] (anchored) and
    traj[t] = one-step prediction from truth[t-1] using params[t-1].
    This mirrors the general models' one-step convention so the same
    'one_step' mode toggle applies uniformly.

    known_params_over_time: optional per-timestep logged known_params, same
    semantics as in simulate_lla_rollout - use the logged value at t instead
    of a single static bank-wide computation.
    """
    print(f"[rollout] LLA one-step: {total} steps")
    num_steps = len(lla_params_over_time)

    bank = build_bank(lla_params_over_time, params_car, num_steps)
    integrator = rk6Factory(jax.device_put(bank.param_bank), dynamics.diffequation, dt)
    static_known_params = bank.get_known_params()

    logged_kp = None
    if known_params_over_time is not None:
        logged_kp = np.asarray(known_params_over_time)

    state0 = recording["state"][0]
    one_step = [np.array(state0)]   # index 0 = truth[0], same as general one-step

    for t in range(total):
        u_t = recording["ctrl"][t]
        kp_t = (logged_kp[t] if logged_kp is not None and t < len(logged_kp)
                else static_known_params)
        # Predict from truth[t] using the t-th LLA params.
        batched_state = np.tile(recording["state"][t], (num_steps, 1))
        next_states = integrator(kp_t, batched_state, u_t)
        one_step.append(np.array(next_states[t]))   # prediction for t+1

    return np.array(one_step[:total])


def simulate_general_m_step(total, recording, general_params_bank, params_car,
                            dt, M, cost_form, mode, ol_reset_interval,
                            full_open_loop=False):
    """For each start time t, roll each fixed model forward up to M steps and
    accumulate the cost_form-weighted squared error vs truth over the horizon.

    Reset behaviour inside the horizon follows `mode`:
      'one_step'  -> re-anchor to truth EVERY step (= sum of M one-step errors)
      'open_loop' -> re-anchor at the existing ol_reset_interval cadence
                     (no internal re-anchor if full_open_loop)
    Each horizon STARTS from truth[t].

    Returns (total, num_models, state_dim): per-dim weighted error summed over
    the horizon, kept separable so the cost-term checkboxes still work.
    """
    num_models = len(general_params_bank)
    print(f"[rollout] general M-step ({mode}): {num_models} models, "
          f"{total} starts x M={M}")
    bank = build_bank(general_params_bank, params_car, num_models)
    lb = history_no_record.LBHistory(
        num_models, dt, np.asarray(cost_form),
        6, rk6Factory, bank, dynamics.diffequation, buffer_size=[0, 0]
    )
    truth = recording["state"]
    ctrl = recording["ctrl"]
    w = np.asarray(cost_form, dtype=float)
    n_truth = len(truth)
    D = truth.shape[1]

    comp = np.zeros((total, num_models, D), dtype=float)
    for t in range(total):
        state = np.tile(truth[t], (num_models, 1))
        for k in range(M):
            ti = t + k
            if ti + 1 >= n_truth:
                break
            if k > 0:
                reset = (mode == 'one_step') or (
                    not full_open_loop and ti % ol_reset_interval == 0)
                if reset:
                    state = np.tile(truth[ti], (num_models, 1))
            lb.predict_states(state, ctrl[ti])
            state = np.array(lb.last_predicted_states)
            err = truth[ti + 1] - state
            err[:, 2] = (err[:, 2] + np.pi) % (2 * np.pi) - np.pi
            comp[t] += (err ** 2) * w[None, :]
    return comp


def simulate_lla_m_step(total, recording, lla_params_over_time, params_car,
                        dt, M, cost_form, mode, ol_reset_interval,
                        full_open_loop=False, known_params_over_time=None):
    """M-step lookahead error for the LLA (per-timestep params) model.

    Same reset semantics as simulate_general_m_step. At horizon step k the
    params used are the ones logged for absolute time ti = t + k.

    known_params_over_time: optional per-timestep logged known_params (see
    simulate_lla_rollout). If given, the integrator at horizon step ti uses
    known_params_over_time[ti] instead of a single static bank-wide value.

    Returns (total, state_dim): per-dim weighted error summed over the horizon.
    WARNING: ~total * M integrator calls -> this is the slow path.
    """
    print(f"[rollout] LLA M-step ({mode}): {total} starts x M={M} (slow)")
    num_steps = len(lla_params_over_time)
    bank = build_bank(lla_params_over_time, params_car, num_steps)
    integrator = rk6Factory(jax.device_put(bank.param_bank),
                            dynamics.diffequation, dt)
    static_known_params = bank.get_known_params()

    logged_kp = None
    if known_params_over_time is not None:
        logged_kp = np.asarray(known_params_over_time)

    truth = recording["state"]
    ctrl = recording["ctrl"]
    w = np.asarray(cost_form, dtype=float)
    n_truth = len(truth)
    D = truth.shape[1]

    comp = np.zeros((total, D), dtype=float)
    for t in range(total):
        state = truth[t]
        for k in range(M):
            ti = t + k
            if ti + 1 >= n_truth or ti >= num_steps:
                break
            if k > 0:
                reset = (mode == 'one_step') or (
                    not full_open_loop and ti % ol_reset_interval == 0)
                if reset:
                    state = truth[ti]
            kp_ti = (logged_kp[ti] if logged_kp is not None and ti < len(logged_kp)
                     else static_known_params)
            batched_state = np.tile(state, (num_steps, 1))
            next_states = integrator(kp_ti, batched_state, ctrl[ti])
            state = np.array(next_states[ti])
            err = truth[ti + 1] - state
            err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
            comp[t] += (err ** 2) * w
    return comp