"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
# from numba import njit, float64, boolean, int64, prange
# from numba.experimental import jitclass
import os


# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("/home/kathy/jax_cache")

print("imports")
import jax
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
from functools import partial
from collections import deque
import time
print("import complete")


# def _get_batch(num_models, x_t, u_t):
#     x_t_batch = np.empty((num_models, x_t.shape[0]), dtype=np.float64)
#     u_t_batch = np.empty((num_models, u_t.shape[0]), dtype=np.float64)
    
#     for i in range(num_models):
#         x_t_batch[i, :] = x_t
#         u_t_batch[i, :] = u_t

#     return x_t_batch, u_t_batch

# spec = [
#     ('num_models', int64),          
#     ('history_length', int64),       
#     ('last_predicted_states', float64[:, :]), 
#     ('running_cost', float64[:]),         
#     ('cost_history', float64[:, :]),       
#     ('queue_index', int64),
#     ('dt', float64),                          
#     ('cost_weights', float64[:]),  
#     ('state_size', int64),                    
# ]
# @jitclass(spec)


print("jit")
@jax.jit
def get_lookback_error(last_predicted_states, x_t, cost_weights, queue_index):
    cost = jnp.sum(jnp.square(x_t[None, :] - last_predicted_states) * cost_weights[None, :], axis = 1)

    return cost

@jax.jit
def find_best_model(running_cost):
    return jnp.argmin(running_cost)

print("jit complete")


class LBHistory:

    def __init__(self, num_models, history_length, dt, cost_weights, state_size, integrator_factory, dynamics_bank, diffeq,
                 buffer_size = None, control_size = 2):
        self.num_models = num_models
        self.history_length = history_length

        self.last_predicted_states = jnp.zeros((self.num_models, state_size), dtype='float16')

        self.running_cost = np.zeros(self.num_models, dtype='float16')
        self.cost_history = np.zeros((self.num_models, self.history_length), dtype='float16')
        self.queue_index = 0
        self.dt = dt
        self.cost_weights = jnp.array(cost_weights)
        self.state_size = state_size
        
        self.dynamics_bank = dynamics_bank
        self.integrator = integrator_factory(
            dynamics_bank.param_bank,
            diffeq,
            dt
        )
        

        # control buffer to account for actuation delay 
        # note that this does not account for actuation speed, which is accounted for via 
        # the nmpc problem setup itself via max acceleratin and steering angle
        self.control_size = 2
        self.buffer_size = np.zeros(control_size) if buffer_size is None else buffer_size
        self.buffer = [deque() for _ in range(control_size)]

    def predict_states(self, x_t, u_t):
        """Batched version of _integrate"""
        buffered_u_t = u_t
        # buffered_u_t = np.zeros_like(u_t)
        # for i in range(self.control_size):
        #     self.buffer[i].append(u_t[i])
        #     if(len(self.buffer[i]) > self.buffer_size[i]):
        #         buffered_u_t[i] = self.buffer[i].popleft()

        t0 = time.perf_counter_ns()
        # buffered_u_t = np.zeros_like(u_t)
        # buffered_u_t = u_t
        # for i in range(self.control_size):
        #     self.buffer[i].append(u_t[i])
        #     if(len(self.buffer[i]) > self.buffer_size[i]):
        #         buffered_u_t[i] = self.buffer[i].popleft()
        
        t1 = time.perf_counter_ns()
        known_params = self.dynamics_bank.get_known_params()
        
        t2 = time.perf_counter_ns()
        self.last_predicted_states = self.integrator(
            known_params,
            x_t,
            buffered_u_t)
        
        t3 = time.perf_counter_ns()

        jax.block_until_ready(self.last_predicted_states)
        t4 = time.perf_counter_ns()
        
        print(f"Buffer: {(t1-t0)*1e-6:.3f}ms, "
          f"GetParams: {(t2-t1)*1e-6:.3f}ms, "
          f"Integrator: {(t3-t2)*1e-6:.3f}ms, "
          f"Sync: {(t4-t3)*1e-6:.3f}ms")
        
        
    def update_lookback_error(self, x_t):
        self.running_cost -= self.cost_history[:, self.queue_index]
        cost = np.array(get_lookback_error(
            self.last_predicted_states,
            x_t, 
            self.cost_weights,
            self.queue_index
        ))
        self.cost_history[:, self.queue_index] = cost
        self.running_cost += cost
        self.queue_index = (self.queue_index + 1) % self.history_length

        return cost
    

    def reset(self):
        self.last_predicted_states = jnp.zeros((self.num_models, self.state_size))
        self.running_cost = np.zeros(self.num_models)
        self.cost_history = np.zeros((self.num_models, self.history_length))
        self.queue_index = 0

    def get_best_model(self):
        return find_best_model(self.running_cost) 
    
    

# integrate vehicle dynamics by 1 step
import numpy as np
# from numba import njit, float64, boolean, int64
# from numba.experimental import jitclass

import jax
import jax.numpy as jnp
from functools import partial


# @njit(parallel=True)
# @njit(fastmath=True)


@partial(jax.jit, static_argnums=(1, 2))  # diffequation and h are static
def odeintRK4_batch(bank_params, diffequation, h, known_params, x0, u):
    def step(b_p, x_t):
        return diffequation(b_p,  known_params,x_t, u)

    def rk4(b_p):
        k1 = h * step(b_p, x0)
        k2 = h * step(b_p, x0 + k1 / 2)
        k3 = h * step(b_p, x0 + k2 / 2)
        k4 = h * step(b_p, x0 + k3)
        return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return jax.vmap(rk4)(bank_params)

def integratorFactory(bank_params, diffequation, h):
    """Returns a function with bank_params, diffequation, and h pre-filled."""
    # Return a partial function that has bank_params, diffequation, h bound
    return partial(odeintRK4_batch, bank_params, diffequation, h)

print("jit diffeq")
@jax.jit
def diffequation(bank_params, known_params, x, u):
    """Optimized for GPU - no conditionals"""
    g = 9.81

    acc = u[0]
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    mass, Iz, lf, lr, roll, pitch = known_params
    
    # Inline force calculation to reduce function call overhead
    Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params
    
    # Forces
    Frx = mass * (acc * Ce - Cm * vx) - Cro - Cd * (vx * vx)
    
    vx_safe = jnp.where(jnp.abs(vx) < 1e-4, 1e-4, vx)
    alphaf = steer - jnp.arctan2((lf * omega + vy), vx_safe)
    alphar = jnp.arctan2((lr * omega - vy), vx_safe)
    
    mask = jnp.abs(vx) >= 1e-4
    alphaf = jnp.where(mask, alphaf, 0.0)
    alphar = jnp.where(mask, alphar, 0.0)
    
    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    return jnp.array([
        vx*jnp.cos(psi) - vy*jnp.sin(psi),
        vx*jnp.sin(psi) + vy*jnp.cos(psi),
        omega,
        1/mass * (Frx - Ffy*jnp.sin(steer)) + vy*omega - g * jnp.sin(pitch),
        1/mass * (Fry + Ffy*jnp.cos(steer)) - vx*omega + g * jnp.sin(roll),
        1/Iz * (Ffy * lf*jnp.cos(steer) - Fry * lr),
    ])
    

@jax.jit
def _calc_forces(bank_params,  x, u):
    acc = u[0]
    steer = u[1]
    psi = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]

    Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm = bank_params
    mass, Iz, lf, lr, pitch, roll =  3.74, 0.04712, 0.15875,0.17145, 0, 0

    Frx = mass * (acc * Ce - Cm * vx ) - Cro - Cd * (vx * vx)

    alphaf = steer - jnp.arctan2((lf*omega + vy), jnp.abs(vx))
    alphar =jnp.arctan2((lr*omega - vy), jnp.abs(vx))
    Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
    Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

    return Ffy, Frx, Fry # each of these should end up being num_models long

print("jit diffeq complete")


print("defining bank")
# @jitclass(spec)
class DBMPacejkaBank():
    def __init__(self, 
                 lf, lr, 
                 mass, Iz, 
                 Bf, Br,
                 Cf, Cr, 
                 Df, Dr, 
                 Cro, Cd,
                 Ce, Cm, 
                 roll, pitch, 
                 num_models
                 ):
        # non-varying parameters
        self.lf = lf
        self.lr = lr
        self.mass = mass
        self.Iz = Iz

        # varying parameters
        self.Bf = Bf
        self.Br = Br
        self.Cf = Cf
        self.Cr = Cr
        self.Df = Df
        self.Dr = Dr
        self.Cro = Cro
        self.Cd = Cd
        self.Ce = Ce
        self.Cm = Cm
        self.num_models = num_models

        self.pitch = pitch
        self.roll = roll

        # non-sampled state parameters

        self.num_models = num_models

        self.param_bank = jnp.stack([
            self.Bf, self.Br, self.Cf, self.Cr, self.Df, self.Dr,
            self.Cro, self.Cd, self.Ce, self.Cm
        ], axis=1)

    def get_known_params(self):
        return jnp.array([self.mass, self.Iz, self.lf, self.lr, self.roll, self.pitch])

    def get_bank_params(self):
        return self.param_bank

    def get_model_params_arr(self, index):
        return np.array([
            self.Bf[index],
            self.Br[index],
            self.Cf[index],
            self.Cr[index],
            self.Df[index],
            self.Dr[index],
            self.Cro[index],
            self.Cd[index],
            self.Ce[index],
            self.Cm[index],
            self.roll,
            self.pitch
        ])
    
print("finished defining bank")

import time
import jax
import jax.numpy as jnp
import numpy as np

# Assuming your integrator, diffequation, _calc_forces, DBMPacejkaBank,
# get_lookback_error, find_best_model, and LBHistory class
# are already defined/imported here

def benchmark_lbhistory_loop():
    print("defining params")
    num_models = 6000
    history_length = 25
    dt = 0.02
    state_size = 6

    print("creating cost weights")
    cost_weights = jnp.ones(state_size)


    print("initializing bank params")

    # Initialize model bank with dummy parameters (replace with your data)
    bank_params = jnp.array(np.random.rand(num_models, 10), dtype=jnp.float32)
    print("initializing bank params")

    print("initializing bank")
    dynamics_bank = DBMPacejkaBank(
        lf=0.15875,
        lr=0.17145,
        mass=3.74,
        Iz=0.04712,
        Bf=bank_params[:, 0],
        Br=bank_params[:, 1],
        Cf=bank_params[:, 2],
        Cr=bank_params[:, 3],
        Df=bank_params[:, 4],
        Dr=bank_params[:, 5],
        Cro=bank_params[:, 6],
        Cd=bank_params[:, 7],
        Ce=bank_params[:, 8],
        Cm=bank_params[:, 9],
        num_models=num_models,
        pitch = 0,
        roll = 0
    )

    print("completed bank")

    print("initializing history")

    # Create LBHistory object
    lb_history = LBHistory(
        num_models, history_length,
        dt, cost_weights,
        state_size, integratorFactory,
        dynamics_bank, diffequation,
        buffer_size = [1, 1]
    )
    print("completed history")

    

    # Initial state and control input
    x_t = jnp.zeros(state_size)
    u_t = jnp.array([1.0, 0.1])  # sample control input


    start = time.perf_counter_ns()
    print("tracing")
    # Warm-up to trigger JIT compilation
    lb_history.predict_states(x_t, u_t)
    lb_history.update_lookback_error(x_t)
    lb_history.get_best_model()
    end = time.perf_counter_ns()
    print(f"complete in {(end-start)* 1e-9} seconds")

    # Benchmark loop
    n_steps = 1000
    predict_times = []

    for _ in range(n_steps):
        start = time.perf_counter_ns()
        lb_history.predict_states(x_t, u_t)
        jnp.min(lb_history.last_predicted_states)
        end = time.perf_counter_ns()
        predict_times.append(end - start)

        lb_history.update_lookback_error(x_t)
        x_t = np.empty_like(lb_history.last_predicted_states[0])

    avg_predict_time_ms = (sum(predict_times) / n_steps) * 1e-6
    worst_predict_time_ms = max(predict_times) * 1e-6
    print(f"Average predict_states time: {avg_predict_time_ms:.4f} ms")
    print(f"Worst predict time: {worst_predict_time_ms:.4f} ms")

if __name__ == "__main__":
    print("starting")
    benchmark_lbhistory_loop()
