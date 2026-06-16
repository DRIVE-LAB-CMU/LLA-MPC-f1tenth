import casadi as ca
from acados_template import AcadosModel,  AcadosOcp, AcadosOcpSolver
import numpy as np
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from scipy.linalg import block_diag

from llampc.params import F110

def export_model(params_car, exact = False):
    model = AcadosModel()

    model.name = "f1tenth"

    x = ca.SX.sym('x', 8) # state: x, y, phi, vx, vy, omega, acceleration, delta
    u = ca.SX.sym('u', 2) # control rate: jerk, steer rate
    p = ca.SX.sym('p', 12)

    x_ref = ca.SX.sym('x_ref', 8)
    

    mass = params_car['mass']
    Iz = params_car['Iz'] 
    lf = params_car['lf']
    lr = params_car['lr']

    g = 9.81

    if not exact: 
        print("NONLINEAR MODEL USED")
        eps = .1
        vx_dyn = ca.sqrt(x[3]**2 + eps)

        beta = (lr / (lf + lr)) * x[7] 

        kin_dx4 = 0.0
        kin_dx5 = (x[3] * ca.sin(beta)) / lr

        alphaf = x[7] - ca.atan2((x[5] * lf + x[4]), vx_dyn)
        alphar = ca.atan2((x[5] * lr - x[4]), vx_dyn)

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dyn_dx4 = (Fry + Ffy * ca.cos(x[7])) / mass - x[3] * x[5] + g * ca.sin(p[10])
        dyn_dx5 = (lf * Ffy * ca.cos(x[7]) - lr * Fry) / Iz

        v_offset = x[3] - 1.0
        weight_dyn = v_offset / ca.sqrt(1 + v_offset**2) # Ranges -1 to 1
        weight_dyn = 0.5 * (1.0 + weight_dyn)            # Ranges 0 to 1
        weight_kin = 1.0 - weight_dyn

        dx0 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2]))
        dx1 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2]))
        dx2 = x[5]

        Frx = mass * x[6] * (p[8] - p[9] * x[3]) - p[6] - p[7] * x[3] * x[3]
        dx3 = (Frx - weight_dyn * Ffy * ca.sin(x[7])) / mass + weight_dyn *(x[4] * x[5]) - g * ca.sin(p[11])

        dx4 = weight_kin * kin_dx4 + weight_dyn * dyn_dx4
        dx5 = weight_kin * kin_dx5 + weight_dyn * dyn_dx5

        dx6 = u[0]
        dx7 = u[1]

    else:
        # 1. Extract states for readability
        psi   = x[2]
        vx    = x[3]
        vy    = x[4]
        omega = x[5]
        accel = x[6]
        steer = x[7]

        # 2. Low-speed singularity protection using if_else
        vmin = 0.02
        vy_safe    = ca.if_else(vx < vmin, 0.0, vy)
        omega_safe = ca.if_else(vx < vmin, 0.0, omega)
        steer_safe = ca.if_else(vx < vmin, 0.0, steer)
        vx_safe    = ca.if_else(vx < vmin, vmin, vx)

        # 3. Compute forces using the "safe" variables
        # We use vx_safe here so drag forces don't behave weirdly at zero speed
        Frx = mass * accel * (p[8] - p[9] * vx_safe) - p[6] - p[7] * vx_safe * vx_safe
        
        # Slip angles evaluate safely because vx_safe is never less than 0.05
        alphaf = steer_safe - ca.atan2(omega_safe * lf + vy_safe, vx_safe)
        alphar = ca.atan2(omega_safe * lr - vy_safe, vx_safe)

        # Pacejka lateral tire forces
        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        # 4. Compute derivatives
        # Use ACTUAL vx and vy for world position so the car can perfectly stop
        dx0 = (vx * ca.cos(psi)) - (vy * ca.sin(psi)) # xdot
        dx1 = (vx * ca.sin(psi)) + (vy * ca.cos(psi)) # ydot
        dx2 = omega                                   # phidot

        # Use SAFE variables for dynamics so the solver doesn't panic at zero speed
        dx3 = ((Frx - Ffy * ca.sin(steer_safe)) / mass) + (vy_safe * omega_safe) # vxdot
        dx4 = ((Fry + Ffy * ca.cos(steer_safe)) / mass) - (vx_safe * omega_safe) # vydot
        dx5 = (lf * Ffy * ca.cos(steer_safe) - lr * Fry) / Iz                    # omegadot
        
        # Control inputs
        dx6 = u[0] # jerk (accel rate)
        dx7 = u[1] # steer rate

    f_expl = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)


    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = ca.vertcat(p, x_ref)

    return model

def create_ocp(model, params_car, steps, horizon):
    ocp = AcadosOcp()
    ocp.model = model

    N = steps #steps
    Tf = horizon # total time horizon

    ocp.solver_options.tf = Tf #set solver settings

    nx = model.x.size()[0]  # This MUST be 8
    nu = model.u.size()[0]  # This is 2
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    w_x = 1.0
    w_y = 1.0
    w_xe = 0.0
    w_ye = 0.0
    w_accel = 0.002
    w_steer = 1
    w_jerk = 0.0
    w_steer_v = 0.0
    # w_vel = 0.001
    Q_flat = [w_x, w_y, 0, 0, 0, 0, w_accel, w_steer]
    R_flat = [w_jerk, w_steer_v]

    Q = np.diag(Q_flat) # nx, for trajectory deviation 6x6
    R = np.diag(R_flat)  # nu, for control smoothness 2x2
    Qf = np.diag([w_xe, w_ye, 0, 0, 0, 0, 0,0])  # Now size 8x8

    ocp.cost.W = np.diag(np.concatenate((Q_flat, R_flat))) #nx, nu, nu, 10x10
    ocp.cost.W_e = Qf

    x_ref = model.p[-8:]  # last 8 parameters
    x = model.x
    u = model.u

    ny = nx + nu # running dimensions
    ny_e = nx #terminal dimension

    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    ocp.cost.yref = np.zeros(ny) # running objective function reference
    ocp.cost.yref_e = np.zeros(ny_e) # terminal objective function reference

    ocp.model.cost_y_expr =  ca.vertcat(
        x - x_ref, # of size nx + nu
        #trajectory deviation and control magnitude (make sure last 2 values of xref are 0s)
        u #control smoothness of size nu
    ) # running objective function value 10 long vector
    ocp.model.cost_y_expr_e = x - x_ref # terminal objective funciton value 8 long
    
    ocp.model.p = model.p  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0]
    ocp.parameter_values = np.zeros((ocp.dims.np, 1))

    ocp.constraints.idxbx = np.array([3, 4,5, 6, 7])
    ocp.constraints.lbx = np.array([-0.5, 
                                -4,
                                -2 * np.pi,
                                params_car['min_acc'], 
                                params_car['min_steer']])

    ocp.constraints.ubx = np.array([params_car['max_v'], 
                                    4,
                                    2* np.pi,
                                    params_car['max_acc'], 
                                    params_car['max_steer']])
    
    # slack on constraints
    w_slack = 100.0       # L2 slack penalty (quadratic)
    w_slack_l1 = 10.0    # L1 slack penalty (linear)
    
    nsbx = 5
    ocp.dims.nsbx = nsbx
    ocp.constraints.idxsbx = np.arange(nsbx)   # slack all idxbx entries

    ocp.cost.zl  = w_slack_l1 * np.ones(nsbx)  # lower L1 weight
    ocp.cost.zu  = w_slack_l1 * np.ones(nsbx)  # upper L1 weight
    ocp.cost.Zl  = w_slack    * np.ones(nsbx)  # lower L2 weight
    ocp.cost.Zu  = w_slack    * np.ones(nsbx)  # upper L2 weight
    #####################################
    
    ocp.constraints.idxbx_0 = np.arange(8) # IMPORTANT FOR RUNTIME
    ocp.constraints.lbx_0 = np.zeros(8)           # placeholder
    ocp.constraints.ubx_0 = np.zeros(8)           # placeholder

    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    
    # slack on terminal constraints
    nsbx_e = 5
    ocp.dims.nsbx_e = nsbx_e
    ocp.constraints.idxsbx_e = np.arange(nsbx_e)

    ocp.cost.zl_e  = w_slack_l1 * np.ones(nsbx_e)
    ocp.cost.zu_e  = w_slack_l1 * np.ones(nsbx_e)
    ocp.cost.Zl_e  = w_slack    * np.ones(nsbx_e)
    ocp.cost.Zu_e  = w_slack    * np.ones(nsbx_e)
    ###############################################
    
    # ocp.constraints.lbu = np.array([-10, -params_car['max_steer_vel']])
    # ocp.constraints.ubu = np.array([10, params_car['max_steer_vel']])
    # ocp.constraints.idxbu = np.array([0, 1]) # 0 is jerk, 1 is steer_vel

    # ocp.constraints.lbu = np.array([-params_car['max_steer_vel']])
    # ocp.constraints.ubu = np.array([params_car['max_steer_vel']])
    # ocp.constraints.idxbu = np.array([1]) # 0 is jerk, 1 is steer_vel

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.regularize_method = 'CONVEXIFY' 
    
    ocp.solver_options.sim_method_num_steps = 5

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_iter_max = 15
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_warm_start = 0    
    
    ocp.solver_options.as_rti_level = 3  # does multiple QP solves per RTI step
    ocp.solver_options.as_rti_iter = 1   # number of iterations
    ocp.solver_options.globalization = 'FIXED_STEP'

    # STABILITY FIXES
    ocp.solver_options.levenberg_marquardt = 1e-4  # Increased damping
    ocp.solver_options.regularize_hessian = 1e-6   # Prevent singular Hessian crashes
    # ocp.solver_options.qp_solver_cond_N = N        # Enable full condensing for small horizons
    ocp.solver_options.hpipm_mode = 'SPEED'       # Failsafe against stiff Pacejka matrices

    return ocp


import casadi as ca
import numpy as np



# =============================================================================
# 6-STATE model, condensed from the user's 8-state f1tenth model.
#
#   8-state original:  x = [x,y,psi,vx,vy,omega, accel, steer],  u = [jerk, steer_rate]
#                      with dx6 = jerk, dx7 = steer_rate (pure integrators)
#
#   6-state condensed: x = [x,y,psi,vx,vy,omega],  u = [accel, steer]  (COMMANDS)
#                      the integrators are removed; accel/steer are controls.
#                      jerk / steer_rate are recovered as the inter-stage
#                      differences (accel_k - accel_{k-1}) / dt  etc., penalized
#                      and rate-limited as expressions in the solver.
#
#   The 6-row dynamics are exactly the `exact=True` branch of the original.
#   Parameter layout p (length 12) is unchanged:
#     p[0]=Bf p[1]=Br p[2]=Cf p[3]=Cr p[4]=Df p[5]=Dr
#     p[6]=Cr0 p[7]=Cr2 p[8]=p8 p[9]=p9 p[10]=grade_long p[11]=grade_lat
# =============================================================================
import casadi as ca
import numpy as np

import numpy as np
import casadi as ca
import numpy as np
import casadi as cs

# =============================================================================
# 1. INLINED DYNAMICS (Parametric Tires & Motors, Baked-in Mass/Geometry)
# =============================================================================
def get_dynamics(x, u, p_tire, m, dxdt_sym):
    """
    Inlines the symbolic operations.
    Tire and motor parameters are now passed dynamically via the p_tire symbolic vector (length 12).
    """
    pwm, steer = u[0], u[1]
    psi, vx, vy, omega = x[2], x[3], x[4], x[5]
    
    # Extract parametric tire and motor variables (matches length 12 layout)
    Bf, Br, Cf, Cr, Df, Dr = p_tire[0], p_tire[1], p_tire[2], p_tire[3], p_tire[4], p_tire[5]
    Cm1, Cm2, Cr0, Cr2     = p_tire[6], p_tire[7], p_tire[8], p_tire[9]

    vmin = 0.05
    vy    = cs.if_else(vx < vmin, 0, vy)
    omega = cs.if_else(vx < vmin, 0, omega)
    steer = cs.if_else(vx < vmin, 0, steer)
    vx    = cs.if_else(vx < vmin, vmin, vx)

    # Motor params are now parametric
    Frx = (Cm1 - Cm2 * vx) * pwm - Cr0 - Cr2 * (vx ** 2)
    
    alphaf = steer - cs.atan2((m['lf'] * omega + vy), vx)
    alphar = cs.atan2((m['lr'] * omega - vy), vx)
    
    # Pacejka equations use the parametric variables
    Ffy = Df * cs.sin(Cf * cs.arctan(Bf * alphaf))
    Fry = Dr * cs.sin(Cr * cs.arctan(Br * alphar))

    dxdt_sym[0] = vx * cs.cos(psi) - vy * cs.sin(psi)
    dxdt_sym[1] = vx * cs.sin(psi) + vy * cs.cos(psi)
    dxdt_sym[2] = omega
    dxdt_sym[3] = 1 / m['mass'] * (Frx - Ffy * cs.sin(steer)) + vy * omega
    dxdt_sym[4] = 1 / m['mass'] * (Fry + Ffy * cs.cos(steer)) - vx * omega
    dxdt_sym[5] = 1 / m['Iz']   * (Ffy * m['lf'] * cs.cos(steer) - Fry * m['lr'])
    
    return dxdt_sym


# =============================================================================
# 2. CREATE IPOPT SOLVER (Functional Closure)
# =============================================================================
def create_ipopt_solver(horizon, Ts, Q, R, base_params):
    """
    Builds a single CasADi NLP problem that accepts tire & motor parameters at runtime.
    """
    n_states  = 6
    n_inputs  = 2
    xref_size = 2

    # --- Symbolic Variables ---
    x0     = cs.SX.sym('x0', n_states, 1)
    xref   = cs.SX.sym('xref', xref_size, horizon + 1)
    uprev  = cs.SX.sym('uprev', 2, 1)
    x      = cs.SX.sym('x', n_states, horizon + 1)
    u      = cs.SX.sym('u', n_inputs, horizon)
    dxdtc  = cs.SX.sym('dxdt', n_states, 1)
    
    # New: 12-element symbolic vector for the Pacejka + Motor variables
    p_tire = cs.SX.sym('p_tire', 12, 1)

    cost_tracking = 0
    cost_actuation = 0
    cost_violation = 0

    # --- Terminal Cost & Initial State ---
    cost_tracking += (x[:xref_size, -1] - xref[:xref_size, -1]).T @ Q @ (x[:xref_size, -1] - xref[:xref_size, -1])
    constraints = x[:, 0] - x0

    # --- Loop 1: Dynamics Continuity ---
    for idh in range(horizon):
        # Pass the p_tire symbolic vector into the dynamics
        dxdt = get_dynamics(x[:, idh], u[:, idh], p_tire, base_params, dxdtc)
        constraints = cs.vertcat(constraints, x[:, idh + 1] - x[:, idh] - Ts * dxdt)

    # --- Loop 2: Path Cost + Actuation + Bounds ---
    for idh in range(horizon):
        deltaU = (u[:, idh] - uprev) if idh == 0 else (u[:, idh] - u[:, idh - 1])

        cost_tracking  += (x[:xref_size, idh + 1] - xref[:xref_size, idh + 1]).T @ Q @ (x[:xref_size, idh + 1] - xref[:xref_size, idh + 1])
        cost_actuation += deltaU.T @ R @ deltaU

        constraints = cs.vertcat(constraints,  u[:, idh] - base_params['max_inputs'])
        constraints = cs.vertcat(constraints, -u[:, idh] + base_params['min_inputs'])
        constraints = cs.vertcat(constraints,  deltaU[1] - base_params['max_rates'][1] * Ts)
        constraints = cs.vertcat(constraints, -deltaU[1] + base_params['min_rates'][1] * Ts)

    # --- Compile Problem ---
    cost  = cost_tracking + cost_actuation + cost_violation
    xvars = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1))
    
    # Append p_tire to the end of the parameter vector
    pvars = cs.vertcat(cs.reshape(x0, -1, 1), cs.reshape(xref, -1, 1), cs.reshape(uprev, -1, 1), p_tire)

    nlp = {'x': xvars, 'p': pvars, 'f': cost, 'g': constraints}
    
    ipoptoptions = {
        'print_level': 0, 
        'print_timing_statistics': 'yes', 
        'max_iter': 100,
        'linear_solver': 'mumps',
        'mumps_pivtol': 1e-6,
        'mumps_mem_percent': 200,
        'mumps_pivot_order': 0
    }
    options = {'expand': True, 'print_time': True, 'ipopt': ipoptoptions}
    problem = cs.nlpsol('nmpc', 'ipopt', nlp, options)

    # --- Closure Solver ---
    # Signature updated to require p_tire_val (which should be length 12)
    def solve(x0_val, xref_val, uprev_val, p_tire_val):
        Aineq = np.zeros([0, 2]); bineq = np.zeros([0, 1])
        
        arg = {}
        arg['p'] = np.concatenate([
            x0_val.reshape(-1, 1),
            xref_val.T.reshape(-1, 1),
            uprev_val.reshape(-1, 1),
            Aineq.T.reshape(-1, 1), # Kept for byte-compatibility
            bineq.T.reshape(-1, 1), # Kept for byte-compatibility
            p_tire_val.reshape(-1, 1)
        ])
        arg['lbx'] = -np.inf * np.ones(n_states * (horizon + 1) + n_inputs * horizon)
        arg['ubx'] =  np.inf * np.ones(n_states * (horizon + 1) + n_inputs * horizon)
        arg['lbg'] = np.concatenate([np.zeros(n_states * (horizon + 1)), -np.inf * np.ones(horizon * 6)])
        arg['ubg'] = np.concatenate([np.zeros(n_states * (horizon + 1)),  np.zeros(horizon * 6)])

        res  = problem(**arg)
        
        # Extract and reshape the predicted states for the full horizon (Shape: 6 x N+1)
        xmpc = res['x'][:n_states * (horizon + 1)].full().reshape(horizon + 1, n_states).T
        
        # Extract and reshape the control inputs for the full horizon (Shape: 2 x N)
        umpc = res['x'][n_states * (horizon + 1):n_states * (horizon + 1) + n_inputs * horizon].full().reshape(horizon, n_inputs).T
        
        # Return both the immediate control to apply and the full predicted trajectory
        return {
            'u': umpc[:, 0],
            'X': xmpc
        }
    return solve

    
def get_solver_directory(solver_config = "default"):
    # package_dir = Path(get_package_share_directory('llampc'))
    package_dir = Path(__file__).parent.resolve()
    solvers_dir = package_dir / 'solvers' / solver_config

    solvers_dir.mkdir(parents=True, exist_ok=True)
    
    return solvers_dir


def setup_mpc(steps, horizon, json_file='f1tenth_acados_ocp.json', solver_config ="default", build=True, params_car=F110):
    
    solver_dir = get_solver_directory(solver_config)
    full_json_path = solver_dir / json_file
    
    original_cwd = os.getcwd()
    p_car = params_car()
    try:
        os.chdir(solver_dir)
        print(f"Generating solver in: {solver_dir}")
        
        f1tenth_model = export_model(p_car, exact = False)
        ocp = create_ocp(f1tenth_model, p_car, steps, horizon)
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=build)
        
        print(f"Solver generated successfully: {full_json_path}")
        return solver
    except Exception as e:
        print(f"Error generating solver: {e}")
        raise
    finally:
        os.chdir(original_cwd) 
        
        
def setup_mpc_ipopt(steps, horizon, params_car=F110):
    
    p_car = params_car()
    try:
        
        w_x = 1.0
        w_y = 1.0
        w_xe = 0.0
        w_ye = 0.0
        w_accel = 0.002
        w_steer = 1
        w_jerk = 0.0
        w_steer_v = 0.0
        # w_vel = 0.001
        Q = np.diag([w_x, w_y])
        R = np.diag([w_jerk, w_steer_v])

        
        solver = create_ipopt_solver(steps, horizon/steps, Q, R, p_car)
        # solver = setupNLP(steps, horizon/steps,np.diag([1, 1]), 
        #                   np.diag([0, 0]), np.diag([0.005, 1]),
        #                   )

        return solver
    except Exception as e:
        print(f"Error generating solver: {e}")
        raise 

import os
import re
import ast
import time
import numpy as np
import casadi as ca
# =================================================================
# 2. FILE PARSER WITH TIMING EXTRACTION
# =================================================================
def parse_mpc_log_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        raw_text = f.read().replace('\xa0', ' ')

    blocks = raw_text.split("x0:[")
    parsed_cases = []

    for idx, block in enumerate(blocks[1:]):
        full_block = "x0:[" + block
        try:
            x0_match = re.search(r"x0:\[(.*?)\]", full_block)
            x0 = np.fromstring(x0_match.group(1).replace('\n', ' '), sep=' ')

            xref_match = re.search(r"xref:\[\[(.*?)]\s*\[(.*?)]\]", full_block, re.DOTALL)
            r1 = np.fromstring(xref_match.group(1).replace('\n', ' '), sep=' ')
            r2 = np.fromstring(xref_match.group(2).replace('\n', ' '), sep=' ')
            xref = np.vstack([r1, r2])

            uprev_match = re.search(r"uprev:\[(.*?)\]", full_block)
            uprev = np.fromstring(uprev_match.group(1).replace('\n', ' '), sep=' ')

            dict_match = re.search(r"params(\{.*?\})", full_block)
            params_line = ast.literal_eval(dict_match.group(1))

            time_match = re.search(r"total\s*\|\s*([\d\.]+)ms", full_block)
            their_time = float(time_match.group(1)) if time_match else None

            parsed_cases.append({
                'params_car': params_line,
                'x0': x0,
                'xref': xref,
                'uprev': uprev,
                'their_time_ms': their_time
            })
        except Exception as err:
            print(f"⚠️ Block parse exception at index {idx}: {err}")

    return parsed_cases


# =================================================================
# 3. CLEAN LIVE REPLAY REPLAY LOOP
# =================================================================
if __name__ == "__main__":
    log_file = "mpc_log.txt"
    test_cases = parse_mpc_log_file(log_file)
    
    if not test_cases:
        print("❌ Workspace processing halted: No log lines loaded.")
        exit()

    initial_snapshot = test_cases[0]['params_car']
    steps = test_cases[0]['xref'].shape[1] - 1
    horizon = steps * 0.05

    print("🔨 Instantiating underlying optimization graph...")
    solve_fcn = create_ipopt_solver(initial_snapshot, steps=steps, horizon=horizon)
    print("✔️ Graph ready. Processing sequential replay iterations:\n")

    for idx, case in enumerate(test_cases):
        line_params = case['params_car']
        p_tire_val = [
            line_params['Bf'],  line_params['Br'],  line_params['Cf'],  line_params['Cr'],  
            line_params['Df'],  line_params['Dr'],  line_params['Cm1'], line_params['Cm2'], 
            line_params['Cr0'], line_params['Cr2'], 0.0,                0.0
        ]

        print(f"=== Running Replay Iteration {idx+36} ===")
        their_time_str = f"{case['their_time_ms']:.2f} ms" if case['their_time_ms'] is not None else "N/A"
        print(f"-> Expected Log Wall-Time: {their_time_str}")

        # The internal C++ CasADi timers will dump their block directly to the screen here:
        results = solve_fcn(case['x0'], case['xref'], case['uprev'], p_tire_val)

        print(f"-> Status: {results['status']} | Iters Taken: {results['iters']}")
        print(f"-> Output Control Vector -> PWM: {results['u'][0]:.5f}, Steer: {results['u'][1]:.5f}\n")