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
    #parameters: Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm, roll, pitch
    x_ref = ca.SX.sym('x_ref', 8)
    

    mass = params_car['mass']
    Iz = params_car['Iz'] 
    lf = params_car['lf']
    lr = params_car['lr']

    g = 9.81

    if not exact: 
        print("APPROX MODEL USED")
        eps = 0.1
        vx_dyn = ca.sqrt(x[3]**2 + eps)

        beta = (lr / (lf + lr)) * x[7] 

        kin_dx4 = 0.0
        kin_dx5 = (x[3] * ca.sin(beta)) / lr

        # Use vx_safe here to prevent NaNs when v = 0
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

        # 5. Differential equations
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

        Frx = mass * x[6]* ( p[8]  -  p[9] * x[3]) - p[6] - p[7] * x[3] * x[3]

        alphaf = x[7] - ca.atan2(x[5] * lf + x[4], x[3])
        alphar = ca.atan2(x[5] * lr - x[4], x[3])

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dx0 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2]))
        dx1 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2]))
        dx2 = x[5]
        dx3 = (Frx -  Ffy * ca.sin(x[7])) / mass + (x[4] * x[5]) - g * ca.sin(p[11])
        dx4 = (Fry + Ffy * ca.cos(x[7])) / mass - x[3] * x[5] + g * ca.sin(p[10])
        dx5 = (lf * Ffy * ca.cos(x[7]) - lr * Fry) / Iz

        dx6 = u[0]
        dx7 = u[1]


    f_expl = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)


    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = ca.vertcat(p, x_ref)
    model.a_long_expr = dx3

    # xdot = ca.SX.sym('xdot', 8) 

    # model.f_impl_expr = xdot - f_expl   # <-- add this
    # model.xdot = xdot                    # <-- add this


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

    w_x = 4.0
    w_y = 4.0
    w_xe = 0.0
    w_ye = 0.0
    w_theta = 0
    w_vel = 0
    w_pwm = 0.1
    w_steer = 0.01
    w_slew = 0.0001
    w_steer_v = 0.001
    
    w_a_long = 0.0
      
    Q_flat = [w_x, w_y, w_theta, w_vel, 0, 0, w_pwm, w_steer, w_a_long]
    R_flat = [w_slew, w_steer_v]

    Q = np.diag(Q_flat) # nx, for trajectory deviation 6x6
    R = np.diag(R_flat)  # nu, for control smoothness 2x2
    Qf = np.diag([w_xe, w_ye, 0, 0, 0, 0, 0,0, 0])  # Now size 8x8

    ocp.cost.W = np.diag(np.concatenate((Q_flat, R_flat))) #nx, nu, nu, 10x10
    ocp.cost.W_e = Qf

    x_ref = model.p[-8:]  # last 8 parameters
    x = model.x
    u = model.u

    ny = nx + nu + 1# running dimensions
    ny_e = nx + 1 #terminal dimension

    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    ocp.cost.yref = np.zeros(ny) # running objective function reference
    ocp.cost.yref_e = np.zeros(ny_e) # terminal objective function reference

    yaw_err = x[2] - x_ref[2]
    #yaw_err_wrapped = ca.atan2(ca.sin(yaw_err), ca.cos(yaw_err))
    
    a_long = model.a_long_expr

    ocp.model.cost_y_expr = ca.vertcat(
        x[0] - x_ref[0],   # x
        x[1] - x_ref[1],   # y
        yaw_err,    # yaw (wrapped)
        x[3] - x_ref[3],   # vx
        x[4] - x_ref[4],   # vy
        x[5] - x_ref[5],   # omega
        x[6] - x_ref[6],   # pwm
        x[7] - x_ref[7],   # steer
        a_long,
        u
    )
    ocp.model.cost_y_expr_e = ca.vertcat(
        x[0] - x_ref[0],   # x
        x[1] - x_ref[1],   # y
        yaw_err,    # yaw (wrapped)
        x[3] - x_ref[3],   # vx
        x[4] - x_ref[4],   # vy
        x[5] - x_ref[5],   # omega
        x[6] - x_ref[6],   # pwm
        x[7] - x_ref[7],   # steer
        a_long
    )
    
    ocp.model.p = model.p  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0]
    ocp.parameter_values = np.zeros((ocp.dims.np, 1))

    ocp.constraints.idxbx = np.array([3, 4,5, 6, 7])
    ocp.constraints.lbx = np.array([-0.5, 
                                -4,
                                -2 * np.pi,
                                -0.1, 
                               params_car['min_steer']])

    ocp.constraints.ubx = np.array([params_car['max_v'], 
                                    4,
                                    2* np.pi,
                                    0.5, 
                                    params_car['max_steer']])
    
    ocp.constraints.lbu = np.array([-0.05, -params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([0.2, params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([0, 1]) # 0 is slew rate, 1 is steer_vel

    # slack on constraints
    w_slack = 100.0       # L2 slack penalty (quadratic)
    w_slack_l1 = 10.0    # L1 slack penalty (linear)
    
    nsbx = 3
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

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    
    # DROPPED FROM 10 to 2 (This makes the solver ~5x faster)
    ocp.solver_options.sim_method_num_steps = 8

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    # ocp.solver_options.nlp_solver_max_iter = 10  # 2-3 iterations
    # ocp.solver_options.globalization = 'FIXED_STEP'
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_warm_start = 1    
    
    # STABILITY FIXES
    ocp.solver_options.levenberg_marquardt = 1e-4  # Increased damping
    ocp.solver_options.regularize_hessian = 1e-6   # Prevent singular Hessian crashes
    # ocp.solver_options.qp_solver_cond_N = N        # Enable full condensing for small horizons
    ocp.solver_options.hpipm_mode = 'SPEED'       # Failsafe against stiff Pacejka matrices

    return ocp



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


import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
def diagnostic_mpc_solve():
    mass, Iz, lf, lr = 3.47, 0.047, 0.149, 0.181
    params_p = [12.0, 12.0, 1.7, 1.7, 32.0, 32.0] 
    params_p = [8.0, 8.0, 1.45, 1.45, 32.0, 32.0] 

    N, dt = 20, 0.04
    
    # Decisions: X (States), U (Controls) -- SLACKS REMOVED
    X = ca.MX.sym('X', 8, N+1)
    U = ca.MX.sym('U', 2, N)

    # Model
    x = ca.MX.sym('x', 8)
    u = ca.MX.sym('u', 2)
    vx, vy, omega, a, delta = x[3], x[4], x[5], x[6], x[7]
    
    # Tiny epsilon to absolutely prevent division by zero in IPOPT line searches
    vx_safe = ca.sqrt(vx**2 + 1e-4)
    
    # --- CRITICAL FIX: The Slip Angles ---
    # Rear slip angle MUST oppose the lateral velocity to be stable
    af = delta - ca.atan2(vy + omega*lf, vx_safe)
    ar = -ca.atan2(vy - omega*lr, vx_safe) 
    
    Ffy = params_p[4] * ca.sin(params_p[2] * ca.atan(params_p[0] * af))
    Fry = params_p[5] * ca.sin(params_p[3] * ca.atan(params_p[1] * ar))

    f = ca.vertcat(
        vx*ca.cos(x[2]) - vy*ca.sin(x[2]),
        vx*ca.sin(x[2]) + vy*ca.cos(x[2]),
        omega,
        a - (Ffy*ca.sin(delta))/mass + vy*omega,
        (Fry + Ffy*ca.cos(delta))/mass - vx*omega,
        (lf*Ffy*ca.cos(delta) - lr*Fry)/Iz,
        u[0], u[1]
    )
    f_func = ca.Function('f', [x, u], [f])

    obj = 0
    g = []
    
    x0_init = [0, 0, 0.05, 1.0, 0, 0, 0, 0] 
    g.append(X[:, 0] - x0_init) 
    
    for k in range(N):
        # Cost Function
        obj += 5000*(X[1, k] - 0.15)**2    # Target Y
        # obj += 100*(X[3, k] - 3.0)**2    # Target Speed
        
        # Stability Costs
        # obj += 200 * (X[2, k])**2        # Penalize Heading Angle
        # obj += 100 * (X[4, k])**2        # Penalize lateral sliding (vy)
        # obj += 100 * (X[5, k])**2        # Penalize yaw rate (omega)
        # obj += 50 * (X[7, k])**2         # Penalize Steering Angle
        
        obj += 0.1 * U[0, k]**2          # Jerk effort
        obj += 5.0 * U[1, k]**2          # Steering rate effort 
        
        # Dynamics (Hard Constraints)
        M = 10 
        dt_sub = dt / M
        
        X_k = X[:,k]
        for _ in range(M):
            k1 = f_func(X_k, U[:,k])
            k2 = f_func(X_k + dt_sub/2*k1, U[:,k])
            k3 = f_func(X_k + dt_sub/2*k2, U[:,k])
            k4 = f_func(X_k + dt_sub*k3, U[:,k])
            X_k = X_k + dt_sub/6*(k1 + 2*k2 + 2*k3 + k4)
            
        g.append(X[:, k+1] - X_k)

    # SOLVER OPTIONS
    opts = {
        'ipopt.max_iter': 500,
        'ipopt.hessian_approximation': 'limited-memory', 
        'ipopt.tol': 1e-3, 
        'ipopt.print_level': 5,
    }

    all_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
    nlp = {'x': all_vars, 'f': obj, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('S', 'ipopt', nlp, opts)
    
    # BOUNDS
    lbx = []
    ubx = []
    
    # State bounds: [x, y, theta, vx, vy, omega, a, delta]
    for _ in range(N+1):
        lbx.extend([-ca.inf, -ca.inf, -ca.inf,  0.1, -ca.inf, -ca.inf, -8.0, -0.6])
        ubx.extend([ ca.inf,  ca.inf,  ca.inf, 20.0,  ca.inf,  ca.inf,  8.0,  0.6])
        
    # Control bounds: [jerk, steer_rate]
    for _ in range(N):
        lbx.extend([-10.0, -2.0])
        ubx.extend([ 10.0,  2.0])
        
    # Initial Guess
    x_guess_list = []
    for k in range(N+1):
        step_guess = list(x0_init)
        step_guess[0] += step_guess[3] * k * dt  # Move X forward based on vx
        x_guess_list.extend(step_guess)
        
    u_guess_list = [0.0] * (2 * N)
    vars_init = x_guess_list + u_guess_list
    
    # Solve 
    sol = solver(x0=vars_init, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    # Result Extraction
    res = sol['x'].full().flatten()
    states = res[:8*(N+1)].reshape(N+1, 8)
    
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    plt.plot(states[:,0], states[:,1], '-o', label='Trajectory')
    plt.axhline(0.15, color='r', linestyle='--', label='Target Line')
    plt.ylabel('Y pos (m)'); plt.legend()
    
    plt.subplot(212)
    plt.plot(states[:,3], label='vx')
    plt.axhline(0.05, color='k', alpha=0.3, label='if_else threshold')
    plt.ylabel('Vel (m/s)'); plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    diagnostic_mpc_solve()
