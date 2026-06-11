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

    x = ca.MX.sym('x', 8) # state: x, y, phi, vx, vy, omega, acceleration, delta
    u = ca.MX.sym('u', 2) # control rate: jerk, steer rate
    p = ca.MX.sym('p', 12)

    x_ref = ca.MX.sym('x_ref', 8)
    

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
        Frx = mass * x[6]* ( p[8]  -  p[9] * x[3]) - p[6] - p[7] * x[3] * x[3]

        alphaf = x[7] - ca.atan2(x[5] * lf + x[4], x[3])
        alphar = ca.atan2(x[5] * lr - x[4], x[3])

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dx0 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2])) #xdot
        dx1 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2])) #ydot
        dx2 = x[5] #phidot
        dx3 = ((Frx - Ffy * ca.sin(x[7])) / mass) + (x[4] * x[5]) #vxdot
        dx4 = ((Fry + Ffy * ca.cos(x[7])) / mass ) - (x[3] * x[5]) #vydot
        dx5 = (lf * Ffy * ca.cos(x[7]) - lr * Fry) / Iz #omegadot
        dx6 = u[0] # jerk
        dx7 = u[1] # steer rate


    f_expl = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)


    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = ca.vertcat(p, x_ref)

    print("\n--- CASADI PRE-SOLVE DIAGNOSTIC ---")
    try:
        test_dyn_func = ca.Function('test_dyn', [model.x, model.u, model.p], [model.f_expl_expr])
        
        x_test = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]) # x, y, yaw, vx, vy, omega, a, steer
        u_test = np.array([0.0, 0.0]) # jerk, steer_rate
        
        p_len = model.p.shape[0]
        p_test = np.ones(p_len) * 0.1 
        p_test[-8:] = x_test # Set x_ref to match x_test
        
        # Evaluate
        dx_eval = test_dyn_func(x_test, u_test, p_test)
        print("Dynamics evaluated at v_x = 5.0:")
        print(dx_eval)
        
        if np.any(np.isnan(dx_eval)) or np.any(np.isinf(dx_eval)):
            print("CRITICAL: Your CasADi dynamics evaluate to NaN/Inf. The solver is dead on arrival.")
        else:
            print("CasADi math is sound. The issue is in the Acados QP/Constraints.")
    except Exception as e:
        print(f"CasADi Function creation failed: {e}")
    print("-----------------------------------\n")

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
    w_accel = 0.01
    w_steer = 0.01
    w_jerk = 0.001
    w_steer_v = 0.1
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
    
    ocp.constraints.lbu = np.array([-10, -params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([10, params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([0, 1]) # 0 is jerk, 1 is steer_vel

    # ocp.constraints.lbu = np.array([-params_car['max_steer_vel']])
    # ocp.constraints.ubu = np.array([params_car['max_steer_vel']])
    # ocp.constraints.idxbu = np.array([1]) # 0 is jerk, 1 is steer_vel

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    
    ocp.solver_options.sim_method_num_steps = 10

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_warm_start = 0    
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    # STABILITY FIXES
    ocp.solver_options.levenberg_marquardt = 1e-4  # Increased damping
    ocp.solver_options.regularize_hessian = 1e-6   # Prevent singular Hessian crashes
    # ocp.solver_options.qp_solver_cond_N = N        # Enable full condensing for small horizons
    ocp.solver_options.hpipm_mode = 'SPEED'       # Failsafe against stiff Pacejka matrices

    return ocp


def create_ipopt_solver(model, params_car, steps, horizon):
    N  = steps
    dt = horizon / N

    x      = model.x
    u      = model.u
    p_sym  = model.p
    f_expr = model.f_expl_expr

    f_func      = ca.Function('f',      [x, u, p_sym], [f_expr])

    w_x = 4.0
    w_y = 4.0
    w_xe = 0.0
    w_ye = 0.0
    w_accel = 0.01
    w_steer = 0.01
    w_jerk = 0.001
    w_steer_v = 0.1

    Q = np.array([w_x, w_y, 0, 0, 0, 0, w_accel, w_steer])
    R = np.array([w_jerk, w_steer_v])

    X_vars = ca.MX.sym('X', 8, N + 1)
    U_vars = ca.MX.sym('U', 2, N)
    p_tire = ca.MX.sym('p_tire', 12)
    X_ref  = ca.MX.sym('Xref',  8, N + 1)

    obj = 0
    g, g_lb, g_ub = [], [], []

    for k in range(N):
        xk   = X_vars[:, k]
        uk   = U_vars[:, k]
        xref = X_ref[:, k]
        p_k  = ca.vertcat(p_tire, xref)

        err  = xk - xref
        obj += ca.dot(err, err * Q)
        obj += ca.dot(uk, uk * R)
        M      = 1
        dt_sub = dt / M
        xk_int = xk
        for _ in range(M):
            k1 = f_func(xk_int,               uk, p_k)
            k2 = f_func(xk_int + dt_sub/2*k1, uk, p_k)
            k3 = f_func(xk_int + dt_sub/2*k2, uk, p_k)
            k4 = f_func(xk_int + dt_sub  *k3, uk, p_k)
            xk_int = xk_int + dt_sub/6 * (k1 + 2*k2 + 2*k3 + k4)

        g.append(X_vars[:, k + 1] - xk_int)
        g_lb.extend([0.0] * 8)
        g_ub.extend([0.0] * 8)

    lbx_list, ubx_list = [], []
    for _ in range(N + 1):
        lbx_list.extend([-ca.inf, -ca.inf, -ca.inf,
                          -0.5, -4.0, -2*np.pi, params_car['min_acc'],
                          params_car['min_steer']])
        ubx_list.extend([ ca.inf,  ca.inf,  ca.inf,
                          params_car['max_v'], 4.0, 2*np.pi, 0.5, params_car['max_acc'],
                          params_car['max_steer']])
    for _ in range(N):
        lbx_list.extend([-10, -params_car['max_steer_vel']])
        ubx_list.extend([10,  params_car['max_steer_vel']])

    lbx = np.array(lbx_list)
    ubx = np.array(ubx_list)

    all_vars   = ca.vertcat(ca.reshape(X_vars, -1, 1),
                            ca.reshape(U_vars, -1, 1))
    all_params = ca.vertcat(p_tire, ca.reshape(X_ref, -1, 1))

    nlp = {'x': all_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': all_params}

    solver = ca.nlpsol('ipopt_mpc', 'ipopt', nlp, {
        'ipopt.max_iter':              500,
        'ipopt.tol':                   1e-4,
        'ipopt.acceptable_tol':        1e-3,
        'ipopt.acceptable_iter':       5,
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.print_level':           0,
        'print_time':                  0,
    })

    def solve(x0_state: np.ndarray,
          p_tire_val: np.ndarray,
          x_ref_traj: np.ndarray) -> dict:
    
        p_val = np.concatenate([p_tire_val, x_ref_traj.flatten(order='F')])
        
        lbx[:8] = x0_state
        ubx[:8] = x0_state

        sol = solver(lbx=lbx, ubx=ubx, lbg=g_lb, ubg=g_ub, p=p_val)

        res      = sol['x'].full().flatten()
        nx_total = 8 * (N + 1)
        X_sol    = res[:nx_total].reshape(8, N + 1, order='F')
        U_sol    = res[nx_total:].reshape(2, N,     order='F')

        return {
            'u':      U_sol[:, 0],
            'X':      X_sol,
            'U':      U_sol,
            'status': solver.stats()['return_status'],
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

