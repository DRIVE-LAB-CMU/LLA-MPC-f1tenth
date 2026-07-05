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

    x = ca.SX.sym('x', 7) # state: x, y, phi, vx, vy, omega, delta
    u = ca.SX.sym('u', 1) # control rate:steer rate
    p = ca.SX.sym('p', 6)
    #parameters: Bf, Br, Cf, Cr, Df, Dr
    x_ref = ca.SX.sym('x_ref', 7)
    

    mass = params_car['mass']
    Iz = params_car['Iz'] 
    lf = params_car['lf']
    lr = params_car['lr']

    g = 9.81

    if not exact: 
        eps = 0.1
        vx_dyn = ca.sqrt(x[3]**2 + eps)

        beta = (lr / (lf + lr)) * x[6] 

        kin_dx4 = 0.0
        kin_dx5 = (x[3] * ca.sin(beta)) / lr

        # Use vx_safe here to prevent NaNs when v = 0
        alphaf = x[6] - ca.atan2((x[5] * lf + x[4]), vx_dyn)
        alphar = ca.atan2((x[5] * lr - x[4]), vx_dyn)

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dyn_dx4 = (Fry + Ffy * ca.cos(x[6])) / mass - x[3] * x[5]
        dyn_dx5 = (lf * Ffy * ca.cos(x[6]) - lr * Fry) / Iz

        v_offset = x[3] - 1.0
        weight_dyn = v_offset / ca.sqrt(1 + v_offset**2) # Ranges -1 to 1
        weight_dyn = 0.5 * (1.0 + weight_dyn)            # Ranges 0 to 1
        weight_kin = 1.0 - weight_dyn

        # 5. Differential equations
        dx0 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2]))
        dx1 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2]))
        dx2 = x[5]

        dx3 = 0

        dx4 = weight_kin * kin_dx4 + weight_dyn * dyn_dx4
        dx5 = weight_kin * kin_dx5 + weight_dyn * dyn_dx5

        dx6 = u[0]

    else:
        alphaf = x[6] - ca.atan2(x[5] * lf + x[4], x[3])
        alphar = ca.atan2(x[5] * lr - x[4], x[3])

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dx0 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2]))
        dx1 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2]))
        dx2 = x[5]
        dx3 = 0
        dx4 = (Fry + Ffy * ca.cos(x[6])) / mass - x[3] * x[5]
        dx5 = (lf * Ffy * ca.cos(x[6]) - lr * Fry) / Iz

        dx6 = u[0]


    f_expl = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6)


    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = ca.vertcat(p, x_ref)

    xdot = ca.SX.sym('xdot', 7) 

    model.f_impl_expr = xdot - f_expl   # <-- add this
    model.xdot = xdot                    # <-- add this


    return model

def create_ocp(model, params_car, steps, horizon):
    ocp = AcadosOcp()
    ocp.model = model

    N = steps #steps
    Tf = horizon # total time horizon

    ocp.solver_options.tf = Tf #set solver settings
    
    nx = model.x.size()[0] 
    nu = model.u.size()[0]
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    w_x = 2.0
    w_y = 2.0
    w_xe = 0.0
    w_ye = 0.0
    w_theta = 0

    w_steer = 0.01
    w_steer_v = 0.001

      
    Q_flat = [w_x, w_y, w_theta, 0, 0, 0, w_steer]
    R_flat = [w_steer_v]

    Q = np.diag(Q_flat) # nx, for trajectory deviation 6x6
    R = np.diag(R_flat)  # nu, for control smoothness 2x2
    Qf = np.diag([w_xe, w_ye, 0, 0, 0, 0, w_steer])

    ocp.cost.W = np.diag(np.concatenate((Q_flat, R_flat))) #nx, nu, nu, 10x10
    ocp.cost.W_e = Qf

    x_ref = model.p[-7:]  # last 8 parameters
    x = model.x
    u = model.u

    ny = nx + nu # running dimensions
    ny_e = nx  #terminal dimension

    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    ocp.cost.yref = np.zeros(ny) # running objective function reference
    ocp.cost.yref_e = np.zeros(ny_e) # terminal objective function reference

    yaw_err = x[2] - x_ref[2]
    yaw_err_wrapped = ca.atan2(ca.sin(yaw_err), ca.cos(yaw_err))

    ocp.model.cost_y_expr = ca.vertcat(
        x[0] - x_ref[0],   # x
        x[1] - x_ref[1],   # y
        yaw_err_wrapped,    # yaw (wrapped)
        x[3] - x_ref[3],   # vx
        x[4] - x_ref[4],   # vy
        x[5] - x_ref[5],   # omega
        x[6] - x_ref[6],   # steer        u
        u[0]
    )
    ocp.model.cost_y_expr_e = ca.vertcat(
        x[0] - x_ref[0],   # x
        x[1] - x_ref[1],   # y
        yaw_err_wrapped,    # yaw (wrapped)
        x[3] - x_ref[3],   # vx
        x[4] - x_ref[4],   # vy
        x[5] - x_ref[5],   # omega
        x[6] - x_ref[6],   # steer
    )
    
    ocp.model.p = model.p  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0]
    ocp.parameter_values = np.zeros((ocp.dims.np, 1))

    ocp.constraints.idxbx = np.array([3, 4, 5, 6])
    ocp.constraints.lbx = np.array([-0.5, 
                                -4,
                                -2 * np.pi,
                               params_car['min_steer']])

    ocp.constraints.ubx = np.array([params_car['max_v'], 
                                    4,
                                    2* np.pi,
                                    params_car['max_steer']])
    
    ocp.constraints.lbu = np.array([-params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([0]) # 0 is slew rate
    
    # slack on constraints
    w_slack = 100.0       # L2 slack penalty (quadratic)
    w_slack_l1 = 10.0    # L1 slack penalty (linear)
    
    nsbx = 4
    ocp.dims.nsbx = nsbx
    ocp.constraints.idxsbx = np.arange(nsbx)   # slack all idxbx entries

    ocp.cost.zl  = w_slack_l1 * np.ones(nsbx)  # lower L1 weight
    ocp.cost.zu  = w_slack_l1 * np.ones(nsbx)  # upper L1 weight
    ocp.cost.Zl  = w_slack    * np.ones(nsbx)  # lower L2 weight
    ocp.cost.Zu  = w_slack    * np.ones(nsbx)  # upper L2 weight
    #####################################
    
    ocp.constraints.idxbx_0 = np.arange(7) # IMPORTANT FOR RUNTIME
    ocp.constraints.lbx_0 = np.zeros(7) # placeholder
    ocp.constraints.ubx_0 = np.zeros(7)           # placeholder

    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    
    # slack on terminal constraints
    nsbx_e = 4
    ocp.dims.nsbx_e = nsbx_e
    ocp.constraints.idxsbx_e = np.arange(nsbx_e)

    ocp.cost.zl_e  = w_slack_l1 * np.ones(nsbx_e)
    ocp.cost.zu_e  = w_slack_l1 * np.ones(nsbx_e)
    ocp.cost.Zl_e  = w_slack    * np.ones(nsbx_e)
    ocp.cost.Zu_e  = w_slack    * np.ones(nsbx_e)
    
    ###############################################

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_num_stages = 4
    
    # DROPPED FROM 10 to 2 (This makes the solver ~5x faster)
    # ocp.solver_options.sim_method_num_steps = 8

    ocp.solver_options.nlp_solver_type = 'SQP'
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

