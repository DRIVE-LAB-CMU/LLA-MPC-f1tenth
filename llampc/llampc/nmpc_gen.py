import casadi as ca
from acados_template import AcadosModel,  AcadosOcp, AcadosOcpSolver
import numpy as np
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from llampc.params import F110

def export_model(params_car):
    model = AcadosModel()

    model.name = "f1tenth"

    x = ca.MX.sym('x', 8) # state: x, y, phi, vx, vy, omega, acceleration, delta
    u = ca.MX.sym('u', 2) # control rate: jerk, steer rate
    p = ca.MX.sym('p', 10)
    #parameters: Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm

    mass = params_car['mass']
    Iz = params_car['Iz'] 
    lf = params_car['lf']
    lr = params_car['lr']

    Frx = mass * (x[7] * p[8]  -  p[9] * x[3]) - p[6] - p[7] * x[3] * x[3]
    #nominal force * Cefficiency - Crolling - Cmotor vx - cdrag vx^2

    alphaf = x[6] - ca.atan2(x[5] * lf + x[4], x[3]+ 1e-8)
    #steer - arctan(omega * lf + vy, vx)
    alphar =  ca.atan2(x[5] * lr - x[4], x[3]+ 1e-8)
    #arctan(omega * lr - vy, vx)

    Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
    Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

    dx1 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2])) #xdot
    dx2 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2])) #ydot
    dx3 = (x[5]) #phidot
    dx4 = (Frx - Ffy * ca.sin(x[6])) / mass + x[4] * x[5] #vxdot
    dx5 = (Fry + Ffy * ca.cos(x[6])) / mass - x[3] * x[5] #vydot
    dx6 = (Ffy * lf * ca.cos(x[6]) - Fry * lr)/ Iz #omegadot
    dx7 = u[0] # steer rate
    dx8 = u[1] # jerk

    # inputs: accel, steer, (x[7] and x[6])
    # input/control rates: jerk and steer rate (u[0] and u[1])


    f_expl = ca.vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8)


    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p

    return model

def create_ocp(model, params_car):
    ocp = AcadosOcp()
    ocp.model = model

    N = 20 #steps
    Tf = 2.0 # total time horizon
    nx, nu = model.x.size()[0] - 2, model.u.size()[0] 
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    w_x = 1.0
    w_y = 1.0
    w_xe = 2.0
    w_ye = 2.0
    w_steer = 1.0
    w_accel = 1.0
    w_jerk = 0
    w_steer_v = 0

    Q = ca.DM(np.diag([w_x, w_y, 0.0, 0.0, 0.0, 0.0])) # nx, for trajectory deviation 6x6
    R = ca.DM(np.diag([w_steer, w_accel]))  # nu, for control magnitude 2x2
    Rd = ca.DM(np.diag([w_jerk, w_steer_v]))  # nu, for control smoothness 2x2
    Qf = ca.DM(np.diag([w_x, w_y, 0.0, 0.0, 0.0, 0.0]))  # nx, for final state deviation 6x6

    ocp.cost.W = ca.block_diag(Q, R, Rd) #nx, nu, nu, 10x10
    ocp.cost.W_e = Qf # cost matrix

    x_ref = ca.MX.sym('x_ref', nx + nu)  # Reference state
    u_prev = ca.MX.sym('u_prev', nu)  # Previous control input
    x = model.x
    u = model.u

    ny = nx + 2 * nu # running dimensions
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
    ocp.model.cost_y_expr_e = x[:6] - x_ref[:6] # terminal objective funciton value 6 long
    
    ocp.model.p = ca.vertcat(model.p, x_ref)  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0] + x_ref.size()[0]

    ocp.constraints.lbx = np.array([params_car['min_v'], params_car['min_acc'], params_car['min_steer'] ])
    ocp.constraints.ubx = np.array([params_car['max_v'], params_car['max_acc'], params_car['max_steer'] ])
    ocp.constraints.idxbx = np.array([3, 6, 7])  # vx, delta, acceleration, steer_Rate

    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

    ocp.constraints.lbu = np.array([-params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([1])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 50

    return ocp

def get_solver_directory(solver_config = "default"):
    package_dir = Path(get_package_share_directory('f1tenth_mpc'))
    solvers_dir = package_dir / 'solvers' / solver_config

    solvers_dir.mkdir(parents=True, exist_ok=True)
    
    return solvers_dir


def setup_mpc(json_file='f1tenth_acados_ocp.json', solver_config ="default", build=True, params_car=F110):
    
    solver_dir = get_solver_directory(solver_config)
    full_json_path = solver_dir / json_file
    
    original_cwd = os.getcwd()
    p_car = params_car()
    try:
        os.chdir(solver_dir)
        print(f"Generating solver in: {solver_dir}")
        
        f1tenth_model = export_model(p_car)
        ocp = create_ocp(f1tenth_model, p_car)
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=build)
        
        print(f"Solver generated successfully: {full_json_path}")
        return solver
    except Exception as e:
        print(f"Error generating solver: {e}")
        raise
    finally:
        os.chdir(original_cwd) 

def setup_mpc_from_json(json_file='f1tenth_acados_ocp.json', solver_config = "default", params_car=F110):
    solver_dir = get_solver_directory(solver_config)
    full_json_path = solver_dir / json_file

    original_cwd = os.getcwd()

    if not full_json_path.exists():
        print(f"JSON file {full_json_path} not found. Building solver from scratch...")
        return setup_mpc(json_file=json_file, solver_config=solver_config, build=True, params_car=params_car)    
    
    try:
        os.chdir(solver_dir)
        # Load solver from existing JSON (no rebuild needed)
        solver = AcadosOcpSolver.create_from_json(json_file)
        print(f"Successfully loaded solver from {json_file}")

        return solver
    except Exception as e:
        print(f"Failed to load from JSON: {e}")
        print("Rebuilding solver from scratch...")
        return setup_mpc(json_file=json_file, build=True, params_car=params_car)
    finally:
        os.chdir(original_cwd)