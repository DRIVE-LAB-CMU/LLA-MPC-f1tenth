import casadi as ca
from acados_template import AcadosModel,  AcadosOcp, AcadosOcpSolver
import numpy as np
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

def export_model():
    model = AcadosModel()
    model.name = "f1tenth"

    x = ca.MX.sym('x', 7) # state: x, y, phi, vx, vy, omega, delta
    u = ca.MX.sym('u', 2) # controls: acceleration, steer_rate
    p = ca.MX.sym('p', 8)
    #parameters: Bf, Br, Cf, Cr, Df, Dr, Cro, Cd
    p_car = ca.MX.sym('pcar', 4)
    #fixed: m, Iz, lf, lr

    Frx = (p_car[0] * u[0]) - p[6] - p[7] * x[3] * x[3]
    #nominal force - Cro - cdvx^2

    alphaf = x[6] - ca.atan2(x[5] * p_car[2] + x[4], x[3]+ 1e-8)
    #steer - arctan(omega * lf + vy, vx)
    alphar =  ca.atan2(x[5] * p_car[3] - x[4], x[3]+ 1e-8)
    #arctan(omega * lr - vy, vx)

    Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
    Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

    dx1 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2])) #xdot
    dx2 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2])) #ydot
    dx3 = (x[5]) #phidot
    dx4 = (Frx - Ffy * ca.sin(x[6])) / p_car[0] + x[4] * x[5] #vxdot
    dx5 = (Fry + Ffy * ca.cos(x[6])) / p_car[0] - x[3] * x[5] #vydot
    dx6 = (Ffy * p_car[2] * ca.cos(x[6]) - Fry * p_car[3])/ p_car[1]
    dx7 = u[1] #steerdot

    f_expl = ca.vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7)


    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = ca.vertcat(p, p_car)

    return model

def create_ocp(model):
    ocp = AcadosOcp()
    ocp.model = model

    N = 20
    Tf = 2.0
    nx, nu = model.x.size()[0], model.u.size()[0]
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'


    Q = ca.DM(np.diag([])) # nx 
    R = ca.DM(np.diag([]))  # nu
    Rd = ca.DM(np.diag([]))  # nu 
    Qf = ca.DM(np.diag([]))  # nx

    ocp.cost.W = ca.block_diag(Q, R, Rd) #nx + nu + nu
    ocp.cost.W_e = Qf # cost matrix


    x_ref = ca.MX.sym('x_ref', nx)  # Reference state
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
        x - x_ref,
        u,
        u - u_prev
    ) # running objective function value
    ocp.model.cost_y_expr_e = x - x_ref # terminal objective funciton value
    
    p_ref = ca.vertcat(x_ref, u_prev)  # reference parameters
    ocp.model.p = ca.vertcat(model.p, p_ref)  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0] + p_ref.size()[0]

    ocp.constraints.lbx = [0.0, -0.35]    # Just the constrained values
    ocp.constraints.ubx = [1.5, 0.35]     # Just the constrained values
    ocp.constraints.idxbx = np.array([3, 6]) 

    ocp.constraints.lbu = np.array([-3.0, -0.5])  # [min_acceleration, min_steer_rate]
    ocp.constraints.ubu = np.array([3.0, 0.5])    # [max_acceleration, max_steer_rate]
    ocp.constraints.idxbu = np.array([0, 1])    

    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

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


def setup_mpc(json_file='f1tenth_acados_ocp.json', solver_config ="default", build=True):
    
    solver_dir = get_solver_directory(solver_config)
    full_json_path = solver_dir / json_file
    
    original_cwd = os.getcwd()
    try:
        os.chdir(solver_dir)
        print(f"Generating solver in: {solver_dir}")
        
        f1tenth_model = export_model()
        ocp = create_ocp(f1tenth_model)
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=build)
        
        print(f"Solver generated successfully: {full_json_path}")
        return solver
    except Exception as e:
        print(f"Error generating solver: {e}")
        raise
    finally:
        os.chdir(original_cwd) 

def setup_mpc_from_json(json_file='f1tenth_acados_ocp.json', solver_config = "default"):
    solver_dir = get_solver_directory(solver_config)
    full_json_path = solver_dir / json_file

    original_cwd = os.getcwd()

    if not full_json_path.exists():
        print(f"JSON file {full_json_path} not found. Building solver from scratch...")
        return setup_mpc(json_file=json_file, solver_config=solver_config, build=True)    
    
    try:
        os.chdir(solver_dir)
        # Load solver from existing JSON (no rebuild needed)
        solver = AcadosOcpSolver.create_from_json(json_file)
        print(f"Successfully loaded solver from {json_file}")

        return solver
    except Exception as e:
        print(f"Failed to load from JSON: {e}")
        print("Rebuilding solver from scratch...")
        return setup_mpc(json_file=json_file, build=True)
    finally:
        os.chdir(original_cwd)