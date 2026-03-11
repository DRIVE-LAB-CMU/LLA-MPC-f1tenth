import casadi as ca
from acados_template import AcadosModel,  AcadosOcp, AcadosOcpSolver
import numpy as np
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from scipy.linalg import block_diag

from llampc.params import F110

def export_model(params_car, linear = False):
    model = AcadosModel()

    model.name = "f1tenth"

    x = ca.MX.sym('x', 8) # state: x, y, phi, vx, vy, omega, acceleration, delta
    u = ca.MX.sym('u', 2) # control rate: jerk, steer rate
    p = ca.MX.sym('p', 12)
    #parameters: Bf, Br, Cf, Cr, Df, Dr, Cro, Cd, Ce, Cm, roll, pitch
    x_ref = ca.MX.sym('x_ref', 8)
    

    mass = params_car['mass']
    Iz = params_car['Iz'] 
    lf = params_car['lf']
    lr = params_car['lr']

    g = 9.81

    if not linear: 
        print("NONLINEAR MODEL USED")
        Frx = mass * x[6] * (p[8]  -  p[9] * x[3]) - p[6] - p[7] * x[3] * x[3]
        #nominal force * Cefficiency - Crolling - Cmotor vx - cdrag vx^2
        
        alphaf = ca.if_else(x[3] < 0.01, 0, x[7] - ca.atan2(x[5] * lf + x[4], x[3]))
        alphar = ca.if_else(x[3] < 0.01, 0, ca.atan2(x[5] * lr - x[4], x[3]))
        #arctan(omega * lr - vy, vx)

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dx0 = (x[3] * ca.cos(x[2])) - (x[4] * ca.sin(x[2])) #xdot
        dx1 = (x[3] * ca.sin(x[2])) + (x[4] * ca.cos(x[2])) #ydot
        dx2 = x[5] #phidot
        dx3 = (Frx - Ffy * ca.sin(x[7])) / mass + x[4] * x[5] - g * ca.sin(p[11]) #vxdot
        dx4 = (Fry + Ffy * ca.cos(x[7])) / mass - x[3] * x[5] + g * ca.sin(p[10])#vydot
        dx5 = (lf * Ffy * ca.cos(x[7]) - lr * Fry) / Iz #omegadot
        dx6 = u[0] # jerk
        dx7 = u[1] # steer rate
    else:
        print("LINEAR MODEL USED")
        Frx = mass * x[6]* ( p[8]  -  p[9] * x[3]) - p[6] - p[7] * x[3] * x[3]

        alphaf = ca.if_else(x[3] < 1e-4, 0, x[7] - ca.atan2(x[5] * lf + x[4], x[3]))
        alphar = ca.if_else(x[3] < 1e-4, 0, ca.atan2(x[5] * lr - x[4], x[3]))

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


    # inputs: accel, steer, (x[6] and x[7])
    # input/control rates: jerk and steer rate (u[0] and u[1])


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
    nx, nu = model.x.size()[0] - 2, model.u.size()[0] 
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # w_x = 2.0
    # w_y = 2.0
    # w_xe = 0
    # w_ye = 0
    # w_steer = .001
    # w_accel = 0.001
    # w_jerk = 0
    # w_steer_v = 0
    

    w_x = 2.0
    w_y = 2.0
    w_xe = 0
    w_ye = 0
    w_steer = 0.03
    w_accel = 0.01
    w_jerk = .001
    w_steer_v = 0.01
    Q_flat = [w_x, w_y, 0.0, 0.0, 0.0, 0.0]
    R_flat = [w_accel, w_steer]
    Rd_flat = [w_jerk, w_steer_v]

    Q = np.diag(Q_flat) # nx, for trajectory deviation 6x6
    R = np.diag(R_flat)  # nu, for control magnitude 2x2
    Rd = np.diag(Rd_flat)  # nu, for control smoothness 2x2
    Qf = np.diag([w_xe, w_ye, 0.0, 0.0, 0.0, 0.0])  # nx, for final state deviation 6x6

    ocp.cost.W = np.diag(np.concatenate((Q_flat, R_flat, Rd_flat))) #nx, nu, nu, 10x10
    ocp.cost.W_e = Qf # cost matrix

    x_ref = model.p[-8:]  # last 8 parameters
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
    
    ocp.model.p = model.p  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0]
    ocp.parameter_values = np.zeros((ocp.dims.np, 1))

    ocp.constraints.lbx = np.array([-1e9, -1e9, -1e9, params_car['min_v'], -1e9, -1e9, params_car['min_acc'], params_car['min_steer'] ])
    ocp.constraints.ubx = np.array([1e9, 1e9, 1e9, params_car['max_v'], 1e9, 1e9, params_car['max_acc'], params_car['max_steer'] ])
    ocp.constraints.idxbx = np.arange(8)  # vx, delta, acceleration, steer_Rate

    ocp.constraints.idxbx_0 = np.arange(8) # IMPORTANT FOR RUNTIME
    ocp.constraints.lbx_0 = np.zeros(8)           # placeholder
    ocp.constraints.ubx_0 = np.zeros(8)           # placeholder

    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

    ocp.constraints.lbu = np.array([-params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([1])

    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 5
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 200 # Sub-steps per dt

    # OPTIMIZATION 7: Relaxed tolerances for speed
    # ocp.solver_options.qp_solver_tol_stat = 1e-4              # Relaxed from 1e-8
    # ocp.solver_options.qp_solver_tol_eq = 1e-4
    # ocp.solver_options.qp_solver_tol_ineq = 1e-4
    # ocp.solver_options.qp_solver_tol_comp = 1e-4
    
    ocp.solver_options.qp_solver_iter_max = 50                 # Limit QP iterations
    ocp.solver_options.print_level = 0                         # No printing
    # ocp.solver_options.qp_solver_warm_start = 2     


    # ocp.solver_options.hpipm_mode = 'SPEED' 

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
        
        f1tenth_model = export_model(p_car, linear = False)
        ocp = create_ocp(f1tenth_model, p_car, steps, horizon)
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=build)
        
        print(f"Solver generated successfully: {full_json_path}")
        return solver
    except Exception as e:
        print(f"Error generating solver: {e}")
        raise
    finally:
        os.chdir(original_cwd) 

def main(args=None):
    solver = setup_mpc(build=True)

if __name__ == '__main__':
    main()
