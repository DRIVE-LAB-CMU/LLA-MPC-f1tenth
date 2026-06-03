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
        eps = .1
        vx_dyn = ca.sqrt(x[3]**2 + eps)

        # eps_v = 0.05   # Transition smoothness
        # v_min = 0.5    # The "floor" velocity

        # # 1. Smoothly clip vx so it never goes below v_min
        # # This is a C2-continuous approximation of max(vx, v_min)
        # vx_dyn = 0.5 * (x[3] + v_min + ca.sqrt((x[3] - v_min)**2 + eps_v))

        # 2. Kinematic model targets (Valid at v ~ 0)
        # Use small-angle approximation for beta to eliminate the dangerous ca.tan() singularity.
        # At low speeds, this is highly accurate and unconditionally safe for the solver.
        beta = (lr / (lf + lr)) * x[7] 

        kin_dx4 = 0.0
        kin_dx5 = (x[3] * ca.sin(beta)) / lr

        # 3. Dynamic Pacejka model (Valid at v > 0.5)
        # Use vx_safe here to prevent NaNs when v = 0
        alphaf = x[7] - ca.atan2((x[5] * lf + x[4]), vx_dyn)
        alphar = ca.atan2((x[5] * lr - x[4]), vx_dyn)

        Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
        Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

        dyn_dx4 = (Fry + Ffy * ca.cos(x[7])) / mass - x[3] * x[5] + g * ca.sin(p[10])
        dyn_dx5 = (lf * Ffy * ca.cos(x[7]) - lr * Fry) / Iz

        # 4. Smoother Blending Function
        # Soften the transition so the Jacobian doesn't spike. 
        # At x[3] = 0, weight_dyn is effectively 0.0, so the solver only "sees" kinematics.
        weight_dyn = 0.5 * (1.0 + ca.tanh(5.0 * (x[3] - 0.5))) 
        weight_kin = 1.0 - weight_dyn

        v_offset = x[3] - 1.0
        weight_dyn = v_offset / ca.sqrt(1 + v_offset**2) # Ranges -1 to 1
        weight_dyn = 0.5 * (1.0 + weight_dyn)            # Ranges 0 to 1
        weight_kin = 1.0 - weight_dyn

        # 4. Pure Algebraic Blending (No if_else, no tanh)
        # v_trans = 0.5    # Midpoint of transition (m/s)
        # eps_b = 0.05     # Sharpness of transition (smaller = sharper)
        # v_diff = x[3] - v_trans

        # # This creates a smooth 0.0 -> 1.0 ramp without transcendental functions
        # # Formula: 0.5 * (1 + x / sqrt(eps + x^2))
        # weight_dyn = 0.5 * (1 + v_diff / ca.sqrt(eps_b + v_diff**2))
        # weight_kin = 1.0 - weight_dyn

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
    # if not linear:
    #     print("NONLINEAR MODEL USED")

    #     # --- 1. Smooth velocity ---
    #     eps_v = 0.01
    #     vx_abs = ca.sqrt(x[3]**2 + eps_v)
    #     v_min  = 0.5
    #     vx_dyn = 0.5 * (vx_abs + v_min + ca.sqrt((vx_abs - v_min)**2 + eps_v))

    #     # --- 2. Kinematic model (valid at low speed) ---
    #     beta = (lr / (lf + lr)) * x[7]  # small-angle approximation
    #     kin_dx4 = 0.0
    #     kin_dx5 = (vx_dyn * ca.sin(beta)) / lr

    #     # --- 3. Dynamic Pacejka model ---
    #     alphaf = x[7] - ca.atan2((x[5]*lf + x[4]), vx_dyn)
    #     alphar = ca.atan2((x[5]*lr - x[4]), vx_dyn)

    #     Ffy_raw = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
    #     Fry_raw = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

    #     # Force scaling to avoid flipping at vx~0
    #     force_scaling = vx_abs / vx_dyn
    #     Ffy = Ffy_raw * force_scaling
    #     Fry = Fry_raw * force_scaling

    #     dyn_dx4 = (Fry + Ffy * ca.cos(x[7])) / mass - vx_dyn * x[5] + g * ca.sin(p[10])
    #     dyn_dx5 = (lf * Ffy * ca.cos(x[7]) - lr * Fry) / Iz

    #     # --- 4. Smooth blending ---
    #     v_trans = 0.5  # transition velocity
    #     eps_b = 0.05   # sharpness
    #     v_diff = vx_dyn - v_trans
    #     weight_dyn = 0.5 * (1 + v_diff / ca.sqrt(eps_b + v_diff**2))
    #     weight_kin = 1 - weight_dyn

    #     # --- 5. Equations of motion ---
    #     dx0 = x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2])
    #     dx1 = x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2])
    #     dx2 = x[5]

    #     # Optional: rolling/aero drag
    #     Frx = mass * x[6]*(p[8] - p[9]*x[3])
    #     dx3 = (Frx - Ffy*ca.sin(x[7]))/mass + x[4]*x[5] - g*ca.sin(p[11])

    #     dx4 = weight_kin*kin_dx4 + weight_dyn*dyn_dx4
    #     dx5 = weight_kin*kin_dx5 + weight_dyn*dyn_dx5

    #     dx6 = u[0]
    #     dx7 = u[1]

    # if not linear:
    #     print("NONLINEAR MODEL USED")

    #     # --- 1. Extract states for clarity ---
    #     vx = x[3]
    #     vy = x[4]
    #     omega = x[5]
    #     a = x[6]
    #     delta = x[7]

    #     # --- 2. Compute slip angles ---
    #     alphaf = delta - ca.atan2(omega*lf + vy, vx)
    #     alphar = ca.atan2(omega*lr - vy, vx)

    #     # --- 3. Lateral forces using Pacejka tire model ---
    #     Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
    #     Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

    #     # --- 4. Longitudinal force (rolling + aero drag) ---
    #     Frx = mass * a * (p[8] - p[9]*vx)

    #     # --- 5. Equations of motion ---
    #     dx0 = vx * ca.cos(x[2]) - vy * ca.sin(x[2])      # x_dot
    #     dx1 = vx * ca.sin(x[2]) + vy * ca.cos(x[2])      # y_dot
    #     dx2 = omega                                      # yaw_dot
    #     dx3 = (Frx - Ffy*ca.sin(delta))/mass + vy*omega - g*ca.sin(p[11])   # vx_dot
    #     dx4 = (Fry + Ffy*ca.cos(delta))/mass - vx*omega + g*ca.sin(p[10])                      # vy_dot
    #     dx5 = (lf*Ffy*ca.cos(delta) - lr*Fry)/Iz                               # omega_dot
    #     dx6 = u[0]  # jerk
    #     dx7 = u[1]  # steer rate

    #     # --- 6. Stack derivatives ---
    #     f_expl = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)
    # if not linear:
    #     print("NONLINEAR MODEL USED")

    #     # --- 1. Extract states for clarity ---
    #     vx = x[3]
    #     vy = x[4]
    #     omega = x[5]
    #     a = x[6]
    #     delta = x[7]

    #     # --- 1b. Low-speed protection ---
    #     vmin = 0.05
    #     vy    = ca.if_else(vx < vmin, 0, vy)
    #     omega = ca.if_else(vx < vmin, 0, omega)
    #     delta = ca.if_else(vx < vmin, 0, delta)
    #     vx    = ca.if_else(vx < vmin, vmin, vx)

    #     # --- 2. Compute slip angles ---
    #     alphaf = delta - ca.atan2(omega*lf + vy, vx)
    #     alphar = ca.atan2(omega*lr - vy, vx)

    #     # --- 3. Lateral forces using Pacejka tire model ---
    #     Ffy = p[4] * ca.sin(p[2] * ca.atan(p[0] * alphaf))
    #     Fry = p[5] * ca.sin(p[3] * ca.atan(p[1] * alphar))

    #     # --- 4. Longitudinal force (rolling + aero drag) ---
    #     Frx = mass * a * (p[8] - p[9]*vx)

    #     # --- 5. Equations of motion ---
    #     dx0 = vx * ca.cos(x[2]) - vy * ca.sin(x[2])              # x_dot
    #     dx1 = vx * ca.sin(x[2]) + vy * ca.cos(x[2])              # y_dot
    #     dx2 = omega                                              # yaw_dot
    #     dx3 = (Frx - Ffy*ca.sin(delta))/mass + vy*omega - g*ca.sin(p[11])  # vx_dot
    #     dx4 = (Fry + Ffy*ca.cos(delta))/mass - vx*omega + g*ca.sin(p[10])  # vy_dot
    #     dx5 = (lf*Ffy*ca.cos(delta) - lr*Fry)/Iz                             # omega_dot
    #     dx6 = u[0]  # jerk
    #     dx7 = u[1]  # steer rate

    #     # --- 6. Stack derivatives ---
    #     f_expl = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)

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
    model.a_long_expr = dx3

    # print("\n--- CASADI PRE-SOLVE DIAGNOSTIC ---")
    # # Create a temporary CasADi function to evaluate your dynamics
    # try:
    #     test_dyn_func = ca.Function('test_dyn', [model.x, model.u, model.p], [model.f_expl_expr])
        
    #     # Setup a realistic initial condition (e.g., vx = 5.0 m/s to avoid division by zero)
    #     x_test = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]) # x, y, yaw, vx, vy, omega, a, steer
    #     u_test = np.array([0.0, 0.0]) # jerk, steer_rate
        
    #     # Generate dummy parameters (Make sure this perfectly matches the length of model.p!)
    #     # Assuming p has your params + 8 x_ref values
    #     p_len = model.p.shape[0]
    #     p_test = np.ones(p_len) * 0.1 
    #     p_test[-8:] = x_test # Set x_ref to match x_test
        
    #     # Evaluate
    #     dx_eval = test_dyn_func(x_test, u_test, p_test)
    #     print("Dynamics evaluated at v_x = 5.0:")
    #     print(dx_eval)
        
    #     if np.any(np.isnan(dx_eval)) or np.any(np.isinf(dx_eval)):
    #         print("CRITICAL: Your CasADi dynamics evaluate to NaN/Inf. The solver is dead on arrival.")
    #     else:
    #         print("CasADi math is sound. The issue is in the Acados QP/Constraints.")
    # except Exception as e:
    #     print(f"CasADi Function creation failed: {e}")
    # print("-----------------------------------\n")

    return model

def create_ocp(model, params_car, steps, horizon):
    ocp = AcadosOcp()
    ocp.model = model

    N = steps #steps
    Tf = horizon # total time horizon

    ocp.solver_options.tf = Tf #set solver settings


    # N = steps #steps
    # Tf = horizon # total time horizon
    nx = model.x.size()[0]  # This MUST be 8
    nu = model.u.size()[0]  # This is 2
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    w_x = 2.0
    w_y = 2.0
    w_theta = 0.0
    w_vel = 0.0
    
    w_xe = 0
    w_ye = 0
    
    w_pwm = 1.0
    w_steer = 0.01
    
    w_slew = 0.0
    w_steer_v = 0.1
    w_a_long = 0.1
      
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
    
    ocp.constraints.lbu = np.array([-0.2, -params_car['max_steer_vel']])
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
    
    ###############################################3

    
    


    
    # ocp.constraints.lbu = np.array([-params_car['max_steer_vel']])
    # ocp.constraints.ubu = np.array([params_car['max_steer_vel']])
    # ocp.constraints.idxbu = np.array([1]) # 0 is jerk, 1 is steer_vel

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    
    # DROPPED FROM 10 to 2 (This makes the solver ~5x faster)
    ocp.solver_options.sim_method_num_steps = 10

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_warm_start = 1    

    # STABILITY FIXES
    ocp.solver_options.levenberg_marquardt = 1e-2  # Increased damping
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
