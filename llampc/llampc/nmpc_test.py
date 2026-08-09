import os, re, ast
import numpy as np
import casadi as cs
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
            Aineq.T.reshape(-1, 1), # Kept for byte-compatibility (evaluates to empty)
            bineq.T.reshape(-1, 1), # Kept for byte-compatibility (evaluates to empty)
            p_tire_val.reshape(-1, 1)
        ])
        arg['lbx'] = -np.inf * np.ones(n_states * (horizon + 1) + n_inputs * horizon)
        arg['ubx'] =  np.inf * np.ones(n_states * (horizon + 1) + n_inputs * horizon)
        arg['lbg'] = np.concatenate([np.zeros(n_states * (horizon + 1)), -np.inf * np.ones(horizon * 6)])
        arg['ubg'] = np.concatenate([np.zeros(n_states * (horizon + 1)),  np.zeros(horizon * 6)])

        res  = problem(**arg)
        
        # Extract and reshape the control inputs for the full horizon
        umpc = res['x'][n_states * (horizon + 1):n_states * (horizon + 1) + n_inputs * horizon].full().reshape(horizon, n_inputs).T
        
        # Return a dictionary with the specific control to apply right now
        return {
            'u': umpc[:, 0]
        }


# =============================================================================
# 3. PARSER
# =============================================================================
def parse_mpc_log_file(file_path):
    with open(file_path, 'r') as f:
        raw = f.read().replace('\xa0', ' ')
    cases = []
    for block in raw.split("checkiter")[1:]:
        try:
            x0 = np.fromstring(re.search(r"x0:\[(.*?)\]", block).group(1).replace('\n', ' '), sep=' ')
            m  = re.search(r"xref:\[\[(.*?)]\s*\[(.*?)]\]", block, re.DOTALL)
            xref  = np.vstack([np.fromstring(m.group(1).replace('\n', ' '), sep=' '),
                               np.fromstring(m.group(2).replace('\n', ' '), sep=' ')])
            uprev = np.fromstring(re.search(r"uprev:\[(.*?)\]", block).group(1).replace('\n', ' '), sep=' ')
            params = ast.literal_eval(re.search(r"params(\{.*?\})", block).group(1))
            t = re.search(r"total\s*\|.*?\(\s*([\d\.]+)ms\)", block)
            logged_ms = float(t.group(1)) if t else 0.0
            cases.append({'params_car': params, 'x0': x0, 'xref': xref, 'uprev': uprev, 'logged_ms': logged_ms})
        except Exception:
            pass
    return cases


# =============================================================================
# 4. EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    cases = parse_mpc_log_file("mpc_log.txt")
    if not cases:
        print("No cases parsed."); raise SystemExit

    steps = cases[0]['xref'].shape[1] - 1
    Ts = 0.02
    Q = np.diag([1.0, 1.0]); R = np.diag([5e-3, 1.0])

    # Extract base static params from the first logged case to build the single solver
    base_params = cases[0]['params_car']
    solve_fn = create_ipopt_solver(steps, Ts, Q, R, base_params)

    print(f"{'step':>4} | {'logged ms':>9} | {'yours ms':>8} | {'iters':>5} | status")
    print("-" * 60)
    
    iters_all, mine_all = [], []
    for i, c in enumerate(cases):
        
        # Package the varying tire parameters dynamically for this step
        pc = c['params_car']
        tire_val = np.array([pc['Df'], pc['Dr'], pc['Bf'], pc['Br'], pc['Cf'], pc['Cr']])
        
        # Feed the packaged parameters directly into the closure solver
        umpc, fval, xmpc, _ = solve_fn(c['x0'], c['xref'], c['uprev'], tire_val)
        
        s = solve_fn.problem.stats()
        it = s.get('iter_count', -1)
        mine = s.get('t_wall_total', 0.0) * 1000.0
        
        iters_all.append(it); mine_all.append(mine)
        print(f"{i:>4} | {c['logged_ms']:>9.2f} | {mine:>8.2f} | {it:>5} | {s.get('return_status','?')}")
        
    print("-" * 60)
    print(f"mean iters {np.mean(iters_all):.1f} | mean yours {np.mean(mine_all):.1f}ms | "
          f"mean logged {np.mean([c['logged_ms'] for c in cases]):.1f}ms")