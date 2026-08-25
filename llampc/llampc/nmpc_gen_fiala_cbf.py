"""F1TENTH Fiala/brush OCP with obstacle avoidance as CBF path constraints.

PARAMETER LAYOUT (per stage):
  p = [ Cf, Cr, muf, mur, Cro | x_ref(6) | (ox, oy, r, active) * N_OBS_SLOTS ]
        0   1   2    3    4     5..10      11 ...
`active` in {0,1} disables a slot ALGEBRAICALLY.  Never disable a slot by
parking it far away: a parked obstacle is a real point in the world, and since
psi = hdot + alpha1*h its influence radius is 2*v/alpha1 (metres), so a
"parked" obstacle brakes the car whenever it drives near that coordinate.
"""

import os
import warnings
from pathlib import Path

import casadi as ca
import numpy as np

from llampc.params import F110

N_OBS_SLOTS  = 8         # max simultaneous obstacles (fixed at codegen)
P_TIRE       = 5         # [Cf, Cr, muf, mur, Cro]
P_XREF       = 6         # [x, y, phi, vx, vy, omega]
P_OBS_STRIDE = 4         # (ox, oy, r, active)

OBS_PARK = (0.0, 0.0, 0.0, 0.0)   # active=0 -> disabled, no phantom geometry
OBS_OFF  = 1e3                    # value of a disabled row (must be < uh)

IDX_XREF = P_TIRE                 # 5
IDX_OBS  = P_TIRE + P_XREF        # 11

# 'Im' is optional: llampc's F110 has none, and its own model divides by Iw.
_REQUIRED_KEYS = ('mass', 'Iz', 'lf', 'lr', 'rw', 'Iw', 'pole_pairs',
                  'gear_ratio', 'lambda', 'min_steer', 'max_steer',
                  'max_steer_vel', 'max_v')

DEFAULT_CAR_WIDTH = 0.30          # F110 has no 'width' key


def n_params(n_obs=N_OBS_SLOTS):
    return P_TIRE + P_XREF + P_OBS_STRIDE * n_obs


def check_params(params_car):
    missing = [k for k in _REQUIRED_KEYS if k not in params_car]
    if missing:
        raise KeyError(
            f"params_car is missing required keys: {missing}. "
            f"Present: {sorted(params_car)}. ('Im' is optional and defaults to "
            f"0.0; Cf/Cr/muf/mur/Cro are RUNTIME values passed through p.)")


def default_body_discs(params_car, car_width=None):
    """Two covering discs at the front and rear axle: [(a_long, radius), ...].
    radius = half-width, so the pair covers a rectangle of length
    (lf + lr + width) and width `car_width`."""
    if car_width is None:
        car_width = float(params_car.get('width', DEFAULT_CAR_WIDTH))
        if 'width' not in params_car:
            warnings.warn(f"car_width not given and params_car has no 'width'; "
                          f"using {DEFAULT_CAR_WIDTH} m. THIS IS THE OBSTACLE "
                          f"SAFETY MARGIN -- pass your measured width.",
                          stacklevel=2)
    lf, lr = params_car['lf'], params_car['lr']
    r = 0.5 * car_width
    return [(float(lf), r), (float(-lr), r)]


# ── Dynamics ───────────────────────────────────────────────────────────
def dynamics_expr(x, u, p_tire, params_car, exact=False):
    """xdot for the 9-state model.
    x = [px, py, phi, vx, vy, omega, omega_w, current, delta]
    u = [d(current)/dt, d(delta)/dt];  p_tire = [Cf, Cr, muf, mur, Cro]"""
    mass, Iz = params_car['mass'], params_car['Iz']
    lf, lr   = params_car['lf'], params_car['lr']
    rw, Iw   = params_car['rw'], params_car['Iw']
    Im       = float(params_car.get('Im', 0.0))
    pole_pairs = params_car['pole_pairs']
    gear_ratio = params_car['gear_ratio']
    lam = params_car['lambda']
    g, L = 9.81, lf + lr

    Cf, Cr, muf, mur, Cro = [p_tire[i] for i in range(P_TIRE)]
    omega_w, current, delta = x[6], x[7], x[8]

    Ffz, Frz = mass*g*lr/L, mass*g*lf/L
    Ffmax, Frmax = muf*Ffz, mur*Frz

    if not exact:
        eps = 0.1
        vx_dyn = ca.sqrt(x[3]**2 + eps)
        alphaf = delta - ca.atan2(x[5]*lf + x[4], vx_dyn)
        alphar = ca.atan2(x[5]*lr - x[4], vx_dyn)
        vx_front = x[3]*ca.cos(delta) + (x[4] + lf*x[5])*ca.sin(delta)
        vx_dyn_f = ca.sqrt(vx_front**2 + eps)
        kappa_r = (rw*omega_w - x[3]) / vx_dyn
        kappa_f = (rw*omega_w - vx_front) / vx_dyn_f
        soft = 1e-2
    else:
        alphaf = delta - ca.atan2(x[5]*lf + x[4], x[3])
        alphar = ca.atan2(x[5]*lr - x[4], x[3])
        kappa_r = (rw*omega_w - x[3]) / x[3]
        vx_front = x[3]*ca.cos(delta) + (x[4] + lf*x[5])*ca.sin(delta)
        kappa_f = (rw*omega_w - vx_front) / vx_front
        soft = 1e-9

    sigma_f = ca.sqrt(ca.tan(alphaf)**2 + kappa_f**2 + soft)
    sigma_r = ca.sqrt(ca.tan(alphar)**2 + kappa_r**2 + soft)

    def brush(C, s, Fmax):
        Cs = C*s
        return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)

    Ff = ca.if_else(sigma_f < 3*Ffmax/Cf, brush(Cf, sigma_f, Ffmax), Ffmax)
    Fr = ca.if_else(sigma_r < 3*Frmax/Cr, brush(Cr, sigma_r, Frmax), Frmax)

    Ffy = Ff*ca.tan(alphaf)/sigma_f
    Ffx = Ff*kappa_f       /sigma_f
    F_roll = Cro * Frz * ca.tanh(x[3] / 0.5)
    Frx = Fr*kappa_r/sigma_r - F_roll
    Fry = Fr*ca.tan(alphar)/sigma_r
    tau_drive = gear_ratio * 1.5 * pole_pairs * lam * current

    return ca.vertcat(
        x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2]),
        x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2]),
        x[5],
        (Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta))/mass + x[4]*x[5],
        (Fry + Ffy*ca.cos(delta) + Ffx*ca.sin(delta))/mass - x[3]*x[5],
        (lf*(Ffy*ca.cos(delta) + Ffx*ca.sin(delta)) - lr*Fry)/Iz,
        (tau_drive - Frx*rw - Ffx*rw)/(Iw + Im),
        u[0], u[1])


def export_model(params_car, exact=False, n_obs=N_OBS_SLOTS):
    from acados_template import AcadosModel
    check_params(params_car)

    model = AcadosModel()
    model.name = "f1tenthfiala_cbf"

    x = ca.MX.sym('x', 9)
    u = ca.MX.sym('u', 2)
    p_tire = ca.MX.sym('p', P_TIRE)
    x_ref  = ca.MX.sym('x_ref', P_XREF)
    p_obs  = ca.MX.sym('p_obs', P_OBS_STRIDE * n_obs)

    f_expl = dynamics_expr(x, u, p_tire, params_car, exact=exact)
    xdot = ca.MX.sym('xdot', 9)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl      # required by IRK
    model.xdot = xdot
    model.x, model.u = x, u
    model.p = ca.vertcat(p_tire, x_ref, p_obs)
    return model


# ── Obstacle CBF expressions ───────────────────────────────────────────
def obstacle_expressions(x, p, n_obs, body_discs, alpha1=None):
    """h_ij   = ||c_j - o_i||^2 - (r_i + rho_j)^2          >= 0
       psi_ij = d/dt h_ij + alpha1*h_ij                    >= 0
    Disc velocity includes the rigid-body transport term omega x a_j, so psi is
    exact for the body, not just the CG.  active=0 replaces the row by OBS_OFF."""
    px, py, phi = x[0], x[1], x[2]
    vx, vy, omega = x[3], x[4], x[5]
    cphi, sphi = ca.cos(phi), ca.sin(phi)

    h_list, psi_list = [], []
    for (a, rho) in body_discs:
        cx, cy = px + a*cphi, py + a*sphi
        vX = vx*cphi - vy*sphi - a*omega*sphi
        vY = vx*sphi + vy*cphi + a*omega*cphi
        for i in range(n_obs):
            ox  = p[IDX_OBS + P_OBS_STRIDE*i + 0]
            oy  = p[IDX_OBS + P_OBS_STRIDE*i + 1]
            rr  = p[IDX_OBS + P_OBS_STRIDE*i + 2]
            act = p[IDX_OBS + P_OBS_STRIDE*i + 3]
            dx_, dy_ = cx - ox, cy - oy
            h_raw = dx_*dx_ + dy_*dy_ - (rr + rho)**2
            h_list.append(act*h_raw + (1.0 - act)*OBS_OFF)
            if alpha1 is not None:
                hdot = 2.0*(dx_*vX + dy_*vY)
                psi_list.append(act*(hdot + alpha1*h_raw)
                                + (1.0 - act)*OBS_OFF)
    return ca.vertcat(*h_list), (ca.vertcat(*psi_list) if psi_list else None)


# ── OCP ────────────────────────────────────────────────────────────────
def create_ocp(model, params_car, steps, horizon,
               n_obs=N_OBS_SLOTS, body_discs=None, car_width=None,
               cbf_alpha=2.0, w_slack_obs=1e4, w_slack_obs_l1=1e3,
               w_slack_psi=1e2, w_slack_psi_l1=1e1,
               integrator='IRK', sim_method_num_steps=2,
               nlp_solver_type='SQP', nlp_solver_max_iter=20,
               globalization=None):
    from acados_template import AcadosOcp
    check_params(params_car)

    ocp = AcadosOcp()
    ocp.model = model
    N, Tf = steps, horizon
    nx, nu = model.x.size()[0], model.u.size()[0]
    np_ = model.p.size()[0]
    assert np_ == n_params(n_obs), f'param size {np_} != {n_params(n_obs)}'

    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    if body_discs is None:
        body_discs = default_body_discs(params_car, car_width)

    # ── cost (unchanged tuning) ───────────────────────────────────────
    ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'

    w_x = w_y = 40.0
    w_xe = w_ye  = 40.0
    w_theta=0.0
    w_vx=10.0
    w_omega =1.0
    w_current, w_steer = 0.01, 0.5
    w_slew, w_steer_v = 0.0, 0.5

    Q_flat = [w_x, w_y, w_theta, w_vx, 0.0, w_omega, w_current, w_steer]
    R_flat = [w_slew, w_steer_v]
    ocp.cost.W = np.diag(np.concatenate((Q_flat, R_flat)))
    ocp.cost.W_e = np.diag([w_xe, w_ye, 0, 0, 0, 0, 0, 0])

    x, u = model.x, model.u
    x_ref = model.p[IDX_XREF:IDX_XREF + P_XREF]   # NOT p[-6:] any more!

    yaw_err = x[2] - x_ref[2]
    yaw_err_wrapped = ca.atan2(ca.sin(yaw_err), ca.cos(yaw_err))
    y_common = ca.vertcat(x[0]-x_ref[0], x[1]-x_ref[1], yaw_err_wrapped,
                          x[3]-x_ref[3], x[4]-x_ref[4], x[5]-x_ref[5],
                          x[7], x[8])
    ocp.model.cost_y_expr   = ca.vertcat(y_common, u)
    ocp.model.cost_y_expr_e = y_common
    ocp.cost.yref   = np.zeros(10)
    ocp.cost.yref_e = np.zeros(8)

    ocp.model.p = model.p
    # Defaults MUST have every slot disabled, else the first solve sees a
    # radius-0 obstacle at the origin and is infeasible.
    ocp.parameter_values = pack_params(np.ones(P_TIRE), np.zeros(P_XREF),
                                       [], n_obs=n_obs)

    # ── box constraints ───────────────────────────────────────────────
    ocp.constraints.idxbx = np.array([3, 4, 5, 7, 8])
    ocp.constraints.lbx = np.array([-0.5, -4, -2*np.pi, -25,
                                    params_car['min_steer']])
    ocp.constraints.ubx = np.array([params_car['max_v'], 4, 2*np.pi, 50,
                                    params_car['max_steer']])
    ocp.constraints.lbu = np.array([-300, -params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([300, params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0 = np.zeros(nx)
    ocp.constraints.ubx_0 = np.zeros(nx)
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

    # ── obstacle CBF path constraints ─────────────────────────────────
    h_expr, psi_expr = obstacle_expressions(x, model.p, n_obs, body_discs,
                                            alpha1=cbf_alpha)
    n_h = h_expr.size()[0]
    n_psi = 0 if psi_expr is None else psi_expr.size()[0]
    con = h_expr if psi_expr is None else ca.vertcat(h_expr, psi_expr)
    nh, BIG = n_h + n_psi, 1e9

    ocp.model.con_h_expr   = con
    ocp.constraints.lh     = np.zeros(nh)
    ocp.constraints.uh     = BIG*np.ones(nh)
    ocp.model.con_h_expr_e = con
    ocp.constraints.lh_e   = np.zeros(nh)
    ocp.constraints.uh_e   = BIG*np.ones(nh)

    # ── slacks: acados orders weights [sbu, sbx, sg, sh, sphi] ────────
    nsbx = 5
    ocp.constraints.idxsbx = np.arange(nsbx)
    ocp.constraints.idxsh  = np.arange(nh)
    obs_l1 = np.concatenate([w_slack_obs_l1*np.ones(n_h),
                             w_slack_psi_l1*np.ones(n_psi)])
    obs_l2 = np.concatenate([w_slack_obs*np.ones(n_h),
                             w_slack_psi*np.ones(n_psi)])
    zl = np.concatenate([10.0*np.ones(nsbx), obs_l1])
    Zl = np.concatenate([100.0*np.ones(nsbx), obs_l2])
    ocp.cost.zl, ocp.cost.zu = zl, zl.copy()
    ocp.cost.Zl, ocp.cost.Zu = Zl, Zl.copy()

    ocp.constraints.idxsbx_e = np.arange(nsbx)
    ocp.constraints.idxsh_e  = np.arange(nh)
    ocp.cost.zl_e, ocp.cost.zu_e = zl.copy(), zl.copy()
    ocp.cost.Zl_e, ocp.cost.Zu_e = Zl.copy(), Zl.copy()

    # ── solver options ────────────────────────────────────────────────
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    if integrator == 'ERK':
        n_st = sim_method_num_steps or max(5, int(np.ceil((Tf/N)/0.002)))
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = n_st
    else:
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_num_stages = 3
        ocp.solver_options.sim_method_num_steps = sim_method_num_steps or 2

    ocp.solver_options.nlp_solver_type = nlp_solver_type
    if nlp_solver_max_iter is not None:
        ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    if globalization is not None:
        ocp.solver_options.globalization = globalization
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.levenberg_marquardt = 1e-4
    # NOTE: `regularize_hessian` is NOT an acados option -- setting it silently
    # created a stray attribute and regularization stayed NO_REGULARIZE.
    ocp.solver_options.hpipm_mode = 'SPEED'
    return ocp


# ── Runtime plumbing ───────────────────────────────────────────────────
def pack_params(tire, x_ref_k, obstacles, n_obs=N_OBS_SLOTS, xy=None):
    """tire (5,), x_ref_k (6,), obstacles = iterable of (ox, oy, r).
    Unused slots are disabled via the activation flag.  If more than n_obs are
    given, the n_obs NEAREST to `xy` are kept and a warning issued -- never a
    silent drop."""
    obs = [np.asarray(o, float).ravel() for o in obstacles]
    for o in obs:
        if o.size != 3:
            raise ValueError(f"each obstacle must be (ox, oy, r); got size "
                             f"{o.size}. For moving obstacles use "
                             f"set_horizon_params.")
    if len(obs) > n_obs:
        if xy is None:
            xy = np.asarray(x_ref_k, float).ravel()[:2]
        obs = sorted(obs, key=lambda o: float(
            np.hypot(o[0]-xy[0], o[1]-xy[1]) - o[2]))[:n_obs]
        warnings.warn(f"pack_params: {len(obstacles)} obstacles > {n_obs} "
                      f"slots; kept the {n_obs} nearest.", stacklevel=2)
    slots  = [np.array([o[0], o[1], o[2], 1.0]) for o in obs]
    slots += [np.asarray(OBS_PARK, float)] * (n_obs - len(slots))
    out = np.concatenate([np.asarray(tire, float).ravel(),
                          np.asarray(x_ref_k, float).ravel(),
                          np.concatenate(slots) if n_obs else np.zeros(0)])
    if out.size != n_params(n_obs):
        raise ValueError(f"pack_params produced {out.size}, expected "
                         f"{n_params(n_obs)}")
    return out


def set_horizon_params(solver, N, tire, ref_traj, obstacles,
                       n_obs=N_OBS_SLOTS, per_stage=None):
    """`obstacles` is a flat list of (ox,oy,r) (static) or a length-(N+1) list
    of such lists (moving -- one predicted set per stage).  per_stage=None
    auto-detects; pass True/False to be explicit."""
    if per_stage is None:
        if len(obstacles) == 0:
            per_stage = False
        else:
            a0 = np.asarray(obstacles[0], dtype=float)
            per_stage = (len(obstacles) == N + 1
                         and (a0.ndim >= 2 or a0.size == 0))
    for k in range(N + 1):
        obs_k = obstacles[k] if per_stage else obstacles
        solver.set(k, "p", pack_params(tire, ref_traj[k], obs_k, n_obs,
                                       xy=np.asarray(ref_traj[k], float)[:2]))


def break_symmetry(obstacles, x, eps=5e-3):
    """Nudge obstacles sitting EXACTLY on the vehicle heading axis.  See
    gotcha #2 -- a head-on obstacle is a saddle and the solver never picks a
    side.  The nudge is perpendicular, a few mm, deterministic."""
    px, py, phi = float(x[0]), float(x[1]), float(x[2])
    c, s = np.cos(phi), np.sin(phi)
    out = []
    for o in obstacles:
        ox, oy, r = float(o[0]), float(o[1]), float(o[2])
        dx_, dy_ = ox - px, oy - py
        ahead = dx_*c + dy_*s
        lat   = -dx_*s + dy_*c
        if ahead > 0.0 and abs(lat) < eps:
            sgn = 1.0 if lat >= 0.0 else -1.0
            ox += -s*sgn*eps
            oy +=  c*sgn*eps
        out.append((ox, oy, r))
    return out


def default_tire_params(params_car, mu=None):
    """Usable [Cf, Cr, muf, mur, Cro] before the estimator is running.  These
    are RUNTIME quantities; F110 holds only a single 'mu'.  Cf/Cr set so the
    brush model saturates near 30% combined slip."""
    if mu is None:
        mu = float(params_car.get('mu', 0.8))
    lf, lr = params_car['lf'], params_car['lr']
    Ffmax = mu * params_car['mass'] * 9.81 * lr / (lf + lr)
    Cf = 3.0 * Ffmax / 0.30
    return np.array([Cf, Cf, mu, mu, 0.02])


def get_solver_directory(solver_config="default"):
    d = Path(__file__).parent.resolve() / 'solvers' / solver_config
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_mpc(steps, horizon, json_file='f1tenth_cbf_ocp.json',
              solver_config="default", build=True, params_car=None,
              n_obs=N_OBS_SLOTS, cbf_alpha=2.0, car_width=None, **ocp_kw):
    from acados_template import AcadosOcpSolver
    if params_car is None:
        params_car = F110
    p_car = params_car() if callable(params_car) else params_car
    check_params(p_car)

    solver_dir = get_solver_directory(solver_config)
    cwd = os.getcwd()
    try:
        os.chdir(solver_dir)
        model = export_model(p_car, exact=False, n_obs=n_obs)
        ocp = create_ocp(model, p_car, steps, horizon, n_obs=n_obs,
                         cbf_alpha=cbf_alpha, car_width=car_width, **ocp_kw)
        return AcadosOcpSolver(ocp, json_file=json_file, build=build)
    finally:
        os.chdir(cwd)