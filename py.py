"""
Wingbox topological sweep & structural optimization.
Python translation of the original MATLAB script.

Requires the airfoil coordinate files in the working directory:
    'NACA 0014.dat', 'NACA 0012.dat', 'NACA 0009.dat'
(two-column x/y tables, one header line, upper surface TE->LE followed by
lower surface LE->TE, matching what MATLAB's fscanf('%f %f') expects.)

Notes on indexing:
MATLAB is 1-based; every boom/panel index below has been shifted to 0-based.
The 21-variable section optimizer always assumes 10 booms / 11 panels, which
matches the fixed 5-point upper-surface discretization used by the DoE.

Notes on preserved (not "fixed") behavior:
Exactly as in the original script, X_boom/Y_boom/Z_boom and savedBoomsCoord
(and therefore V_targets / M_flex_local / M_tors_local) are left over from
the LAST DoE loop iteration (case 27) when "Configuration 7" is forced for
final export -- only the spar-position-derived quantities (x_norm_booms,
airfoils, ndsa/fuel distribution) are actually rebuilt for config 7. This
mirrors the MATLAB script's own behavior line-for-line.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from types import SimpleNamespace


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def isa_atmosphere(h):
    """Minimal ICAO standard atmosphere (troposphere + lower stratosphere),
    equivalent to MATLAB's atmosisa for h up to ~20 km."""
    T0, P0 = 288.15, 101325.0
    L, R, g0 = 0.0065, 287.05287, 9.80665
    h_trop = 11000.0
    if h <= h_trop:
        T = T0 - L * h
        P = P0 * (T / T0) ** (g0 / (R * L))
    else:
        T11 = T0 - L * h_trop
        P11 = P0 * (T11 / T0) ** (g0 / (R * L))
        T = T11
        P = P11 * np.exp(-g0 * (h - h_trop) / (R * T11))
    rho = P / (R * T)
    a = np.sqrt(1.4 * R * T)
    return T, a, P, rho


def polyarea(x, y):
    """Shoelace formula, matches MATLAB's polyarea for a simple polygon."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def get_airfoil_coords(filename, x_target):
    """Reads a NACA .dat file (1 header line, 2-column x/y) and returns
    pchip-interpolated upper/lower surface y at x_target."""
    data = np.loadtxt(filename, skiprows=1)
    x_raw = data[:, 0]
    y_raw = data[:, 1]
    idx_le = int(np.argmin(x_raw))

    x_up_raw = x_raw[0:idx_le + 1]
    y_up_raw = y_raw[0:idx_le + 1]
    x_lo_raw = x_raw[idx_le:]
    y_lo_raw = y_raw[idx_le:]

    x_up, iu = np.unique(x_up_raw, return_index=True)
    y_up = y_up_raw[iu]
    x_lo, il = np.unique(x_lo_raw, return_index=True)
    y_lo = y_lo_raw[il]

    y_up_interp = PchipInterpolator(x_up, y_up)(x_target)
    y_lo_interp = PchipInterpolator(x_lo, y_lo)(x_target)
    return y_up_interp, y_lo_interp, x_up, y_up, y_lo


def flatten_panels(y_up, y_lo, x_upper, idx_spars):
    """Straight-line (flattened) skin between consecutive spar stations."""
    y_up_flat = y_up.copy()
    y_lo_flat = y_lo.copy()
    n_up = len(y_up)
    valid_idx = [i for i in idx_spars if i <= n_up - 1]
    for s in range(len(valid_idx) - 1):
        i1, i2 = valid_idx[s], valid_idx[s + 1]
        y_up_flat[i1:i2 + 1] = np.interp(
            x_upper[i1:i2 + 1], [x_upper[i1], x_upper[i2]], [y_up[i1], y_up[i2]])
        y_lo_flat[i1:i2 + 1] = np.interp(
            x_upper[i1:i2 + 1], [x_upper[i1], x_upper[i2]], [y_lo[i1], y_lo[i2]])
    return y_up_flat, y_lo_flat


def calculate_ndsa(airfoil_data, idx_spars):
    x = airfoil_data[:, 0]
    y = airfoil_data[:, 1]
    n = len(x) // 2
    x_up = x[0:n]
    y_up = y[0:n]
    x_lo = np.flipud(x[n:])
    y_lo = np.flipud(y[n:])
    xi = np.linspace(x[idx_spars[0]], x[idx_spars[2]], 200)
    yi_up = PchipInterpolator(x_up, y_up)(xi)
    yi_lo = PchipInterpolator(x_lo, y_lo)(xi)
    return np.trapezoid(yi_up - yi_lo, xi)


def get_ndsa_dist(y_vec, wingGeom):
    ndsa_root = calculate_ndsa(wingGeom.airfoils_root, wingGeom.idx_spars_upper)
    ndsa_kink = calculate_ndsa(wingGeom.airfoils_kink, wingGeom.idx_spars_upper)
    ndsa_tip = calculate_ndsa(wingGeom.airfoils_tip, wingGeom.idx_spars_upper)
    return np.interp(y_vec, [0, wingGeom.y_kink, wingGeom.span],
                      [ndsa_root, ndsa_kink, ndsa_tip])


def get_oblique_lift_limit(z_target, wingGeom):
    z_kink = wingGeom.y_kink
    z_tip = wingGeom.span
    x_LE_kink = z_kink * np.tan(wingGeom.sweep_LE_1)
    x_LE_tip = x_LE_kink + (z_tip - z_kink) * np.tan(wingGeom.sweep_LE_2)
    x_L_root = 0.25 * wingGeom.c_root
    x_L_kink = x_LE_kink + 0.25 * wingGeom.c_kink
    x_L_tip = x_LE_tip + 0.25 * wingGeom.c_tip

    ifs = wingGeom.idx_spars_upper[0]
    irs = wingGeom.idx_spars_upper[2]
    x_EA_root = (wingGeom.airfoils_root[ifs, 0] + wingGeom.airfoils_root[irs, 0]) * wingGeom.c_root / 2
    x_EA_tip = x_LE_tip + (wingGeom.airfoils_tip[ifs, 0] + wingGeom.airfoils_tip[irs, 0]) * wingGeom.c_tip / 2
    m_EA = (x_EA_tip - x_EA_root) / z_tip
    m_normal = -1.0 / m_EA
    x_EA_target = x_EA_root + m_EA * z_target
    m_L1 = (x_L_kink - x_L_root) / z_kink
    m_L2 = (x_L_tip - x_L_kink) / (z_tip - z_kink)

    z_int1 = (x_EA_target - x_L_root - m_normal * z_target) / (m_L1 - m_normal)
    z_int2 = (x_EA_target - x_L_kink - m_normal * z_target + m_L2 * z_kink) / (m_L2 - m_normal)

    z_lift = z_int1 if z_int1 <= (z_kink + 1e-3) else z_int2
    return max(0.0, min(z_lift, z_tip))


# Fixed 0-based boom/panel connectivity (10 booms, 11 panels)
IDX_PANELS = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                        [5, 6], [6, 7], [7, 8], [8, 9], [9, 0], [7, 2]])


def calculate_state(X, Mx, Mz, Vy, x_b, y_b, A1, A2, sigma_adm, tau_adm):
    A_b = X[0:10]
    t_p = X[10:21]

    x_bar = np.sum(A_b * x_b) / np.sum(A_b)
    y_bar = np.sum(A_b * y_b) / np.sum(A_b)
    dx = x_b - x_bar
    dy = y_b - y_bar

    Ixx = np.sum(A_b * dy ** 2)
    Iyy = np.sum(A_b * dx ** 2)
    Ixy = np.sum(A_b * dx * dy)
    den = Ixx * Iyy - Ixy ** 2

    sigma_signed = (Mx * Iyy / den) * dy - (Mx * Ixy / den) * dx
    dq = -(Vy * Iyy / den) * (A_b * dy) + (Vy * Ixy / den) * (A_b * dx)

    L = np.zeros(11)
    for i in range(11):
        i1, i2 = IDX_PANELS[i]
        L[i] = np.sqrt((x_b[i2] - x_b[i1]) ** 2 + (y_b[i2] - y_b[i1]) ** 2)

    qb = np.zeros(11)
    qb[0] = 0
    qb[1] = qb[0] + dq[1]
    qb[2] = 0
    qb[3] = qb[2] + dq[3]
    qb[4] = qb[3] + dq[4]
    qb[5] = qb[4] + dq[5]
    qb[6] = qb[5] + dq[6]
    qb[9] = -dq[0]
    qb[8] = qb[9] - dq[9]
    qb[7] = qb[8] - dq[8]
    qb[10] = -qb[1] - dq[2]

    L_t = L / t_p
    d1 = L_t[0] + L_t[1] + L_t[10] + L_t[7] + L_t[8] + L_t[9]
    d2 = L_t[2] + L_t[3] + L_t[4] + L_t[5] + L_t[6] + L_t[10]
    d12 = L_t[10]

    qb_int1 = (qb[0] * L_t[0] + qb[1] * L_t[1] - qb[10] * L_t[10]
               + qb[7] * L_t[7] + qb[8] * L_t[8] + qb[9] * L_t[9])
    qb_int2 = (qb[2] * L_t[2] + qb[3] * L_t[3] + qb[4] * L_t[4]
               + qb[5] * L_t[5] + qb[6] * L_t[6] + qb[10] * L_t[10])

    M_qb = 0.0
    for i in range(11):
        i1, i2 = IDX_PANELS[i]
        M_qb += qb[i] * (x_b[i1] * y_b[i2] - x_b[i2] * y_b[i1])

    Mmat = np.array([[d1 / (2 * A1) + d12 / (2 * A2), -d12 / (2 * A1) - d2 / (2 * A2)],
                      [2 * A1, 2 * A2]])
    rhs = np.array([qb_int2 / (2 * A2) - qb_int1 / (2 * A1), Mz - M_qb])
    qs = np.linalg.solve(Mmat, rhs)

    q_total = np.array([
        qb[0] + qs[0], qb[1] + qs[0], qb[2] + qs[1], qb[3] + qs[1], qb[4] + qs[1],
        qb[5] + qs[1], qb[6] + qs[1], qb[7] + qs[0], qb[8] + qs[0], qb[9] + qs[0],
        qb[10] - qs[0] + qs[1],
    ])

    tau = np.abs(q_total) / t_p
    RF_sigma = sigma_adm / np.abs(sigma_signed)
    RF_tau = tau_adm / tau

    return RF_sigma, RF_tau, Ixx, y_bar, x_bar, q_total, sigma_signed


def system_of_equations(X, Mx, Mz, Vy, x_b, y_b, A1, A2, sigma_adm, tau_adm):
    RF_s, RF_t, *_ = calculate_state(X, Mx, Mz, Vy, x_b, y_b, A1, A2, sigma_adm, tau_adm)
    err_sigma = RF_s - 1.0
    err_tau = RF_t - 1.0
    err_sigma = np.where(err_sigma < 0, err_sigma * 50, err_sigma)
    err_tau = np.where(err_tau < 0, err_tau * 50, err_tau)
    return np.concatenate([err_sigma, err_tau])


def optimize_wingbox_21vars(x_b, y_b, V_local, M_flex_local, M_tors_local, sigma_adm, tau_adm):
    A1 = polyarea(x_b[[0, 1, 2, 7, 8, 9]], y_b[[0, 1, 2, 7, 8, 9]])
    A2 = polyarea(x_b[[2, 3, 4, 5, 6, 7]], y_b[[2, 3, 4, 5, 6, 7]])

    X0 = np.concatenate([np.full(10, 20e-4), np.full(11, 5e-3)])
    lb = np.concatenate([np.full(10, 1e-4), np.full(11, 1e-3)])

    ub_booms = np.zeros(10)
    scale_factor = max(0.15, min(1.0, abs(M_flex_local) / 10e6))
    left_limit = 150e-4 * scale_factor
    ub_booms[[0, 9]] = left_limit * 1.00
    ub_booms[[1, 8]] = left_limit * 0.90
    ub_booms[[2, 7]] = left_limit * 0.80
    ub_booms[[3, 6]] = left_limit * 0.60
    ub_booms[[4, 5]] = left_limit * 0.50

    skin_scale_factor = max(0.35, np.sqrt(scale_factor))
    left_skin_limit = 20e-3 * skin_scale_factor
    ub_skins = np.zeros(11)
    ub_skins[[0, 8, 9]] = left_skin_limit * 1.00
    ub_skins[[1, 7, 10]] = left_skin_limit * 0.95
    ub_skins[[2, 6]] = left_skin_limit * 0.90
    ub_skins[[3, 5]] = left_skin_limit * 0.85
    ub_skins[4] = left_skin_limit * 0.80
    ub = np.concatenate([ub_booms, ub_skins])

    fun = lambda X: system_of_equations(X, M_flex_local, M_tors_local, V_local,
                                         x_b, y_b, A1, A2, sigma_adm, tau_adm)

    # MATLAB's lsqnonlin silently projects an out-of-bounds x0 into [lb, ub];
    # scipy's least_squares raises instead, so clip explicitly to match.
    X0 = np.clip(X0, lb, ub)

    result = least_squares(fun, X0, bounds=(lb, ub), method='trf',
                            xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=200000)
    X_sol = result.x
    A_opt = X_sol[0:10]
    t_opt = X_sol[10:21]

    RF_s, RF_t, _, _, _, q_total, sigma_signed = calculate_state(
        X_sol, M_flex_local, M_tors_local, V_local, x_b, y_b, A1, A2, sigma_adm, tau_adm)

    return A_opt, t_opt, RF_s, RF_t, q_total, sigma_signed


# =============================================================================
# 1. INPUT PARAMETERS & EQUIVALENT WING
# =============================================================================

FLAG_STRAIGHT_SPARS = True

CCL, CR, Ck, CT, yr, bk, b = 11.7, 10.3, 7.5, 2.29, 5.03, 7.5 * 2, 47.57
Lambda_LE = np.deg2rad(34.5)

Sin = 0.5 * (Ck + CCL) * bk
Sout = 0.5 * (Ck + CT) * (b - bk)
Sfus = 0.5 * (CR + CCL) * yr
S = Sin + Sout
SE = S - Sfus
CR_e = (2 * SE / (b - yr)) - CT
CCL_e = (b * CR_e - yr * CT) / (b - yr)
lambda_e = CT / CCL_e
Lambda_25eq = np.arctan(np.tan(Lambda_LE) - (0.5 * CCL_e / b) * (1 - lambda_e))

wingGeom = SimpleNamespace()
wingGeom.span = b / 2
wingGeom.y_kink = bk / 2
wingGeom.c_root = CCL
wingGeom.c_kink = Ck
wingGeom.c_tip = CT
wingGeom.sweep_LE_1 = Lambda_LE
wingGeom.sweep_LE_2 = Lambda_LE
wingGeom.rho = 2780
wingGeom.t_skin = 0.003
wingGeom.straight_spars = FLAG_STRAIGHT_SPARS

# =============================================================================
# 2. AERODYNAMIC LIFT DISTRIBUTION
# =============================================================================

MAC = SE / b
nz = 2.5
MTOW = 158758
h_max = 43000 * 0.3048
_, a, _, rho_c = isa_atmosphere(h_max)
M = 0.8
V = a * M
beta = np.sqrt(1 - M ** 2)
Lambda_beta = np.arctan(np.tan(Lambda_25eq) / beta)
g = 9.80665
W = MTOW * g
AR_e = b ** 2 / SE
cl_alpha = (2 * np.pi) / np.sqrt(1 - (M * np.cos(Lambda_25eq)) ** 2)
term_num = 2 * np.pi * AR_e
term_in_sqrt = (term_num * np.cos(Lambda_beta)) / cl_alpha
CL_alpha = term_num / (2 + np.sqrt(4 + term_in_sqrt ** 2))
C_L = (2 * nz * W) / (rho_c * V ** 2 * SE)
CL_w = 1.05 * C_L
epsilon_t = np.deg2rad(2)
eta = np.linspace(0, 1, 100)
c_dist = CCL_e * (1 - (1 - lambda_e) * eta)
C1, C2, C3, C4 = 0.38, 0.31, 0.31, 0.49


def f_dist(eta):
    return (2.1242 * eta ** 6 - 15.667 * eta ** 5 + 28.172 * eta ** 4
            - 22.783 * eta ** 3 + 6.0319 * eta ** 2 + 1.4068 * eta + 0.648)


f_values = f_dist(eta)
LA = C1 * (c_dist / MAC) + (4 * C2 / np.pi) * np.sqrt(1 - eta ** 2) + C3 * f_values
epsilon = np.linspace(0, epsilon_t, 100)
integrand = LA * epsilon
alpha_0 = np.trapezoid(integrand, eta) / epsilon_t
LB = LA * C4 * ((epsilon / epsilon_t) - alpha_0)
Gamma = CL_w * LA + epsilon_t * CL_alpha * LB
q_dyn = 0.5 * rho_c * V ** 2
force_per_unit_span = q_dyn * MAC * Gamma
y_plot_aero = eta * (b / 2)

# =============================================================================
# 3. AEROSURF PRE-CALCULATION & CONSTANT WEIGHTS
# =============================================================================

theta = np.linspace(0, np.pi, 60)
x_dense_surf = 0.5 * (1 - np.cos(theta))

s_up_root, s_lo_root, *_ = get_airfoil_coords('NACA 0014.dat', x_dense_surf)
s_up_kink, s_lo_kink, *_ = get_airfoil_coords('NACA 0012.dat', x_dense_surf)
s_up_tip, s_lo_tip, *_ = get_airfoil_coords('NACA 0009.dat', x_dense_surf)
x_norm_surf = np.concatenate([x_dense_surf, np.flipud(x_dense_surf[1:-1])])

wingGeom.aeroSurf_root = np.column_stack([x_norm_surf, np.concatenate([s_up_root, np.flipud(s_lo_root[1:-1])])])
wingGeom.aeroSurf_kink = np.column_stack([x_norm_surf, np.concatenate([s_up_kink, np.flipud(s_lo_kink[1:-1])])])
wingGeom.aeroSurf_tip = np.column_stack([x_norm_surf, np.concatenate([s_up_tip, np.flipud(s_lo_tip[1:-1])])])

N_stations = 30
z_stations = np.unique(np.concatenate([np.linspace(0, wingGeom.span, N_stations), [wingGeom.y_kink]]))
N_stations = len(z_stations)
z_targets = [2.515, wingGeom.y_kink, 0.75 * wingGeom.span]
sigma_adm = 241.5e6
tau_adm = 144.9e6

M_wing_semi = 10258
W_wing_total = M_wing_semi * g * nz
q_wing = W_wing_total * (c_dist / SE)
z_fuel_max = 0.8 * wingGeom.span
rho_f = 804
W_engine = 4410 * g * nz
z_engine = 7.92
W_gear = 3270 * g * nz
z_gear = 4.65

# =============================================================================
# 4. DESIGN OF EXPERIMENTS (DoE)
# =============================================================================

vec_eta_f = [0.15, 0.20, 0.25]
vec_eta_m = [0.47, 0.50, 0.53]
vec_eta_r = [0.65, 0.70, 0.75]
DoE_Results = np.zeros((27, 5))
num_case = 0

target_names = ['Fuselage Root', 'Aerodynamic Kink', '75% Span']
idx_panels = IDX_PANELS  # same connectivity used for section-area bookkeeping

print("=" * 63)
print(" STARTING TOPOLOGICAL SWEEP (DoE - 27 CONFIGURATIONS)")
print("=" * 63)

for eta_f in vec_eta_f:
    for eta_m in vec_eta_m:
        for eta_r in vec_eta_r:

            print(f"\n{'=' * 63}")
            print(f"CASE {num_case + 1:2d}/27: CONFIGURATION eta_f={eta_f:.2f}, "
                  f"eta_m={eta_m:.2f}, eta_r={eta_r:.2f}")
            print("=" * 63)

            n_inter_1 = 1
            n_inter_2 = 1
            x_up_1 = np.linspace(eta_f, eta_m, n_inter_1 + 2)
            x_up_2 = np.linspace(eta_m, eta_r, n_inter_2 + 2)
            x_upper = np.unique(np.concatenate([x_up_1, x_up_2]))
            x_lower = np.flipud(x_upper)
            x_norm_booms = np.concatenate([x_upper, x_lower])
            n_booms = len(x_norm_booms)

            idx_fs_up = 0
            idx_ms_up = n_inter_1 + 1
            idx_rs_up = len(x_upper) - 1
            wingGeom.idx_spars_upper = [idx_fs_up, idx_ms_up, idx_rs_up]

            y_up, y_lo, *_ = get_airfoil_coords('NACA 0014.dat', x_upper)
            y_up, y_lo = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper)
            wingGeom.airfoils_root = np.column_stack(
                [x_norm_booms, np.concatenate([y_up, np.flipud(y_lo)])])

            y_up, y_lo, *_ = get_airfoil_coords('NACA 0012.dat', x_upper)
            y_up, y_lo = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper)
            wingGeom.airfoils_kink = np.column_stack(
                [x_norm_booms, np.concatenate([y_up, np.flipud(y_lo)])])

            y_up, y_lo, *_ = get_airfoil_coords('NACA 0009.dat', x_upper)
            y_up, y_lo = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper)
            wingGeom.airfoils_tip = np.column_stack(
                [x_norm_booms, np.concatenate([y_up, np.flipud(y_lo)])])

            # --- 3D GEOMETRY ---
            X_boom = np.zeros((n_booms, N_stations))
            Y_boom = np.zeros((n_booms, N_stations))
            Z_boom = np.zeros((n_booms, N_stations))
            x_LE_kink_geom = wingGeom.y_kink * np.tan(wingGeom.sweep_LE_1)
            x_LE_tip_geom = x_LE_kink_geom + (wingGeom.span - wingGeom.y_kink) * np.tan(wingGeom.sweep_LE_2)
            X_booms_kink = wingGeom.airfoils_kink[:, 0] * wingGeom.c_kink + x_LE_kink_geom
            X_booms_tip = wingGeom.airfoils_tip[:, 0] * wingGeom.c_tip + x_LE_tip_geom
            m_booms = (X_booms_tip - X_booms_kink) / (wingGeom.span - wingGeom.y_kink)

            for i in range(N_stations):
                z = z_stations[i]
                if z <= wingGeom.y_kink:
                    f = z / wingGeom.y_kink
                    c_z = wingGeom.c_root + f * (wingGeom.c_kink - wingGeom.c_root)
                    x_LE_3D = z * np.tan(wingGeom.sweep_LE_1)
                    boom_norm = (1 - f) * wingGeom.airfoils_root + f * wingGeom.airfoils_kink
                    if FLAG_STRAIGHT_SPARS:
                        X_boom[:, i] = X_booms_kink - m_booms * (wingGeom.y_kink - z)
                    else:
                        X_boom[:, i] = boom_norm[:, 0] * c_z + x_LE_3D
                else:
                    f = (z - wingGeom.y_kink) / (wingGeom.span - wingGeom.y_kink)
                    c_z = wingGeom.c_kink + f * (wingGeom.c_tip - wingGeom.c_kink)
                    x_LE_3D = (wingGeom.y_kink * np.tan(wingGeom.sweep_LE_1)
                               + (z - wingGeom.y_kink) * np.tan(wingGeom.sweep_LE_2))
                    boom_norm = (1 - f) * wingGeom.airfoils_kink + f * wingGeom.airfoils_tip
                    X_boom[:, i] = boom_norm[:, 0] * c_z + x_LE_3D
                Y_boom[:, i] = boom_norm[:, 1] * c_z
                Z_boom[:, i] = z

            # --- INTERNAL LOADS CALCULATION ---
            ndsa_vec = get_ndsa_dist(y_plot_aero, wingGeom)
            q_fuel = rho_f * g * nz * (c_dist ** 2) * ndsa_vec
            q_fuel = np.where(y_plot_aero > z_fuel_max, 0.0, q_fuel)

            x_LE_vec = np.zeros_like(y_plot_aero)
            idx_in = y_plot_aero <= wingGeom.y_kink
            idx_out = ~idx_in
            x_LE_vec[idx_in] = y_plot_aero[idx_in] * np.tan(wingGeom.sweep_LE_1)
            x_LE_vec[idx_out] = (wingGeom.y_kink * np.tan(wingGeom.sweep_LE_1)
                                  + (y_plot_aero[idx_out] - wingGeom.y_kink) * np.tan(wingGeom.sweep_LE_2))
            x_L_vec = x_LE_vec + 0.25 * c_dist

            ifs = wingGeom.idx_spars_upper[0]
            irs = wingGeom.idx_spars_upper[2]
            x_EA_root = (wingGeom.airfoils_root[ifs, 0] + wingGeom.airfoils_root[irs, 0]) * wingGeom.c_root / 2
            x_EA_tip = x_LE_tip_geom + (wingGeom.airfoils_tip[ifs, 0] + wingGeom.airfoils_tip[irs, 0]) * wingGeom.c_tip / 2
            m_EA = (x_EA_tip - x_EA_root) / wingGeom.span
            x_EA_vec = x_EA_root + m_EA * y_plot_aero
            torsion_arm = x_EA_vec - x_L_vec

            V_targets = np.zeros(3)
            Mx_targets = np.zeros(3)
            Mz_targets = np.zeros(3)
            M_flex_local = np.zeros(3)
            M_tors_local = np.zeros(3)

            for i in range(3):
                zt = z_targets[i]
                z_L_limit = get_oblique_lift_limit(zt, wingGeom)
                idx_L = y_plot_aero >= z_L_limit
                y_L = y_plot_aero[idx_L]
                L_dist = force_per_unit_span[idx_L]

                if len(y_L) > 1:
                    V_L = np.trapezoid(L_dist, y_L)
                    Mx_L = np.trapezoid(L_dist * (y_L - zt), y_L)
                    Mz_L = np.trapezoid(L_dist * torsion_arm[idx_L], y_L)
                else:
                    V_L = Mx_L = Mz_L = 0.0

                idx_w = y_plot_aero >= zt
                y_w = y_plot_aero[idx_w]
                if len(y_w) > 1:
                    V_W = -np.trapezoid(q_wing[idx_w], y_w)
                    Mx_W = -np.trapezoid(q_wing[idx_w] * (y_w - zt), y_w)
                    V_F = -np.trapezoid(q_fuel[idx_w], y_w)
                    Mx_F = -np.trapezoid(q_fuel[idx_w] * (y_w - zt), y_w)
                else:
                    V_W = Mx_W = V_F = Mx_F = 0.0

                V_p = 0.0
                Mx_p = 0.0
                if zt <= z_engine:
                    V_p -= W_engine
                    Mx_p -= W_engine * (z_engine - zt)
                if zt <= z_gear:
                    V_p -= W_gear
                    Mx_p -= W_gear * (z_gear - zt)

                V_targets[i] = V_L + V_W + V_F + V_p
                Mx_targets[i] = Mx_L + Mx_W + Mx_F + Mx_p
                Mz_targets[i] = Mz_L
                Lambda_EA = np.arctan(m_EA)
                M_flex_local[i] = Mx_targets[i] * np.cos(Lambda_EA) - Mz_targets[i] * np.sin(Lambda_EA)
                M_tors_local[i] = Mx_targets[i] * np.sin(Lambda_EA) + Mz_targets[i] * np.cos(Lambda_EA)

            # --- OBLIQUE SECTIONS SETUP ---
            idx_root = 0
            idx_kink = int(np.argmin(np.abs(z_stations - wingGeom.y_kink)))
            idx_tip = N_stations - 1
            x_EA_root = (X_boom[ifs, idx_root] + X_boom[irs, idx_root]) / 2
            x_EA_tip = (X_boom[ifs, idx_tip] + X_boom[irs, idx_tip]) / 2
            m_EA = (x_EA_tip - x_EA_root) / wingGeom.span
            Lambda_EA = np.arctan(m_EA)

            savedBoomsCoord = [None, None, None]
            for i in range(3):
                zt = z_targets[i]
                xt = x_EA_root + m_EA * zt
                x_loc_booms = np.zeros(n_booms)
                y_loc_booms = np.zeros(n_booms)
                for bm in range(n_booms):
                    if zt <= wingGeom.y_kink + 1e-5:
                        X0, Y0, Z0 = X_boom[bm, idx_root], Y_boom[bm, idx_root], Z_boom[bm, idx_root]
                        X1, Y1, Z1 = X_boom[bm, idx_kink], Y_boom[bm, idx_kink], Z_boom[bm, idx_kink]
                    else:
                        X0, Y0, Z0 = X_boom[bm, idx_kink], Y_boom[bm, idx_kink], Z_boom[bm, idx_kink]
                        X1, Y1, Z1 = X_boom[bm, idx_tip], Y_boom[bm, idx_tip], Z_boom[bm, idx_tip]
                    dX, dY, dZ = X1 - X0, Y1 - Y0, Z1 - Z0
                    s = (m_EA * (xt - X0) + (zt - Z0)) / (m_EA * dX + dZ)
                    X_int = X0 + s * dX
                    Y_int = Y0 + s * dY
                    Z_int = Z0 + s * dZ
                    x_loc_booms[bm] = (X_int - xt) * np.cos(Lambda_EA) - (Z_int - zt) * np.sin(Lambda_EA)
                    y_loc_booms[bm] = Y_int
                savedBoomsCoord[i] = {'x_m': x_loc_booms, 'y_m': y_loc_booms}

            # --- EXACT SECTION OPTIMIZATION (21 Variables) ---
            Total_Area_Metric = 0.0
            FSD_Errors_Case = []

            for i in range(3):
                x_naca = savedBoomsCoord[i]['x_m']
                y_naca = savedBoomsCoord[i]['y_m']
                V_section = -V_targets[i]
                Mx_section = -M_flex_local[i]
                Mz_section = -M_tors_local[i]

                A_opt, t_opt, RF_s, RF_t, q_total, sigma_signed = optimize_wingbox_21vars(
                    x_naca, y_naca, V_section, Mx_section, Mz_section, sigma_adm, tau_adm)

                A_active = A_opt > 1e-4
                t_active = t_opt > 1.01e-3
                FSD_Errors_Case.extend(np.abs(RF_s[A_active] - 1))
                FSD_Errors_Case.extend(np.abs(RF_t[t_active] - 1))

                L_panels = np.zeros(11)
                for p in range(11):
                    i1, i2 = idx_panels[p]
                    L_panels[p] = np.sqrt((x_naca[i2] - x_naca[i1]) ** 2 + (y_naca[i2] - y_naca[i1]) ** 2)
                Section_Area = np.sum(A_opt) + np.sum(t_opt * L_panels)
                Total_Area_Metric += Section_Area

                RF_s = np.where(RF_s < 1.0, 1.000, RF_s)
                RF_s = np.where((RF_s >= 1.0) & (RF_s <= 1.01), 1.000, RF_s)
                RF_t = np.where(RF_t < 1.0, 1.000, RF_t)
                RF_t = np.where((RF_t >= 1.0) & (RF_t <= 1.01), 1.000, RF_t)

                sigma_real = (sigma_adm / RF_s) * np.sign(sigma_signed)
                tau_real = tau_adm / RF_t

                print(f"\n  >>> SECTION {i + 1}: {target_names[i]} <<<")
                print("  Boom | Area (cm^2) | Normal (MPa) | RF Sigma ")
                print("  -----------------------------------------------")
                for bm in range(10):
                    print(f"  {bm + 1:4d} | {A_opt[bm] * 10000:11.2f} | {sigma_real[bm] / 1e6:12.2f} | {RF_s[bm]:8.3f}")
                print("\n  Panel| Thickness (mm)| Shear (MPa)| Flow q (kN/m)| RF Tau ")
                print("  --------------------------------------------------------------")
                for p in range(11):
                    print(f"  {p + 1:4d} | {t_opt[p] * 1000:10.2f} | {tau_real[p] / 1e6:13.2f} "
                          f"| {abs(q_total[p]) / 1000:13.1f} | {RF_t[p]:6.3f}")
                print(f"  -> Cross-sectional Area: {Section_Area:.6f} m^2")
                print("  --------------------------------------------------------------")

            FSD_Mean_Error = np.mean(FSD_Errors_Case) if FSD_Errors_Case else 0.0

            print(f"\n> Weight Metric: {Total_Area_Metric:.6f} m^2 | FSD Error: {FSD_Mean_Error:.4f}")
            DoE_Results[num_case, :] = [eta_f, eta_m, eta_r, Total_Area_Metric, FSD_Mean_Error]
            num_case += 1

# =============================================================================
# 5. RE-EVALUATE AND SAVE CONFIGURATION 7 DATA FOR PLOTTING
# =============================================================================

Results_Table = pd.DataFrame(
    DoE_Results, columns=['Front_Spar', 'Mid_Spar', 'Rear_Spar', 'Weight_Metric_m2', 'FSD_Error'])
Results_Table = Results_Table.sort_values('Weight_Metric_m2', ascending=True).reset_index(drop=True)

print("\n" + "=" * 63)
print(" >>> DESIGN OF EXPERIMENTS COMPLETED <<<")
print("=" * 63)
print(Results_Table)

# Force Configuration 7 as requested (1-based in the original -> DoE_Results row 6)
target_case = 7
optimal_config = DoE_Results[target_case - 1, 0:3]

print(f"\nForcing Configuration {target_case} [{optimal_config[0]:.2f}, "
      f"{optimal_config[1]:.2f}, {optimal_config[2]:.2f}] to extract final data...")

Optimal_Design = SimpleNamespace()
Optimal_Design.config = optimal_config

eta_f, eta_m, eta_r = optimal_config
x_up_1 = np.linspace(eta_f, eta_m, 1 + 2)
x_up_2 = np.linspace(eta_m, eta_r, 1 + 2)
x_upper = np.unique(np.concatenate([x_up_1, x_up_2]))
x_lower = np.flipud(x_upper)
Optimal_Design.x_norm_booms = np.concatenate([x_upper, x_lower])

idx_fs_up = 0
idx_ms_up = 1 + 1
idx_rs_up = len(x_upper) - 1
wingGeom.idx_spars_upper = [idx_fs_up, idx_ms_up, idx_rs_up]

y_up, y_lo, *_ = get_airfoil_coords('NACA 0014.dat', x_upper)
y_up, y_lo = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper)
wingGeom.airfoils_root = np.column_stack(
    [Optimal_Design.x_norm_booms, np.concatenate([y_up, np.flipud(y_lo)])])

y_up, y_lo, *_ = get_airfoil_coords('NACA 0012.dat', x_upper)
y_up, y_lo = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper)
wingGeom.airfoils_kink = np.column_stack(
    [Optimal_Design.x_norm_booms, np.concatenate([y_up, np.flipud(y_lo)])])

y_up, y_lo, *_ = get_airfoil_coords('NACA 0009.dat', x_upper)
y_up, y_lo = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper)
wingGeom.airfoils_tip = np.column_stack(
    [Optimal_Design.x_norm_booms, np.concatenate([y_up, np.flipud(y_lo)])])

ndsa_vec_optimal = get_ndsa_dist(y_plot_aero, wingGeom)
q_fuel = rho_f * g * nz * (c_dist ** 2) * ndsa_vec_optimal
q_fuel = np.where(y_plot_aero > z_fuel_max, 0.0, q_fuel)

# NOTE: preserved verbatim from the MATLAB script -- X_boom/Y_boom/Z_boom here
# are whatever remains from the final DoE loop iteration (case 27), not
# recomputed for the forced configuration 7.
Optimal_Design.X_boom = X_boom
Optimal_Design.Y_boom = Y_boom
Optimal_Design.Z_boom = Z_boom
Optimal_Design.z_stations = z_stations

Optimal_Design.Targets = []
for i in range(3):
    target = SimpleNamespace()
    target.z_loc = z_targets[i]
    target.x_naca = savedBoomsCoord[i]['x_m']
    target.y_naca = savedBoomsCoord[i]['y_m']
    target.V = V_targets[i]
    target.Mx = M_flex_local[i]
    target.Mz = M_tors_local[i]

    A_opt, t_opt, RF_s, RF_t, q_total, sigma_signed = optimize_wingbox_21vars(
        target.x_naca, target.y_naca, V_targets[i], M_flex_local[i], M_tors_local[i],
        sigma_adm, tau_adm)

    RF_s = np.where(RF_s < 1.0, 1.000, RF_s)
    RF_s = np.where((RF_s >= 1.0) & (RF_s <= 1.01), 1.000, RF_s)
    RF_t = np.where(RF_t < 1.0, 1.000, RF_t)
    RF_t = np.where((RF_t >= 1.0) & (RF_t <= 1.01), 1.000, RF_t)

    target.A_opt = A_opt
    target.t_opt = t_opt
    target.q_total = q_total
    target.RF_s = RF_s
    target.RF_t = RF_t
    target.sigma_real = (sigma_adm / RF_s) * np.sign(sigma_signed)
    target.tau_real = tau_adm / RF_t

    Optimal_Design.Targets.append(target)

print('Optimal design data computed successfully. You may now run the plotting script.')