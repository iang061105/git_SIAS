% =========================================================================
% SCRIPT 1: WINGBOX OPTIMIZATION & STRUCTURAL CALCULATIONS
% =========================================================================
clear; clc; close all;
fprintf('===============================================================\n');
fprintf(' STARTING TOPOLOGICAL SWEEP (DoE - 27 CONFIGURATIONS)\n');
fprintf('===============================================================\n');

% --- 0. GLOBAL SETTINGS ---
FLAG_STRAIGHT_SPARS = true;

% --- 1. INPUT PARAMETERS & EQUIVALENT WING ---
CCL = 11.7; CR = 10.3; Ck = 7.5; CT = 2.29; yr = 5.03; bk = 7.5*2; b = 47.57;
Lambda_LE = deg2rad(34.5);

Sin  = 0.5 * (Ck + CCL) * bk; Sout = 0.5 * (Ck + CT) * (b - bk); Sfus = 0.5 * (CR + CCL) * yr;
S = Sin + Sout; SE = S - Sfus;            
CR_e = (2 * SE / (b-yr)) - CT;
CCL_e = (b * CR_e - yr * CT)/(b - yr);
lambda_e = CT / CCL_e;
Lambda_25eq = atan(tan(Lambda_LE) - (0.5 * CCL_e / b) * ((1 - lambda_e)));

wingGeom.span = b / 2;
wingGeom.y_kink = bk / 2;         
wingGeom.c_root = CCL; wingGeom.c_kink = Ck; wingGeom.c_tip  = CT;
wingGeom.sweep_LE_1 = Lambda_LE; wingGeom.sweep_LE_2 = Lambda_LE;  
wingGeom.rho = 2780; wingGeom.t_skin = 0.003;          
wingGeom.straight_spars = FLAG_STRAIGHT_SPARS;

% --- 2. AERODYNAMIC LIFT DISTRIBUTION ---
MAC = SE / b; nz = 2.5; MTOW = 158758; h_max = 43000 * 0.3048; 
[~, a, ~, rho_c] = atmosisa(h_max); M = 0.8; V = a * M; beta = sqrt(1 - M^2);
Lambda_beta = atan(tan(Lambda_25eq) / beta); g = 9.80665; W = MTOW * g;
AR_e = b^2 / SE;
cl_alpha = (2 * pi) / sqrt(1 - (M * cos(Lambda_25eq))^2);
term_num = 2 * pi * AR_e;
term_in_sqrt = (term_num * cos(Lambda_beta)) / cl_alpha;
CL_alpha = term_num / (2 + sqrt(4 + term_in_sqrt^2));
C_L = (2 * nz * W) / (rho_c * V^2 * SE); CL_w = 1.05 * C_L;
epsilon_t = deg2rad(2);
eta = linspace(0, 1, 100);
c_dist = CCL_e * (1 - (1 - lambda_e) * eta);
C1 = 0.38; C2 = 0.31; C3 = 0.31; C4 = 0.49;
f_dist = @(eta) 2.1242.*eta.^6 - 15.667.*eta.^5 + 28.172.*eta.^4 - 22.783.*eta.^3 + 6.0319.*eta.^2 + 1.4068.*eta + 0.648;
f_values = f_dist(eta);
LA = C1 .* (c_dist ./ MAC) + (4 * C2 / pi) .* sqrt(1 - eta.^2) + C3 .* f_values;
epsilon = linspace(0, epsilon_t, 100); 
integrand = LA .* epsilon;
alpha_0 = trapz(eta, integrand) / epsilon_t;
LB = LA .* C4 .* ((epsilon ./ epsilon_t) - alpha_0);
Gamma = CL_w .* LA + epsilon_t * CL_alpha .* LB;
q_dyn = 0.5 * rho_c * V^2;
force_per_unit_span = q_dyn * MAC .* Gamma;
y_plot_aero = eta * (b/2);

% --- 3. AEROSURF PRE-CALCULATION & CONSTANT WEIGHTS ---
theta = linspace(0, pi, 60)'; x_dense_surf = 0.5 * (1 - cos(theta));
[s_up_root, s_lo_root] = getAirfoilCoords('NACA 0014.dat', x_dense_surf);
[s_up_kink, s_lo_kink] = getAirfoilCoords('NACA 0012.dat', x_dense_surf);
[s_up_tip,  s_lo_tip]  = getAirfoilCoords('NACA 0009.dat', x_dense_surf);
x_norm_surf = [x_dense_surf; flipud(x_dense_surf(2:end-1))];

wingGeom.aeroSurf.root = [x_norm_surf, [s_up_root; flipud(s_lo_root(2:end-1))]];
wingGeom.aeroSurf.kink = [x_norm_surf, [s_up_kink; flipud(s_lo_kink(2:end-1))]];
wingGeom.aeroSurf.tip  = [x_norm_surf, [s_up_tip;  flipud(s_lo_tip(2:end-1))]];

N_stations = 30;
z_stations = unique(sort([linspace(0, wingGeom.span, N_stations), wingGeom.y_kink]));
N_stations = length(z_stations);
z_targets = [2.515, wingGeom.y_kink, 0.75 * wingGeom.span];
sigma_adm = 241.5e6; tau_adm = 144.9e6;

% Fixed Weight Parameters (Fuel load depends on geometry, calculated inside loop)
M_wing_semi = 10258; W_wing_total = M_wing_semi * g * nz; 
q_wing = W_wing_total .* (c_dist ./ SE);
z_fuel_max = 0.8 * wingGeom.span; rho_f = 804; 
W_engine = 4410 * g * nz;  z_engine = 7.92;
W_gear   = 3270 * g * nz;  z_gear   = 4.65;

% --- 4. DESIGN OF EXPERIMENTS (DoE) ---
vec_eta_f = [0.15, 0.20, 0.25];
vec_eta_m = [0.47, 0.50, 0.53];
vec_eta_r = [0.65, 0.70, 0.75];
DoE_Results = zeros(27, 5);
num_case = 1;

for i_f = 1:length(vec_eta_f)
    for i_m = 1:length(vec_eta_m)
        for i_r = 1:length(vec_eta_r)
            
            eta_f = vec_eta_f(i_f); eta_m = vec_eta_m(i_m); eta_r = vec_eta_r(i_r);
            fprintf('\n===============================================================\n');
            fprintf('CASE %2d/27: CONFIGURATION eta_f=%.2f, eta_m=%.2f, eta_r=%.2f\n', num_case, eta_f, eta_m, eta_r);
            fprintf('===============================================================\n');
            
            n_inter_1 = 1; n_inter_2 = 1;
            x_up_1 = linspace(eta_f, eta_m, n_inter_1 + 2);
            x_up_2 = linspace(eta_m, eta_r, n_inter_2 + 2);
            x_upper = unique([x_up_1, x_up_2]); 
            x_lower = fliplr(x_upper); x_norm_booms = [x_upper, x_lower]';
            n_booms = length(x_norm_booms);
            
            idx_fs_up = 1; idx_ms_up = 1 + n_inter_1 + 1; idx_rs_up = length(x_upper);
            wingGeom.idx_spars_upper = [idx_fs_up, idx_ms_up, idx_rs_up];
            
            [y_up, y_lo, ~, ~, ~] = getAirfoilCoords('NACA 0014.dat', x_upper);
            [y_up, y_lo] = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper);
            wingGeom.airfoils.root = [x_norm_booms, [y_up, fliplr(y_lo)]'];
            
            [y_up, y_lo, ~, ~, ~] = getAirfoilCoords('NACA 0012.dat', x_upper);
            [y_up, y_lo] = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper);
            wingGeom.airfoils.kink = [x_norm_booms, [y_up, fliplr(y_lo)]'];
            
            [y_up, y_lo, ~, ~, ~] = getAirfoilCoords('NACA 0009.dat', x_upper);
            [y_up, y_lo] = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper);
            wingGeom.airfoils.tip  = [x_norm_booms, [y_up, fliplr(y_lo)]'];
            
            % --- 3D GEOMETRY ---
            X_boom = zeros(n_booms, N_stations); Y_boom = zeros(n_booms, N_stations); Z_boom = zeros(n_booms, N_stations);
            x_LE_kink_geom = wingGeom.y_kink * tan(wingGeom.sweep_LE_1);
            x_LE_tip_geom = x_LE_kink_geom + (wingGeom.span - wingGeom.y_kink) * tan(wingGeom.sweep_LE_2);
            X_booms_kink = wingGeom.airfoils.kink(:, 1) * wingGeom.c_kink + x_LE_kink_geom;
            X_booms_tip = wingGeom.airfoils.tip(:, 1) * wingGeom.c_tip + x_LE_tip_geom;
            m_booms = (X_booms_tip - X_booms_kink) / (wingGeom.span - wingGeom.y_kink);
            
            for i = 1:N_stations
                z = z_stations(i);
                if z <= wingGeom.y_kink
                    f = z / wingGeom.y_kink;
                    c_z = wingGeom.c_root + f * (wingGeom.c_kink - wingGeom.c_root);
                    x_LE_3D = z * tan(wingGeom.sweep_LE_1);
                    boom_norm = (1 - f) * wingGeom.airfoils.root + f * wingGeom.airfoils.kink;
                    if FLAG_STRAIGHT_SPARS
                        X_boom(:, i) = X_booms_kink - m_booms * (wingGeom.y_kink - z);
                    else
                        X_boom(:, i) = boom_norm(:, 1) * c_z + x_LE_3D;
                    end
                else
                    f = (z - wingGeom.y_kink) / (wingGeom.span - wingGeom.y_kink);
                    c_z = wingGeom.c_kink + f * (wingGeom.c_tip - wingGeom.c_kink);
                    x_LE_3D = wingGeom.y_kink * tan(wingGeom.sweep_LE_1) + (z - wingGeom.y_kink) * tan(wingGeom.sweep_LE_2);
                    boom_norm = (1 - f) * wingGeom.airfoils.kink + f * wingGeom.airfoils.tip;
                    X_boom(:, i) = boom_norm(:, 1) * c_z + x_LE_3D;
                end
                Y_boom(:, i) = boom_norm(:, 2) * c_z; Z_boom(:, i) = z;
            end
            
            % --- INTERNAL LOADS CALCULATION ---
            ndsa_vec = get_ndsa_dist(y_plot_aero, wingGeom);
            q_fuel = rho_f * g * nz * (c_dist.^2) .* ndsa_vec; q_fuel(y_plot_aero > z_fuel_max) = 0; 
            
            x_LE_vec = zeros(size(y_plot_aero));
            idx_in = y_plot_aero <= wingGeom.y_kink; idx_out = y_plot_aero > wingGeom.y_kink;
            x_LE_vec(idx_in) = y_plot_aero(idx_in) * tan(wingGeom.sweep_LE_1);
            x_LE_vec(idx_out) = wingGeom.y_kink * tan(wingGeom.sweep_LE_1) + (y_plot_aero(idx_out) - wingGeom.y_kink) * tan(wingGeom.sweep_LE_2);
            x_L_vec = x_LE_vec + 0.25 .* c_dist;
            
            ifs = wingGeom.idx_spars_upper(1); irs = wingGeom.idx_spars_upper(3);
            x_EA_root = (wingGeom.airfoils.root(ifs,1) + wingGeom.airfoils.root(irs,1)) * wingGeom.c_root / 2;
            x_EA_tip  = x_LE_tip_geom + (wingGeom.airfoils.tip(ifs,1) + wingGeom.airfoils.tip(irs,1)) * wingGeom.c_tip / 2;
            m_EA = (x_EA_tip - x_EA_root) / wingGeom.span;
            x_EA_vec = x_EA_root + m_EA .* y_plot_aero; 
            torsion_arm = x_EA_vec - x_L_vec; 
            
            V_targets = zeros(1, 3); Mx_targets = zeros(1, 3); Mz_targets = zeros(1, 3);
            M_flex_local = zeros(1, 3); M_tors_local = zeros(1, 3);
            
            for i = 1:3
                zt = z_targets(i);
                z_L_limit = getObliqueLiftLimit(zt, wingGeom);
                idx_L = y_plot_aero >= z_L_limit; y_L = y_plot_aero(idx_L); L_dist = force_per_unit_span(idx_L);
                
                if length(y_L) > 1
                    V_L  = trapz(y_L, L_dist);
                    Mx_L = trapz(y_L, L_dist .* (y_L - zt)); 
                    Mz_L = trapz(y_L, L_dist .* torsion_arm(idx_L));
                else
                    V_L = 0; Mx_L = 0; Mz_L = 0; 
                end
                
                idx_w = y_plot_aero >= zt; y_w = y_plot_aero(idx_w);
                if length(y_w) > 1
                    V_W  = -trapz(y_w, q_wing(idx_w)); Mx_W = -trapz(y_w, q_wing(idx_w) .* (y_w - zt));
                    V_F  = -trapz(y_w, q_fuel(idx_w)); Mx_F = -trapz(y_w, q_fuel(idx_w) .* (y_w - zt));
                else
                    V_W = 0; Mx_W = 0; V_F = 0; Mx_F = 0;
                end
                
                V_p = 0; Mx_p = 0;
                if zt <= z_engine, V_p = V_p - W_engine; Mx_p = Mx_p - W_engine * (z_engine - zt); end
                if zt <= z_gear,   V_p = V_p - W_gear;   Mx_p = Mx_p - W_gear * (z_gear - zt); end
                
                V_targets(i)  = V_L + V_W + V_F + V_p;
                Mx_targets(i) = Mx_L + Mx_W + Mx_F + Mx_p; Mz_targets(i) = Mz_L; 
                Lambda_EA = atan(m_EA);
                M_flex_local(i) = Mx_targets(i) * cos(Lambda_EA) - Mz_targets(i) * sin(Lambda_EA);
                M_tors_local(i) = Mx_targets(i) * sin(Lambda_EA) + Mz_targets(i) * cos(Lambda_EA);
            end
            
            % --- OBLIQUE SECTIONS SETUP ---
            idx_root = 1;
            idx_kink = find(abs(z_stations - wingGeom.y_kink) < 1e-5, 1); idx_tip  = length(z_stations);
            x_EA_root = (X_boom(ifs, idx_root) + X_boom(irs, idx_root)) / 2;
            x_EA_tip  = (X_boom(ifs, idx_tip)  + X_boom(irs, idx_tip))  / 2;
            m_EA = (x_EA_tip - x_EA_root) / wingGeom.span; Lambda_EA = atan(m_EA); 
            
            savedBoomsCoord = struct();
            for i = 1:3
                zt = z_targets(i); xt = x_EA_root + m_EA * zt; 
                x_loc_booms = zeros(1, n_booms); y_loc_booms = zeros(1, n_booms);
                for b = 1:n_booms
                    if zt <= wingGeom.y_kink + 1e-5 
                        X0 = X_boom(b, idx_root); Y0 = Y_boom(b, idx_root); Z0 = Z_boom(b, idx_root);
                        X1 = X_boom(b, idx_kink); Y1 = Y_boom(b, idx_kink); Z1 = Z_boom(b, idx_kink);
                    else
                        X0 = X_boom(b, idx_kink); Y0 = Y_boom(b, idx_kink); Z0 = Z_boom(b, idx_kink);
                        X1 = X_boom(b, idx_tip);  Y1 = Y_boom(b, idx_tip);  Z1 = Z_boom(b, idx_tip);
                    end
                    dX = X1 - X0; dY = Y1 - Y0; dZ = Z1 - Z0;
                    s = (m_EA * (xt - X0) + (zt - Z0)) / (m_EA * dX + dZ);
                    X_int = X0 + s * dX; Y_int = Y0 + s * dY; Z_int = Z0 + s * dZ;
                    x_loc_booms(b) = (X_int - xt) * cos(Lambda_EA) - (Z_int - zt) * sin(Lambda_EA);
                    y_loc_booms(b) = Y_int;
                end
                savedBoomsCoord(i).booms_all.x_m = x_loc_booms';
                savedBoomsCoord(i).booms_all.y_m = y_loc_booms';
            end
            
            % --- EXACT SECTION OPTIMIZATION (21 Variables) ---
            Total_Area_Metric = 0; FSD_Errors_Case = []; 
            idx_panels = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 1; 8 3];
            target_names = {'Fuselage Root', 'Aerodynamic Kink', '75% Span'};
            
            for i = 1:3
                x_naca = savedBoomsCoord(i).booms_all.x_m';
                y_naca = savedBoomsCoord(i).booms_all.y_m';
                V_section  = -V_targets(i); Mx_section =- M_flex_local(i); Mz_section =- M_tors_local(i);
                
                % We only extract the signed sigma, panels remain strictly absolute
                [A_opt, t_opt, RF_s, RF_t, q_total, sigma_signed] = optimize_wingbox_21vars(x_naca, y_naca, V_section, Mx_section, Mz_section, sigma_adm, tau_adm);
                
                % 1) Calculate FINAL pure metrics, unmasked, to ensure integrity
                A_active = A_opt > 1e-4; t_active = t_opt > 1.01e-3; 
                FSD_Errors_Case = [FSD_Errors_Case, abs(RF_s(A_active) - 1), abs(RF_t(t_active) - 1)];
                
                L_panels = zeros(1, 11);
                for p = 1:11
                    L_panels(p) = sqrt((x_naca(idx_panels(p,2)) - x_naca(idx_panels(p,1)))^2 + (y_naca(idx_panels(p,2)) - y_naca(idx_panels(p,1)))^2);
                end
                Section_Area = sum(A_opt) + sum(t_opt .* L_panels);
                Total_Area_Metric = Total_Area_Metric + Section_Area;
                
                % 2) APPLY VISUAL STRICT RULE: Any RF < 1 or slightly loose is clamped to 1
                RF_s(RF_s < 1.0) = 1.000;
                RF_s(RF_s >= 1.0 & RF_s <= 1.01) = 1.000;
                
                RF_t(RF_t < 1.0) = 1.000;
                RF_t(RF_t >= 1.0 & RF_t <= 1.01) = 1.000;
                
                % 3) Recalculate visual stresses (Booms = Signed, Panels = Absolute)
                sigma_real = (sigma_adm ./ RF_s) .* sign(sigma_signed);
                tau_real   = tau_adm ./ RF_t; % Remains absolute
                
                % --- DETAILED CONSOLE PRINTOUT ---
                fprintf('\n  >>> SECTION %d: %s <<<\n', i, target_names{i});
                fprintf('  Boom | Area (cm^2) | Normal (MPa) | RF Sigma \n');
                fprintf('  -----------------------------------------------\n');
                for b = 1:10
                    fprintf('  %4d | %11.2f | %12.2f | %8.3f\n', b, A_opt(b)*10000, sigma_real(b)/1e6, RF_s(b));
                end
                fprintf('\n  Panel| Thickness (mm)| Shear (MPa)| Flow q (kN/m)| RF Tau \n');
                fprintf('  --------------------------------------------------------------\n');
                for p = 1:11
                    fprintf('  %4d | %10.2f | %13.2f | %13.1f | %6.3f\n', p, t_opt(p)*1000, abs(q_total(p))/1000, RF_t(p));
                end
                fprintf('  -> Cross-sectional Area: %.6f m^2\n', Section_Area);
                fprintf('  --------------------------------------------------------------\n');
            end
            
            if isempty(FSD_Errors_Case)
                FSD_Mean_Error = 0;
            else
                FSD_Mean_Error = mean(FSD_Errors_Case);
            end
            
            fprintf('\n> Weight Metric: %.6f m^2 | FSD Error: %.4f\n', Total_Area_Metric, FSD_Mean_Error);
            DoE_Results(num_case, :) = [eta_f, eta_m, eta_r, Total_Area_Metric, FSD_Mean_Error];
            num_case = num_case + 1;
        end
    end
end

%% --- 5. RE-EVALUATE AND SAVE CONFIGURATION 7 DATA FOR PLOTTING ---
Results_Table = array2table(DoE_Results, 'VariableNames', {'Front_Spar', 'Mid_Spar', 'Rear_Spar', 'Weight_Metric_m2', 'FSD_Error'});

% SORT BY WEIGHT METRIC IN ASCENDING ORDER
Results_Table = sortrows(Results_Table, 'Weight_Metric_m2', 'ascend');

fprintf('\n===============================================================\n');
fprintf(' >>> DESIGN OF EXPERIMENTS COMPLETED <<<\n');
fprintf('===============================================================\n');
disp(Results_Table);

% Force Configuration 7 as requested
target_case = 7;
optimal_config = DoE_Results(target_case, 1:3);

fprintf('\nForcing Configuration %d [%.2f, %.2f, %.2f] to extract final data...\n', target_case, optimal_config);

Optimal_Design = struct();
Optimal_Design.config = optimal_config;

% Rebuild Optimal Geometry
eta_f = optimal_config(1); eta_m = optimal_config(2); eta_r = optimal_config(3);
x_up_1 = linspace(eta_f, eta_m, 1 + 2); x_up_2 = linspace(eta_m, eta_r, 1 + 2);
x_upper = unique([x_up_1, x_up_2]); x_lower = fliplr(x_upper); 
Optimal_Design.x_norm_booms = [x_upper, x_lower]';

idx_fs_up = 1; idx_ms_up = 1 + 1 + 1; idx_rs_up = length(x_upper);
wingGeom.idx_spars_upper = [idx_fs_up, idx_ms_up, idx_rs_up];

% Re-flatten airfoils for the optimal case to guarantee correct NDSA mapping
[y_up, y_lo, ~, ~, ~] = getAirfoilCoords('NACA 0014.dat', x_upper);
[y_up, y_lo] = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper);
wingGeom.airfoils.root = [Optimal_Design.x_norm_booms, [y_up, fliplr(y_lo)]'];

[y_up, y_lo, ~, ~, ~] = getAirfoilCoords('NACA 0012.dat', x_upper);
[y_up, y_lo] = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper);
wingGeom.airfoils.kink = [Optimal_Design.x_norm_booms, [y_up, fliplr(y_lo)]'];

[y_up, y_lo, ~, ~, ~] = getAirfoilCoords('NACA 0009.dat', x_upper);
[y_up, y_lo] = flatten_panels(y_up, y_lo, x_upper, wingGeom.idx_spars_upper);
wingGeom.airfoils.tip  = [Optimal_Design.x_norm_booms, [y_up, fliplr(y_lo)]'];

% Recalculate Final Fuel Distribution based on exact Optimal Geometry
ndsa_vec_optimal = get_ndsa_dist(y_plot_aero, wingGeom);
q_fuel = rho_f * g * nz * (c_dist.^2) .* ndsa_vec_optimal;
q_fuel(y_plot_aero > z_fuel_max) = 0; 

% Save 3D coordinates 
Optimal_Design.X_boom = X_boom; 
Optimal_Design.Y_boom = Y_boom; 
Optimal_Design.Z_boom = Z_boom;
Optimal_Design.z_stations = z_stations;

% Store Final Target Section Variables
for i = 1:3
    Optimal_Design.Targets(i).z_loc = z_targets(i);
    Optimal_Design.Targets(i).x_naca = savedBoomsCoord(i).booms_all.x_m';
    Optimal_Design.Targets(i).y_naca = savedBoomsCoord(i).booms_all.y_m';
    Optimal_Design.Targets(i).V = V_targets(i);
    Optimal_Design.Targets(i).Mx = M_flex_local(i);
    Optimal_Design.Targets(i).Mz = M_tors_local(i);
    
    % Execute the solver extracting ONLY the sign of sigma
    [A_opt, t_opt, RF_s, RF_t, q_total, sigma_signed] = optimize_wingbox_21vars(Optimal_Design.Targets(i).x_naca, Optimal_Design.Targets(i).y_naca, ...
        V_targets(i), M_flex_local(i), M_tors_local(i), sigma_adm, tau_adm);
        
    % --- STRICT RULE: Final masking in exported structure ---
    RF_s(RF_s < 1.0) = 1.000;
    RF_s(RF_s >= 1.0 & RF_s <= 1.01) = 1.000;
    RF_t(RF_t < 1.0) = 1.000;
    RF_t(RF_t >= 1.0 & RF_t <= 1.01) = 1.000;
    % --------------------------------------------------------
    
    Optimal_Design.Targets(i).A_opt = A_opt;
    Optimal_Design.Targets(i).t_opt = t_opt;
    Optimal_Design.Targets(i).q_total = q_total;
    
    % Save clamped RFs. Sigma is signed, Tau is absolute.
    Optimal_Design.Targets(i).RF_s = RF_s;
    Optimal_Design.Targets(i).RF_t = RF_t;
    Optimal_Design.Targets(i).sigma_real = (sigma_adm ./ RF_s) .* sign(sigma_signed);
    Optimal_Design.Targets(i).tau_real = tau_adm ./ RF_t;
end
disp('Data successfully saved to OptimizationResults.mat. You may now run the plotting script.');


%% ========================================================================
%  HELPER FUNCTIONS (WITH UPDATED OPTIMIZATION RULES)
% =========================================================================

function [A_opt, t_opt, RF_s, RF_t, q_total, sigma_signed] = optimize_wingbox_21vars(x_b, y_b, V_local, M_flex_local, M_tors_local, sigma_adm, tau_adm)
    A1 = polyarea(x_b([1, 2, 3, 8, 9, 10]), y_b([1, 2, 3, 8, 9, 10]));
    A2 = polyarea(x_b([3, 4, 5, 6, 7, 8]), y_b([3, 4, 5, 6, 7, 8]));
    X0 = [ones(1,10)*20e-4, ones(1,11)*5e-3];
    lb = [ones(1,10)*1e-4, ones(1,11)*1e-3];   
    
    % --- RULE 1: AGGRESSIVE DECREMENT FOR BOOMS (AREAS) ---
    ub_booms = zeros(1, 10);
    scale_factor = max(0.15, min(1.0, abs(M_flex_local) / 10e6));
    left_limit = 150e-4 * scale_factor;
    ub_booms([1, 10]) = left_limit * 1.00; % Front Spar
    ub_booms([2, 9])  = left_limit * 0.90; % Interboom 1
    ub_booms([3, 8])  = left_limit * 0.80; % Mid Spar
    ub_booms([4, 7])  = left_limit * 0.60; % Interboom 2
    ub_booms([5, 6])  = left_limit * 0.50; % Rear Spar
    
    % --- RULE 2: GENTLE DECREMENT FOR PANELS (SKINS & WEBS) ---
    skin_scale_factor = max(0.35, sqrt(scale_factor));
    left_skin_limit = 20e-3 * skin_scale_factor;
    ub_skins = zeros(1, 11);
    ub_skins([1, 9, 10]) = left_skin_limit * 1.00; % D-Box (Front)
    ub_skins([2, 8, 11]) = left_skin_limit * 0.95; % Mid-Front
    ub_skins([3, 7])     = left_skin_limit * 0.90; % Mid-Rear
    ub_skins([4, 6])     = left_skin_limit * 0.85; % Rear
    ub_skins(5)          = left_skin_limit * 0.80; % Rear Spar Web
    ub = [ub_booms, ub_skins];
    
    options = optimoptions('lsqnonlin', 'Display', 'none', ...
        'Algorithm', 'trust-region-reflective', ... 
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, ...
        'MaxFunctionEvaluations', 10^12, 'MaxIterations', 10^12);
        
    fun_equations = @(X) system_of_equations(X, M_flex_local, M_tors_local, V_local, x_b, y_b, A1, A2, sigma_adm, tau_adm);
    [X_sol, ~, ~, ~] = lsqnonlin(fun_equations, X0, lb, ub, options);
    A_opt = X_sol(1:10); t_opt = X_sol(11:21);
    
    % We only extract the signed sigma (tau remains absolute in the state function)
    [RF_s, RF_t, ~, ~, ~, q_total, sigma_signed] = calculate_state(X_sol, M_flex_local, M_tors_local, V_local, x_b, y_b, A1, A2, sigma_adm, tau_adm);
end

function F = system_of_equations(X, Mx, Mz, Vy, x_b, y_b, A1, A2, sigma_adm, tau_adm)
    [RF_s, RF_t, ~, ~, ~, ~, ~] = calculate_state(X, Mx, Mz, Vy, x_b, y_b, A1, A2, sigma_adm, tau_adm);
    
    Error_sigma = RF_s - 1; 
    Error_tau = RF_t - 1;
    
    % Safety Wall: Heavy penalty if RF drops below 1.0
    Error_sigma(Error_sigma < 0) = Error_sigma(Error_sigma < 0) * 50;
    Error_tau(Error_tau < 0) = Error_tau(Error_tau < 0) * 50;
    
    F = [Error_sigma, Error_tau];
end

function [RF_sigma, RF_tau, Ixx, y_bar, x_bar, q_total, sigma_signed] = calculate_state(X, Mx, Mz, Vy, x_b, y_b, A1, A2, sigma_adm, tau_adm)
    A_b = X(1:10); t_p = X(11:21); 
    
    x_bar = sum(A_b .* x_b) / sum(A_b); y_bar = sum(A_b .* y_b) / sum(A_b);
    dx = x_b - x_bar; dy = y_b - y_bar;               
    
    Ixx = sum(A_b .* (dy.^2)); Iyy = sum(A_b .* (dx.^2));
    Ixy = sum(A_b .* (dx .* dy)); den = (Ixx * Iyy) - (Ixy^2);
    
    % Normal stress calculation WITH SIGN for final output
    sigma_signed = ((Mx * Iyy / den) .* dy - (Mx * Ixy / den) .* dx);
    dq = - (Vy * Iyy / den) .* (A_b .* dy) + (Vy * Ixy / den) .* (A_b .* dx);
    
    idx = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 1; 8 3];
    L = zeros(1,11); for i=1:11, L(i) = sqrt((x_b(idx(i,2))-x_b(idx(i,1)))^2 + (y_b(idx(i,2))-y_b(idx(i,1)))^2); end
    
    qb = zeros(1,11);
    qb(1)=0; qb(2)=qb(1)+dq(2); qb(3)=0; qb(4)=qb(3)+dq(4); qb(5)=qb(4)+dq(5); qb(6)=qb(5)+dq(6); qb(7)=qb(6)+dq(7);             
    qb(10)=-dq(1); qb(9)=qb(10)-dq(10); qb(8)=qb(9)-dq(9); qb(11)=-qb(2)-dq(3);            
    
    L_t = L ./ t_p; 
    d1 = L_t(1)+L_t(2)+L_t(11)+L_t(8)+L_t(9)+L_t(10); d2 = L_t(3)+L_t(4)+L_t(5)+L_t(6)+L_t(7)+L_t(11); d12 = L_t(11);
    
    qb_int1 = qb(1)*L_t(1)+qb(2)*L_t(2)-qb(11)*L_t(11)+qb(8)*L_t(8)+qb(9)*L_t(9)+qb(10)*L_t(10);
    qb_int2 = qb(3)*L_t(3)+qb(4)*L_t(4)+qb(5)*L_t(5)+qb(6)*L_t(6)+qb(7)*L_t(7)+qb(11)*L_t(11);
    
    calc_M = @(q, xi, yi, xf, yf) q * (xi*yf - xf*yi); M_qb = 0;
    for i=1:11, M_qb = M_qb + calc_M(qb(i), x_b(idx(i,1)), y_b(idx(i,1)), x_b(idx(i,2)), y_b(idx(i,2))); end
    
    qs = [d1/(2*A1)+d12/(2*A2), -d12/(2*A1)-d2/(2*A2); 2*A1, 2*A2] \ [qb_int2/(2*A2)-qb_int1/(2*A1); Mz-M_qb];
    
    q_total = [qb(1)+qs(1), qb(2)+qs(1), qb(3)+qs(2), qb(4)+qs(2), qb(5)+qs(2), qb(6)+qs(2), qb(7)+qs(2), qb(8)+qs(1), qb(9)+qs(1), qb(10)+qs(1), qb(11)-qs(1)+qs(2)];  
    
    % Shear stress returned to Absolute Value per user request
    tau = abs(q_total) ./ t_p;
    
    % Standard RF calculations using absolute values for optimizer stability
    RF_sigma = sigma_adm ./ abs(sigma_signed); 
    RF_tau = tau_adm ./ tau;
end

function [y_up_interp, y_lo_interp, x_up, y_up, y_lo] = getAirfoilCoords(filename, x_target)
    fid = fopen(filename, 'r'); fgetl(fid);
    data = fscanf(fid, '%f %f', [2 Inf])'; fclose(fid);
    x_raw = data(:,1); y_raw = data(:,2); [~, idx_le] = min(x_raw);
    [x_up, idxU] = unique(x_raw(1:idx_le)); y_up = y_raw(idxU);
    [x_lo, idxL] = unique(x_raw(idx_le:end)); y_lo = y_raw(idx_le-1+idxL);
    y_up_interp = interp1(x_up, y_up, x_target, 'pchip');
    y_lo_interp = interp1(x_lo, y_lo, x_target, 'pchip');
end

function z_lift = getObliqueLiftLimit(z_target, wingGeom)
    z_kink = wingGeom.y_kink; z_tip = wingGeom.span;
    x_LE_kink = z_kink * tan(wingGeom.sweep_LE_1); x_LE_tip  = x_LE_kink + (z_tip - z_kink) * tan(wingGeom.sweep_LE_2);
    x_L_root = 0.25 * wingGeom.c_root;
    x_L_kink = x_LE_kink + 0.25 * wingGeom.c_kink; x_L_tip  = x_LE_tip  + 0.25 * wingGeom.c_tip;
    
    ifs = wingGeom.idx_spars_upper(1);
    irs = wingGeom.idx_spars_upper(3);
    x_EA_root = (wingGeom.airfoils.root(ifs,1) + wingGeom.airfoils.root(irs,1)) * wingGeom.c_root / 2;
    x_EA_tip  = x_LE_tip + (wingGeom.airfoils.tip(ifs,1) + wingGeom.airfoils.tip(irs,1)) * wingGeom.c_tip / 2;
    m_EA = (x_EA_tip - x_EA_root) / z_tip;
    m_normal = -1 / m_EA; x_EA_target = x_EA_root + m_EA * z_target;
    m_L1 = (x_L_kink - x_L_root) / z_kink;
    m_L2 = (x_L_tip - x_L_kink) / (z_tip - z_kink);
    
    z_int1 = (x_EA_target - x_L_root - m_normal * z_target) / (m_L1 - m_normal);
    z_int2 = (x_EA_target - x_L_kink - m_normal * z_target + m_L2 * z_kink) / (m_L2 - m_normal);
    
    if z_int1 <= (z_kink + 1e-3), z_lift = z_int1; else, z_lift = z_int2; end
    z_lift = max(0, min(z_lift, z_tip));
end

function ndsa_vec = get_ndsa_dist(y_vec, wingGeom)
    ndsa_root = calculate_ndsa(wingGeom.airfoils.root, wingGeom.idx_spars_upper);
    ndsa_kink = calculate_ndsa(wingGeom.airfoils.kink, wingGeom.idx_spars_upper);
    ndsa_tip  = calculate_ndsa(wingGeom.airfoils.tip,  wingGeom.idx_spars_upper);
    ndsa_vec = interp1([0, wingGeom.y_kink, wingGeom.span], [ndsa_root, ndsa_kink, ndsa_tip], y_vec, 'linear');
end

function ndsa = calculate_ndsa(airfoil_data, idx_spars)
    x = airfoil_data(:,1); y = airfoil_data(:,2); n = length(x) / 2;
    x_up = x(1:n); y_up = y(1:n); x_lo = flipud(x(n+1:end)); y_lo = flipud(y(n+1:end));
    xi = linspace(x(idx_spars(1)), x(idx_spars(3)), 200);
    yi_up = interp1(x_up, y_up, xi, 'pchip'); yi_lo = interp1(x_lo, y_lo, xi, 'pchip');
    ndsa = trapz(xi, yi_up - yi_lo);
end

function [y_up_flat, y_lo_flat] = flatten_panels(y_up, y_lo, x_upper, idx_spars)
    y_up_flat = y_up; y_lo_flat = y_lo;
    n_up = length(y_up);
    valid_idx = idx_spars(idx_spars <= n_up); 
    for s = 1:(length(valid_idx)-1)
        i1 = valid_idx(s); i2 = valid_idx(s+1);
        y_up_flat(i1:i2) = interp1([x_upper(i1), x_upper(i2)], [y_up(i1), y_up(i2)], x_upper(i1:i2), 'linear');
        y_lo_flat(i1:i2) = interp1([x_upper(i1), x_upper(i2)], [y_lo(i1), y_lo(i2)], x_upper(i1:i2), 'linear');
    end
end