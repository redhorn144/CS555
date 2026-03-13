% CS 555 HW 2 - Part 1b (Corrected for Boundary Derivatives)
% 1D Burgers Equation: Tracking Maximum Spatial Derivative

%% Parameters & Setup
N = 200;
nu = 1 / (100 * pi);   % Viscosity
dx = 1 / N;            % Uniform spacing
x = linspace(0, 1, N+1)';
u0 = sin(pi * x);      % Initial condition

% Timestep calculation
CFL = 0.1;
c_max = 1.0;           % Max wave speed
dt = CFL * dx / c_max; % dt = 0.0005
t_end = 2.0;
nsteps = round(t_end / dt);

% Interior nodes count
nx = N - 1;
M = speye(nx);

% 3-point stencils for interior nodes 
e = ones(nx, 1);
% 2nd derivative operator (Diffusion)
A = (nu / dx^2) * spdiags([e, -2*e, e], [-1, 0, 1], nx, nx);
% 1st derivative operator (Convection)
D = (1 / (2*dx)) * spdiags([-e, e], [-1, 1], nx, nx);

%% Initialization for BDF/EXT
% History arrays for the 3 previous steps
u_hist = zeros(nx, 3);
N_hist = zeros(nx, 3);

u_int = u0(2:end-1);
u_hist(:, 1) = u_int;
% Convective term N(u) = -u * u_x 
N_hist(:, 1) = -u_int .* (D * u_int);

% Arrays to track s(t)
s_t = zeros(nsteps, 1);
t_array = (1:nsteps) * dt;

%% Time Stepping Loop (BDFk/EXTk)
for n = 1:nsteps
    k = min(n, 3); % Boot from k=1, to k=2, then stay at k=3 
    
    % Coefficients based on the assignment snippet 
    if k == 1
        b0 = 1;      b1 = -1;  b2 = 0;     b3 = 0; 
        a1 = 1;      a2 = 0;   a3 = 0;
    elseif k == 2
        b0 = 1.5;    b1 = -2;  b2 = 0.5;   b3 = 0;
        a1 = 2;      a2 = -1;  a3 = 0;
    elseif k == 3
        b0 = 11/6;   b1 = -3;  b2 = 1.5;   b3 = -1/3;
        a1 = 3;      a2 = -3;  a3 = 1;
    end
    
    % Update the implicit operator matrix H during boot-up 
    if n <= 3
        H = b0 * M - dt * A;
    end
    
    % Build the Right Hand Side 
    RHS = -(b1 * u_hist(:,1) + b2 * u_hist(:,2) + b3 * u_hist(:,3)) ...
          + dt * (a1 * N_hist(:,1) + a2 * N_hist(:,2) + a3 * N_hist(:,3));
      
    % Solve for the new interior field
    u_new = H \ RHS;
    
    % --- PART 1b DERIVATIVE CALCULATION ---
    u_full = [0; u_new; 0]; 
    
    % Derivative at interior points 
    u_prime_int = D * u_new;
    
    % 2nd-order one-sided derivative at x = 0 (Forward difference)
    u_prime_0 = (-3*u_full(1) + 4*u_full(2) - u_full(3)) / (2*dx);
    
    % 2nd-order one-sided derivative at x = 1 (Backward difference)
    u_prime_N = (3*u_full(end) - 4*u_full(end-1) + u_full(end-2)) / (2*dx);
    
    % Find the global max absolute slope across the entire domain
    s_t(n) = max(abs([u_prime_0; u_prime_int; u_prime_N]));
    
    % Shift history arrays backward
    u_hist(:, 3) = u_hist(:, 2);
    u_hist(:, 2) = u_hist(:, 1);
    u_hist(:, 1) = u_new;
    
    N_hist(:, 3) = N_hist(:, 2);
    N_hist(:, 2) = N_hist(:, 1);
    N_hist(:, 1) = -u_new .* (D * u_new);
end

%% Analysis & Plotting
[s_star, max_idx] = max(s_t);
t_star = t_array(max_idx);

fprintf('Timestep size dt = %.5f\n', dt);
fprintf('Maximum slope s* = %.5f\n', s_star);
fprintf('Time of maximum t* = %.5f\n', t_star);

% Plot s(t) vs time
figure('Name', 'Maximum Absolute Spatial Derivative');
plot(t_array, s_t, 'k-', 'LineWidth', 1.5);
grid on;
title('Maximum Absolute Slope s(t) vs. Time');
xlabel('Time t');
ylabel('s(t) = max |du/dx|');