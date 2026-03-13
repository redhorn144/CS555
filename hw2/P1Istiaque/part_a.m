% CS 555 HW 2 - Part 1a
% 1D Burgers Equation using Finite Differences and BDF3/EXT3

%% Parameters
N = 200;
nu = 1 / (100 * pi);   % Viscosity from Basdevant et al.
dx = 1 / N;            % Uniform spacing
x = linspace(0, 1, N+1)';
u_init = sin(pi * x);  % Initial condition

% Time stepping parameters
CFL = 0.1;
dt = CFL * dx;         % dt = 0.0005 for stability
t_end = 2.0;
nsteps = round(t_end / dt);
plot_steps = round(0.1 / dt); % Step interval to plot every 0.1s

%% Matrix Construction (Interior points only)
Nx = N - 1; 
e = ones(Nx, 1);
M = speye(Nx);

% Diffusion operator A (nu * d^2/dx^2) using a 3-point stencil
A = (nu / dx^2) * spdiags([e, -2*e, e], [-1, 0, 1], Nx, Nx);

% Derivative operator D (d/dx) using a 3-point stencil
D = (1 / (2*dx)) * spdiags([-e, e], [-1, 1], Nx, Nx);

%% Initialization
u_hist = zeros(Nx, 3);
N_hist = zeros(Nx, 3);

% Set initial conditions for the history arrays
u_int = u_init(2:end-1);
u_hist(:, 1) = u_int;
N_hist(:, 1) = -u_int .* (D * u_int); % Convective form: -u * du/dx

% Setup the plot
figure('Name', '1D Burgers Equation', 'Position', [100, 100, 800, 600]);
hold on; grid on;
plot(x, u_init, 'k-', 'LineWidth', 1.5, 'DisplayName', 't = 0.0');
colors = parula(21); 
c_idx = 2;

%% Time Stepping Loop (BDFk / EXTk)
for n = 1:nsteps
    
    % Booting off BDF1/2 for the first steps, then BDF3
    k = min(n, 3);
    
    % BDF / EXT coefficients
    if k == 1
        b0 = 1; b1 = -1; b2 = 0; b3 = 0;
        a1 = 1; a2 = 0; a3 = 0;
    elseif k == 2
        b0 = 1.5; b1 = -2; b2 = 0.5; b3 = 0;
        a1 = 2; a2 = -1; a3 = 0;
    elseif k == 3
        b0 = 11/6; b1 = -3; b2 = 1.5; b3 = -1/3;
        a1 = 3; a2 = -3; a3 = 1;
    end
    
    % Recompute the implicit operator H during the boot-up steps
    if n <= 3
        H = b0 * M - dt * A; 
    end
    
    % Build the Right Hand Side using historical states
    RHS = -(b1 * u_hist(:,1) + b2 * u_hist(:,2) + b3 * u_hist(:,3)) ...
          + dt * (a1 * N_hist(:,1) + a2 * N_hist(:,2) + a3 * N_hist(:,3));
      
    % Solve the linear system for the new interior values
    u_new = H \ RHS;
    
    % Shift history arrays
    u_hist(:, 3) = u_hist(:, 2);
    u_hist(:, 2) = u_hist(:, 1);
    u_hist(:, 1) = u_new;
    
    N_hist(:, 3) = N_hist(:, 2);
    N_hist(:, 2) = N_hist(:, 1);
    N_hist(:, 1) = -u_new .* (D * u_new);
    
    % Plotting at specified 0.1 intervals
    if mod(n, plot_steps) == 0
        t_current = n * dt;
        u_full = [0; u_new; 0]; % Reapply homogeneous Dirichlet BCs
        plot(x, u_full, 'Color', colors(c_idx,:), 'DisplayName', sprintf('t = %.1f', t_current));
        c_idx = c_idx + 1;
    end
end

% Format the final plot
title('1D Burgers Equation (Uniform Spacing, N=200)');
xlabel('x');
ylabel('u(x,t)');
legend('Location', 'eastoutside');