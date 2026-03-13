% CS 555 HW 2 - Part 1d
% 1D Burgers Equation: Conservation Form at Finest Resolution

%% Parameters & Setup
N = 2048;              % Finest resolution from preceding parts
nu = 1 / (100 * pi);   % Viscosity
dx = 1 / N;            % Uniform spacing
x = linspace(0, 1, N+1)';
u0 = sin(pi * x);      % Initial condition

% Timestep calculation
CFL = 0.1;
c_max = 1.0;           
dt = CFL * dx / c_max; 
t_end = 2.0;
nsteps = round(t_end / dt);

% Interior nodes count
nx = N - 1;
M = speye(nx);

% 3-point stencils for interior nodes
e = ones(nx, 1);
A = (nu / dx^2) * spdiags([e, -2*e, e], [-1, 0, 1], nx, nx);
D = (1 / (2*dx)) * spdiags([-e, e], [-1, 1], nx, nx);

%% Initialization for BDF/EXT
u_hist = zeros(nx, 3);
N_hist = zeros(nx, 3);

u_int = u0(2:end-1);
u_hist(:, 1) = u_int;

% --- CONSERVATION FORM INITIALIZATION ---
% N(u) = -0.5 * d(u^2)/dx
N_hist(:, 1) = -0.5 * (D * (u_int.^2));

% Arrays to track s(t)
s_t = zeros(nsteps, 1);
t_array = (1:nsteps) * dt;

fprintf('Running Part 1d: Conservation Form (N = %d)...\n', N);

%% Time Stepping Loop (BDFk/EXTk)
for n = 1:nsteps
    k = min(n, 3); 
    
    % BDF/EXT Coefficients
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
    
    if n <= 3
        H = b0 * M - dt * A;
    end
    
    % Build the Right Hand Side 
    RHS = -(b1 * u_hist(:,1) + b2 * u_hist(:,2) + b3 * u_hist(:,3)) ...
          + dt * (a1 * N_hist(:,1) + a2 * N_hist(:,2) + a3 * N_hist(:,3));
      
    u_new = H \ RHS;
    
    % Derivative calculation to find s(t)
    u_full = [0; u_new; 0]; 
    u_prime_int = D * u_new;
    u_prime_0 = (-3*u_full(1) + 4*u_full(2) - u_full(3)) / (2*dx);
    u_prime_N = (3*u_full(end) - 4*u_full(end-1) + u_full(end-2)) / (2*dx);
    
    s_t(n) = max(abs([u_prime_0; u_prime_int; u_prime_N]));
    
    % Shift history arrays backward
    u_hist(:, 3) = u_hist(:, 2);
    u_hist(:, 2) = u_hist(:, 1);
    u_hist(:, 1) = u_new;
    
    % --- CONSERVATION FORM UPDATE ---
    N_hist(:, 3) = N_hist(:, 2);
    N_hist(:, 2) = N_hist(:, 1);
    N_hist(:, 1) = -0.5 * (D * (u_new.^2));
end

%% Analysis
[s_star, max_idx] = max(s_t);
t_star = t_array(max_idx);

fprintf('Finest Resolution N = %d\n', N);
fprintf('Timestep size dt = %.5e\n', dt);
fprintf('Maximum slope s* = %.5f\n', s_star);
fprintf('Time of maximum t* = %.5f\n', t_star);