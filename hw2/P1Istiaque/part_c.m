% CS 555 HW 2 - Part 1c
% Convergence study of s* as a function of N (Uniform Spacing)

%% Setup
s_analytical = 152.00516;  % Exact max slope from Table 3
nu = 1 / (100 * pi);       % Viscosity
CFL = 0.1;
c_max = 1.0;
t_end = 2.0;

k_vals = 5:11;
N_vals = 2.^k_vals;
num_runs = length(N_vals);

% Preallocate arrays for the results table
s_star_computed = zeros(num_runs, 1);
rel_error = zeros(num_runs, 1);
nsteps_array = zeros(num_runs, 1);

fprintf('Running convergence study for uniform spacing...\n');
fprintf('------------------------------------------------------------\n');
fprintf('   N      nsteps         s* rel.err.\n');
fprintf('------------------------------------------------------------\n');

%% Main Loop over N
for i = 1:num_runs
    N = N_vals(i);
    dx = 1 / N;
    dt = CFL * dx / c_max; 
    nsteps = round(t_end / dt);
    nsteps_array(i) = nsteps;
    
    x = linspace(0, 1, N+1)';
    u0 = sin(pi * x);
    
    nx = N - 1;
    M = speye(nx);
    e = ones(nx, 1);
    
    % Operators
    A = (nu / dx^2) * spdiags([e, -2*e, e], [-1, 0, 1], nx, nx);
    D = (1 / (2*dx)) * spdiags([-e, e], [-1, 1], nx, nx);
    
    % Initialization
    u_hist = zeros(nx, 3);
    N_hist = zeros(nx, 3);
    
    u_int = u0(2:end-1);
    u_hist(:, 1) = u_int;
    N_hist(:, 1) = -u_int .* (D * u_int);
    
    max_s_current_run = 0; % Track max slope for this specific N
    
    % Time Stepping
    for n = 1:nsteps
        k = min(n, 3);
        
        if k == 1
            b0 = 1; b1 = -1; b2 = 0; b3 = 0; a1 = 1; a2 = 0; a3 = 0;
        elseif k == 2
            b0 = 1.5; b1 = -2; b2 = 0.5; b3 = 0; a1 = 2; a2 = -1; a3 = 0;
        elseif k == 3
            b0 = 11/6; b1 = -3; b2 = 1.5; b3 = -1/3; a1 = 3; a2 = -3; a3 = 1;
        end
        
        if n <= 3
            H = b0 * M - dt * A;
        end
        
        RHS = -(b1 * u_hist(:,1) + b2 * u_hist(:,2) + b3 * u_hist(:,3)) ...
              + dt * (a1 * N_hist(:,1) + a2 * N_hist(:,2) + a3 * N_hist(:,3));
          
        u_new = H \ RHS;
        
        % Derivative calculation including boundaries
        u_full = [0; u_new; 0];
        u_prime_int = D * u_new;
        u_prime_0 = (-3*u_full(1) + 4*u_full(2) - u_full(3)) / (2*dx);
        u_prime_N = (3*u_full(end) - 4*u_full(end-1) + u_full(end-2)) / (2*dx);
        
        % Update maximum slope
        current_max_s = max(abs([u_prime_0; u_prime_int; u_prime_N]));
        if current_max_s > max_s_current_run
            max_s_current_run = current_max_s;
        end
        
        % Shift histories
        u_hist(:, 3) = u_hist(:, 2);
        u_hist(:, 2) = u_hist(:, 1);
        u_hist(:, 1) = u_new;
        
        N_hist(:, 3) = N_hist(:, 2);
        N_hist(:, 2) = N_hist(:, 1);
        N_hist(:, 1) = -u_new .* (D * u_new);
    end
    
    % Store and calculate errors
    s_star_computed(i) = max_s_current_run;
    rel_error(i) = abs(max_s_current_run - s_analytical) / s_analytical;
    
    % Print row to command window dynamically
    fprintf('%4d %10d %15.5f %15.5e\n', N, nsteps, s_star_computed(i), rel_error(i));
end
fprintf('------------------------------------------------------------\n');