% CS 555 HW 2 - Part 1e
% Convergence study across 4 cases (Uniform vs Chebyshev, Convective vs Conservation)

%% Setup
s_analytical = 152.00516;  % Analytical max slope from paper
nu = 1 / (100 * pi);       % Viscosity
CFL = 0.1;
c_max = 1.0;
t_end = 2.0;

k_vals = 5:11;
N_vals = 2.^k_vals;

case_names = {
    'i. Uniform spacing, convective form', ...
    'ii. Chebyshev spacing, convective form', ...
    'iii. Uniform spacing, conservation form', ...
    'iv. Chebyshev spacing, conservation form'
};

%% Main Loop Over the 4 Cases
for c = 1:4
    fprintf('\nCase %s\n', case_names{c});
    fprintf('------------------------------------------------------------------\n');
    fprintf('   N      nsteps         s* rel.err.      ratio\n');
    fprintf('------------------------------------------------------------------\n');
    
    prev_err = NaN; % Initialize previous error for ratio calculation
    
    for i = 1:length(N_vals)
        N = N_vals(i);
        
        % As instructed, use uniform dx for calculating dt in all cases
        dt = CFL * (1/N) / c_max;
        nsteps = round(t_end / dt);
        
        % 1. Grid Spacing
        if c == 1 || c == 3
            % Uniform Spacing
            x = linspace(0, 1, N+1)';
        else
            % Chebyshev Spacing
            j = (0:N)';
            x = (1 - cos(pi * j / N)) / 2;
        end
        
        u0 = sin(pi * x);
        nx = N - 1;
        M = speye(nx);
        
        % 2. Matrix Construction (A and D for arbitrary spacing)
        A = sparse(nx, nx);
        D = sparse(nx, nx);
        
        for row = 1:nx
            idx = row + 1; % Index in full 'x' array
            hm = x(idx) - x(idx-1); % h minus
            hp = x(idx+1) - x(idx); % h plus
            
            % D matrix (1st derivative operator)
            if row > 1
                D(row, row-1) = -1 / (hm + hp);
            end
            if row < nx
                D(row, row+1) =  1 / (hm + hp);
            end
            
            % A matrix (2nd derivative diffusion operator)
            coeff = nu * 2 / (hm + hp);
            if row > 1
                A(row, row-1) = coeff / hm;
            end
            A(row, row) = -coeff * (1/hm + 1/hp);
            if row < nx
                A(row, row+1) = coeff / hp;
            end
        end
        
        % 3. Initialization
        u_hist = zeros(nx, 3);
        N_hist = zeros(nx, 3);
        u_int = u0(2:end-1);
        u_hist(:, 1) = u_int;
        
        if c == 1 || c == 2
            % Convective form: -u * u_x
            N_hist(:, 1) = -u_int .* (D * u_int);
        else
            % Conservation form: -0.5 * (u^2)_x
            N_hist(:, 1) = -0.5 * (D * (u_int.^2));
        end
        
        max_s = 0;
        
        % 4. Time Stepping Loop
        for n = 1:nsteps
            k_bdf = min(n, 3);
            if k_bdf == 1
                b0 = 1; b1 = -1; b2 = 0; b3 = 0; a1 = 1; a2 = 0; a3 = 0;
            elseif k_bdf == 2
                b0 = 1.5; b1 = -2; b2 = 0.5; b3 = 0; a1 = 2; a2 = -1; a3 = 0;
            elseif k_bdf == 3
                b0 = 11/6; b1 = -3; b2 = 1.5; b3 = -1/3; a1 = 3; a2 = -3; a3 = 1;
            end
            
            if n <= 3
                H = b0 * M - dt * A;
            end
            
            RHS = -(b1 * u_hist(:,1) + b2 * u_hist(:,2) + b3 * u_hist(:,3)) ...
                  + dt * (a1 * N_hist(:,1) + a2 * N_hist(:,2) + a3 * N_hist(:,3));
              
            u_new = H \ RHS;
            
            % 5. Derivative Calculation (Including non-uniform boundaries)
            u_full = [0; u_new; 0];
            u_prime_int = D * u_new;
            
            h1 = x(2) - x(1); h2 = x(3) - x(2);
            u_prime_0 = -(2*h1 + h2)/(h1*(h1+h2))*u_full(1) + (h1+h2)/(h1*h2)*u_full(2) - h1/(h2*(h1+h2))*u_full(3);
            
            h1 = x(end) - x(end-1); h2 = x(end-1) - x(end-2);
            u_prime_N = (2*h1 + h2)/(h1*(h1+h2))*u_full(end) - (h1+h2)/(h1*h2)*u_full(end-1) + h1/(h2*(h1+h2))*u_full(end-2);
            
            curr_max = max(abs([u_prime_0; u_prime_int; u_prime_N]));
            if curr_max > max_s
                max_s = curr_max;
            end
            
            % Shift history backward
            u_hist(:, 3) = u_hist(:, 2);
            u_hist(:, 2) = u_hist(:, 1);
            u_hist(:, 1) = u_new;
            
            % Compute nonlinear terms for history
            if c == 1 || c == 2
                N_hist(:, 3) = N_hist(:, 2);
                N_hist(:, 2) = N_hist(:, 1);
                N_hist(:, 1) = -u_new .* (D * u_new);
            else
                N_hist(:, 3) = N_hist(:, 2);
                N_hist(:, 2) = N_hist(:, 1);
                N_hist(:, 1) = -0.5 * (D * (u_new.^2));
            end
        end
        
        % 6. Error tracking
        rel_err = abs(max_s - s_analytical) / s_analytical;
        if i == 1
            ratio = NaN;
        else
            ratio = prev_err / rel_err;
        end
        prev_err = rel_err;
        
        fprintf('%4d %10d %15.5f %15.5e %10.3f\n', N, nsteps, max_s, rel_err, ratio);
    end
    fprintf('------------------------------------------------------------------\n');
end