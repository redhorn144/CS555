% --- Parameters ---
L = 1.0;
c = 1.0;
nu = 1e-3;
f = 1.0;
E = 20;          
N = E + 1;       


s_vals = 0.5:0.01:0.95;          % Range of s values to test
errors = zeros(length(s_vals), 1); % Array to store max errors

for k = 1:length(s_vals)
    s = s_vals(k);
    
    % Mesh Generation
    L1 = L * (1 - s) / (1 - s^E);   
    Le = L1 * s.^(0:E-1);           
    x = [0, cumsum(Le)]';           
    
    % Initialize Global Matrices
    K = sparse(N, N); 
    F = zeros(N, 1);
    
    % Assembly Process
    for e = 1:E
        h_e = Le(e); 
        
        A_e = (1.0 / h_e) * [1.0, -1.0; -1.0, 1.0];
        C_e = 0.5 * [-1.0, 1.0; -1.0, 1.0];
        K_e = nu * A_e + c * C_e;
        F_e = (f * h_e / 2.0) * [1.0; 1.0];
        
        idx = [e, e+1]; 
        K(idx, idx) = K(idx, idx) + K_e;
        F(idx) = F(idx) + F_e;
    end
    
    % Apply Dirichlet Boundary Conditions
    K(1, :) = 0.0; K(1, 1) = 1.0; F(1) = 0.0;
    K(end, :) = 0.0; K(end, end) = 1.0; F(end) = 0.0;
    
    % Solve the Linear System
    u_h = K \ F;
    
    % Analytical Solution
    u_ex = x - exp((c / nu) * (x - L));
    u_ex(1) = 0.0;   
    u_ex(end) = 0.0; 
    
    % Calculate Error
    interior_idx = 2:(N-1);
    rel_error = abs(u_ex(interior_idx) - u_h(interior_idx)) ./ abs(u_ex(interior_idx));
    errors(k) = max(rel_error);
end

% --- Find Optimal s ---
[min_err, min_idx] = min(errors);
opt_s = s_vals(min_idx);

fprintf('Optimal s value: %.2f\n', opt_s);
fprintf('Minimum maximum pointwise relative error: %.4e\n', min_err);

% --- Plotting ---
figure;
plot(s_vals, errors, 'k-', 'LineWidth', 1.5); hold on;
plot(opt_s, min_err, 'r*', 'MarkerSize', 10, 'LineWidth', 1.5);
title('Error vs. Scale Factor (s) for E=20');
xlabel('Scale Factor (s)');
ylabel('Max Pointwise Relative Error');
legend('Error Curve', sprintf('Minimum (s=%.2f)', opt_s), 'Location', 'North');
grid on;
hold off;