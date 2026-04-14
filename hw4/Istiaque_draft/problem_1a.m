% --- Parameters ---
L = 1.0;
c = 1.0;
nu = 1e-3;
f = 1.0;
E = 100;         % Number of elements
N = E + 1;       % Number of global nodes
h = L / E;       % Uniform element length

x = linspace(0, L, N)';


K = sparse(N, N); 
F = zeros(N, 1);

% --- Assembly Process ---
for e = 1:E
    % Local matrices
    A_e = (1.0 / h) * [1.0, -1.0; -1.0, 1.0];
    C_e = 0.5 * [-1.0, 1.0; -1.0, 1.0];
    
    % Combined local stiffness
    K_e = nu * A_e + c * C_e;
    
    % Local load vector
    F_e = (f * h / 2.0) * [1.0; 1.0];
    
   
    idx = [e, e+1]; 
    K(idx, idx) = K(idx, idx) + K_e;
    F(idx) = F(idx) + F_e;
end

% --- Apply Dirichlet Boundary Conditions ---
% u(0) = 0
K(1, :) = 0.0;
K(1, 1) = 1.0;
F(1) = 0.0;

% u(L) = 0
K(end, :) = 0.0;
K(end, end) = 1.0;
F(end) = 0.0;

% --- Solve the Linear System ---
u_h = K \ F;

% --- Analytical Solution ---
% Using the stable approximation to avoid exp(1000) overflow
u_ex = x - exp((c / nu) * (x - L));
u_ex(1) = 0.0;   % Enforce exact BC at x=0
u_ex(end) = 0.0; % Enforce exact BC at x=1

% --- Calculate Error ---
interior_idx = 2:(N-1);
rel_error = abs(u_ex(interior_idx) - u_h(interior_idx)) ./ abs(u_ex(interior_idx));
max_rel_error = max(rel_error);

fprintf('Maximum pointwise relative error: %.4e\n', max_rel_error);

% --- Plotting ---
figure;
plot(x, u_ex, 'k-', 'LineWidth', 1.5); hold on;
plot(x, u_h, 'ro-', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
title('Steady-State Advection-Diffusion (Uniform E=100)');
xlabel('x');
ylabel('u(x)');
legend('Analytical', 'Linear FEM', 'Location', 'NorthWest');
grid on;
hold off;