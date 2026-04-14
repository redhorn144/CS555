% --- Parameters ---
L = 1.0;
c = 1.0;
nu = 1e-3;
f = 1.0;
E = 20;          % Number of nonuniform elements
N = E + 1;       % Number of global nodes
s = 0.7;         % Geometric progression scale factor

% --- Mesh Generation ---
L1 = L * (1 - s) / (1 - s^E);   % Calculate first element length
Le = L1 * s.^(0:E-1);           % Array of all element lengths
x = [0, cumsum(Le)]';           % Nodal coordinates

% --- Initialize Global Matrices ---
K = sparse(N, N); 
F = zeros(N, 1);

% --- Assembly Process ---
for e = 1:E
    h_e = Le(e); % Local element length
    
    % Local matrices
    A_e = (1.0 / h_e) * [1.0, -1.0; -1.0, 1.0];
    C_e = 0.5 * [-1.0, 1.0; -1.0, 1.0];
    
    % Combined local stiffness
    K_e = nu * A_e + c * C_e;
    
    % Local load vector
    F_e = (f * h_e / 2.0) * [1.0; 1.0];
   
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
u_ex = x - exp((c / nu) * (x - L));
u_ex(1) = 0.0;   
u_ex(end) = 0.0; 

% --- Calculate Error ---
interior_idx = 2:(N-1);
rel_error = abs(u_ex(interior_idx) - u_h(interior_idx)) ./ abs(u_ex(interior_idx));
max_rel_error = max(rel_error);

fprintf('Maximum pointwise relative error: %.4e\n', max_rel_error);

% --- Plotting ---
figure;
plot(x, u_ex, 'k-', 'LineWidth', 1.5); hold on;
plot(x, u_h, 'bo-', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
title('Steady-State Advection-Diffusion (Nonuniform E=20, s=0.7)');
xlabel('x');
ylabel('u(x)');
legend('Analytical', 'Linear FEM', 'Location', 'NorthWest');
grid on;
hold off;