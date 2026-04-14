L = 1.0; c = 1.0; nu = 1e-3; f = 0.0;
E = 100; N = E + 1; h = L / E;
dt = 0.001; t_end = 1.5;
n_steps = round(t_end / dt);

x = linspace(0, L, N)';

A = sparse(N, N); C = sparse(N, N); B = sparse(N, N);

for e = 1:E
    A_e = (1.0 / h) * [1.0, -1.0; -1.0, 1.0];
    C_e = 0.5 * [-1.0, 1.0; -1.0, 1.0];
    B_e = (h / 6.0) * [2.0, 1.0; 1.0, 2.0];
    
    idx = [e, e+1];
    A(idx, idx) = A(idx, idx) + A_e;
    C(idx, idx) = C(idx, idx) + C_e;
    B(idx, idx) = B(idx, idx) + B_e;
end

U = zeros(N, n_steps + 1);

LHS1 = (1/dt)*B + nu*A;
LHS2 = (3/(2*dt))*B + nu*A;
LHS3 = (11/(6*dt))*B + nu*A;

for n = 1:n_steps
    t_next = n * dt;
    
    if n == 1
        RHS = (1/dt)*B*U(:, n) - c*C*U(:, n);
        LHS = LHS1;
    elseif n == 2
        RHS = (1/dt)*B*(2*U(:, n) - 0.5*U(:, n-1)) - c*C*(2*U(:, n) - U(:, n-1));
        LHS = LHS2;
    else
        RHS = (1/dt)*B*(3*U(:, n) - 1.5*U(:, n-1) + (1/3)*U(:, n-2)) - ...
              c*C*(3*U(:, n) - 3*U(:, n-1) + U(:, n-2));
        LHS = LHS3;
    end
    
    LHS_bc = LHS;
    LHS_bc(1, :) = 0; LHS_bc(1, 1) = 1; RHS(1) = sin(pi * t_next);
    LHS_bc(end, :) = 0; LHS_bc(end, end) = 1; RHS(end) = 0;
    
    U(:, n+1) = LHS_bc \ RHS;
end

figure;
plot(x, U(:, end), 'b-', 'LineWidth', 1.5);
title('Unsteady Advection-Diffusion at t=1.5 (Dirichlet BC)');
xlabel('x'); ylabel('u(x, t=1.5)');
grid on;