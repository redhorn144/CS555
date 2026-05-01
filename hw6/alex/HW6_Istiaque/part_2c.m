% part_2c.m
% Calculates and prints the final table for F and xp/L

hdr; 

Ex = 120; Ey = 40;
L = .0041; W = .15625*L; T = .09375*L; Ta_val = pi/180;
h1 = 3.70e-7; h2 = 2.50e-7; patm = 101325;
rho = 1.225; nu_air = 1.5e-5; mu = rho*nu_air; U = 20;

[x, y, t] = box_elem(Ex, Ey, L, W); E = 2*Ex*Ey; nv = 3;
[AL, BL, Q, t, areaL] = abqfem([x y], t);
nb = size(Q, 2); xL = x(t'); yL = y(t'); xe = sum(xL, 1)'/nv; ye = sum(yL, 1)'/nv;
order = 1; Nq = 3; [z, w] = trigausspoints(Nq); rq = z(:,1); sq = z(:,2); Bh = diag(w);
nq = length(rq); Dr = zeros(nq, 3); Ds = Dr; Jq = zeros(nq, 3);
for k = 1:3
    [Dr(:,k), Ds(:,k)] = basis_deriv_12(rq, sq, k, order);
    Jq(:,k) = basis_tri_12(rq, sq, k, order);
end
Xr = Dr*xL; Yr = Dr*yL; Xs = Ds*xL; Ys = Ds*yL;
Jac = Xr.*Ys - Xs.*Yr; Rx = Ys ./ Jac; Sx = -Yr ./ Jac;
Bq = Bh*Jac; Ie = speye(E); DLr = kron(Ie, Dr'); DLs = kron(Ie, Ds'); JL = kron(Ie, Jq);

% 1a: Incomp 2D
he = profile_taper_flat(xe, L, T, Ta_val, h1, h2);
nu_mat = (1./(12*mu))*(he.^3);
An = kron(spdiags(nu_mat, 0, E, E), speye(3)) * AL;
hq = Jq*profile_taper_flat(xL, L, T, Ta_val, h1, h2);
Uh = (U/2)*(Bq.*hq);
CL = (DLr*spdiags(reshape(Rx.*Uh, nq*E, 1), 0, nq*E, nq*E) + DLs*spdiags(reshape(Sx.*Uh, nq*E, 1), 0, nq*E, nq*E)) * JL;
Cb = Q'*CL*Q;
R_2D = restriction(nb, boundedges([x,y], t));
A = R_2D*(Q'*An*Q)*R_2D'; C = R_2D*Cb*R_2D';
Pb_1a = R_2D' * (A \ (R_2D*(Cb*ones(nb,1))));
Bb = Q'*BL*Q;
F_1a = sum(Bb * Pb_1a); xp_1a = (x' * Bb * Pb_1a) / F_1a;

% 1b: Incomp 1D
he_1b = profile_taper_flat(xe, L, T, 0, h1, h2);
nu_mat_1b = (1./(12*mu))*(he_1b.^3);
An_1b = kron(spdiags(nu_mat_1b, 0, E, E), speye(3)) * AL;
hq_1b = Jq*profile_taper_flat(xL, L, T, 0, h1, h2);
Uh_1b = (U/2)*(Bq.*hq_1b);
CL_1b = (DLr*spdiags(reshape(Rx.*Uh_1b, nq*E, 1), 0, nq*E, nq*E) + DLs*spdiags(reshape(Sx.*Uh_1b, nq*E, 1), 0, nq*E, nq*E)) * JL;
Cb_1b = Q'*CL_1b*Q;
R_1D = restriction(nb, unique([find(abs(x) < 1e-10); find(abs(x - L) < 1e-10)]));
A_1b = R_1D*(Q'*An_1b*Q)*R_1D';
Pb_1b = R_1D' * (A_1b \ (R_1D*(Cb_1b*ones(nb,1))));
F_1b = sum(Bb * Pb_1b); xp_1b = (x' * Bb * Pb_1b) / F_1b;

% 2a: Comp 2D
Pb_2a = zeros(nb, 1);
rhs_comp = R_2D * (Cb * (patm * ones(nb, 1)));
h3 = he.^3;
for iter = 1:15
    pa_e = ( (Pb_2a(t(:,1)) + Pb_2a(t(:,2)) + Pb_2a(t(:,3)))/3 ) + patm;
    nu_val = (pa_e .* h3) ./ (12 * mu);
    An_c = kron(spdiags(nu_val, 0, E, E), speye(3)) * AL;
    Pb_new = R_2D' * ((R_2D*(Q'*An_c*Q)*R_2D' - C) \ rhs_comp);
    if max(abs(Pb_new - Pb_2a)) < 1e-3, Pb_2a = Pb_new; break; end
    Pb_2a = Pb_new;
end
F_2a = sum(Bb * Pb_2a); xp_2a = (x' * Bb * Pb_2a) / F_2a;

% Print Table
fprintf('                PART 2C RESULTS TABLE                  \n');
fprintf('%-25s | %-10s | %-10s\n', 'Case', 'Load F (N)', 'xp/L');
fprintf('-------------------------------------------------------\n');
fprintf('%-25s | %-10.4f | %-10.4f\n', '1a: Incompressible 2D', F_1a, xp_1a/L);
fprintf('%-25s | %-10.4f | %-10.4f\n', '1b: Incompressible 1D', F_1b, xp_1b/L);
fprintf('%-25s | %-10.4f | %-10.4f\n', '2a: Compressible 2D', F_2a, xp_2a/L);
fprintf('=======================================================\n\n');