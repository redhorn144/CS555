% part_2b.m
% Side-by-side comparison of pressure distributions

hdr; 
%close(1); 

% Grid settings
Ex = 120; Ey = 40;
L = .0041; W = .15625*L;
[x, y, t] = box_elem(Ex, Ey, L, W); 
E = 2*Ex*Ey; nv = 3; 

% Common Physical Parameters
rho = 1.225; nu_air = 1.5e-5; mu = rho*nu_air; U = 20; 
T = .09375*L; Ta_val = pi/180; h1 = 3.70e-7; h2 = 2.50e-7; patm = 101325;

% Pre-setup for FEM
[AL, BL, Q, t, areaL] = abqfem([x y], t);
nb = size(Q, 2);
xL = x(t'); yL = y(t'); 
xe = sum(xL, 1)'/nv; ye = sum(yL, 1)'/nv; 
order = 1; Nq = 3; [z, w] = trigausspoints(Nq); rq = z(:,1); sq = z(:,2); Bh = diag(w);
nq = length(rq); Dr = zeros(nq, 3); Ds = Dr; Jq = zeros(nq, 3);
for k = 1:3
    [Dr(:,k), Ds(:,k)] = basis_deriv_12(rq, sq, k, order);
    Jq(:,k) = basis_tri_12(rq, sq, k, order);
end
Xr = Dr*xL; Yr = Dr*yL; Xs = Ds*xL; Ys = Ds*yL;
Jac = Xr.*Ys - Xs.*Yr; Rx = Ys ./ Jac; Sx = -Yr ./ Jac;
Bq = Bh*Jac; Ie = speye(E); DLr = kron(Ie, Dr'); DLs = kron(Ie, Ds'); JL = kron(Ie, Jq);

% CASE 1a: Incompressible 2D Taper-Flat
he = profile_taper_flat(xe, L, T, Ta_val, h1, h2);
nu_mat = (1./(12*mu))*(he.^3);
An = kron(spdiags(nu_mat, 0, E, E), speye(3)) * AL;
hq = Jq*profile_taper_flat(xL, L, T, Ta_val, h1, h2);
Uh = (U/2)*(Bq.*hq);
CL = (DLr*spdiags(reshape(Rx.*Uh, nq*E, 1), 0, nq*E, nq*E) + ...
      DLs*spdiags(reshape(Sx.*Uh, nq*E, 1), 0, nq*E, nq*E)) * JL;
Cb = Q'*CL*Q;
boundary_nodes = boundedges([x,y], t);
R_2D = restriction(nb, boundary_nodes);
A = R_2D*(Q'*An*Q)*R_2D'; C = R_2D*Cb*R_2D';
rhs_1a = R_2D*(Cb*ones(nb,1));
Pb_1a = R_2D' * (A \ rhs_1a);

% CASE 1b: Incompressible 1D Wedge (No Taper, Neumann edges)
he_1b = profile_taper_flat(xe, L, T, 0, h1, h2);
nu_mat_1b = (1./(12*mu))*(he_1b.^3);
An_1b = kron(spdiags(nu_mat_1b, 0, E, E), speye(3)) * AL;
hq_1b = Jq*profile_taper_flat(xL, L, T, 0, h1, h2);
Uh_1b = (U/2)*(Bq.*hq_1b);
CL_1b = (DLr*spdiags(reshape(Rx.*Uh_1b, nq*E, 1), 0, nq*E, nq*E) + ...
         DLs*spdiags(reshape(Sx.*Uh_1b, nq*E, 1), 0, nq*E, nq*E)) * JL;
Cb_1b = Q'*CL_1b*Q;
R_1D = restriction(nb, unique([find(abs(x) < 1e-10); find(abs(x - L) < 1e-10)]));
A_1b = R_1D*(Q'*An_1b*Q)*R_1D'; C_1b = R_1D*Cb_1b*R_1D';
rhs_1b = R_1D*(Cb_1b*ones(nb,1));
Pb_1b = R_1D' * (A_1b \ rhs_1b);

% CASE 2a: Compressible 2D Taper-Flat
Pb_2a = zeros(nb, 1);
h3 = he.^3; % for taper-flat
rhs_comp = R_2D * (Cb * (patm * ones(nb, 1)));
for iter = 1:15
    pa_e = ( (Pb_2a(t(:,1)) + Pb_2a(t(:,2)) + Pb_2a(t(:,3)))/3 ) + patm;
    nu_val = (pa_e .* h3) ./ (12 * mu);
    An_c = kron(spdiags(nu_val, 0, E, E), speye(3)) * AL;
    P_free = (R_2D*(Q'*An_c*Q)*R_2D' - C) \ rhs_comp;
    Pb_new = R_2D' * P_free;
    if max(abs(Pb_new - Pb_2a)) < 1e-3, Pb_2a = Pb_new; break; end
    Pb_2a = Pb_new;
end

% PLOTTING
f = figure(4); clf(f); 
% Made the figure slightly wider so the subplots have room to breathe
set(f, 'Position', [100 100 1500 400]); 
fs = 12;

subplot(1,3,1);
Pmax_1a = max(Pb_1a);
trimesh(t, x, y, W*Pb_1a/Pmax_1a); 
axis([0 L -W/1.9 W/1.9 0 W]); % Enforced explicit bounding box
axis equal;
view([-37.5, 30]); % Enforced default 3D viewing angle
title({'(1a) Incompressible 2D', sprintf('Pmax = %.1f Pa', Pmax_1a)}, 'FontSize', fs);
zlabel('W*p/p_{max}');

subplot(1,3,2);
Pmax_1b = max(Pb_1b);
trimesh(t, x, y, W*Pb_1b/Pmax_1b); 
axis([0 L -W/1.9 W/1.9 0 W]); % Enforced explicit bounding box
axis equal;
view([-37.5, 30]);
title({'(1b) Incompressible 1D', sprintf('Pmax = %.1f Pa', Pmax_1b)}, 'FontSize', fs);

subplot(1,3,3);
Pmax_2a = max(Pb_2a);
trimesh(t, x, y, W*Pb_2a/Pmax_2a); 
axis([0 L -W/1.9 W/1.9 0 W]); % Enforced explicit bounding box
axis equal;
view([-37.5, 30]);
title({'(2a) Compressible 2D', sprintf('Pmax = %.1f Pa', Pmax_2a)}, 'FontSize', fs);

drawnow; pause(0.5);
saveas(f, 'part_2b.png');
fprintf('Figure 4 generated and saved as part_2b.png\n');