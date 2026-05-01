% part_1a.m
% Solution for HW6 Part 1a: Computing Load and Center of Pressure

hdr;

Ex = 120;
Ey = 40;

rho = 1.225;         % kg/m^3      density of air
nu  = 1.5e-5;        % m^2/s       kinematic viscosity
mu  = rho*nu;        %             dynamic viscosity
U   = 20;            % m/s         speed of plate
L   = .0041;         % m           slider length
W   = .15625*L;      % m           slider width (not used in 1D)
T   = .09375*L;      % m           slider taper length
Ta  = pi/180;        % m           Taper angle (1-deg)

h1  = 3.70e-7;       % m           1/2-micron leading  edge gap
h2  = 2.50e-7;       % m           1/4-micron trailing edge gap
gamma = (h1-h2)/L;   %             pitch angle of slider << 1
patm= 101325;        %             Pressure in N/m^2
n_per_gram= 9.81e-3; % N/g         Weight of a 1 gram mass

[x,y,t] = box_elem(Ex,Ey,L,W);
E  = 2*Ex*Ey;

nv = 3;            %% Linear triangles for elements

xL=x(t'); yL=y(t');
xe=sum(xL,1)'/nv;   %% Use Centroids for Laplacian quadrature points
ye=sum(yL,1)'/nv;

[AL,BL,Q,t,areaL]=abqfem([x y],t);
Ab=Q'*AL*Q;
Bb=Q'*BL*Q;
nb=size(Q,2);

he = profile_taper_flat(xe,L,T,Ta,h1,h2); %% Evaluate on quad points
h3 = he.*he.*he;
nu = (1./(12*mu))*h3;  
nu = spdiags(nu,0,E,E);

Iv = speye(nv);
nu = kron(nu,Iv);
An = nu*AL;        %% AL, each element scaled by local nu


order=1;  % order=2 is not yet working
nv=3*order;

Nq=3; % 8 is Max allowable for trigausspoints
[z,w]=trigausspoints(Nq); rq=z(:,1);sq=z(:,2);  Bh=diag(w);

nq=length(rq); Dr=zeros(nq,nv); Ds=Dr; Jq=zeros(nq,nv);
for k=1:nv; 
 [Dr(:,k), Ds(:,k)]=basis_deriv_12(rq,sq,k,order);  % Differentiate to fine mesh
  Jq(:,k)         =basis_tri_12  (rq,sq,k,order);  % Interpolate to fine mesh
end

Xr  = Dr*xL; Yr = Dr*yL; Xs  = Ds*xL; Ys = Ds*yL;
Jac = Xr.*Ys - Xs.*Yr; Jmin =min(min(Jac)); if Jmin <= 0; error('vanishing Jacobian'); end
Rx  = Ys ./ Jac; Ry = -Xs ./ Jac; Sy  = Xr ./ Jac; Sx = -Yr ./ Jac;
Bq  = Bh*Jac;

hL  = profile_taper_flat(xL,L,T,Ta,h1,h2); %% Evaluate on quad points
hq  = Jq*hL;
Uh  = (U/2)*(Bq.*hq);

Ie  = speye(E);
DLr = kron(Ie,Dr');
DLs = kron(Ie,Ds');
ULr = spdiags(reshape(Rx.*Uh,nq*E,1),0,nq*E,nq*E);
ULs = spdiags(reshape(Sx.*Uh,nq*E,1),0,nq*E,nq*E);
JL  = kron(Ie,Jq);

CL  = (DLr*ULr + DLs*ULs)*JL;
Cb  = Q'*CL*Q;

boundary_nodes = boundedges([x,y],t);
R = restriction(nb,boundary_nodes);
A = R*(Q'*An*Q)*R';
B = R*Bb*R';
C = R*Cb*R';

rhs = R*(Cb*ones(nb,1));
P   = A \ rhs;
Pb  = R'*P;

% Plotting
Pmax=max(max(abs(Pb)));
figure(1);
trimesh(t,x,y,W*Pb/Pmax);
axis([0 L -W/1.9 W/1.9 0 W]);
axis equal;

% Added font size definition 
fs = 14; 

xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
zlabel('W*p/p_{max}','FontSize',fs);
title('Incompressible Case','FontSize',fs);
drawnow;
pause;

% PART 1a: Compute Load (F) and Center-of-Pressure (xp)
F  = sum(Bb * Pb);
xp = (x' * Bb * Pb) / F;

fprintf('--- Part 1a (Taper-Flat Incompressible) ---\n');
fprintf('Computed Load, F        = %e N\n', F);
fprintf('Center of Pressure, xp  = %e m\n', xp);

% VERIFICATION: Re-solve with wedge config (Ta=0, Neumann on top/bottom)
% to show F_numerical matches the analytical F_tilde.
alpha   = h1 / h2;
Lambda  = 6 * mu * U * L / (h2^2);
F_tilde = (Lambda * L * W) / (1 - alpha)^2 * (log(alpha) + 2 * (1 - alpha) / (1 + alpha));

Ta_v  = 0;
he_v  = profile_taper_flat(xe, L, T, Ta_v, h1, h2);
nu_v  = spdiags((1./(12*mu))*he_v.^3, 0, E, E);
An_v  = kron(nu_v, speye(3)) * AL;

hL_v  = profile_taper_flat(xL, L, T, Ta_v, h1, h2);
hq_v  = Jq * hL_v;
Uh_v  = (U/2) * (Bq .* hq_v);
ULr_v = spdiags(reshape(Rx.*Uh_v, nq*E, 1), 0, nq*E, nq*E);
ULs_v = spdiags(reshape(Sx.*Uh_v, nq*E, 1), 0, nq*E, nq*E);
Cb_v  = Q' * (DLr*ULr_v + DLs*ULs_v) * JL * Q;

R_v   = restriction(nb, unique([find(abs(x) < 1e-10); find(abs(x - L) < 1e-10)]));
P_v   = (R_v*(Q'*An_v*Q)*R_v') \ (R_v*(Cb_v*ones(nb,1)));
Pb_v  = R_v' * P_v;
F_v   = sum(Bb * Pb_v);

fprintf('\n--- Verification (Ta=0, Neumann top/bottom) ---\n');
fprintf('Numerical Load (wedge), F = %e N\n', F_v);
fprintf('Analytical Load, F_tilde  = %e N\n', F_tilde);
fprintf('Relative Error            = %.2f%%\n\n', 100*abs(F_v - F_tilde)/F_tilde);