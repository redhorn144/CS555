% part_2a.m
% Solution for HW6 Part 2a: Compressible Taper-Flat Slider

hdr;

Ex = 120;
Ey = 40;

% --- Physical parameters ---
rho = 1.225;         
nu  = 1.5e-5;        
mu  = rho*nu;        
U   = 20;            
L   = .0041;         
W   = .15625*L;      
T   = .09375*L;      
Ta  = pi/180;        % Taper angle is back to 1-degree for Part 2a

h1  = 3.70e-7;       
h2  = 2.50e-7;       
gamma = (h1-h2)/L;   
patm= 101325;        

[x,y,t] = box_elem(Ex,Ey,L,W);
E  = 2*Ex*Ey;
nv = 3;            

xL=x(t'); yL=y(t');
xe=sum(xL,1)'/nv;   
ye=sum(yL,1)'/nv;

[AL,BL,Q,t,areaL]=abqfem([x y],t);
Ab=Q'*AL*Q;
Bb=Q'*BL*Q;
nb=size(Q,2);

% Evaluate h^3 at quadrature points
he = profile_taper_flat(xe,L,T,Ta,h1,h2); 
h3 = he.*he.*he;

order=1;  
nv=3*order;
Nq=3; 
[z,w]=trigausspoints(Nq); rq=z(:,1);sq=z(:,2);  Bh=diag(w);

nq=length(rq); Dr=zeros(nq,nv); Ds=Dr; Jq=zeros(nq,nv);
for k=1:nv 
 [Dr(:,k), Ds(:,k)]=basis_deriv_12(rq,sq,k,order);  
  Jq(:,k)         =basis_tri_12  (rq,sq,k,order);  
end

Xr  = Dr*xL; Yr = Dr*yL; Xs  = Ds*xL; Ys = Ds*yL;
Jac = Xr.*Ys - Xs.*Yr; Jmin =min(min(Jac)); if Jmin <= 0; error('vanishing Jacobian'); end
Rx  = Ys ./ Jac; Ry = -Xs ./ Jac; Sy  = Xr ./ Jac; Sx = -Yr ./ Jac;
Bq  = Bh*Jac;

hL  = profile_taper_flat(xL,L,T,Ta,h1,h2); 
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

% COMPRESSIBLE NONLINEAR SOLVER

% Pre-compute constant matrices for the compressible equation
% Eq 16: R*(A - C)*R' * P_free = R*C * P_atm_vec
C_comp = R * Cb * R';
patm_vec = patm * ones(nb, 1);
rhs_comp = R * (Cb * patm_vec);

% Initial guess: relative pressure p = 0, so pa = patm
Pb = zeros(nb, 1);
pa = Pb + patm;

fprintf('--- Part 2a: Compressible Solver Iterations ---\n');
max_iters = 15;
tol = 1e-3;

Iv = speye(3); % 3 nodes per triangle

for iter = 1:max_iters
    % 1. Evaluate absolute pressure at element centroids (average of 3 nodes)
    pa_e = (pa(t(:,1)) + pa(t(:,2)) + pa(t(:,3))) / 3;
    
    % 2. Update the variable viscosity matrix (nu) using absolute pressure
    % nu = (pa * h^3) / (12 * mu)
    nu_val = (pa_e .* h3) ./ (12 * mu);  
    nu_val = spdiags(nu_val, 0, E, E);
    nu_val = kron(nu_val, Iv);
    
    % 3. Rebuild the Stiffness Matrix (A)
    An = nu_val * AL;
    A_comp = R * (Q' * An * Q) * R';
    
    % 4. Form LHS matrix: R*(A - C)*R'
    LHS = A_comp - C_comp;
    
    % 5. Solve for the free nodes
    P_free = LHS \ rhs_comp;
    
    % 6. Reconstruct full relative pressure p and update pa
    Pb_new = R' * P_free;
    
    % Check convergence
    diff = max(max(abs(Pb_new - Pb)));
    fprintf('Iteration %d: max pressure change = %e\n', iter, diff);
    
    Pb = Pb_new;
    pa = Pb + patm;
    
    if diff < tol
        fprintf('Converged in %d iterations!\n\n', iter);
        break;
    end
end


% Load and center-of-pressure are computed using RELATIVE pressure (Pb)
F_comp = sum(Bb * Pb); 
xp_comp = (x' * Bb * Pb) / F_comp;

fprintf('--- Part 2a Results ---\n');
fprintf('Trailing height (h2) : %e m\n', h2);
fprintf('Pitch (gamma)        : %e radians\n', gamma);
fprintf('Load (F)             : %e N\n', F_comp);
fprintf('Center of Pressure   : %e m\n', xp_comp);

%  PLOTTING 
f = figure(3);
clf(f);
Pmax = max(max(abs(Pb)));
trimesh(t, x, y, W * Pb / Pmax);
axis([0 L -W/1.9 W/1.9 0 W]);
axis equal;

fs = 14;
xlabel('x', 'FontSize', fs);
ylabel('y', 'FontSize', fs);
zlabel('W*p/p_{max}', 'FontSize', fs);
title('Compressible Case', 'FontSize', fs);

drawnow; 
pause(0.5); 

% Automatically save the figure 
saveas(f, 'part_2a.png');
fprintf('\nFigure successfully rendered and saved as ''part_2a.png''.\n');