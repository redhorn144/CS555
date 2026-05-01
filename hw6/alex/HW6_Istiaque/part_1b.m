% part_1b.m
% Solution for HW6 Part 1b: 1D Wedge Bearing Convergence Verification

hdr;

% Test different grid resolutions for the x-direction
Ex_vals = [15, 30, 60, 120];
Ey = 40; % Keep y-resolution constant
errors = zeros(length(Ex_vals), 1);

fprintf('--- Part 1b: 1D Wedge Convergence Study ---\n');

for i = 1:length(Ex_vals)
    Ex = Ex_vals(i);
    
    % --- Physical parameters ---
    rho = 1.225;         
    nu  = 1.5e-5;        
    mu  = rho*nu;        
    U   = 20;            
    L   = .0041;         
    W   = .15625*L;      
    T   = .09375*L;      
    
    % 1. REMOVE TAPER FOR 1D WEDGE VERIFICATION
    Ta  = 0;             

    h1  = 3.70e-7;       
    h2  = 2.50e-7;       
    
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

    he = profile_taper_flat(xe,L,T,Ta,h1,h2); 
    h3 = he.*he.*he;
    nu_val = (1./(12*mu))*h3;  
    nu_val = spdiags(nu_val,0,E,E);

    Iv = speye(nv);
    nu_val = kron(nu_val,Iv);
    An = nu_val*AL;        

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

    % 2. ENFORCE 1D BOUNDARY CONDITIONS (Neumann on top/bottom)
    left_nodes = find(abs(x) < 1e-10);
    right_nodes = find(abs(x - L) < 1e-10);
    boundary_nodes_1D = unique([left_nodes; right_nodes]);
    
    R = restriction(nb, boundary_nodes_1D);
    
    A = R*(Q'*An*Q)*R';
    B = R*Bb*R';
    C = R*Cb*R';

    rhs = R*(Cb*ones(nb,1));
    P   = A \ rhs;
    Pb  = R'*P;
    
    % === COMPUTE ANALYTICAL SOLUTION ===
    alpha = h1 / h2;
    Lambda = 6 * mu * U * L / (h2^2);
    H_x = alpha + (1 - alpha) * (x / L);
    
    % Equation (4) 
    p_analytical = (alpha * Lambda / (1 - alpha^2)) * (1./H_x.^2 - 1/alpha^2) - (Lambda / (1 - alpha)) * (1./H_x - 1/alpha);
    
    % Calculate max pointwise error
    errors(i) = max(abs(Pb - p_analytical));
    fprintf('Ex = %3d, Max Error = %e\n', Ex, errors(i));
end

%  PLOTTING 
f = figure(2);
clf(f); 
loglog(Ex_vals, errors, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
fs = 14;
xlabel('Number of Elements (E_x)', 'FontSize', fs);
ylabel('Max Pointwise Error in p(x)', 'FontSize', fs);
title('1D Wedge Bearing Convergence', 'FontSize', fs);

drawnow; 
pause(0.5); 

saveas(f, 'part_1b.png');
fprintf('\nFigure successfully rendered and saved as ''part_1b.png''.\n');

% Estimate and print the order of convergence (slope of log-log line)
p_order = polyfit(log(Ex_vals), log(errors'), 1);
fprintf('Estimated Convergence Rate (slope): %.2f\n', p_order(1));
fprintf('(A slope of ~ -2.0 indicates second-order convergence)\n\n');