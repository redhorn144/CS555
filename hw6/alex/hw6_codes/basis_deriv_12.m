
function [phir,phis]=basis_deriv_12(r,s,k,order) % Return gradient of phi_k at points [r,s]
   
if order==1; % linear triangles

   if k==1;
      phir = -1 + 0*r;
      phis = -1 + 0*r;
   elseif k==2;
      phir =  1 + 0*r;
      phis =  0 + 0*r;
   elseif k==3;
      phir =  0 + 0*r;
      phis =  1 + 0*r;
   end;

else;  % 2nd-order triangles
   
   %% if k==1; phi=(1-r-s).*(1-2*r-2*s); end;  %% phi_1 -- phi_6
   %% if k==2; phi=r.*(2*r-1);           end;
   %% if k==3; phi=s.*(2*s-1);           end;
   %% if k==4; phi=4.*r.*(1-r-s);        end;
   %% if k==5; phi=4.*r.*s;              end;
   %% if k==6; phi=4.*s.*(1-r-s);        end;
   
   if k==1;
      phir = -(1-2*r-2*s) - 2*(1-r-s);
      phis = -(1-2*r-2*s) - 2*(1-r-s);
   elseif k==2;
      phir = (4*r-1);
      phis = 0*r;
   elseif k==3;
      phir = 0*r;
      phis = (4*s-1);
   elseif k==4;
      phir = 4*(1-r-s) - 4*r;
      phis = -4*r;
   elseif k==5;
      phir = 4*s;
      phis = 4*r;
   elseif k==6;
      phir = -4*s;
      phis = 4*(1-r-s) - 4*s;
   end;
   
end;
