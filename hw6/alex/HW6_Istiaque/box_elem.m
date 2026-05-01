function [xb,yb,t]=box_elem(Ex,Ey,L,W);

hdr

[z,w]=zwgll(Ex); Ex1=Ex+1; x1=L*(z+1)/2;
[z,w]=zwgll(Ey); Ey1=Ey+1; y1=W*(z+0)/2;

[z,w]=zwuni(Ex); Ex1=Ex+1; x1=L*(z+1)/2;
[z,w]=zwuni(Ey); Ey1=Ey+1; y1=W*(z+0)/2;

[Xb,Yb]=ndgrid(x1,y1);
nb = Ex1*Ey1;
E  = 2*Ex*Ey;    %% E = number of triangles

xb=reshape(Xb,nb,1);
yb=reshape(Yb,nb,1);
p = [1:Ex1*Ey1]'; p=reshape(p,Ex1,Ey1);

t  = zeros(E,3);
e=0;
for j=1:Ey;
for i=1:Ex;

%   if j > Ex/2;
    if j < 0;

       e=e+1;
       t(e,1) = p(i  ,j  );
       t(e,2) = p(i+1,j  );
       t(e,3) = p(i+1,j+1);

       e=e+1;
       t(e,1) = p(i  ,j  );
       t(e,2) = p(i+1,j+1);
       t(e,3) = p(i  ,j+1);

    else;

       e=e+1;
       t(e,1) = p(i  ,j  );
       t(e,2) = p(i+1,j  );
       t(e,3) = p(i  ,j+1);

       e=e+1;
       t(e,1) = p(i+1,j  );
       t(e,2) = p(i+1,j+1);
       t(e,3) = p(i  ,j+1);
    end;

end;
end;

u = xb+0*xb.*yb;
trimesh(t,xb,yb,u); view(00,90);
xlabel('X',fs,20);
ylabel('Y',fs,20);
axis equal; 
drawnow;

p=[xb yb];  % Std Matlab format
