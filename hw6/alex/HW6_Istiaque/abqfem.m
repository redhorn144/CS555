function [AL,BL,Q,t,area]=abqfem(p,t) % Build 2D FEM stiffness, mass, & assembly matrix

  nt = size(t,1); nl=3*nt;

  y23 = p(t(:,2),2) - p(t(:,3),2);
  y31 = p(t(:,3),2) - p(t(:,1),2);
  y12 = p(t(:,1),2) - p(t(:,2),2);
  x32 = p(t(:,3),1) - p(t(:,2),1);
  x13 = p(t(:,1),1) - p(t(:,3),1);
  x21 = p(t(:,2),1) - p(t(:,1),1);
  area  = 0.5*(x21.*y31 - y12.*x13) ;
  aream = min(area);
  areaM = max(area);


  eflip = find(area<0);      % Check for negative Jacobians
  nflip = length(eflip);
  if nflip > 0; 
     temp       = t(eflip,2);
     t(eflip,2) = t(eflip,3);
     t(eflip,3) = temp;
  end;

  y23 = p(t(:,2),2) - p(t(:,3),2);
  y31 = p(t(:,3),2) - p(t(:,1),2);
  y12 = p(t(:,1),2) - p(t(:,2),2);
  x32 = p(t(:,3),1) - p(t(:,2),1);
  x13 = p(t(:,1),1) - p(t(:,3),1);
  x21 = p(t(:,2),1) - p(t(:,1),1);
  area4i = 1./area; area4i = 0.25*area4i;



  i0 = (0:nt-1)';
  i1 = (1:nt)';
  AL = spalloc(nl,nl,9*nl); BL = AL;
  A1 = zeros(3,3,nt);       B1 = A1;

  A1(1,1,:) = area4i.*( y23.*y23+x32.*x32 );
  A1(1,2,:) = area4i.*( y23.*y31+x32.*x13 );
  A1(1,3,:) = area4i.*( y23.*y12+x32.*x21 );
  A1(2,1,:) = area4i.*( y31.*y23+x13.*x32 );
  A1(2,2,:) = area4i.*( y31.*y31+x13.*x13 );
  A1(2,3,:) = area4i.*( y31.*y12+x13.*x21 );
  A1(3,1,:) = area4i.*( y12.*y23+x21.*x32 );
  A1(3,2,:) = area4i.*( y12.*y31+x21.*x13 );
  A1(3,3,:) = area4i.*( y12.*y12+x21.*x21 );

  dmass=0;         % Full (local) mass matix
  dmass=1;         % Diagonal mass matrix
  if dmass==0; 
     B1(1,1,:) = area/6;
     B1(1,2,:) = area/12;
     B1(1,3,:) = area/12;
     B1(2,1,:) = area/12;
     B1(2,2,:) = area/6;
     B1(2,3,:) = area/12;
     B1(3,1,:) = area/12;
     B1(3,2,:) = area/12;
     B1(3,3,:) = area/6;
  else
     B1(1,1,:) = area/3;
     B1(2,2,:) = area/3;
     B1(3,3,:) = area/3;
  end;

% for e=0:nt-1;        THIS APPROACH IS WAY TOO SLOW
%   AL(3*e+(1:3),3*e+(1:3)) = A1(:,:,e+1);
%   BL(3*e+(1:3),3*e+(1:3)) = B1(:,:,e+1);
% end;

d0=zeros(3,nt);        % main diagonal
d1=zeros(3,nt);        % 1st lower diagonal
d2=zeros(3,nt);        % 2nd lower diagonal

d0(1,:)=A1(1,1,:);
d0(2,:)=A1(2,2,:);
d0(3,:)=A1(3,3,:);     d0=reshape(d0,nl,1);
d1(1,:)=A1(2,1,:);
d1(2,:)=A1(3,2,:);     d1=reshape(d1,nl,1);
d2(1,:)=A1(3,1,:);     d2=reshape(d2,nl,1);

AL=spdiags([d2 d1],-2:-1,nl,nl);
AL=AL+AL';
Ad=spdiags(d0,0:0,nl,nl);
AL=AL+Ad;

d0=zeros(3,nt);        % main diagonal
d1=zeros(3,nt);        % 1st lower diagonal
d2=zeros(3,nt);        % 2nd lower diagonal

d0(1,:)=B1(1,1,:);
d0(2,:)=B1(2,2,:);
d0(3,:)=B1(3,3,:);     d0=reshape(d0,nl,1);
d1(1,:)=B1(2,1,:);
d1(2,:)=B1(3,2,:);     d1=reshape(d1,nl,1);
d2(1,:)=B1(3,1,:);     d2=reshape(d2,nl,1);

BL=spdiags([d2 d1],-2:-1,nl,nl);
BL=BL+BL';
Ad=spdiags(d0,0:0,nl,nl);
BL=BL+Ad;

Q  = sparse(1:nl,reshape(t',nl,1),1);

