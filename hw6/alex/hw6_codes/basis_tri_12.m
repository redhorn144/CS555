function phi=basis_tri_12(r,s,k,order)

%
%  This is permuted from other basis_tri1 usage
%
%           ^                                       ^                           
%           |   LOCAL ORDERING, order=2             |   LOCAL ORDERING, order=1
%           |                                     3 o                
%         3 o                                       | \
%           | \                                     |   \
%           |   \                                   |     \
%           |     \                                 |       \
%         6 o       o 5                             |         \
%           |         \                             |           \
%           |           \                           |             \
%           |             \                         o--------------o-->
%           o-------o------o-->                     1              2
%           1       4      2                     



if order==1;

   if k==1; phi=(1-r-s);  end;
   if k==2; phi=r;        end;
   if k==3; phi=s;        end;

elseif order==2;

   if k==1; phi=(1-r-s).*(1-2*r-2*s); end;
   if k==2; phi=r.*(2*r-1);           end;
   if k==3; phi=s.*(2*s-1);           end;
   if k==4; phi=4.*r.*(1-r-s);        end;
   if k==5; phi=4.*r.*s;              end;
   if k==6; phi=4.*s.*(1-r-s);        end;

else

   error('Stopping. order > 2 not supported in basis_tri_12.m')

endif
