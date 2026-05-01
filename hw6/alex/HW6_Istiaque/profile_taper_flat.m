function h = profile_taper_flat(x,L,T,Ta,h1,h2); %% Evaluate on quad points

h    = h2+((h1-h2)/L)*(L-x);       % Height on the flat

i    = find(x<T); 

h(i) = h(i)-(x(i)-T)*tan(Ta);      % Add taper section for x < T 

