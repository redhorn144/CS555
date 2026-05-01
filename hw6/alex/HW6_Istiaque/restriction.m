function R = restriction(nb,boundary_list);


all_nodes = [1:nb]';
interior  = setdiff(all_nodes,boundary_list);

R         = speye(nb);
R         = R(interior,:); %% Keep only interior columns

