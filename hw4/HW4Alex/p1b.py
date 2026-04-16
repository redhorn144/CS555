import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from FEM import assemble, build_mesh_geometric, build_R

L = 1
c = 1
E = 20
nu = 1e-3

geometric_elements = build_mesh_geometric(E, L=L)

A_bar, B_bar, C_bar = assemble(geometric_elements)

R = build_R(E)

A = R @ A_bar @ R.T
B = R @ B_bar @ R.T
C = R @ C_bar @ R.T

#steady advection diffusion
#c u_x - nu u_xx = f
# Au + Cu = (A + C) u = Bf
f = np.ones(len(geometric_elements))  # source term at interior nodes
u = sp.sparse.linalg.spsolve(nu * A + C, R @ (B_bar @ f))

# Analytical solution: -nu u'' + c u' = 1, u(0)=u(1)=0
# u(x) = (1 - exp(Pe*x)) / (c*(exp(Pe) - 1)) + x/c,  Pe = c/nu
x = geometric_elements
x_int = x[1:-1]
Pe = c / nu
u_exact = x_int / c - (L / c) * np.exp(Pe * (x_int - L))

rel_error = np.abs(u - u_exact) / np.abs(u_exact)
max_rel_error = np.max(rel_error)
print(f"Maximum pointwise relative error: {max_rel_error:.6e}")

# Build full solution vector (prepend/append boundary values) for plotting
u_full = np.concatenate(([0.0], u, [0.0]))
u_exact_full = np.concatenate(([0.0], u_exact, [0.0]))

# Plot the solution
plt.plot(x, u_exact_full, 'k-', linewidth=1.5, label='Analytical Solution')
plt.plot(x, u_full, 'bo-', markersize=5, markerfacecolor='b', label='Numerical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Steady Advection-Diffusion Solution')
plt.legend()
plt.grid()
plt.savefig('1b.png')



