import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from FEM import assemble, build_mesh_geometric, build_R

L = 1
c = 1
E = 20
nu = 10e-3

geometric_elements = build_mesh_geometric(E, L=L)

A_bar, B_bar, C_bar = assemble(geometric_elements)

R = build_R(E)

A = R @ A_bar @ R.T
B = R @ B_bar @ R.T
C = R @ C_bar @ R.T

#steady advection diffusion
#c u_x - nu u_xx = f
# Au + Cu = (A + C) u = Bf
f = np.ones(E - 1)  # source term at interior nodes
u = sp.sparse.linalg.spsolve(nu * A + C, B @ f)

# Analytical solution: -nu u'' + c u' = 1, u(0)=u(1)=0
# u(x) = (1 - exp(Pe*x)) / (c*(exp(Pe) - 1)) + x/c,  Pe = c/nu
x = geometric_elements
x_int = x[1:-1]
Pe = c / nu
u_exact = (1 - np.exp(Pe * x_int)) / (c * (np.exp(Pe) - 1)) + x_int / c

max_error = np.max(np.abs(u - u_exact))
print(f"Maximum error vs analytical solution: {max_error:.6e}")

# Plot the solution
plt.plot(x_int, u, label='Numerical Solution')
plt.plot(x_int, u_exact, '--', label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Steady Advection-Diffusion Solution')
plt.legend()
plt.grid()
plt.savefig('1b.png')



