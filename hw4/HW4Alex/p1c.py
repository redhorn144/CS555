import numpy as np
import scipy.optimize as opt
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from FEM import assemble, build_mesh_geometric, build_R

L = 1.0
c = 1.0
E = 20
nu = 1e-3   # was incorrectly 10e-3 (=0.01); correct value is 1e-3 (=0.001)
Pe = c / nu


def max_rel_error(s):
    nodes = build_mesh_geometric(E, L=L, s=s)
    A_bar, B_bar, C_bar = assemble(nodes, c=c)
    R = build_R(E)
    A = R @ A_bar @ R.T
    B = R @ B_bar @ R.T
    C = R @ C_bar @ R.T

    f = np.ones(len(nodes))
    u = spla.spsolve(nu * A + C, R @ (B_bar @ f))

    x_int = nodes[1:-1]
    u_exact = x_int - np.exp(Pe * (x_int - L))

    return np.max(np.abs(u - u_exact) / np.abs(u_exact))


# Minimize over s in (0, 1) — s must stay away from 0 and 1
result = opt.minimize_scalar(max_rel_error, bounds=(0.3, 0.999), method='bounded')
s_opt = result.x
err_opt = result.fun
print(f"Optimal s = {s_opt:.6f}")
print(f"Minimum max relative error = {err_opt:.6e}")

# Error curve over a wide range for plotting
s_vals = np.linspace(0.3, 0.999, 200)
errors = np.array([max_rel_error(s) for s in s_vals])

plt.figure()
plt.plot(s_vals, errors, 'k-', linewidth=1.5, label='Error curve')
plt.axvline(s_opt, color='r', linestyle='--', label=f'Optimal s = {s_opt:.4f}')
plt.scatter([s_opt], [err_opt], color='r', zorder=5)
plt.xlabel('Scale factor s')
plt.ylabel('Max pointwise relative error')
plt.title(f'Error vs. s  (E={E}, Pe={Pe:.0f})')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.savefig('1c.png')
plt.close()