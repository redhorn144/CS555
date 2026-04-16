import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FEM import assemble, build_mesh_uniform

L = 1.0
c = 1.0
nu = 1e-3
E = 100
dt = 0.001
t_end = 1.5
n_steps = round(t_end / dt)

nodes = build_mesh_uniform(E, L=L)
A_bar, B_bar, C_bar = assemble(nodes, c=c)
N = E + 1  # total nodes


def apply_bc_mat(mat):
    """Enforce only the left Dirichlet BC; right BC is natural (Neumann)."""
    m = mat.tolil()
    m[0, :] = 0;  m[0, 0] = 1
    return m.tocsr()


# Precompute and factorize the three LHS matrices (BC fixed, so do this once)
LHS1_bc = apply_bc_mat((1.0 / dt) * B_bar + nu * A_bar)
LHS2_bc = apply_bc_mat((3.0 / (2 * dt)) * B_bar + nu * A_bar)
LHS3_bc = apply_bc_mat((11.0 / (6 * dt)) * B_bar + nu * A_bar)

solve1 = spla.factorized(LHS1_bc)
solve2 = spla.factorized(LHS2_bc)
solve3 = spla.factorized(LHS3_bc)

# Time march — store every 10th step for the GIF (~150 frames total)
gif_stride = 10
frames_U = []
frames_t = []

U_prev3 = np.zeros(N)
U_prev2 = np.zeros(N)
U_prev1 = np.zeros(N)

for n in range(1, n_steps + 1):
    t_next = n * dt

    if n == 1:
        RHS = (1.0 / dt) * B_bar @ U_prev1 - c * C_bar @ U_prev1
        RHS[0] = np.sin(np.pi * t_next)
        U_new = solve1(RHS)
    elif n == 2:
        RHS = (1.0 / dt) * B_bar @ (2 * U_prev1 - 0.5 * U_prev2) \
              - c * C_bar @ (2 * U_prev1 - U_prev2)
        RHS[0] = np.sin(np.pi * t_next)
        U_new = solve2(RHS)
    else:
        RHS = (1.0 / dt) * B_bar @ (3 * U_prev1 - 1.5 * U_prev2 + (1.0 / 3.0) * U_prev3) \
              - c * C_bar @ (3 * U_prev1 - 3 * U_prev2 + U_prev3)
        RHS[0] = np.sin(np.pi * t_next)
        U_new = solve3(RHS)

    U_prev3 = U_prev2
    U_prev2 = U_prev1
    U_prev1 = U_new.copy()

    if n % gif_stride == 0:
        frames_U.append(U_prev1.copy())
        frames_t.append(t_next)

# Static plot at t=1.5
plt.figure()
plt.plot(nodes, U_prev1, 'm-', linewidth=1.5)
plt.title('Unsteady Advection-Diffusion at t=1.5 (Neumann right BC)')
plt.xlabel('x')
plt.ylabel('u(x, t=1.5)')
plt.grid()
plt.savefig('2b.png')
plt.close()

# Animated GIF
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.grid()
line, = ax.plot([], [], 'm-', linewidth=1.5)
title = ax.set_title('')


def init():
    line.set_data([], [])
    title.set_text('')
    return line, title


def update(i):
    line.set_data(nodes, frames_U[i])
    title.set_text(f'Unsteady Advection-Diffusion (Neumann right BC)  t = {frames_t[i]:.3f}')
    return line, title


ani = animation.FuncAnimation(
    fig, update, frames=len(frames_U), init_func=init,
    interval=50, blit=True
)
ani.save('2b.gif', writer='pillow', fps=20)
plt.close()
print('Saved 2b.gif')