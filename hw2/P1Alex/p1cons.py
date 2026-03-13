# Solving Burgers equation using a three point finite difference stencil for A and D and bdf3/ext3 for time stepping.
# Conservation form: N(u) = (1/2) d(u^2)/dx  instead of convective u * du/dx
# domain [0, 1]
# initial condition u(x, 0) = sin(pi * x)
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from numba import njit


################################
# Params
################################
#node_type = "uniform"
node_type = "chebyshev"
N     = 15000
dt    = 1e-3
tfinal = 2
useCFL = True

###############################
# Functions
###############################
def Nodes(N, type = "uniform"):
    if type == "uniform":
        return np.linspace(0, 1, N + 1)
    elif type == "chebyshev":
        j = np.arange(N + 1)
        return 0.5 * (1 - np.cos(np.pi * j / N))
    else:
        raise ValueError("Unknown node type")

def build_A(x, nu=(10**-2)/np.pi):
    """Diffusion operator: nu * d^2u/dx^2, with homogeneous Dirichlet BCs."""
    N = len(x) - 1
    A = np.zeros((N + 1, N + 1))
    for j in range(1, N):
        hm = x[j]     - x[j - 1]
        hp = x[j + 1] - x[j]
        A[j, j - 1] =  2 / (hm * (hm + hp))
        A[j, j]     = -2 / (hm * hp)
        A[j, j + 1] =  2 / (hp * (hm + hp))
    A *= nu
    return A

def build_D(x):
    """First derivative operator: du/dx, central difference, Dirichlet BCs."""
    N = len(x) - 1
    D = np.zeros((N + 1, N + 1))
    for j in range(1, N):
        D[j, j - 1] = -1 / (x[j + 1] - x[j - 1])
        D[j, j + 1] =  1 / (x[j + 1] - x[j - 1])
    return D


@njit(cache=True)
def tri_matvec(lower, upper, u):
    """Compute (D @ u) using only the off-diagonal bands of D (O(N) instead of O(N^2))."""
    n = len(u)
    out = np.zeros(n)
    for j in range(1, n - 1):
        out[j] = lower[j] * u[j - 1] + upper[j] * u[j + 1]
    return out


@njit(cache=True)
def thomas_solve(lo, diag, up, d):
    """Thomas algorithm: O(N) solve for a tridiagonal system."""
    n = len(d)
    c = np.empty(n)
    x = np.empty(n)
    c[0] = up[0] / diag[0]
    x[0] = d[0]  / diag[0]
    for i in range(1, n):
        denom = diag[i] - lo[i] * c[i - 1]
        c[i] = up[i] / denom
        x[i] = (d[i] - lo[i] * x[i - 1]) / denom
    for i in range(n - 2, -1, -1):
        x[i] -= c[i] * x[i + 1]
    return x


def extract_bands(M):
    """Extract lower, diagonal, upper bands from a tridiagonal matrix."""
    n = M.shape[0]
    lo   = np.zeros(n)   # lo[0]   unused
    diag = np.diag(M).copy()
    up   = np.zeros(n)   # up[-1]  unused
    for i in range(1, n):
        lo[i] = M[i, i - 1]
    for i in range(n - 1):
        up[i] = M[i, i + 1]
    return lo, diag, up

def H_bands(b0, A_lo, A_diag, A_up, dt):
    """Compute tridiagonal bands of H = b0*I - dt*A."""
    return -dt * A_lo, b0 - dt * A_diag, -dt * A_up

def cfl_dt(x, u, cfl=0.1):
    """Return dt satisfying CFL = max|u| * dt / dx_min <= cfl."""
    #dx_min = np.min(np.diff(x))
    dx_min = 1/len(x)
    max_u  = np.max(np.abs(u))
    if max_u == 0:
        raise ValueError("max|u| = 0; cannot determine dt from CFL.")
    return cfl * dx_min / max_u


###############################
# Solver
###############################

#setup

x = Nodes(N, node_type)
A = build_A(x)
D = build_D(x)

# Extract bands once — avoids O(N^2) matvecs and matrix rebuilds in the loop
A_lo, A_diag, A_up = extract_bands(A)
D_lo, _,      D_up = extract_bands(D)   # D has no diagonal

u0 = np.sin(np.pi * x)

#CFL dt
if useCFL:
    dt = cfl_dt(x, u0, 0.05)
nsteps = int(np.ceil(tfinal / dt))
print(dt)
# Warm up numba JIT before the timed loop (use valid H bands — A_diag[0]=0 causes div/0)
_H_lo, _H_diag, _H_up = H_bands(1.0, A_lo, A_diag, A_up, dt)
_ = tri_matvec(D_lo, D_up, u0)
_ = thomas_solve(_H_lo, _H_diag, _H_up, u0)

# history: [u^{n-2}, u^{n-1}, u^n]
u_hist = [u0.copy(), u0.copy(), u0.copy()]

# Storage for plots
t_snap   = np.arange(0.0, tfinal + 1e-10, 0.1)   # t = 0.0, 0.1, ..., 2.0  (1a)
u_snaps  = {0.0: u0.copy()}                        # snapshots keyed by time
t_hist   = [0.0]                                   # time at each step       (1b)
s_hist   = [np.max(np.abs(tri_matvec(D_lo, D_up, u0)))]   # s(t) = max|du/dx|  (1b)

# Precompute which step indices correspond to snapshots
snap_steps = set(int(round(ts / dt)) for ts in t_snap if ts > 0)

#Main Solver Loop
k = 1
H_lo = H_diag = H_up = None
for n in range(nsteps):
    #### BDF/EXT ######
    if k == 1:
        b0 = 1;      b1 = -1;  b2 = 0;    b3 = 0;      a1 = 1; a2 =  0; a3 = 0
        H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
        k += 1
    elif k == 2:
        b0 = 1.5;    b1 = -2;  b2 = 0.5;  b3 = 0;      a1 = 2; a2 = -1; a3 = 0
        H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
        k += 1
    else:
        b0 = 11/6;   b1 = -3;  b2 = 1.5;  b3 = -1/3;   a1 = 3; a2 = -3; a3 = 1
        if k == 3:   # build H bands once for BDF3, reuse every remaining step
            H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
            k += 1   # sentinel: k=4 means H bands are already cached

    # Nonlinear advection — conservation form: N(u) = (1/2) * D @ (u^2)
    N0 = 0.5 * tri_matvec(D_lo, D_up, u_hist[2]**2)
    N1 = 0.5 * tri_matvec(D_lo, D_up, u_hist[1]**2)
    N2 = 0.5 * tri_matvec(D_lo, D_up, u_hist[0]**2)

    # RHS: BDF history + EXT-extrapolated advection
    rhs = -(b1 * u_hist[2] + b2 * u_hist[1] + b3 * u_hist[0]) \
          - dt * (a1 * N0 + a2 * N1 + a3 * N2)

    # Solve tridiagonal system via Thomas algorithm — O(N) instead of O(N^3)
    u_new = thomas_solve(H_lo, H_diag, H_up, rhs)

    # Shift history
    u_hist = [u_hist[1], u_hist[2], u_new]

    t = (n + 1) * dt
    step = n + 1

    # Record s(t) every step (1b)
    t_hist.append(t)
    s_hist.append(np.max(np.abs(tri_matvec(D_lo, D_up, u_new))))

    # Record snapshots at t = 0.1, 0.2, ..., 2.0 (1a) — O(1) lookup vs O(|t_snap|)
    if step in snap_steps:
        ts = round(t, 10)
        u_snaps[ts] = u_new.copy()

###############################
# Plots
###############################

# 1a: solution snapshots
fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.cm.viridis
for ts in sorted(u_snaps):
    color = cmap(ts / tfinal)
    ax.plot(x, u_snaps[ts], color=color, label=f't={ts:.1f}')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, tfinal))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='t')
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title(f'Burgers equation (conservation) — N={N}, {node_type} grid, dt={dt}')
plt.tight_layout()
plt.savefig(f"./P1Alex/1a_cons_solution_{node_type}.png", dpi=150)
plt.show()

# 1b: s(t) = max|du/dx|
t_hist = np.array(t_hist)
s_hist = np.array(s_hist)
i_max  = np.argmax(s_hist)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_hist, s_hist, 'b-', linewidth=1.5)
ax.axvline(t_hist[i_max], color='r', linestyle='--', label=f't* = {t_hist[i_max]:.4f}')
ax.scatter([t_hist[i_max]], [s_hist[i_max]], color='r', zorder=5,
           label=f's* = {s_hist[i_max]:.5f}')
ax.set_xlabel('t')
ax.set_ylabel('s(t) = max|du/dx|')
ax.set_title(f's(t) conservation — N={N}, {node_type} grid, dt={dt}')
ax.legend()
plt.tight_layout()
plt.savefig(f"./P1Alex/1b_cons_smax_{node_type}.png", dpi=150)
plt.show()

print(f"s*  = {s_hist[i_max]:.5f}")
print(f"pi t*  = {(t_hist[i_max]*np.pi):.4f}")
