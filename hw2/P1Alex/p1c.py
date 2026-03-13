# Part 1c: convergence of s* = max_t max_x |du/dx| vs N
# Analytical reference: s* = 152.00516
# Chebyshev nodes, BDF3/EXT3, CFL = 0.05

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

S_EXACT   = 152.00516
CFL       = 0.1
TFINAL    = 2.0
NODE_TYPE = "chebyshev"

N_VALUES  = [25, 50, 100, 200, 400, 800, 1500, 3000, 6000, 10000, 15000]

# ─────────────────────────────────────────────
# Shared numerics (same as p1a.py)
# ─────────────────────────────────────────────

def Nodes(N, type="uniform"):
    if type == "uniform":
        return np.linspace(0, 1, N + 1)
    elif type == "chebyshev":
        j = np.arange(N + 1)
        return 0.5 * (1 - np.cos(np.pi * j / N))
    else:
        raise ValueError("Unknown node type")

def build_A(x, nu=(1e-2) / np.pi):
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
    N = len(x) - 1
    D = np.zeros((N + 1, N + 1))
    for j in range(1, N):
        D[j, j - 1] = -1 / (x[j + 1] - x[j - 1])
        D[j, j + 1] =  1 / (x[j + 1] - x[j - 1])
    return D

@njit(cache=True)
def tri_matvec(lower, upper, u):
    n = len(u)
    out = np.zeros(n)
    for j in range(1, n - 1):
        out[j] = lower[j] * u[j - 1] + upper[j] * u[j + 1]
    return out

@njit(cache=True)
def thomas_solve(lo, diag, up, d):
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
    n = M.shape[0]
    lo   = np.zeros(n)
    diag = np.diag(M).copy()
    up   = np.zeros(n)
    for i in range(1, n):
        lo[i] = M[i, i - 1]
    for i in range(n - 1):
        up[i] = M[i, i + 1]
    return lo, diag, up

def H_bands(b0, A_lo, A_diag, A_up, dt):
    return -dt * A_lo, b0 - dt * A_diag, -dt * A_up

def cfl_dt(x, u, cfl=0.1):
    dx_min = 1 / len(x)
    max_u  = np.max(np.abs(u))
    if max_u == 0:
        raise ValueError("max|u| = 0; cannot determine dt from CFL.")
    return cfl * dx_min / max_u

# ─────────────────────────────────────────────
# Solver: returns s* for a given N
# ─────────────────────────────────────────────

def run(N):
    x  = Nodes(N, NODE_TYPE)
    A  = build_A(x)
    D  = build_D(x)
    A_lo, A_diag, A_up = extract_bands(A)
    D_lo, _,      D_up = extract_bands(D)

    u0 = np.sin(np.pi * x)
    dt = cfl_dt(x, u0, CFL)
    nsteps = int(np.ceil(TFINAL / dt))

    u_hist = [u0.copy(), u0.copy(), u0.copy()]
    s_max  = np.max(np.abs(tri_matvec(D_lo, D_up, u0)))

    k = 1
    H_lo = H_diag = H_up = None
    for step in range(nsteps):
        if k == 1:
            b0=1;     b1=-1;  b2=0;   b3=0;     a1=1; a2= 0; a3=0
            H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
            k += 1
        elif k == 2:
            b0=1.5;   b1=-2;  b2=0.5; b3=0;     a1=2; a2=-1; a3=0
            H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
            k += 1
        else:
            b0=11/6;  b1=-3;  b2=1.5; b3=-1/3;  a1=3; a2=-3; a3=1
            if k == 3:
                H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
                k += 1

        N0 = u_hist[2] * tri_matvec(D_lo, D_up, u_hist[2])
        N1 = u_hist[1] * tri_matvec(D_lo, D_up, u_hist[1])
        N2 = u_hist[0] * tri_matvec(D_lo, D_up, u_hist[0])

        rhs = -(b1 * u_hist[2] + b2 * u_hist[1] + b3 * u_hist[0]) \
              - dt * (a1 * N0 + a2 * N1 + a3 * N2)

        u_new  = thomas_solve(H_lo, H_diag, H_up, rhs)
        u_hist = [u_hist[1], u_hist[2], u_new]

        s = np.max(np.abs(tri_matvec(D_lo, D_up, u_new)))
        if s > s_max:
            s_max = s

    return s_max, dt, nsteps

# ─────────────────────────────────────────────
# JIT warm-up on a tiny problem
# ─────────────────────────────────────────────
_x0  = Nodes(10, NODE_TYPE)
_u0  = np.sin(np.pi * _x0)
_A0  = build_A(_x0);  _alo, _adiag, _aup = extract_bands(_A0)
_D0  = build_D(_x0);  _dlo, _,      _dup = extract_bands(_D0)
_Hlo, _Hd, _Hup = H_bands(1.0, _alo, _adiag, _aup, 1e-3)
_ = tri_matvec(_dlo, _dup, _u0)
_ = thomas_solve(_Hlo, _Hd, _Hup, _u0)

# ─────────────────────────────────────────────
# Sweep over N
# ─────────────────────────────────────────────
print(f"{'N':>7}  {'dt':>12}  {'nsteps':>8}  {'s*':>14}  {'rel_err':>12}")
print("-" * 62)

results = []
for N in N_VALUES:
    s_star, dt, nsteps = run(N)
    rel_err = abs(s_star - S_EXACT) / S_EXACT
    results.append((N, dt, nsteps, s_star, rel_err))
    print(f"{N:>7}  {dt:>12.6e}  {nsteps:>8}  {s_star:>14.6f}  {rel_err:>12.3e}")

# ─────────────────────────────────────────────
# Plot: relative error vs N
# ─────────────────────────────────────────────
Ns      = np.array([r[0] for r in results])
rel_err = np.array([r[4] for r in results])

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(Ns, rel_err, 'bo-', linewidth=1.5, markersize=5, label='rel. error')

# reference slope lines
for order, ls in [(1, '--'), (2, ':')]:
    ref = rel_err[0] * (Ns[0] / Ns) ** order
    ax.loglog(Ns, ref, color='gray', linestyle=ls, label=f'O(N^-{order})')

ax.set_xlabel('N')
ax.set_ylabel('Relative error  |s* - s_exact| / s_exact')
ax.set_title(f'Part 1c — convergence of s*  (CFL={CFL}, {NODE_TYPE})')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('./P1Alex/1c_convergence.png', dpi=150)
plt.show()
