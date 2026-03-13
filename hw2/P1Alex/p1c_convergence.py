# Part 1c: convergence tables for all four cases
# N = 2^k, k = 5..11  →  N = 32, 64, 128, 256, 512, 1024, 2048
# Cases: (uniform|chebyshev) x (convective|conservation)
# Analytical reference: s* = 152.00516,  CFL = 0.1

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

S_EXACT  = 152.00516
CFL      = 0.1
TFINAL   = 2.0
N_VALUES = [2**k for k in range(5, 12)]   # 32 … 2048

CASES = [
    ("uniform",   "convective",   "i.   Uniform,    convective"),
    ("chebyshev", "convective",   "ii.  Chebyshev,  convective"),
    ("uniform",   "conservation", "iii. Uniform,    conservation"),
    ("chebyshev", "conservation", "iv.  Chebyshev,  conservation"),
]

# ─────────────────────────────────────────────
# Numerics
# ─────────────────────────────────────────────

def Nodes(N, node_type="uniform"):
    if node_type == "uniform":
        return np.linspace(0, 1, N + 1)
    j = np.arange(N + 1)
    return 0.5 * (1 - np.cos(np.pi * j / N))

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
        raise ValueError("max|u| = 0")
    return cfl * dx_min / max_u

# ─────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────

def run(N, node_type, form):
    """Return (s_star, dt, nsteps) for one grid/form combination."""
    x  = Nodes(N, node_type)
    A  = build_A(x)
    D  = build_D(x)
    A_lo, A_diag, A_up = extract_bands(A)
    D_lo, _,      D_up = extract_bands(D)

    u0     = np.sin(np.pi * x)
    dt     = cfl_dt(x, u0, CFL)
    nsteps = int(np.ceil(TFINAL / dt))

    u_hist = [u0.copy(), u0.copy(), u0.copy()]
    s_max  = np.max(np.abs(tri_matvec(D_lo, D_up, u0)))

    k = 1
    H_lo = H_diag = H_up = None
    for step in range(nsteps):
        if k == 1:
            b0=1;    b1=-1;  b2=0;   b3=0;    a1=1; a2= 0; a3=0
            H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
            k += 1
        elif k == 2:
            b0=1.5;  b1=-2;  b2=0.5; b3=0;    a1=2; a2=-1; a3=0
            H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
            k += 1
        else:
            b0=11/6; b1=-3;  b2=1.5; b3=-1/3; a1=3; a2=-3; a3=1
            if k == 3:
                H_lo, H_diag, H_up = H_bands(b0, A_lo, A_diag, A_up, dt)
                k += 1

        if form == "convective":
            N0 = u_hist[2] * tri_matvec(D_lo, D_up, u_hist[2])
            N1 = u_hist[1] * tri_matvec(D_lo, D_up, u_hist[1])
            N2 = u_hist[0] * tri_matvec(D_lo, D_up, u_hist[0])
        else:  # conservation
            N0 = 0.5 * tri_matvec(D_lo, D_up, u_hist[2]**2)
            N1 = 0.5 * tri_matvec(D_lo, D_up, u_hist[1]**2)
            N2 = 0.5 * tri_matvec(D_lo, D_up, u_hist[0]**2)

        rhs = -(b1 * u_hist[2] + b2 * u_hist[1] + b3 * u_hist[0]) \
              - dt * (a1 * N0 + a2 * N1 + a3 * N2)

        u_new  = thomas_solve(H_lo, H_diag, H_up, rhs)
        u_hist = [u_hist[1], u_hist[2], u_new]

        s = np.max(np.abs(tri_matvec(D_lo, D_up, u_new)))
        if s > s_max:
            s_max = s

    return s_max, dt, nsteps

# ─────────────────────────────────────────────
# JIT warm-up
# ─────────────────────────────────────────────
_x  = Nodes(10, "uniform")
_u  = np.sin(np.pi * _x)
_Al, _Ad, _Au = extract_bands(build_A(_x))
_Dl, _,   _Du = extract_bands(build_D(_x))
_Hl, _Hd, _Hu = H_bands(1.0, _Al, _Ad, _Au, 1e-3)
_ = tri_matvec(_Dl, _Du, _u)
_ = thomas_solve(_Hl, _Hd, _Hu, _u)

# ─────────────────────────────────────────────
# Run all four cases and print tables
# ─────────────────────────────────────────────
all_results = {}   # label -> list of (N, nsteps, s_star, rel_err, ratio)

HDR = f"{'N':>6}  {'nsteps':>8}  {'s*':>14}  {'rel.err.':>12}  {'ratio':>8}"
SEP = "-" * 56

for node_type, form, label in CASES:
    print(f"\n{'='*56}")
    print(f"  Case {label}")
    print(f"{'='*56}")
    print(HDR)
    print(SEP)

    rows     = []
    prev_err = None
    for N in N_VALUES:
        s_star, dt, nsteps = run(N, node_type, form)
        rel_err   = abs(s_star - S_EXACT) / S_EXACT
        ratio     = prev_err / rel_err if prev_err is not None else float('nan')
        rows.append((N, nsteps, s_star, rel_err, ratio))
        ratio_str = f"{ratio:8.3f}" if not np.isnan(ratio) else "     ---"
        print(f"{N:>6}  {nsteps:>8}  {s_star:>14.6f}  {rel_err:>12.3e}  {ratio_str}")
        prev_err = rel_err

    all_results[label] = rows

# ─────────────────────────────────────────────
# Plot: relative error vs N for all four cases
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

styles = ['bo-', 'rs-', 'g^-', 'mD-']
Ns = np.array(N_VALUES)

for (node_type, form, label), style in zip(CASES, styles):
    errs = np.array([r[3] for r in all_results[label]])
    ax.loglog(Ns, errs, style, linewidth=1.5, markersize=5, label=label)

# Reference slopes anchored to first data point of case i
ref_err = all_results[CASES[0][2]][0][3]
for order, ls, col in [(1, '--', 'gray'), (2, ':', 'black')]:
    ax.loglog(Ns, ref_err * (Ns[0] / Ns)**order, color=col,
              linestyle=ls, linewidth=1, label=f'O(N^-{order})')

ax.set_xlabel('N')
ax.set_ylabel('Relative error  |s* - s_exact| / s_exact')
ax.set_title(f'Part 1c — convergence of s*  (CFL={CFL}, s_exact={S_EXACT})')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('./P1Alex/1c_convergence.png', dpi=150)
plt.show()
