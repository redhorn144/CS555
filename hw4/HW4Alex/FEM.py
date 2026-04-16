import numpy as np
import scipy.sparse as sp


def build_mesh_uniform(E, L=1.0):
    """Return E+1 uniformly spaced nodal coordinates on [0, L]."""
    return np.linspace(0, L, E + 1)


def build_mesh_geometric(E, L=1.0, s=0.7):
    """
    Return E+1 nodal coordinates on [0, L] with a geometric progression
    of element lengths: L_e = s * L_{e-1}.
    """
    # L_1 * (1 + s + s^2 + ... + s^{E-1}) = L
    if abs(s - 1.0) < 1e-14:
        return np.linspace(0, L, E + 1)
    L1 = L * (1 - s) / (1 - s**E)
    lengths = L1 * s ** np.arange(E)
    nodes = np.zeros(E + 1)
    nodes[1:] = np.cumsum(lengths)
    nodes[-1] = L  # enforce exact endpoint
    return nodes


# ---------------------------------------------------------------------------
# Local element matrices
# ---------------------------------------------------------------------------

def local_stiffness(Le):
    """
    Local stiffness matrix A^e (2x2) for a linear element of length Le.

    A^e_pq = int_{Omega_e} dl_p/dx * dl_q/dx dx = (1/Le) * [[1,-1],[-1,1]]
    """
    return (1.0 / Le) * np.array([[ 1.0, -1.0],
                                   [-1.0,  1.0]])


def local_mass(Le):
    """
    Local mass matrix B^e (2x2) for a linear element of length Le.

    B^e_pq = int_{Omega_e} l_p * l_q dx = (Le/6) * [[2,1],[1,2]]
    """
    return (Le / 6.0) * np.array([[2.0, 1.0],
                                   [1.0, 2.0]])


def local_advection(Le, c=1.0):
    """
    Local advection matrix C^e (2x2) for a linear element of length Le.

    C^e_pq = int_{Omega_e} l_p * c * dl_q/dx dx = (c/2) * [[-1,1],[-1,1]]
    """
    return (c / 2.0) * np.array([[-1.0,  1.0],
                                  [-1.0,  1.0]])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(nodes, c=1.0):
    """
    Assemble the global (Neumann) stiffness A_bar, mass B_bar, and advection
    C_bar matrices from the nodal coordinates.

    Parameters
    ----------
    nodes : array of length E+1
        Global nodal coordinates (must include boundary nodes).
    c : float
        Advection speed.

    Returns
    -------
    A_bar, B_bar, C_bar : (E+1, E+1) sparse CSR matrices
    """
    E = len(nodes) - 1
    N = E  # number of interior + boundary DOFs = E+1, index 0..N

    rows_A, cols_A, vals_A = [], [], []
    rows_B, cols_B, vals_B = [], [], []
    rows_C, cols_C, vals_C = [], [], []

    for e in range(E):
        Le = nodes[e + 1] - nodes[e]
        Ae = local_stiffness(Le)
        Be = local_mass(Le)
        Ce = local_advection(Le, c)
        idx = [e, e + 1]  # global node indices for this element
        for p in range(2):
            for q in range(2):
                rows_A.append(idx[p]); cols_A.append(idx[q]); vals_A.append(Ae[p, q])
                rows_B.append(idx[p]); cols_B.append(idx[q]); vals_B.append(Be[p, q])
                rows_C.append(idx[p]); cols_C.append(idx[q]); vals_C.append(Ce[p, q])

    size = N + 1
    A_bar = sp.csr_matrix((vals_A, (rows_A, cols_A)), shape=(size, size))
    B_bar = sp.csr_matrix((vals_B, (rows_B, cols_B)), shape=(size, size))
    C_bar = sp.csr_matrix((vals_C, (rows_C, cols_C)), shape=(size, size))

    return A_bar, B_bar, C_bar


# ---------------------------------------------------------------------------
# Restriction matrix (enforces homogeneous Dirichlet BCs)
# ---------------------------------------------------------------------------

def build_R(N, bc='dirichlet_both'):
    """
    Build restriction matrix R that maps global DOFs (0..N) to interior DOFs.

    Parameters
    ----------
    N : int
        Number of elements (so N+1 global nodes, indices 0..N).
    bc : str
        'dirichlet_both' : remove nodes 0 and N  (u(0)=u(L)=0)
        'dirichlet_left' : remove node 0 only     (u(0)=0, Neumann at x=L)

    Returns
    -------
    R : sparse CSR matrix of shape (n_free, N+1)
    """
    if bc == 'dirichlet_both':
        interior = np.arange(1, N)
    elif bc == 'dirichlet_left':
        interior = np.arange(1, N + 1)
    else:
        raise ValueError(f"Unknown bc type: {bc!r}")

    R = sp.eye(N + 1, format='csr')[interior, :]
    return R


# ---------------------------------------------------------------------------
# Restricted operators
# ---------------------------------------------------------------------------

def restrict(A_bar, B_bar, C_bar, R):
    """
    Apply the restriction R to obtain operators on the interior DOFs.

    Returns A, B, C as sparse CSR matrices where:
        A = R @ A_bar @ R.T
        B = R @ B_bar @ R.T
        C = R @ C_bar @ R.T
    """
    A = R @ A_bar @ R.T
    B = R @ B_bar @ R.T
    C = R @ C_bar @ R.T
    return A, B, C
