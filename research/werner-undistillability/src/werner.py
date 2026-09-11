"""Core routines for the n-copy distillability problem for Werner states.

Notation and derivation
-----------------------
Werner state on C^d (x) C^d, with F the swap operator:

    rho_t = (I + t F) / (d^2 + t d),      t in [-1, 1].

Partial transpose: F^Gamma = d Q with Q = |Phi><Phi|, |Phi> = d^{-1/2} sum_i |ii>.
So rho_t^Gamma = (I + t d Q)/(d^2 + t d), which is NPT iff t < -1/d.  For Werner
states PPT <=> separable, so the NPT range is exactly t in [-1, -1/d).

Throughout we write c = -t, so the NPT range is c in (1/d, 1].

Distillability criterion (Horodecki-Horodecki-Horodecki; DiVincenzo et al.):
rho is n-copy distillable iff there is a Schmidt-rank-<=2 vector |phi> in
(C^{d^n} (x) C^{d^n}) with <phi| (rho^{(x)n})^Gamma |phi> < 0.

Since (rho^{(x)n})^Gamma = (rho^Gamma)^{(x)n} and, up to reordering tensor
factors (which preserves Schmidt rank across the A:B cut),

    (rho_{-c}^Gamma)^{(x)n}  ~  J(Lambda_c)^{(x)n} = J(Lambda_c^{(x)n}),

where J is the Choi matrix and Lambda_c : M_d -> M_d is the Werner-Holevo type map

    Lambda_c(X) = Tr(X) I_d - c X.

Using "Lambda is k-positive <=> J(Lambda) is positive on Schmidt rank <= k vectors":

    (R1)  rho_{-c} is n-copy UNdistillable  <=>  Lambda_c^{(x)n} is 2-positive.

Reference points: Lambda_c is positive iff c <= 1, completely positive iff
c <= 1/d (the PPT boundary), and k-positive iff c <= 1/k.

Expanding Lambda_c^{(x)n} = prod_i (Tr_i(.) I_i - c id_i) gives, for a state rho
on C^2 (x) (C^d)^{(x)n} (index 0 = qubit, 1..n = qudits),

    T_c(rho) := (id_2 (x) Lambda_c^{(x)n})(rho)
              = sum_{S subset [n]} (-c)^{|S|} rho_{0S} (x) I_{S^c}.

T_c is self-adjoint w.r.t. the Hilbert-Schmidt inner product, so with
B_c(psi, chi) := <chi| T_c(|psi><psi|) |chi> we get the symmetric biquadratic form

    (R2)  B_c(psi, chi) = sum_{S subset [n]} (-c)^{|S|} Tr[ rho^psi_{0S} rho^chi_{0S} ]
                        = <psi (x) chi| F_0 (x) (I - c F)^{(x)n} |psi (x) chi>,

and rho_{-c} is n-copy undistillable iff B_c(psi, chi) >= 0 for all psi, chi in
H = C^2 (x) (C^d)^{(x)n}.

On the diagonal chi = psi, using that a pure state has equal purities for
complementary marginals (Tr rho_{0S}^2 = Tr rho_{S^c}^2):

    (R3)  B_c(psi, psi) = sum_{R subset [n]} (-c)^{n-|R|} Tr[ (rho^psi_R)^2 ],

a statement purely about the marginal purities of a pure state of one qubit and
n qudits (R ranges over subsets of the qudits only; rho_emptyset = 1).

Define the n-copy threshold
    c_n(d) = sup { c : Lambda_c^{(x)n} is 2-positive }.
c_1(d) = 1/2 exactly; c_n is non-increasing in n; c_n >= 1/d always.  The NPT
bound entanglement conjecture is that lim_n c_n(d) > 1/d for d >= 3.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

# ----------------------------------------------------------------- basic objects


def swap(d: int) -> np.ndarray:
    """The swap operator F on C^d (x) C^d."""
    F = np.zeros((d, d, d, d))
    for i in range(d):
        for j in range(d):
            F[i, j, j, i] = 1.0
    return F.reshape(d * d, d * d)


def werner(t: float, d: int) -> np.ndarray:
    """Werner state rho_t = (I + t F)/(d^2 + t d) on C^d (x) C^d."""
    return (np.eye(d * d) + t * swap(d)) / (d * d + t * d)


def partial_transpose(rho: np.ndarray, d: int) -> np.ndarray:
    """Partial transpose on the second factor of C^d (x) C^d."""
    return rho.reshape(d, d, d, d).transpose(0, 3, 2, 1).reshape(d * d, d * d)


def choi(c: float, d: int) -> np.ndarray:
    """Choi matrix J(Lambda_c) = sum_ij Lambda_c(|i><j|) (x) |i><j| = I - c d Q."""
    q = np.zeros((d * d, d * d))
    for i in range(d):
        for j in range(d):
            q[i * d + i, j * d + j] = 1.0  # = d Q
    return np.eye(d * d) - c * q


# ------------------------------------------------- the map T_c on operators


def apply_T(rho_tensor: np.ndarray, d: int, n: int, c: float) -> np.ndarray:
    """(id_2 (x) Lambda_c^{(x)n}) applied to an operator.

    rho_tensor has shape (2,) + (d,)*n + (2,) + (d,)*n : the first n+1 axes are
    the "out" (ket) indices, the last n+1 the "in" (bra) indices.  Applies
    X -> Tr(X) I - c X on each qudit factor in turn, which costs n passes over
    the array instead of the 2^n subset sum.
    """
    A = rho_tensor
    for k in range(1, n + 1):
        B = np.moveaxis(A, [k, n + 1 + k], [-2, -1])
        tr = np.trace(B, axis1=-2, axis2=-1)
        B = tr[..., None, None] * np.eye(d) - c * B
        A = np.moveaxis(B, [-2, -1], [k, n + 1 + k])
    return A


def T_matrix(psi: np.ndarray, d: int, n: int, c: float) -> np.ndarray:
    """T_c(|psi><psi|) as a (2 d^n) x (2 d^n) Hermitian matrix.

    psi is a flat vector of length 2 d^n (layout: qubit slowest, then qudits).
    """
    shape = (2,) + (d,) * n
    v = psi.reshape(shape)
    rho = np.multiply.outer(v, v.conj())
    out = apply_T(rho, d, n, c)
    m = 2 * d**n
    return out.reshape(m, m)


def B_form(psi: np.ndarray, chi: np.ndarray, d: int, n: int, c: float) -> float:
    """The biquadratic form B_c(psi, chi) of (R2)."""
    T = T_matrix(psi, d, n, c)
    return float(np.real(chi.conj() @ T @ chi))


def B_form_marginals(psi: np.ndarray, chi: np.ndarray, d: int, n: int, c: float) -> float:
    """B_c(psi, chi) computed independently via the subset sum of (R2).

    Kept as a cross-check on apply_T; costs 2^n marginals.
    """
    shape = (2,) + (d,) * n
    v, w = psi.reshape(shape), chi.reshape(shape)
    total = 0.0
    for mask in range(1 << n):
        S = [i + 1 for i in range(n) if (mask >> i) & 1]
        keep = [0] + S
        traced = [a for a in range(n + 1) if a not in keep]
        rv = np.tensordot(v, v.conj(), axes=(traced, traced))
        rw = np.tensordot(w, w.conj(), axes=(traced, traced))
        dim = 2 * d ** len(S)
        rv = rv.reshape(dim, dim)
        rw = rw.reshape(dim, dim)
        total += ((-c) ** len(S)) * np.real(np.trace(rv @ rw))
    return float(total)


def B_diagonal_purities(psi: np.ndarray, d: int, n: int, c: float) -> float:
    """B_c(psi, psi) via the marginal-purity formula (R3)."""
    shape = (2,) + (d,) * n
    v = psi.reshape(shape)
    total = 0.0
    for mask in range(1 << n):
        R = [i + 1 for i in range(n) if (mask >> i) & 1]
        traced = [a for a in range(n + 1) if a not in R]
        r = np.tensordot(v, v.conj(), axes=(traced, traced))
        dim = d ** len(R)
        r = r.reshape(dim, dim)
        total += ((-c) ** (n - len(R))) * np.real(np.trace(r @ r))
    return float(total)


# ------------------------------------------------------------------ the see-saw


def lambda_min_vec(M: np.ndarray) -> tuple[float, np.ndarray]:
    w, V = sla.eigh(M, subset_by_index=[0, 0])
    return float(w[0]), V[:, 0]


def seesaw(d: int, n: int, c: float, rng: np.random.Generator, iters: int = 300,
           tol: float = 1e-13, psi0: np.ndarray | None = None):
    """Alternating minimisation of B_c(psi, chi) over unit psi, chi.

    Because B_c(psi, chi) = <chi|T_c(psi psi^*)|chi> = <psi|T_c(chi chi^*)|psi>,
    fixing one argument and minimising the other is an eigenvalue problem, so the
    iteration is monotonically non-increasing.  Returns (value, psi, chi).
    """
    m = 2 * d**n
    if psi0 is None:
        psi = rng.normal(size=m) + 1j * rng.normal(size=m)
        psi = psi / np.linalg.norm(psi)
    else:
        psi = psi0 / np.linalg.norm(psi0)
    val = np.inf
    chi = psi
    for _ in range(iters):
        _, chi = lambda_min_vec(T_matrix(psi, d, n, c))
        v2, psi = lambda_min_vec(T_matrix(chi, d, n, c))
        if val - v2 < tol:
            val = v2
            break
        val = v2
    return val, psi, chi


def g_n(d: int, n: int, c: float, restarts: int = 20, seed: int = 0, iters: int = 300):
    """Estimate g_n(c) = min over unit psi, chi of B_c(psi, chi) by multistart see-saw."""
    rng = np.random.default_rng(seed)
    best = np.inf
    arg = None
    for _ in range(restarts):
        v, psi, chi = seesaw(d, n, c, rng, iters=iters)
        if v < best:
            best, arg = v, (psi, chi)
    return best, arg


def threshold(d: int, n: int, lo: float = 0.0, hi: float = 1.0, restarts: int = 20,
              seed: int = 0, tol: float = 1e-7, iters: int = 300) -> float:
    """Bisect for c_n(d): the largest c with g_n(c) >= 0.

    The see-saw only ever exhibits distillability, so this returns an UPPER bound
    on the true n-copy threshold c_n(d)."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        v, _ = g_n(d, n, mid, restarts=restarts, seed=seed, iters=iters)
        if v < -1e-12:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------- normalised (ratio) see-saw
#
# B_c(psi, chi) = sum_k (-c)^k A_k(psi, chi) with
#     A_k = sum_{|S|=k} Tr[rho^psi_{0S} rho^chi_{0S}] >= 0,
# so B_c >= 0 has a large degenerate zero set: whenever the qubit marginals of
# psi and chi have orthogonal supports every A_k vanishes and B_c = 0 exactly.
# A plain see-saw is attracted to that set.  Dividing by A_0 = Tr[rho^psi_0
# rho^chi_0] = <chi| (rho^psi_0 (x) I) |chi> removes it: A_0 = 0 forces B_c = 0,
# and elsewhere sign(B_c) = sign(B_c / A_0).  Minimising the ratio for fixed psi
# is the generalised eigenproblem (T_c(psi), rho^psi_0 (x) I).

def qubit_marginal_op(psi: np.ndarray, d: int, n: int) -> np.ndarray:
    """rho_0^psi (x) I as a (2 d^n) x (2 d^n) matrix."""
    v = psi.reshape(2, d**n)
    r0 = v @ v.conj().T
    return np.kron(r0, np.eye(d**n))


def ratio_min_vec(psi: np.ndarray, d: int, n: int, c: float, ridge: float = 1e-12):
    T = T_matrix(psi, d, n, c)
    N = qubit_marginal_op(psi, d, n)
    N = N + ridge * np.eye(N.shape[0])
    w, V = sla.eigh(T, N, subset_by_index=[0, 0])
    v = V[:, 0]
    return float(w[0]), v / np.linalg.norm(v)


def seesaw_ratio(d: int, n: int, c: float, rng: np.random.Generator, iters: int = 400,
                 tol: float = 1e-12, psi0: np.ndarray | None = None):
    """Alternating minimisation of R_c(psi, chi) = B_c(psi, chi) / A_0(psi, chi)."""
    m = 2 * d**n
    if psi0 is None:
        psi = rng.normal(size=m) + 1j * rng.normal(size=m)
    else:
        psi = psi0.astype(complex)
    psi = psi / np.linalg.norm(psi)
    val = np.inf
    chi = psi
    for _ in range(iters):
        _, chi = ratio_min_vec(psi, d, n, c)
        v2, psi = ratio_min_vec(chi, d, n, c)
        if val - v2 < tol:
            val = v2
            break
        val = v2
    return val, psi, chi


def gt_n(d: int, n: int, c: float, restarts: int = 20, seed: int = 0, iters: int = 400,
         warm: np.ndarray | None = None):
    """min over psi, chi of R_c(psi, chi).  Negative <=> n-copy distillable."""
    rng = np.random.default_rng(seed)
    best, arg = np.inf, None
    starts = [warm] * (1 if warm is not None else 0) + [None] * restarts
    for s in starts:
        v, psi, chi = seesaw_ratio(d, n, c, rng, iters=iters, psi0=s)
        if v < best:
            best, arg = v, (psi, chi)
    return best, arg
