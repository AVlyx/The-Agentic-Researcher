"""E011: (a) verify the swap identity and the 2x2 Gram identity; (b) minimise q_k over
RANK-ONE C, to see how much margin the diagonal of the Gram matrix carries."""
import numpy as np
from itertools import combinations
from functools import reduce
from scipy.optimize import minimize

rng = np.random.default_rng(11)


def ptrace(C, dims, S):
    k = len(dims); T = C.reshape(tuple(dims) + tuple(dims)); dims = list(dims)
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i); k -= 1; dims = dims[:i] + dims[i+1:]
    D = int(np.prod(dims)) if dims else 1
    return T.reshape(D, D)


def subsets(k):
    return [S for r in range(k + 1) for S in combinations(range(k), r)]


def q(C, dims, c):
    return sum((-c) ** len(S) * np.linalg.norm(ptrace(C, dims, S)) ** 2
               for S in subsets(len(dims)))


# ---------- (a1) swap identity:  q_k = Tr[(C (x) C^dag) prod_j (F_j - c I_j)] ----------
def swap_identity(dims, c, C):
    k = len(dims); D = int(np.prod(dims))
    ops = []
    for j, dj in enumerate(dims):
        F = np.eye(dj * dj).reshape(dj, dj, dj, dj).transpose(1, 0, 2, 3).reshape(dj*dj, dj*dj)
        ops.append(F - c * np.eye(dj * dj))
    W = reduce(np.kron, ops)                       # on (H_1 H_1')(H_2 H_2')...
    # reorder W from interleaved (j,j') to block (all j)(all j')
    W = W.reshape([d for dj in dims for d in (dj, dj)] * 2)
    perm = list(range(0, 2*k, 2)) + list(range(1, 2*k, 2))
    W = W.transpose(perm + [p + 2*k for p in perm]).reshape(D*D, D*D)
    M = np.kron(C, C.conj().T)
    return np.trace(M @ W).real


for dims in ([2, 2], [3, 2], [2, 3, 2]):
    D = int(np.prod(dims)); c = 0.37
    C = (rng.normal(size=(D, 2)) + 1j*rng.normal(size=(D, 2))) @ \
        (rng.normal(size=(D, 2)) + 1j*rng.normal(size=(D, 2))).conj().T
    a, b = q(C, dims, c), swap_identity(dims, c, C)
    print(f"[{'PASS' if abs(a-b) < 1e-9*max(1,abs(a)) else 'FAIL'}] swap identity dims={dims}"
          f"  q={a:.10f}  Tr[(C x C^d)W]={b:.10f}")

# ---------- (a2) Gram identity:  q_k = 1^T G 1,  G_ii' from rank-one pieces ----------
for dims in ([3, 2], [2, 2, 2]):
    D = int(np.prod(dims)); c = 0.41
    U = rng.normal(size=(D, 2)) + 1j*rng.normal(size=(D, 2))
    V = rng.normal(size=(D, 2)) + 1j*rng.normal(size=(D, 2))
    C = U @ V.conj().T
    G = np.zeros((2, 2), complex)
    for i in range(2):
        for ip in range(2):
            tot = 0
            for S in subsets(len(dims)):
                Ai = ptrace(np.outer(U[:, i], V[:, i].conj()), dims, S)
                Ap = ptrace(np.outer(U[:, ip], V[:, ip].conj()), dims, S)
                tot += (-c) ** len(S) * np.trace(Ap.conj().T @ Ai)
            G[ip, i] = tot
    a, b = q(C, dims, c), np.ones(2) @ G @ np.ones(2)
    print(f"[{'PASS' if abs(a-b.real) < 1e-9*max(1,abs(a)) else 'FAIL'}] Gram identity dims={dims}"
          f"  q={a:.10f}  1^T G 1={b.real:.10f}")

# ---------- (b) minimise q_k over rank-one C = |u><v| ----------
print("\nmin of q_k(-c,C)/||C||^2 over RANK-ONE C   (rank-2 min = theta_k(c))")
print(f"{'k':>2s} {'d':>2s} {'c':>5s} {'rank1 min':>12s}   {'theta_k(c)=(1-2c)(1-c)^(k-1)':>28s}")
for k in (1, 2, 3):
    for d in (2, 3, 4):
        for c in (0.5,):
            dims = [d] * k; D = d ** k
            def f(p):
                u = p[:2*D:2] + 1j*p[1:2*D:2]; v = p[2*D::2] + 1j*p[2*D+1::2]
                C = np.outer(u, v.conj()); n = np.linalg.norm(C)
                if n < 1e-9:
                    return 0.0
                return q(C, dims, c) / n**2
            best = np.inf
            for _ in range(30):
                p0 = rng.normal(size=4*D)
                r = minimize(f, p0, method="L-BFGS-B", options=dict(maxiter=2000))
                u = r.x[:2*D:2] + 1j*r.x[1:2*D:2]; v = r.x[2*D::2] + 1j*r.x[2*D+1::2]
                C = np.outer(u, v.conj())
                if np.linalg.norm(C) > 1e-9:
                    best = min(best, q(C, dims, c) / np.linalg.norm(C)**2)  # numpy recheck
            print(f"{k:2d} {d:2d} {c:5.2f} {best:12.6f}   {(1-2*c)*(1-c)**(k-1):28.6f}")
