"""E012: the N-form.  q_k(-c,C) = l1^2 P1 + l2^2 P2 + 2 l1 l2 Z  with
N = (x)_j (I_j - c F_j) > 0,  P_i = <a_i b_i|N|a_i b_i>,  Z = Re<a_2 b_1|N|a_1 b_2>.
Hence  theta_k(c) >= 0  <=>  P1 P2 >= Z^2  for all orthonormal a_1,a_2 / b_1,b_2.
(1) verify the identity against the subset-sum definition;
(2) minimise P1 P2 - Z^2 over ENTANGLED a_i, b_i at c=1/2.
"""
import numpy as np
from itertools import combinations
from functools import reduce
from scipy.optimize import minimize

rng = np.random.default_rng(2718)


def ptrace(C, dims, S):
    k = len(dims); T = C.reshape(tuple(dims) + tuple(dims)); dims = list(dims)
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i); k -= 1; dims = dims[:i] + dims[i+1:]
    D = int(np.prod(dims)) if dims else 1
    return T.reshape(D, D)


def q_sub(C, dims, c):
    return sum((-c) ** len(S) * np.linalg.norm(ptrace(C, dims, S)) ** 2
               for r in range(len(dims) + 1) for S in combinations(range(len(dims)), r))


def Nop(d, k, c):
    """N = (x)_j (I - c F) on (H_A (x) H_B) with H_A = (C^d)^{(x)k}, sites interleaved
    then permuted to A-block / B-block ordering."""
    F = np.eye(d*d).reshape(d, d, d, d).transpose(1, 0, 2, 3).reshape(d*d, d*d)
    W = reduce(np.kron, [np.eye(d*d) - c * F] * k)
    W = W.reshape([d] * (4 * k))
    perm = list(range(0, 2*k, 2)) + list(range(1, 2*k, 2))
    return W.transpose(perm + [p + 2*k for p in perm]).reshape(d**(2*k), d**(2*k))


def PZ(N, a1, a2, b1, b2):
    kb = lambda x, y: np.kron(x, y)
    P1 = (kb(a1, b1).conj() @ N @ kb(a1, b1)).real
    P2 = (kb(a2, b2).conj() @ N @ kb(a2, b2)).real
    Z = (kb(a2, b1).conj() @ N @ kb(a1, b2)).real
    return P1, P2, Z


# ---- (1) identity check ----
for d, k in ((2, 2), (3, 2), (2, 3)):
    dims = [d] * k; D = d ** k; c = 0.43; N = Nop(d, k, c)
    A = rng.normal(size=(D, 2)) + 1j*rng.normal(size=(D, 2))
    B = rng.normal(size=(D, 2)) + 1j*rng.normal(size=(D, 2))
    Qa, Ra = np.linalg.qr(A); Qb, Rb = np.linalg.qr(B)
    a1, a2 = Qa[:, 0], Qa[:, 1]; b1, b2 = Qb[:, 0], Qb[:, 1]
    l1, l2 = 0.8, 0.6
    C = l1 * np.outer(a1, b1.conj()) + l2 * np.outer(a2, b2.conj())
    P1, P2, Z = PZ(N, a1, a2, b1, b2)
    lhs, rhs = q_sub(C, dims, c), l1**2*P1 + l2**2*P2 + 2*l1*l2*Z
    print(f"[{'PASS' if abs(lhs-rhs) < 1e-9*max(1,abs(lhs)) else 'FAIL'}] N-form d={d} k={k}"
          f"   subset-sum={lhs:.10f}  N-form={rhs:.10f}")

# ---- (2) minimise P1 P2 - Z^2 over entangled a_i, b_i at c = 1/2 ----
print("\nmin of (P1*P2 - Z^2) over orthonormal a1,a2 / b1,b2   [>=0 <=> theta_k(1/2)>=0]")
print(f"{'k':>2s} {'d':>2s} {'min P1P2-Z^2':>14s} {'min q at lam=1/sqrt2':>21s}")
c = 0.5
for k in (1, 2, 3):
    for d in (2, 3, 4):
        if d ** k > 70:
            continue
        D = d ** k; N = Nop(d, k, c)
        def unpack(p):
            A = (p[:2*D] + 1j*p[2*D:4*D]).reshape(D, 2)
            B = (p[4*D:6*D] + 1j*p[6*D:8*D]).reshape(D, 2)
            Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
            return Qa[:, 0], Qa[:, 1], Qb[:, 0], Qb[:, 1]
        def f(p):
            P1, P2, Z = PZ(N, *unpack(p))
            return P1 * P2 - Z ** 2
        best, bestq = np.inf, np.inf
        for _ in range(60):
            r = minimize(f, rng.normal(size=8*D), method="L-BFGS-B",
                         options=dict(maxiter=4000, ftol=1e-15))
            P1, P2, Z = PZ(N, *unpack(r.x))          # independent numpy re-evaluation
            best = min(best, P1 * P2 - Z ** 2)
            bestq = min(bestq, 0.5 * P1 + 0.5 * P2 + Z)
        print(f"{k:2d} {d:2d} {best:14.8f} {bestq:21.8f}")
