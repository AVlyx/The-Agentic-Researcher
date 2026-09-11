"""E010: sharp dimension-free tradeoff  max { s1 - a*s2 }  over rank-<=2 C, k=3.
s1 = sum_j x_j, s2 = sum_{j<l} x_jl, x_S = ||Tr_S C||^2/||C||^2.
Pure numpy + scipy L-BFGS, many restarts.  Gives the sharp f(a) in  s1 <= f(a) + a*s2.
"""
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

rng = np.random.default_rng(7)


def ptrace(C, dims, S):
    k = len(dims); T = C.reshape(tuple(dims) + tuple(dims)); dims = list(dims)
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i); k -= 1; dims = dims[:i] + dims[i+1:]
    D = int(np.prod(dims)) if dims else 1
    return T.reshape(D, D)


def stats(C, dims):
    n2 = np.linalg.norm(C) ** 2
    x = {S: np.linalg.norm(ptrace(C, dims, S)) ** 2 / n2
         for r in range(4) for S in combinations(range(3), r)}
    s1 = sum(x[(j,)] for j in range(3))
    s2 = sum(x[s] for s in [(0, 1), (0, 2), (1, 2)])
    return s1, s2, x


def unpack(p, D):
    n = 2 * D * 2
    U = (p[:n:2] + 1j * p[1:n:2]).reshape(D, 2)
    V = (p[n::2] + 1j * p[n + 1::2]).reshape(D, 2)
    return U @ V.conj().T


def run(dims, a, restarts=40, seedC=None):
    D = int(np.prod(dims))
    best, bestC = -np.inf, None
    starts = []
    if seedC is not None:
        U, s, Vh = np.linalg.svd(seedC)
        starts.append((U[:, :2] * s[:2], Vh[:2].conj().T))
    for _ in range(restarts):
        starts.append((rng.normal(size=(D, 2)) + 1j * rng.normal(size=(D, 2)),
                       rng.normal(size=(D, 2)) + 1j * rng.normal(size=(D, 2))))
    for U0, V0 in starts:
        p0 = np.concatenate([np.stack([U0.real.ravel(), U0.imag.ravel()], 1).ravel(),
                             np.stack([V0.real.ravel(), V0.imag.ravel()], 1).ravel()])
        f = lambda p: -(lambda t: t[0] - a * t[1])(stats(unpack(p, D), dims)[:2]) \
            if np.linalg.norm(unpack(p, D)) > 1e-8 else 0.0
        r = minimize(f, p0, method="L-BFGS-B",
                     options=dict(maxiter=3000, ftol=1e-14, gtol=1e-12))
        C = unpack(r.x, D)
        if np.linalg.norm(C) < 1e-8:
            continue
        s1, s2, _ = stats(C, dims)          # independent numpy re-evaluation
        if s1 - a * s2 > best:
            best, bestC = s1 - a * s2, C
    return best, bestC


print("sharp f(a) = max over rank-2 C of (s1 - a*s2), by local dim")
print(f"{'a':>5s} " + " ".join(f"{'d='+str(d):>10s}" for d in (2, 3, 4)) + "   argmax(d=4) x_j | x_jl | t")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    row, lastC, lastdims = [], None, None
    for d in (2, 3, 4):
        dims = [d] * 3
        v, C = run(dims, a, restarts=25 if d < 4 else 15, seedC=lastC if lastdims == dims else None)
        row.append(v); lastC, lastdims = C, dims
    s1, s2, x = stats(lastC, [4, 4, 4])
    xj = [round(x[(j,)], 3) for j in range(3)]
    xjl = [round(x[s], 3) for s in [(0, 1), (0, 2), (1, 2)]]
    print(f"{a:5.2f} " + " ".join(f"{v:10.6f}" for v in row) +
          f"   {xj} | {xjl} | {round(x[(0,1,2)],3)}")
