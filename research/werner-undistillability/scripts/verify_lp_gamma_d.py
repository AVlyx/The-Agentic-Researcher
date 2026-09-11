"""Dimension-aware LP bound on theta_k(c): gamma_k(d).

The dimension-free LP in verify_lp_gamma.py is loose in an identifiable way: at its optimum
it sets x_S = 0 for all |S| = k-1 while keeping x_{[k]} = |Tr C|^2 = 2, which is impossible
(Tr_S C = 0 forces Tr C = 0).  The missing inequalities are genuinely dimension-dependent,
so we add them and report gamma_k(d) rather than a dimension-free constant.

Extra valid constraints, for C of rank <= 2 on (C^d)^{(x)k}.  Write r_S for a bound on
rank(Tr_S C).  Since Tr_S(uv^dag) is the product of the H_{S^c} x H_S reshapes of u and v,
    rank(Tr_S C) <= 2 min(d^{|S|}, d^{k-|S|}) =: r_S .
Then, with x_S = ||Tr_S C||^2/||C||^2:

 (D1)  |Tr C|^2 <= rank(Tr_S C) ||Tr_S C||^2        =>   x_{[k]} <= r_S x_S ,
       from |Tr M| <= sqrt(rank M) ||M||_HS applied to M = Tr_S C (note Tr C = Tr(Tr_S C)).
 (D2)  for S subset S',  ||Tr_{S'} C||^2 = ||Tr_{S' \ S}(Tr_S C)||^2 <= r_S ||Tr_S C||^2,
       i.e.  x_{S'} <= r_S x_S ,
       by the rank lemma applied to Tr_S C.

Both are added to the dimension-free LP.  The optimum stays a valid lower bound on
theta_k(c), now for the given d.

Run:  python scripts/verify_lp_gamma_d.py
"""
from __future__ import annotations

import sys, os, json
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

rng = np.random.default_rng(2718)
HERE = os.path.join(os.path.dirname(__file__), "..")
FAILS = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def rand_rank_r(dims, r):
    D = int(np.prod(dims))
    A = rng.normal(size=(D, r)) + 1j * rng.normal(size=(D, r))
    B = rng.normal(size=(D, r)) + 1j * rng.normal(size=(D, r))
    return A @ B.conj().T


def ptrace(C, dims, S):
    k = len(dims)
    T = C.reshape(tuple(dims) + tuple(dims))
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i)
        k -= 1
        dims = dims[:i] + dims[i + 1:]
    D = int(np.prod(dims)) if dims else 1
    return T.reshape(D, D)


def rbound(S, k, d):
    return 2 * min(d ** len(S), d ** (k - len(S)))


# ------------------------------------------------- numerical check of D1 and D2
w1 = w2 = 0.0
wrank = 0.0
for k in (2, 3):
    for d in (2, 3, 4):
        dims = [d] * k
        for _ in range(150):
            C = rand_rank_r(dims, 2)
            n2 = np.linalg.norm(C) ** 2
            t = abs(np.trace(C)) ** 2 / n2
            for r in range(1, k + 1):
                for S in combinations(range(k), r):
                    xS = np.linalg.norm(ptrace(C, dims, S)) ** 2 / n2
                    rk = np.linalg.matrix_rank(ptrace(C, dims, S), tol=1e-9)
                    wrank = max(wrank, rk / rbound(S, k, d))
                    if xS > 1e-12:
                        w1 = max(w1, t / (rbound(S, k, d) * xS))
                    for r2 in range(r + 1, k + 1):
                        for S2 in combinations(range(k), r2):
                            if not set(S).issubset(S2):
                                continue
                            xS2 = np.linalg.norm(ptrace(C, dims, S2)) ** 2 / n2
                            if xS > 1e-12:
                                w2 = max(w2, xS2 / (rbound(S, k, d) * xS))
check("rank(Tr_S C) <= 2 min(d^|S|, d^{k-|S|})", wrank <= 1 + 1e-9, f"(max {wrank:.4f})")
check("D1  x_[k] <= r_S x_S", w1 <= 1 + 1e-9, f"(max ratio {w1:.6f})")
check("D2  x_S' <= r_S x_S for S subset S'", w2 <= 1 + 1e-9, f"(max ratio {w2:.6f})")


# --------------------------------------------------------------------- the LP
def lp_bound_d(k, c, d, dimension_free=False):
    subs = []
    for r in range(k + 1):
        subs.extend(combinations(range(k), r))
    idx = {S: i for i, S in enumerate(subs)}
    n = len(subs)
    obj = np.array([(-c) ** len(S) for S in subs])
    full = tuple(range(k))

    A_ub, b_ub = [], []
    # FHPV with a spectator
    for ra in range(1, k):
        for A in combinations(range(k), ra):
            rest = [s for s in range(k) if s not in A]
            for rb in range(1, len(rest) + 1):
                for B in combinations(rest, rb):
                    row = np.zeros(n)
                    row[idx[A]] += 1
                    row[idx[B]] += 1
                    row[idx[tuple(sorted(A + B))]] -= 0.5
                    A_ub.append(row)
                    b_ub.append(2.0)
    if not dimension_free:
        for r in range(1, k + 1):
            for S in combinations(range(k), r):
                rS = rbound(S, k, d)
                if S != full:                                   # D1
                    row = np.zeros(n)
                    row[idx[full]] += 1
                    row[idx[S]] -= rS
                    A_ub.append(row)
                    b_ub.append(0.0)
                for r2 in range(r + 1, k + 1):                  # D2
                    for S2 in combinations(range(k), r2):
                        if not set(S).issubset(S2):
                            continue
                        row = np.zeros(n)
                        row[idx[S2]] += 1
                        row[idx[S]] -= rS
                        A_ub.append(row)
                        b_ub.append(0.0)
    bounds = [(1.0, 1.0) if S == () else (0.0, 2.0) for S in subs]
    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds,
                  method="highs")
    assert res.status == 0, res.message
    return res.fun


def root(f, lo=1e-6, hi=0.5):
    if f(hi) > 0:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


print()
print("gamma_k(d): c <= gamma_k(d) => rho(d,-c) is k-copy undistillable.")
print("NPT needs c > 1/d, so the window is non-empty iff gamma_k(d) > 1/d.")
print()
print(f"{'k':>2} {'d':>3} {'1/d':>8} {'gamma_k(d)':>12} {'dim-free':>10}   window?")
table = []
for k in (3, 4, 5):
    for d in (3, 4, 5, 6, 8):
        g = root(lambda c: lp_bound_d(k, c, d))
        gf = root(lambda c: lp_bound_d(k, c, d, dimension_free=True))
        win = g > 1.0 / d
        table.append(dict(k=k, d=d, gamma=g, gamma_free=gf, window=bool(win)))
        print(f"{k:>2} {d:>3} {1/d:>8.4f} {g:>12.6f} {gf:>10.6f}   "
              f"{'YES' if win else 'no'}")
    print()

# --------------------------- sanity: bound must not exceed the padded upper bound
ok = True
for k in (3, 4):
    for d in (3, 4):
        for c in (0.1, 0.2, 0.3, 0.4):
            ok &= lp_bound_d(k, c, d) <= (1 - 2 * c) * (1 - c) ** (k - 1) + 1e-9
check("LP_d bound <= padded upper bound", ok)

# --------------------------- sanity: bound must not exceed sampled rank-2 values
ok2 = True
tight = np.inf
for k in (2, 3):
    for d in (3, 4):
        dims = [d] * k
        for c in (0.15, 0.3):
            lo = lp_bound_d(k, c, d)
            for _ in range(300):
                C = rand_rank_r(dims, 2)
                tot = 0.0
                for r in range(k + 1):
                    for S in combinations(range(k), r):
                        tot += ((-c) ** r) * np.linalg.norm(ptrace(C, dims, S)) ** 2
                val = tot / np.linalg.norm(C) ** 2
                tight = min(tight, val - lo)
                ok2 &= val >= lo - 1e-9
check("LP_d bound <= q_k(-c,C)/||C||^2 for random rank-2 C", ok2,
      f"(min slack {tight:+.4f})")

with open(os.path.join(HERE, "data", "lp_gamma_d.json"), "w") as f:
    json.dump(table, f, indent=1)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed; wrote data/lp_gamma_d.json")
