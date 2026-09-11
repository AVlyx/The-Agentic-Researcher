"""Verify every step of the recursive lower bound on theta_k(c), and hence on gamma_k.

Notation: for an operator C on (C^d)^{(x)k},
    q_k(alpha, C) = sum_{S subset [k]} alpha^{|S|} ||Tr_S C||_HS^2 ,
    theta_k(c)    = min { q_k(-c, C)/||C||^2 : rank C <= 2 } .
gamma_k denotes any constant with c <= gamma_k => theta_k(c) >= 0, i.e. every Werner
state rho_alpha with alpha >= -gamma_k is k-copy undistillable, in every dimension.

Chain of claims (each checked below):

 (C1) rank C <= r  =>  ||Tr_S C||^2 <= r ||C||^2  for every S.
 (C2) rank C <= r  =>  |Tr C|^2 <= r ||C||^2.
 (C3) Fraser-Huber-Pozsgay-Vona (arXiv:2607.24309), used as an input, not reproved:
      rank C <= r  =>  ||Tr_1 C||^2 + ||Tr_2 C||^2 <= r ||C||^2 + (1/r)|Tr C|^2.
 (C4) Recursion: q_k(alpha, C) = sum_{m,m'} q_{k-1}(alpha, C^{(mm')})
                                 + alpha q_{k-1}(alpha, Tr_k C),
      where C^{(mm')} = (I (x) <m|) C (I (x) |m'>) are the blocks of C in site k.
      Each block has rank <= rank C, and sum_{m,m'} ||C^{(mm')}||^2 = ||C||^2.
 (C5) theta_2(c) >= (1-c)(1-2c) for 0 <= c <= 1/2  [from (C3) with r=2 and (C2)].
      Combined with the padded upper bound this gives theta_2(c) = (1-2c)(1-c) exactly.
 (C6) q_{k-1}(-c, X) <= ||X||^2 + sum over even |S| >= 2 ... , concretely
      q_{k-1}(-c, Tr_k C) <= ((1+c)^{k-1} + (1-c)^{k-1}) ||C||^2 for rank C <= 2,
      by keeping only the even-|S| terms and applying (C1).
 (C7) Hence theta_k(c) >= theta_{k-1}(c) - c((1+c)^{k-1} + (1-c)^{k-1}), and with (C5),
          theta_3(c) >= 1 - 5c + 2c^2 - 2c^3.

Run:  python scripts/verify_gamma.py
"""
from __future__ import annotations

import sys, os
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402

rng = np.random.default_rng(20260909)
FAILS = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def rand_rank_r(dims, r):
    """Random operator of rank <= r on the tensor product of `dims`."""
    D = int(np.prod(dims))
    A = rng.normal(size=(D, r)) + 1j * rng.normal(size=(D, r))
    B = rng.normal(size=(D, r)) + 1j * rng.normal(size=(D, r))
    return A @ B.conj().T


def ptrace(C, dims, S):
    """Partial trace of C over the sites in S (0-indexed)."""
    k = len(dims)
    T = C.reshape(tuple(dims) + tuple(dims))
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i)
        k -= 1
        dims = dims[:i] + dims[i + 1:]
    D = int(np.prod(dims)) if dims else 1
    return T.reshape(D, D)


def q(alpha, C, dims):
    k = len(dims)
    tot = 0.0
    for r in range(k + 1):
        for S in combinations(range(k), r):
            tot += (alpha ** r) * np.linalg.norm(ptrace(C, dims, S)) ** 2
    return float(np.real(tot))


# ------------------------------------------------------------------ C1 and C2
worst1 = worst2 = 0.0
for d in (2, 3, 4):
    for k in (1, 2, 3):
        dims = [d] * k
        for r in (1, 2, 3):
            for _ in range(60):
                C = rand_rank_r(dims, r)
                n2 = np.linalg.norm(C) ** 2
                for m in range(1, k + 1):
                    for S in combinations(range(k), m):
                        worst1 = max(worst1, np.linalg.norm(ptrace(C, dims, S)) ** 2 / (r * n2))
                worst2 = max(worst2, abs(np.trace(C)) ** 2 / (r * n2))
check("C1  ||Tr_S C||^2 <= r ||C||^2", worst1 <= 1 + 1e-9, f"(max ratio {worst1:.6f})")
check("C2  |Tr C|^2 <= r ||C||^2", worst2 <= 1 + 1e-9, f"(max ratio {worst2:.6f})")

# ------------------------------------------------------------------------- C3
worst3 = 0.0
for d1 in (2, 3, 4):
    for d2 in (2, 3, 4):
        for r in (1, 2, 3, 4):
            for _ in range(60):
                C = rand_rank_r([d1, d2], r)
                lhs = (np.linalg.norm(ptrace(C, [d1, d2], (0,))) ** 2
                       + np.linalg.norm(ptrace(C, [d1, d2], (1,))) ** 2)
                rhs = r * np.linalg.norm(C) ** 2 + abs(np.trace(C)) ** 2 / r
                worst3 = max(worst3, lhs / rhs)
check("C3  Fraser et al. partial-trace inequality", worst3 <= 1 + 1e-9,
      f"(max lhs/rhs {worst3:.6f})")

# ------------------------------------------------------------------------- C4
ok4 = True
for d in (2, 3):
    for k in (2, 3):
        dims = [d] * k
        for _ in range(15):
            C = rand_rank_r(dims, 2)
            alpha = -0.37
            T = C.reshape(tuple(dims) + tuple(dims))
            # blocks in the LAST site
            lhs = q(alpha, C, dims)
            sub = dims[:-1]
            Dsub = int(np.prod(sub)) if sub else 1
            acc = 0.0
            ranks_ok = True
            for m in range(d):
                for mp in range(d):
                    blk = T[..., m, :, mp] if False else None
            # build blocks explicitly
            Tr = C.reshape(Dsub, d, Dsub, d)
            for m in range(d):
                for mp in range(d):
                    blk = Tr[:, m, :, mp]
                    ranks_ok &= (np.linalg.matrix_rank(blk, tol=1e-9) <= 2)
                    acc += q(alpha, blk, sub)
            trk = ptrace(C, dims, (k - 1,))
            rhs = acc + alpha * q(alpha, trk, sub)
            ok4 &= np.isclose(lhs, rhs, rtol=1e-9, atol=1e-9) and ranks_ok
            # Frobenius decomposition
            tot = sum(np.linalg.norm(Tr[:, m, :, mp]) ** 2
                      for m in range(d) for mp in range(d))
            ok4 &= np.isclose(tot, np.linalg.norm(C) ** 2)
check("C4  block recursion + rank + Frobenius decomposition", ok4)

# ------------------------------------------------------------------------- C5
ok5 = True
worst5 = np.inf
for c in (0.05, 0.2, 0.35, 0.45, 0.5):
    for d in (2, 3, 4):
        for _ in range(400):
            C = rand_rank_r([d, d], 2)
            val = q(-c, C, [d, d]) / np.linalg.norm(C) ** 2
            worst5 = min(worst5, val - (1 - c) * (1 - 2 * c))
            ok5 &= val >= (1 - c) * (1 - 2 * c) - 1e-9
check("C5  theta_2(c) >= (1-c)(1-2c)", ok5, f"(min slack {worst5:+.3e})")

# ------------------------------------------------------------------------- C6
ok6 = True
worst6 = 0.0
for c in (0.1, 0.2, 0.3, 0.5):
    for d in (2, 3):
        for k in (3, 4):
            dims = [d] * k
            sub = dims[:-1]
            for _ in range(120):
                C = rand_rank_r(dims, 2)
                X = ptrace(C, dims, (k - 1,))
                lhs = q(-c, X, sub)
                rhs = ((1 + c) ** (k - 1) + (1 - c) ** (k - 1)) * np.linalg.norm(C) ** 2
                worst6 = max(worst6, lhs / rhs)
                ok6 &= lhs <= rhs + 1e-9
check("C6  q_{k-1}(-c, Tr_k C) <= ((1+c)^{k-1}+(1-c)^{k-1})||C||^2", ok6,
      f"(max lhs/rhs {worst6:.6f})")

# ------------------------------------------------------------------------- C7
print()
print("Resulting lower bounds on theta_k(c) and the constants gamma_k:")


def bound_poly(k):
    """theta_k(c) >= f_k(c) as a callable, from theta_2 >= (1-c)(1-2c) and (C7)."""
    def f(c):
        v = (1 - c) * (1 - 2 * c)
        for j in range(3, k + 1):
            v -= c * ((1 + c) ** (j - 1) + (1 - c) ** (j - 1))
        return v
    return f


def root(f, lo=1e-6, hi=0.5):
    if f(hi) > 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


published = {2: 0.5, 3: 1 / 6, 4: 0.1241036, 5: 0.0981112}
ok7 = True
for k in (2, 3, 4, 5, 6):
    f = bound_poly(k)
    g = root(f)
    pub = published.get(k)
    tag = ""
    if pub is not None:
        tag = f"   published gamma_{k} = {pub:.6f}   {'BETTER' if g > pub + 1e-9 else 'not better'}"
    dmin = int(np.ceil(1 / g)) if g > 0 else None
    dmin = dmin + 1 if dmin is not None and abs(1 / dmin - g) < 1e-12 else dmin
    # smallest d with 1/d < g, i.e. the NPT & k-copy-undistillable window is non-empty
    dmin = next((d for d in range(2, 500) if 1.0 / d < g), None)
    print(f"  k={k}: theta_k(c) >= f_k(c) > 0 for c < {g:.6f};"
          f"  NPT window non-empty from d >= {dmin}{tag}")

# cross-check the k=3 polynomial written in the report
f3 = bound_poly(3)
ok7 &= np.allclose([f3(c) for c in (0.1, 0.2, 0.3)],
                   [1 - 5 * c + 2 * c ** 2 - 2 * c ** 3 for c in (0.1, 0.2, 0.3)])
check("C7  k=3 bound equals 1 - 5c + 2c^2 - 2c^3", ok7)

# ---------------- consistency: the bound must not exceed the measured theta_3
ok8 = True
for c in (0.1, 0.2):
    lo = f3(c)
    up = (1 - 2 * c) * (1 - c) ** 2      # padded upper bound
    ok8 &= lo <= up + 1e-12
check("C8  lower bound <= padded upper bound at k=3", ok8)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed")
