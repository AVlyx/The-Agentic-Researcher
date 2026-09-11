"""LP bound on theta_k(c) using BOTH spectator-lifted partial-trace inequalities.

Bharti-Gajjala-Haug (arXiv:2607.24479) prove, for rank-<=2 C and a COMPLEMENTARY
bipartition J, J^c of all k systems (their Eqs. 81-82, with N = ||C||^2, T = |Tr C|^2,
A_J = ||Tr_J C||^2):

    (sum)  A_J + A_{J^c} <= 2N + T/2                                          [their (81)]
    (diff) |A_J - A_{J^c}| <= 2N - T/2                                        [their (82)]

Both are stated only for complementary pairs.  Each lifts to arbitrary disjoint A, B with
the remaining systems E left untouched, by writing C in blocks over E (each block is a
compression of C, hence still rank <= 2), applying the bipartite inequality to every block
on H_A (x) H_B, and summing, using
    sum_{m,m'} ||Tr_A C^{(mm')}||^2 = ||Tr_A C||^2,  sum ||C^{(mm')}||^2 = ||C||^2,
    sum |Tr C^{(mm')}|^2 = ||Tr_{A u B} C||^2 .
For the difference version, additionally |sum_i z_i| <= sum_i |z_i|.  This gives

    (S)  x_A + x_B <= 2 + x_{A u B}/2,
    (D)  |x_A - x_B| <= 2 - x_{A u B}/2,        for all disjoint non-empty A, B,

with x_S = ||Tr_S C||^2/||C||^2 and x_empty = 1.  Minimising sum_S (-c)^{|S|} x_S over
these constraints is a linear program and lower-bounds theta_k(c), dimension-free.

verify_lp_gamma.py used (S) only.  This script adds (D).

Run:  python scripts/verify_lp_gamma2.py
"""
from __future__ import annotations

import sys, os, json
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

rng = np.random.default_rng(161803)
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


# ------------------------------- numerical check of the spectator (S) and (D)
wS = wD = 0.0
for k in (2, 3, 4):
    for d in (2, 3):
        dims = [d] * k
        for _ in range(150):
            C = rand_rank_r(dims, 2)
            n2 = np.linalg.norm(C) ** 2
            for ra in range(1, k):
                for A in combinations(range(k), ra):
                    rest = [s for s in range(k) if s not in A]
                    for rb in range(1, len(rest) + 1):
                        for B in combinations(rest, rb):
                            xA = np.linalg.norm(ptrace(C, dims, A)) ** 2 / n2
                            xB = np.linalg.norm(ptrace(C, dims, B)) ** 2 / n2
                            xAB = np.linalg.norm(
                                ptrace(C, dims, tuple(sorted(A + B)))) ** 2 / n2
                            wS = max(wS, (xA + xB) / (2 + xAB / 2))
                            rhsD = 2 - xAB / 2
                            if rhsD > 1e-12:
                                wD = max(wD, abs(xA - xB) / rhsD)
check("(S) x_A + x_B <= 2 + x_{AuB}/2  (spectator sum)", wS <= 1 + 1e-9,
      f"(max ratio {wS:.6f})")
check("(D) |x_A - x_B| <= 2 - x_{AuB}/2  (spectator difference)", wD <= 1 + 1e-9,
      f"(max ratio {wD:.6f})")


# --------------------------------------------------------------------- the LP
def lp_bound(k, c, use_diff=True):
    subs = []
    for r in range(k + 1):
        subs.extend(combinations(range(k), r))
    idx = {S: i for i, S in enumerate(subs)}
    n = len(subs)
    obj = np.array([(-c) ** len(S) for S in subs])
    A_ub, b_ub = [], []
    for ra in range(1, k):
        for A in combinations(range(k), ra):
            rest = [s for s in range(k) if s not in A]
            for rb in range(1, len(rest) + 1):
                for B in combinations(rest, rb):
                    AB = tuple(sorted(A + B))
                    row = np.zeros(n)                      # (S)
                    row[idx[A]] += 1
                    row[idx[B]] += 1
                    row[idx[AB]] -= 0.5
                    A_ub.append(row)
                    b_ub.append(2.0)
                    if use_diff:                            # (D), both signs
                        for sgn in (+1, -1):
                            row = np.zeros(n)
                            row[idx[A]] += sgn
                            row[idx[B]] -= sgn
                            row[idx[AB]] += 0.5
                            A_ub.append(row)
                            b_ub.append(2.0)
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
print("gamma_k from the LP, with and without the difference family (D):")
print(f"{'k':>2} {'(S) only':>11} {'(S)+(D)':>11} {'published':>11}  NPT window from")
published = {2: 0.5, 3: 1 / 6, 4: 0.1241036, 5: 0.0981112}
table = []
for k in (2, 3, 4, 5, 6, 7):
    g_s = root(lambda c: lp_bound(k, c, use_diff=False))
    g_sd = root(lambda c: lp_bound(k, c, use_diff=True))
    dmin = next((d for d in range(2, 500) if 1.0 / d < g_sd), None)
    pub = published.get(k)
    table.append(dict(k=k, gamma_S=g_s, gamma_SD=g_sd, gamma_published=pub, d_min=dmin))
    ps = f"{pub:.6f}" if pub else "---"
    print(f"{k:>2} {g_s:>11.6f} {g_sd:>11.6f} {ps:>11}  d >= {dmin}")

# ------------------------------------------------------ soundness sanity checks
ok = True
for k in (2, 3, 4):
    for c in (0.05, 0.15, 0.25, 0.35, 0.45):
        ok &= lp_bound(k, c) <= (1 - 2 * c) * (1 - c) ** (k - 1) + 1e-9
check("LP bound <= padded upper bound on theta_k", ok)

ok2 = True
tight = np.inf
for k in (2, 3):
    for d in (2, 3, 4):
        dims = [d] * k
        for c in (0.1, 0.25, 0.35):
            lo = lp_bound(k, c)
            for _ in range(200):
                C = rand_rank_r(dims, 2)
                tot = 0.0
                for r in range(k + 1):
                    for S in combinations(range(k), r):
                        tot += ((-c) ** r) * np.linalg.norm(ptrace(C, dims, S)) ** 2
                val = tot / np.linalg.norm(C) ** 2
                tight = min(tight, val - lo)
                ok2 &= val >= lo - 1e-9
check("LP bound <= q_k(-c,C)/||C||^2 for random rank-2 C", ok2,
      f"(min slack {tight:+.4f})")

with open(os.path.join(HERE, "data", "lp_gamma2.json"), "w") as f:
    json.dump(table, f, indent=1)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed; wrote data/lp_gamma2.json")
