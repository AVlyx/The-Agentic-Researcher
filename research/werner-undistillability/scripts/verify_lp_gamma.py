"""A linear-programming lower bound on theta_k(c), giving much larger constants gamma_k.

Key lemma (FHPV with a spectator).  Let C have rank <= 2 on H_A (x) H_B (x) H_E, with
A, B disjoint and non-empty and E an arbitrary set of spectator sites.  Then

    ||Tr_A C||^2 + ||Tr_B C||^2  <=  2||C||^2 + (1/2)||Tr_{A u B} C||^2 .          (L)

Proof.  Write C in blocks over E: C^{(mm')} = (I (x) <m|) C (I (x) |m'>).  Each block is a
compression of C, so rank C^{(mm')} <= 2, and the Fraser-Huber-Pozsgay-Vona inequality
(arXiv:2607.24309) with r = 2 applies to each block on H_A (x) H_B:
    ||Tr_A C^{(mm')}||^2 + ||Tr_B C^{(mm')}||^2 <= 2||C^{(mm')}||^2 + (1/2)|Tr C^{(mm')}|^2 .
Summing over m, m' and using
    sum ||Tr_A C^{(mm')}||^2 = ||Tr_A C||^2,   sum ||C^{(mm')}||^2 = ||C||^2,
    sum |Tr C^{(mm')}|^2 = ||Tr_{A u B} C||^2
gives (L).  The case E = empty is FHPV itself.

Consequence.  Put x_S = ||Tr_S C||^2 / ||C||^2 for S subset [k], so x_empty = 1 and
q_k(-c,C)/||C||^2 = sum_S (-c)^{|S|} x_S.  Then

    theta_k(c) >= min { sum_S (-c)^{|S|} x_S :  0 <= x_S <= 2,  x_empty = 1,
                        x_A + x_B <= 2 + x_{A u B}/2  for all disjoint non-empty A,B },

a linear program in 2^k variables.  It is a relaxation (every rank-<=2 C gives a feasible
point), hence a valid dimension-free lower bound, and its optimal value is certified by LP
duality.

Run:  python scripts/verify_lp_gamma.py
"""
from __future__ import annotations

import sys, os, json
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

rng = np.random.default_rng(31415)
FAILS = []
HERE = os.path.join(os.path.dirname(__file__), "..")


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------- numerical check of (L)
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


worst = 0.0
for k in (2, 3, 4):
    for d in (2, 3):
        dims = [d] * k
        sites = list(range(k))
        for _ in range(120):
            C = rand_rank_r(dims, 2)
            n2 = np.linalg.norm(C) ** 2
            for ra in range(1, k):
                for A in combinations(sites, ra):
                    rest = [s for s in sites if s not in A]
                    for rb in range(1, len(rest) + 1):
                        for B in combinations(rest, rb):
                            lhs = (np.linalg.norm(ptrace(C, dims, A)) ** 2
                                   + np.linalg.norm(ptrace(C, dims, B)) ** 2)
                            rhs = (2 * n2
                                   + 0.5 * np.linalg.norm(
                                       ptrace(C, dims, tuple(sorted(A + B)))) ** 2)
                            worst = max(worst, lhs / rhs)
check("(L) FHPV with a spectator, all disjoint A,B", worst <= 1 + 1e-9,
      f"(max lhs/rhs {worst:.6f})")

# also re-check the plain rank bound x_S <= 2
worst_rank = 0.0
for k in (2, 3, 4):
    for d in (2, 3):
        dims = [d] * k
        for _ in range(80):
            C = rand_rank_r(dims, 2)
            n2 = np.linalg.norm(C) ** 2
            for r in range(1, k + 1):
                for S in combinations(range(k), r):
                    worst_rank = max(worst_rank,
                                     np.linalg.norm(ptrace(C, dims, S)) ** 2 / (2 * n2))
check("x_S <= 2 (rank lemma)", worst_rank <= 1 + 1e-9, f"(max {worst_rank:.6f})")


# ------------------------------------------------------------------- the LP
def lp_bound(k, c):
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
                    row = np.zeros(n)
                    row[idx[A]] += 1
                    row[idx[B]] += 1
                    row[idx[tuple(sorted(A + B))]] -= 0.5
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
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


print()
print("LP lower bounds on theta_k(c) and the resulting constants gamma_k:")
published = {2: 0.5, 3: 1 / 6, 4: 0.1241036, 5: 0.0981112}
recursion = {2: 0.5, 3: 0.214451, 4: 0.145384, 5: 0.110812}
table = []
for k in (2, 3, 4, 5, 6):
    g = root(lambda c: lp_bound(k, c))
    dmin = next((d for d in range(2, 500) if 1.0 / d < g), None)
    pub = published.get(k)
    rec = recursion.get(k)
    table.append(dict(k=k, gamma_lp=g, gamma_recursion=rec, gamma_published=pub,
                      d_min=dmin))
    extra = ""
    if pub:
        extra += f"  published {pub:.6f}"
    if rec:
        extra += f"  recursion {rec:.6f}"
    print(f"  k={k}: gamma_k >= {g:.6f}   NPT window non-empty from d >= {dmin}{extra}")

# ------------------------------------- sanity: LP bound must not exceed the truth
ok = True
for k in (2, 3):
    for c in (0.05, 0.15, 0.25, 0.35, 0.45):
        lo = lp_bound(k, c)
        up = (1 - 2 * c) * (1 - c) ** (k - 1)      # padded upper bound on theta_k
        ok &= lo <= up + 1e-9
check("LP bound <= padded upper bound on theta_k", ok)

# ------------------------------- sanity: LP bound <= q_k of random rank-2 samples
ok2 = True
tight = np.inf
for k in (2, 3):
    for d in (2, 3, 4):
        dims = [d] * k
        for c in (0.1, 0.25, 0.3):
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

with open(os.path.join(HERE, "data", "lp_gamma.json"), "w") as f:
    json.dump(table, f, indent=1)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed; wrote data/lp_gamma.json")
