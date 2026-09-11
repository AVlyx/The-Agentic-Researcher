"""Why the open sector is easy: spread local reductions suppress the odd terms.

q_k(-1/2, C) = sum_S (-1/2)^{|S|} ||Tr_S C||^2.  The only negative contributions come from
the ODD subsets S.  The following lemma says those are small exactly when the local
reductions are spread out, which is the regime E008 probes.

Lemma (spread bound).  For unit vectors a, b in H_i (x) H_rest,
    ||Tr_i(|a><b|)||_HS  <=  sqrt( lambda_max(rho_i^b) ) ,
and symmetrically with rho_i^a, where rho_i^x is the reduction of |x><x| to site i.

Proof.  Write a = sum_m |m>_i |a^{(m)}>, b = sum_m |m>_i |b^{(m)}>, so
Tr_i(|a><b|) = sum_m |a^{(m)}><b^{(m)}| = A B^dag with A, B the matrices of columns
a^{(m)}, b^{(m)}.  Then ||A B^dag||_HS <= ||A||_HS ||B||_op = ||a|| sigma_max(B), and
sigma_max(B)^2 = lambda_max(B^dag B) = lambda_max(rho_i^b) since B^dag B is the Gram matrix
(<b^{(m)}|b^{(m')}>), i.e. the reduction to site i.

Consequence for rank-two C = sum_a sigma_a |a_a><b_a| with sum sigma_a^2 = 1:
    ||Tr_i C||^2 <= (sum_a sigma_a sqrt(L_i))^2 <= 2 L_i,   L_i := max_a lambda_max(rho_i^{b_a}).
A maximally mixed local reduction has L_i = 1/d, so the odd terms are suppressed by 1/d.
This is the mechanism behind E008: on the full-local-support sector the negative terms
shrink and q_3 acquires a margin.

How far it goes on its own: dropping the (non-negative) even terms and using |Tr C|^2 <= 2,
    q_3 >= 1 - (1/2) sum_i ||Tr_i C||^2 - (1/8)|Tr C|^2 >= 3/4 - 3L,  L := max_i L_i,
which is positive only for L < 1/4, while the most spread case in d=3 is L = 1/3.  So the
mechanism is real but this crude form falls short by a constant factor; closing that gap
(by keeping the even terms) is the concrete remaining task.

Run:  python scripts/verify_spread_lemma.py
"""
from __future__ import annotations

import sys, os
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

rng = np.random.default_rng(11235)
FAILS = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def ptrace(C, dims, S):
    k = len(dims)
    T = C.reshape(tuple(dims) + tuple(dims))
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i)
        k -= 1
        dims = dims[:i] + dims[i + 1:]
    return T.reshape(int(np.prod(dims)) if dims else 1, -1)


# --------------------------------------------------------------- the lemma
worst = 0.0
for k in (2, 3):
    for d in (2, 3, 4):
        dims = [d] * k
        D = d ** k
        for _ in range(400):
            a = rng.normal(size=D) + 1j * rng.normal(size=D)
            b = rng.normal(size=D) + 1j * rng.normal(size=D)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            C = np.outer(a, b.conj())
            for i in range(k):
                lhs = np.linalg.norm(ptrace(C, dims, (i,)))
                B = b.reshape([d if j == i else 1 for j in range(k)] + [-1]) \
                    if False else None
                # rho_i^b
                bt = b.reshape((d,) * k)
                ax = [i] + [j for j in range(k) if j != i]
                M = np.transpose(bt, ax).reshape(d, -1)
                rho_i = M @ M.conj().T
                rhs = np.sqrt(np.linalg.eigvalsh(rho_i).max())
                worst = max(worst, lhs / rhs)
check("||Tr_i(|a><b|)||_HS <= sqrt(lambda_max(rho_i^b))", worst <= 1 + 1e-9,
      f"(max lhs/rhs {worst:.6f})")

# ------------------------------------------- consequence for rank-two C
worst2 = 0.0
for k in (3,):
    for d in (2, 3, 4):
        dims = [d] * k
        D = d ** k
        for _ in range(400):
            A = rng.normal(size=(D, 2)) + 1j * rng.normal(size=(D, 2))
            B = rng.normal(size=(D, 2)) + 1j * rng.normal(size=(D, 2))
            C = A @ B.conj().T
            n2 = np.linalg.norm(C) ** 2
            # singular values and right factors
            U, s, Vh = np.linalg.svd(C)
            s = s[:2]
            for i in range(k):
                Ls = []
                for r in range(2):
                    bt = Vh[r].conj().reshape((d,) * k)
                    ax = [i] + [j for j in range(k) if j != i]
                    M = np.transpose(bt, ax).reshape(d, -1)
                    Ls.append(np.linalg.eigvalsh(M @ M.conj().T).max())
                L = max(Ls)
                lhs = np.linalg.norm(ptrace(C, dims, (i,))) ** 2 / n2
                rhs = 2 * L
                worst2 = max(worst2, lhs / rhs)
check("||Tr_i C||^2 <= 2 L_i ||C||^2 for rank-two C", worst2 <= 1 + 1e-9,
      f"(max lhs/rhs {worst2:.6f})")

# ---------------------- how much margin the crude form gives, vs. what is needed
print()
print("Crude bound  q_3 >= 3/4 - 3L  (L = max_i L_i); most spread case is L = 1/d:")
for d in (2, 3, 4, 6, 12):
    L = 1.0 / d
    print(f"  d={d:2d}:  L=1/d={L:.4f}   3/4 - 3L = {0.75 - 3 * L:+.4f}"
          f"   {'positive' if 0.75 - 3 * L > 0 else 'NOT positive -- gap'}")
print("So the mechanism alone settles the maximally-spread case only for d >= 5;")
print("E008 measures the true margin at d=3,4 and finds it positive throughout.")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed")
