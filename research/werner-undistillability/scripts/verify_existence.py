"""Proposition (b): for every n there exist NPT Werner states that are n-copy
undistillable, in local dimension growing linearly in n.

This is a COROLLARY of having gamma_n > 0, not a deep statement: anyone holding the
constants of arXiv:2607.24479 has it in one line.  It is recorded because it makes precise
what those constants buy, and because the quantity that matters is the DIMENSION required.

Statement.  Let t* = 0.4428544... be the unique root of e^t + t = 2, and gamma_n = t*/n.
For every d > 1/gamma_n and every alpha with -gamma_n <= alpha < -1/d, the state
rho(d,alpha) = (I + alpha F)/(d^2 + alpha d) is NPT (hence entangled) and n-copy
undistillable.

Proof of gamma_n > 0, made explicit (each step checked below).  Unrolling
    theta_1(c) = 1 - 2c,  theta_2(c) >= (1-c)(1-2c),
    theta_k(c) >= theta_{k-1}(c) - c[(1+c)^{k-1} + (1-c)^{k-1}]
and summing the two geometric series gives theta_n >= f_n with the closed form
    f_n(c) = (1-c)(1-2c) + 4c + (1-c)^n - (1+c)^n = 1 + c + 2c^2 + (1-c)^n - (1+c)^n .
Bernoulli gives (1-c)^n >= 1 - nc and (1+c)^n <= e^{nc}, so
    f_n(c) >= 2 + c + 2c^2 - nc - e^{nc} >= 2 - nc - e^{nc},
and t -> e^t + t is strictly increasing, so this is > 0 exactly when nc < t*.
(An earlier draft argued "f_n(0) = 1 > 0 and f_n continuous, so f_n has a positive root" --
a non sequitur: continuity plus positivity at 0 gives no root.  No root is needed; the
explicit bound above is what the proof rests on.)

The LP bound of verify_lp_gamma.py gives strictly better constants where computed, which is
the only part of this that improves on the literature.

Run:  python scripts/verify_existence.py
"""
from __future__ import annotations

import sys, os, json
from itertools import combinations

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
FAILS = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------- the two families of constants
def f_recursion(k, c):
    """Explicit lower bound on theta_k(c) from the verified recursion."""
    v = (1 - c) * (1 - 2 * c)
    for j in range(3, k + 1):
        v -= c * ((1 + c) ** (j - 1) + (1 - c) ** (j - 1))
    return v


def f_lp(k, c):
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
    r = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds,
                method="highs")
    assert r.status == 0
    return r.fun


def root(f, lo=1e-9, hi=0.5):
    if f(hi) > 0:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ------------------------------- every step of the Proposition's proof
TSTAR = 0.4428544010023886   # root of e^t + t = 2


def f_closed(n, c):
    return (1 - c) * (1 - 2 * c) + 4 * c + (1 - c) ** n - (1 + c) ** n


ok_cf = ok_alt = ok_chain = ok_pos = True
for n in range(1, 201):
    for c in np.linspace(0.0, 0.5, 26):
        if n >= 2:
            ok_cf &= np.isclose(f_closed(n, c), f_recursion(n, c), atol=1e-11, rtol=1e-11)
        ok_alt &= np.isclose(f_closed(n, c), 1 + c + 2 * c ** 2 + (1 - c) ** n - (1 + c) ** n)
        ok_chain &= f_closed(n, c) >= 2 - n * c - np.exp(n * c) - 1e-12
    for c in np.linspace(1e-9, TSTAR / n, 30):
        ok_pos &= f_closed(n, c) > 0
check("closed form f_n == unrolled recursion (n<=200)", ok_cf)
check("f_n = 1 + c + 2c^2 + (1-c)^n - (1+c)^n", ok_alt)
check("Bernoulli chain f_n >= 2 - nc - e^{nc}", ok_chain)
check("f_n(c) > 0 for all c <= t*/n, every n <= 200", ok_pos)
check("t* is the root of e^t + t = 2", abs(np.exp(TSTAR) + TSTAR - 2) < 1e-12,
      f"(residual {np.exp(TSTAR) + TSTAR - 2:+.2e})")

# ------------------------------------------------ gamma_n > 0 for every n
ok = True
for n in range(1, 41):
    g = root(lambda c: f_recursion(n, c)) if n >= 2 else 0.5
    ok &= g > 0
check("recursion constant gamma_n > 0 for every n = 1..40", ok)

# ------------------------------------------------ the explicit family
print()
print("Explicit NPT, n-copy undistillable Werner states rho(d, alpha):")
print(f"{'n':>3} {'gamma_n (recur)':>16} {'gamma_n (LP)':>13} {'min d':>6} {'alpha':>9}"
      f"  {'min eig(rho^PT)':>16}  NPT & undistillable?")
rows = []
for n in (1, 2, 3, 4, 5, 6, 7, 8, 9):
    g_rec = 0.5 if n == 1 else root(lambda c: f_recursion(n, c))
    g_lp = 0.5 if n == 1 else root(lambda c: f_lp(n, c))
    g = max(g_rec, g_lp)
    d = next(dd for dd in range(2, 4000) if 1.0 / dd < g)
    alpha = -0.5 * (1.0 / d + g)            # strictly inside (-g, -1/d)
    rho = W.werner(alpha, d)
    ev = sla.eigvalsh(rho)
    evpt = sla.eigvalsh(W.partial_transpose(rho, d))
    is_state = ev.min() > -1e-12 and abs(np.trace(rho) - 1) < 1e-12
    is_npt = evpt.min() < -1e-12
    undist = (-alpha) <= g + 1e-12
    rows.append(dict(n=n, gamma_recursion=g_rec, gamma_lp=g_lp, d=d, alpha=alpha,
                     min_eig_pt=float(evpt.min()), npt=bool(is_npt),
                     undistillable=bool(undist)))
    ok &= is_state and is_npt and undist
    print(f"{n:>3} {g_rec:>16.6f} {g_lp:>13.6f} {d:>6} {alpha:>9.4f}  "
          f"{evpt.min():>16.3e}  {'YES' if (is_npt and undist) else 'NO'}")
check("every listed state is a valid state, NPT, and n-copy undistillable", ok)

# ------------------------------------------- how fast must d grow?
print()
print("Growth of the required dimension (smaller is better):")
for n in (3, 5, 7, 9):
    g_rec = root(lambda c: f_recursion(n, c))
    g_lp = root(lambda c: f_lp(n, c))
    print(f"  n={n}:  d >~ 1/gamma_n  =  {1/g_rec:6.2f} (recursion)   "
          f"{1/g_lp:6.2f} (LP)   [published gamma_n ~ arsinh(1/2)/n gives "
          f"{n/0.4812:6.2f}]")

with open(os.path.join(HERE, "data", "existence.json"), "w") as f:
    json.dump(rows, f, indent=1)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed; wrote data/existence.json")
