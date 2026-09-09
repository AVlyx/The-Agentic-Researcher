"""Which families of subset-isotypic projectors are jointly measurable?

For a block B subset [n], P^{(B)}_lambda is the image of the central element
z_lambda = (f^lambda/k!) sum_sigma chi^lambda(sigma) sigma  in  Z(C[S_k])
under the algebra homomorphism

    Delta_B : C[S_k] -> C[S_k]^{(x)n},   sigma |-> (x)_i (sigma if i in B else e).

CLAIM (laminar commutation).  [P^{(S)}_lambda, P^{(T)}_mu] = 0 whenever S subset T,
T subset S, or S cap T = empty.  For genuinely overlapping S, T (all three of
S cap T, S \\ T, T \\ S nonempty) they generically do NOT commute.

Proof of the positive half: if S subset T then, for each fixed sigma in the support
of z_mu, the S-coordinates contribute Delta_S(z_lambda sigma) vs Delta_S(sigma z_lambda),
equal because z_lambda is central and Delta_S is a homomorphism; the T \\ S coordinates
are untouched.  Disjoint blocks act on disjoint tensor factors.

This script checks the claim EXACTLY (integer arithmetic in the group algebra, which
is representation-independent and so covers all local dimensions at once), and then
confirms the negative half operationally on a Hilbert space where the S_k action is
faithful (d >= k).

Usage:  uv run python scripts/verify_commutation.py
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from math import factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mws  # noqa: E402
from schur_weyl import dim_specht, partitions  # noqa: E402
from schur_weyl.character import character  # noqa: E402
from schur_weyl.symmetric_group import (  # noqa: E402
    permutation_compose,
    permutation_cycle_type,
    permutation_identity,
)

# ---------------------------------------------------------------------------
# exact arithmetic in C[S_k]^{(x)n}
# ---------------------------------------------------------------------------
# An element is a dict {(sigma_1,...,sigma_n): int coefficient}.


def delta(lam: tuple[int, ...], block: tuple[int, ...], n: int, k: int) -> dict:
    """Delta_B(z_lambda), scaled by k!/f^lambda so all coefficients are integers.

    The scaling is a nonzero constant and does not affect commutators.
    """
    e = permutation_identity(k)
    out: dict = {}
    for sigma in itertools.permutations(range(k)):
        chi = character(lam, permutation_cycle_type(sigma))
        if chi == 0:
            continue
        key = tuple(sigma if i in block else e for i in range(n))
        out[key] = out.get(key, 0) + chi
    return out


def mul(a: dict, b: dict, n: int) -> dict:
    out: dict = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            key = tuple(permutation_compose(ka[i], kb[i]) for i in range(n))
            out[key] += va * vb
    return {k: v for k, v in out.items() if v != 0}


def commutes(a: dict, b: dict, n: int) -> bool:
    ab, ba = mul(a, b, n), mul(b, a, n)
    return ab == ba


def relation(S: frozenset, T: frozenset) -> str:
    if S <= T or T <= S:
        return "nested"
    if not (S & T):
        return "disjoint"
    return "overlapping"


# ---------------------------------------------------------------------------


def check_group_algebra(n: int, k: int) -> tuple[int, dict]:
    """Exact commutator check over all block pairs and all partition pairs."""
    lams = list(partitions(k))
    subsets = [
        frozenset(s)
        for r in range(1, n + 1)
        for s in itertools.combinations(range(n), r)
    ]
    stats: dict[str, dict[str, int]] = {
        r: {"commute": 0, "fail": 0} for r in ("nested", "disjoint", "overlapping")
    }
    violations = 0
    cache: dict = {}
    for S, T in itertools.combinations_with_replacement(subsets, 2):
        rel = relation(S, T)
        for lam, mu in itertools.product(lams, repeat=2):
            kaS, kaT = (tuple(sorted(S)), lam), (tuple(sorted(T)), mu)
            for key in (kaS, kaT):
                if key not in cache:
                    cache[key] = delta(key[1], key[0], n, k)
            ok = commutes(cache[kaS], cache[kaT], n)
            stats[rel]["commute" if ok else "fail"] += 1
            # the claim: nested/disjoint MUST commute
            if rel in ("nested", "disjoint") and not ok:
                violations += 1
                print(f"    VIOLATION {sorted(S)} {lam} vs {sorted(T)} {mu} ({rel})")
    return violations, stats


def check_operational(k: int = 3, d: int = 3) -> None:
    """Confirm the negative half on a Hilbert space where C[S_k] acts faithfully.

    n = 3, S = {0,1}, T = {1,2}: show P^S_lam P^T_mu psi != P^T_mu P^S_lam psi.
    """
    n = 3
    dims = (d, d, d)
    rng = np.random.default_rng(7)
    psi = mws.random_pure_state(dims, rng)
    T0 = mws.copy_tensor(psi, k)
    S, T = (0, 1), (1, 2)
    print(f"  n=3, dims={dims}, k={k}, S={S}, T={T} (overlapping)")
    worst = 0.0
    for lam, mu in itertools.product(partitions(k), repeat=2):
        a = mws.apply_isotypic(mws.apply_isotypic(T0, mu, T, n, k), lam, S, n, k)
        b = mws.apply_isotypic(mws.apply_isotypic(T0, lam, S, n, k), mu, T, n, k)
        diff = float(np.abs(a - b).max())
        scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-300)
        if diff > 1e-10:
            worst = max(worst, diff / scale)
            print(
                f"    [P^S_{lam}, P^T_{mu}] psi != 0   "
                f"relative deviation = {diff / scale:.3e}"
            )
    print(f"  max relative deviation over all (lam, mu): {worst:.3e}")

    # and the nested/disjoint cases must vanish on the same state
    print(f"  control: nested S={S}, T=(0,1,2) and disjoint S=(0,), T=(1,)")
    for S2, T2 in [((0, 1), (0, 1, 2)), ((0,), (1,)), ((0,), (0, 1))]:
        worst2 = 0.0
        for lam, mu in itertools.product(partitions(k), repeat=2):
            a = mws.apply_isotypic(mws.apply_isotypic(T0, mu, T2, n, k), lam, S2, n, k)
            b = mws.apply_isotypic(mws.apply_isotypic(T0, lam, S2, n, k), mu, T2, n, k)
            worst2 = max(worst2, float(np.abs(a - b).max()))
        print(f"    S={S2}, T={T2}: max |[.,.]psi| = {worst2:.3e}")


def check_complement_rule(dims: tuple[int, ...], k: int, seed: int = 3) -> None:
    """For a PURE state, lambda^{(S)} = lambda^{(S^c)} with probability 1."""
    n = len(dims)
    rng = np.random.default_rng(seed)
    psi = mws.random_pure_state(dims, rng)
    T0 = mws.copy_tensor(psi, k)
    print(f"  dims={dims}, k={k}")
    for r in range(1, n // 2 + 1):
        for S in itertools.combinations(range(n), r):
            Sc = tuple(i for i in range(n) if i not in S)
            worst = 0.0
            for lam in partitions(k):
                a = mws.apply_isotypic(T0, lam, S, n, k)
                b = mws.apply_isotypic(T0, lam, Sc, n, k)
                worst = max(worst, float(np.abs(a - b).max()))
            print(f"    S={S} vs S^c={Sc}: max |P^S_lam psi - P^Sc_lam psi| = {worst:.3e}")


def main() -> int:
    print("=" * 78)
    print("C1  exact commutator check in C[S_k]^{(x)n}")
    print("=" * 78)
    total_violations = 0
    for n, k in [(3, 3), (4, 3), (3, 4), (4, 4), (5, 3)]:
        v, stats = check_group_algebra(n, k)
        total_violations += v
        line = "  ".join(
            f"{rel}: {s['commute']} commute / {s['fail']} fail"
            for rel, s in stats.items()
        )
        print(f"  n={n} k={k}   {line}")
    print(f"  laminar-commutation violations: {total_violations}  "
          f"{'PASS' if total_violations == 0 else 'FAIL'}")

    print()
    print("=" * 78)
    print("C2  operational check of the negative half (faithful action, d >= k)")
    print("=" * 78)
    check_operational(k=3, d=3)

    print()
    print("=" * 78)
    print("C3  complement rule  lambda^{(S)} = lambda^{(S^c)} on pure states")
    print("=" * 78)
    check_complement_rule((2, 2, 2), 3)
    check_complement_rule((2, 3, 2, 2), 3)

    return 1 if total_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
