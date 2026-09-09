"""Does SEQUENTIAL Schur sampling on overlapping blocks couple the outcomes?

Disjoint copy-groups do not help: on psi^{(x)N} with N = mk split into m groups, the
joint distribution factorises into the per-group marginals, so measuring a different
tree per group yields only the conjunction of the per-tree conditions.

The one route non-commutation does not immediately kill is SEQUENTIAL measurement of
two overlapping blocks S, T on the SAME k copies.  After measuring S the state leaves
Sym^k(H), so the support theorem does not apply to the composite, and

    r(lam, mu) = rank( P^(T)_mu P^(S)_lam |_{Sym^k(H)} )

need not factor through the two individual supports.  If there are pairs (lam, mu) with
both marginals nonzero but r(lam, mu) = 0, sequential sampling certifies a genuinely
coupled constraint on the two cuts -- exactly the kind CHLW-type obstructions need.

This script measures, for n = 4 and overlapping blocks:
  Q1  the marginal supports of each block alone,
  Q2  the sequential support r(lam, mu) > 0, in both orders,
  Q3  whether  supp_seq  is a PROPER subset of  supp_S x supp_T.

Usage:  uv run python scripts/sequential_support.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mws  # noqa: E402
from schur_weyl import partitions  # noqa: E402


def _rank(vecs: list[np.ndarray], atol: float = 1e-8) -> int:
    M = np.stack(vecs)
    ev = np.clip(np.linalg.eigvalsh(M @ M.conj().T), 0.0, None)
    return int(np.sum(ev > atol**2))


def analyse(dims: tuple[int, ...], k: int, S: tuple[int, ...], T: tuple[int, ...],
            n_states: int = 40, seed: int = 31) -> None:
    n = len(dims)
    rng = np.random.default_rng(seed)
    copies = [mws.copy_tensor(mws.random_pure_state(dims, rng), k)
              for _ in range(n_states)]
    dS = int(np.prod([dims[i] for i in S]))
    dT = int(np.prod([dims[i] for i in T]))
    lamS = list(partitions(k, max_height=min(dS, k)))
    lamT = list(partitions(k, max_height=min(dT, k)))

    print(f"  dims={dims} k={k}  S={S} T={T}  "
          f"(overlap={tuple(sorted(set(S) & set(T)))})")

    # Q1: marginal supports
    suppS = {lam for lam in lamS
             if _rank([mws.apply_isotypic(C, lam, S, n, k).reshape(-1) for C in copies])}
    suppT = {mu for mu in lamT
             if _rank([mws.apply_isotypic(C, mu, T, n, k).reshape(-1) for C in copies])}
    print(f"    marginal support S: {sorted(suppS)}")
    print(f"    marginal support T: {sorted(suppT)}")

    # Q2/Q3: sequential support, both orders
    for first, second, fb, sb, label in (
        (S, T, "S", "T", "S then T"),
        (T, S, "T", "S", "T then S"),
    ):
        lam_first = lamS if first is S else lamT
        lam_second = lamT if second is T else lamS
        supp_seq = set()
        for lam in lam_first:
            for mu in lam_second:
                vecs = [
                    mws.apply_isotypic(
                        mws.apply_isotypic(C, lam, first, n, k), mu, second, n, k
                    ).reshape(-1)
                    for C in copies
                ]
                if _rank(vecs):
                    supp_seq.add((lam, mu))
        prod = {(lam, mu)
                for lam in (suppS if first is S else suppT)
                for mu in (suppT if second is T else suppS)}
        killed = sorted(prod - supp_seq)
        extra = sorted(supp_seq - prod)
        print(f"    [{label}] |supp_seq|={len(supp_seq)}  "
              f"|supp_{fb} x supp_{sb}|={len(prod)}  "
              f"killed={len(killed)}  extra={len(extra)}")
        if killed:
            print(f"      COUPLED pairs (each marginal fine, composite zero): {killed}")


def main() -> int:
    print("=" * 78)
    print("Sequential sampling on overlapping blocks: is the support coupled?")
    print("=" * 78)
    # AB vs AC  (the CHLW pattern), and AB vs BC
    for dims, k, S, T in [
        ((2, 2, 2, 2), 3, (0, 1), (0, 2)),
        ((2, 2, 2, 2), 3, (0, 1), (1, 2)),
        ((2, 2, 2, 2), 4, (0, 1), (0, 2)),
        ((3, 2, 2, 2), 3, (0, 1), (0, 2)),
        ((2, 2, 2, 2, 2), 3, (0, 1), (0, 2)),
    ]:
        analyse(dims, k, S, T)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
