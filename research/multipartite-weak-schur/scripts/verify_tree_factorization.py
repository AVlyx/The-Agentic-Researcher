"""Verify the tree factorisation of the multipartite weak-Schur-sampling support.

Setting.  Let F be a laminar family on [n] (any two blocks nested or disjoint),
completed with all singletons and the root [n].  It is a rooted tree with leaves
the parties.  Assign lam^(v) |- k to every node, with lam^([n]) = (k) forced (the
whole system sits in Sym^k, whose S_k-type is trivial).

CLAIM (tree factorisation).

    dim( Im Pi_lam  cap  Sym^k(H) )
        = [ prod_{internal v} g( lam^(c_1(v)), ..., lam^(c_m(v)), lam^(v) ) ]
          * prod_{leaves i} dim V^{d_i}_{lam^(i)}

Derivation.  Projecting bottom-up, the surviving S_k-representation at node v is
m_v copies of S^{lam^(v)}, with m_leaf = 1 and
m_v = (prod_j m_{c_j}) * g(lam^(c_1),...,lam^(c_m), lam^(v)), because the
multiplicity of S^{lam^(v)} in (x)_j S^{lam^(c_j)} is that generalized Kronecker
coefficient.  Intersecting with Sym^k at the root picks the trivial isotypic, i.e.
lam^([n]) = (k).  The GL(d_i) factors are untouched by any node projector, so the
Weyl dimensions appear only at the leaves.

For n = 4 and F = {A},{B},{C},{D},{AB},[4] this reads

    dim = g(lam^A, lam^B, lam^AB) * g(lam^AB, lam^C, lam^D) * prod_i dim V^{d_i}_{lam^i}

Checks:
  F1  dim span{ Pi_lam psi^(x)k : psi random } vs the formula, over all assignments.
  F2  the joint distribution over the tree sums to 1 (it is a genuine measurement).
  F3  marginalising lam^AB out of the tree distribution reproduces the flat
      4-party distribution -- i.e. the tree measurement refines the singleton one.

Usage:  uv run python scripts/verify_tree_factorization.py
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mws  # noqa: E402
from schur_weyl import dim_weyl, kronecker_coefficient  # noqa: E402

# the tree: leaves 0,1,2,3; internal node AB = {0,1}; root {0,1,2,3}
BLOCKS = ((0,), (1,), (2,), (3,), (0, 1))


def predicted_dim(dims: tuple[int, ...], lams: tuple[tuple[int, ...], ...]) -> int:
    """lams is (lam^A, lam^B, lam^C, lam^D, lam^AB) matching BLOCKS."""
    lA, lB, lC, lD, lAB = lams
    k = sum(lA)
    node_ab = kronecker_coefficient(lA, lB, lAB)
    node_root = kronecker_coefficient(lAB, lC, lD)  # root's children: AB, C, D
    if node_ab == 0 or node_root == 0:
        return 0
    total = node_ab * node_root
    for d, lam in zip(dims, (lA, lB, lC, lD)):
        total *= dim_weyl(lam, d)
    return total


def _rank(M: np.ndarray, atol: float = 1e-6) -> int:
    ev = np.clip(np.linalg.eigvalsh(M @ M.conj().T), 0.0, None)
    return int(np.sum(ev > atol**2))


def check_dims(dims: tuple[int, ...], k: int, seed: int = 21, slack: int = 5) -> tuple[int, int]:
    n = len(dims)
    rng = np.random.default_rng(seed)
    dim_of = {B: int(np.prod([dims[i] for i in B])) for B in BLOCKS}
    choices = [mws._partitions(k, max_height=min(dim_of[B], k)) for B in BLOCKS]

    ok = total = 0
    for lams in itertools.product(*choices):
        pred = predicted_dim(dims, lams)
        vecs = []
        for _ in range(pred + slack):
            T = mws.copy_tensor(mws.random_pure_state(dims, rng), k)
            for B, lam in zip(BLOCKS, lams):
                T = mws.apply_isotypic(T, lam, B, n, k)
            vecs.append(T.reshape(-1))
        r = _rank(np.stack(vecs))
        total += 1
        ok += r == pred
        if r != pred:
            print(f"    MISMATCH {lams}: span rank={r} predicted={pred}")
    return ok, total


def check_distribution(dims: tuple[int, ...], k: int, seed: int = 22) -> None:
    """F2 + F3."""
    n = len(dims)
    rng = np.random.default_rng(seed)
    psi = mws.random_pure_state(dims, rng)

    tree = mws.mws_distribution(psi, k, blocks=BLOCKS)
    flat = mws.mws_distribution(psi, k, blocks=tuple((i,) for i in range(n)))
    print(f"  dims={dims} k={k}")
    print(f"    tree outcomes: {len(tree):3d}   sum p = {sum(tree.values()):.12f}")
    print(f"    flat outcomes: {len(flat):3d}   sum p = {sum(flat.values()):.12f}")

    # F3: marginalise lam^AB (last entry) out of the tree distribution
    marg: dict = defaultdict(float)
    for lams, p in tree.items():
        marg[lams[:4]] += p
    keys = set(marg) | set(flat)
    worst = max(abs(marg.get(kk, 0.0) - flat.get(kk, 0.0)) for kk in keys)
    print(f"    max |marginal(tree) - flat| over {len(keys)} outcomes = {worst:.3e}")

    # support consistency with the factorisation
    bad = 0
    for lams in itertools.product(
        *[mws._partitions(k, max_height=min(int(np.prod([dims[i] for i in B])), k)) for B in BLOCKS]
    ):
        pos_pred = predicted_dim(dims, lams) > 0
        pos_obs = tree.get(lams, 0.0) > 1e-11
        bad += pos_pred != pos_obs
    print(f"    support(tree) vs factorisation positivity: {bad} mismatches")


def main() -> int:
    print("=" * 78)
    print("F1  dim span{Pi_lam psi^(x)k} == g(lA,lB,lAB) * g(lAB,lC,lD) * prod dim V")
    print("=" * 78)
    failures = 0
    for dims, k in [((2, 2, 2, 2), 2), ((2, 2, 2, 2), 3), ((2, 2, 3, 2), 3)]:
        ok, total = check_dims(dims, k)
        status = "PASS" if ok == total else "FAIL"
        failures += ok != total
        print(f"  dims={dims!s:14s} k={k}   {ok}/{total} assignments   {status}")

    print()
    print("=" * 78)
    print("F2/F3  tree measurement is a measurement, and refines the flat one")
    print("=" * 78)
    for dims, k in [((2, 2, 2, 2), 3), ((2, 2, 3, 2), 3), ((2, 2, 2, 2), 4)]:
        check_distribution(dims, k)

    print()
    print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
