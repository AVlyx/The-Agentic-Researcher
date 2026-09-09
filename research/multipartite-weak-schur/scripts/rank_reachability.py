"""Which Schmidt-rank profiles are reachable, via Kronecker positivity?

Corollary of the support theorem.  Work in local dimensions d_i = r_i.  Then
ell(lam^(i)) <= rank rho_i <= d_i = r_i, so

    a rank profile r = (r_1,...,r_n) is achievable by a pure state
      <=>  there exist k and partitions lam^(i) |- k with ell(lam^(i)) = r_i
           and g(lam^(1),...,lam^(n)) > 0.

(=>)  weak Schur sampling concentrates lam^(i)/k on spec rho_i, which has exactly
      r_i nonzero entries, so for large k some outcome has ell(lam^(i)) = r_i for
      every i simultaneously; that outcome has g > 0 by the support theorem.
(<=)  the support theorem gives a state psi in (x)_i C^{r_i} with p_psi(lam) > 0,
      hence rank rho_i >= ell(lam^(i)) = r_i, and <= r_i by construction.

So enumerating Kronecker positivity by ROW LENGTHS decides rank achievability.
This script does that for:

  R1  n = 3 singletons: g(lam,mu,nu) > 0            vs the polygon inequalities
  R2  n = 4 singletons: g(lam^A,..,lam^D) > 0       vs the polygon inequalities
  R3  n = 4 on the tree {A},{B},{AB},{C},{D}: a SINGLE lam^AB must satisfy
      g(lam^A,lam^B,lam^AB) > 0 AND g(lam^AB,lam^C,lam^D) > 0 simultaneously.
      Does the constraint decouple into the two node-local polygon conditions,
      or does sharing lam^AB create a genuinely new obstruction?

Usage:  uv run python scripts/rank_reachability.py [KMAX]
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schur_weyl import kronecker_coefficient, partitions  # noqa: E402


def by_length(k: int) -> dict[int, list[tuple[int, ...]]]:
    out: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for lam in partitions(k):
        out[len(lam)].append(lam)
    return out


def polygon(r: tuple[int, ...]) -> bool:
    """r_i <= prod_{j != i} r_j for all i  (Carlini-Kleppe admissibility)."""
    total = 1
    for x in r:
        total *= x
    return all(ri * ri <= total for ri in r)


# ---------------------------------------------------------------------------
# R1 / R2: flat n-party support, indexed by row lengths
# ---------------------------------------------------------------------------


def reachable_flat(n: int, kmax: int, rmax: int) -> dict[tuple[int, ...], int]:
    """{ length-tuple -> smallest k at which it is realised by a positive g }."""
    found: dict[tuple[int, ...], int] = {}
    for k in range(1, kmax + 1):
        bl = by_length(k)
        lengths = [ell for ell in bl if ell <= rmax]
        for ells in itertools.product(lengths, repeat=n):
            if ells in found:
                continue
            for lams in itertools.product(*[bl[e] for e in ells]):
                if kronecker_coefficient(*lams) > 0:
                    found[ells] = k
                    break
    return found


# ---------------------------------------------------------------------------
# R3: the tree {A},{B} -> AB, and {AB},{C},{D} -> root
# ---------------------------------------------------------------------------


def reachable_tree_ab(kmax: int, rmax: int) -> dict[tuple[int, ...], int]:
    """{ (r_A,r_B,r_C,r_D,r_AB) -> smallest k }, requiring one shared lam^AB.

    dim(Im Pi cap Sym^k) = g(lam^A,lam^B,lam^AB) * g(lam^AB,lam^C,lam^D)
                           * prod_leaves dim V^{d_i}_{lam^(i)}
    so positivity needs BOTH node factors positive for the SAME lam^AB.
    """
    found: dict[tuple[int, ...], int] = {}
    for k in range(1, kmax + 1):
        bl = by_length(k)
        parts = list(partitions(k))
        # for each nu, which (ell, ell) pairs can sit below it?
        below: dict[tuple[int, ...], set[tuple[int, int]]] = defaultdict(set)
        for nu in parts:
            for lam, mu in itertools.combinations_with_replacement(parts, 2):
                if kronecker_coefficient(lam, mu, nu) > 0:
                    below[nu].add((len(lam), len(mu)))
                    below[nu].add((len(mu), len(lam)))
        for nu in parts:
            if len(nu) > rmax:
                continue
            pairs = {p for p in below[nu] if max(p) <= rmax}
            for (a, b), (c, d) in itertools.product(pairs, repeat=2):
                key = (a, b, c, d, len(nu))
                if key not in found:
                    found[key] = k
    return found


# ---------------------------------------------------------------------------


def main() -> int:
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rmax = 4

    print("=" * 78)
    print(f"R1  n=3: reachable (ell lam, ell mu, ell nu) with g > 0,  k <= {kmax}")
    print("=" * 78)
    got = reachable_flat(3, kmax, rmax)
    poly = {r for r in itertools.product(range(1, rmax + 1), repeat=3) if polygon(r)}
    print(f"  reachable: {len(got)}   polygon-admissible (r<= {rmax}): {len(poly)}")
    print(f"  reachable \\ polygon : {sorted(set(got) - poly)}")
    print(f"  polygon \\ reachable : {sorted(poly - set(got))}")
    ks = sorted({v for v in got.values()})
    print(f"  smallest k used: {min(ks)}..{max(ks)}")

    print()
    print("=" * 78)
    print(f"R2  n=4: reachable (ell lam^A,..,ell lam^D) with g > 0,  k <= {kmax}")
    print("=" * 78)
    got4 = reachable_flat(4, kmax, rmax)
    poly4 = {r for r in itertools.product(range(1, rmax + 1), repeat=4) if polygon(r)}
    print(f"  reachable: {len(got4)}   polygon-admissible: {len(poly4)}")
    print(f"  reachable \\ polygon : {sorted(set(got4) - poly4)}")
    print(f"  polygon \\ reachable : {sorted(poly4 - set(got4))}")

    print()
    print("=" * 78)
    print(f"R3  n=4 tree with lam^AB shared,  k <= {kmax}")
    print("=" * 78)
    tree = reachable_tree_ab(kmax, rmax)
    # does it decouple?  predicted = polygon(a,b,e) and polygon(e,c,d)
    decoupled = {
        (a, b, c, d, e)
        for a, b, c, d, e in itertools.product(range(1, rmax + 1), repeat=5)
        if polygon((a, b, e)) and polygon((e, c, d))
    }
    print(f"  reachable: {len(tree)}   node-local polygon prediction: {len(decoupled)}")
    extra = sorted(set(tree) - decoupled)
    missing = sorted(decoupled - set(tree))
    print(f"  reachable \\ decoupled : {extra[:20]}{' ...' if len(extra) > 20 else ''}")
    print(f"  decoupled \\ reachable : {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    print(f"  (counts: extra={len(extra)}, missing={len(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
