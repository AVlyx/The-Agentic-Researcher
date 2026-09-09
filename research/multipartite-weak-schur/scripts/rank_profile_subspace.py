"""Rank profiles across ALL bipartitions, via a linear relaxation.

Idea (due to the project owner).  Instead of asking for a JOINT measurement of
overlapping cuts -- impossible, the projectors do not commute -- ask for a single
state on which each cut's projection behaves as required.  For a target profile
r = (r_S)_S over bipartitions S:

    upper bounds   rank rho_S <= r_S    <=>   P^(S)_mu psi^{(x)k} = 0 for all
                                              mu with ell(mu) > r_S
    lower bounds   rank rho_S >= r_S    <=>   P^(S)_lam psi^{(x)k} != 0 for some
                                              lam with ell(lam) = r_S

The upper bounds are LINEAR, so

    W_r = Sym^k(H)  cap  intersection over S, ell(mu) > r_S  of  ker P^(S)_mu

is a subspace.  The lower bounds then come free by genericity: each
{v : P^(S)_lam v != 0} is dense open in W_r whenever P^(S)_lam|_{W_r} != 0, and a
finite intersection of dense opens is nonempty.  So we compute W_r by linear algebra
and read off the profile CERTIFIED by a generic element of it:

    cert_S(v) = max{ ell(mu) : P^(S)_mu v != 0 }.

RELAXATION.  A generic v in W_r need not be of the form psi^{(x)k}, so
cert(v) == r is necessary, not sufficient, for r to be achievable.  If instead
cert_S(v) < r_S for some S (or W_r = 0), the profile is genuinely IMPOSSIBLE --
an obstruction, valid because every psi with profile r would give such a v.

Implementation note: rather than stacking the constraint operators (huge), we form
the PSD Gram matrix G = sum_{S,mu} B^dag P^(S)_mu B over a basis B of Sym^k(H);
ker G in coefficient space is exactly the intersection of the kernels.

Usage:  uv run python scripts/rank_profile_subspace.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mws  # noqa: E402
from schur_weyl import partitions  # noqa: E402


def canonical_cuts(n: int) -> list[tuple[int, ...]]:
    """One representative per bipartition S | S^c, excluding the trivial cut."""
    seen, out = set(), []
    for r in range(1, n):
        for S in itertools.combinations(range(n), r):
            Sc = tuple(i for i in range(n) if i not in S)
            if S in seen or Sc in seen:
                continue
            seen.add(S)
            out.append(S)
    return out


def analyse(dims: tuple[int, ...], k: int, target: dict[tuple[int, ...], int],
            label: str, seed: int = 13) -> None:
    n = len(dims)
    B = mws.sym_basis(dims, k)                  # (m,) + copy layout
    m = B.shape[0]
    Bflat = B.reshape(m, -1)
    cuts = canonical_cuts(n)
    dim_of = {S: int(np.prod([dims[i] for i in S])) for S in cuts}

    # --- upper bounds: build the PSD Gram matrix of all "must vanish" projectors
    G = np.zeros((m, m))
    n_constraints = 0
    for S in cuts:
        rS = target[S]
        cap = min(dim_of[S], int(np.prod([dims[i] for i in range(n) if i not in S])), k)
        for mu in partitions(k, max_height=cap):
            if len(mu) <= rS:
                continue
            PB = mws.apply_isotypic(B, mu, S, n, k, lead=1)
            G += PB.reshape(m, -1) @ Bflat.T
            n_constraints += 1

    # --- W_r = ker G  (in coefficient coordinates)
    if n_constraints == 0:
        null = np.eye(m)
    else:
        G = (G + G.T) / 2
        ev, evec = np.linalg.eigh(G)
        null = evec[:, ev <= 1e-8 * max(ev.max(), 1.0)]
    dimW = null.shape[1]

    print(f"  {label}")
    print(f"    target {{{', '.join(f'{S}:{target[S]}' for S in cuts)}}}")
    print(f"    {n_constraints} vanishing constraints,  dim Sym^k = {m},  dim W_r = {dimW}")
    if dimW == 0:
        print("    => W_r = 0: NO state has all these ranks bounded above.  IMPOSSIBLE.")
        return

    # --- generic element of W_r, and the profile it certifies
    rng = np.random.default_rng(seed)
    v = (Bflat.T @ (null @ rng.normal(size=dimW))).reshape(B.shape[1:])
    scale = np.linalg.norm(v)
    cert, verdict = {}, "consistent"
    for S in cuts:
        best = 0
        for mu in partitions(k, max_height=min(dim_of[S], k)):
            if np.linalg.norm(mws.apply_isotypic(v, mu, S, n, k)) > 1e-8 * scale:
                best = max(best, len(mu))
        cert[S] = best
        if best < min(target[S], k):
            verdict = "IMPOSSIBLE"
    print(f"    certified by generic v: {{{', '.join(f'{S}:{cert[S]}' for S in cuts)}}}")
    if verdict == "IMPOSSIBLE":
        short = [S for S in cuts if cert[S] < min(target[S], k)]
        print(f"    => generic v cannot reach r_S on {short}.  Profile IMPOSSIBLE.")
    else:
        print("    => relaxation feasible (necessary condition passes; not a proof)")


def main() -> int:
    dims = (2, 2, 2, 2)
    A, Bc, C, D = (0,), (1,), (2,), (3,)
    AB, AC, AD = (0, 1), (0, 2), (0, 3)

    def prof(ab, ac, ad, loc=2):
        return {A: loc, Bc: loc, C: loc, D: loc, AB: ab, AC: ac, AD: ad}

    for k in (3, 4):
        print("=" * 78)
        print(f"n=4 qubits, k={k}   (ranks >= k are not distinguishable at this k)")
        print("=" * 78)
        analyse(dims, k, prof(2, 2, 1), "CHLW ray 3   (2,2,2,2 | 2,2,1)  -- known UNACHIEVABLE")
        analyse(dims, k, prof(4, 4, 1), "control      (2,2,2,2 | 4,4,1)  -- product eta_AD (x) theta_BC")
        analyse(dims, k, prof(2, 2, 2), "control      (2,2,2,2 | 2,2,2)")
        analyse(dims, k, prof(4, 3, 3), "GMO test     (2,2,2,2 | 4,3,3)  -- max attained once")
        analyse(dims, k, prof(4, 4, 4), "generic      (2,2,2,2 | 4,4,4)")
        print()
    return 0


def battery(k: int = 4, seed: int = 13) -> None:
    """All (r_AB, r_AC, r_AD) with r_AB >= r_AC >= r_AD, local ranks 2, n=4 qubits.

    Caches the per-(cut, mu) Gram blocks so each profile is just a sum + eigh.
    """
    dims = (2, 2, 2, 2)
    n = 4
    cuts = canonical_cuts(n)
    pair_cuts = [S for S in cuts if len(S) == 2]
    B = mws.sym_basis(dims, k)
    m = B.shape[0]
    Bflat = B.reshape(m, -1)
    cache: dict = {}

    def gram(S, mu):
        if (S, mu) not in cache:
            PB = mws.apply_isotypic(B, mu, S, n, k, lead=1)
            cache[(S, mu)] = PB.reshape(m, -1) @ Bflat.T
        return cache[(S, mu)]

    rng = np.random.default_rng(seed)
    print(f"n=4 qubits, k={k}, local ranks all 2, dim Sym^k = {m}")
    print(f"{'profile':>14}  {'#constr':>7}  {'dim W':>6}  {'certified':>14}  verdict")
    for a, b, c in itertools.combinations_with_replacement(range(1, 5), 3):
        a, b, c = sorted((a, b, c), reverse=True)
        target = {S: 2 for S in cuts if len(S) == 1}
        target.update(dict(zip(pair_cuts, (a, b, c))))
        G = np.zeros((m, m))
        nc = 0
        for S in pair_cuts:
            for mu in partitions(k, max_height=min(4, k)):
                if len(mu) > target[S]:
                    G += gram(S, mu)
                    nc += 1
        if nc:
            ev, evec = np.linalg.eigh((G + G.T) / 2)
            null = evec[:, ev <= 1e-8 * max(ev.max(), 1.0)]
        else:
            null = np.eye(m)
        dimW = null.shape[1]
        v = (Bflat.T @ (null @ rng.normal(size=dimW))).reshape(B.shape[1:])
        sc = np.linalg.norm(v)
        cert = []
        for S in pair_cuts:
            best = max((len(mu) for mu in partitions(k, max_height=min(4, k))
                        if np.linalg.norm(mws.apply_isotypic(v, mu, S, n, k)) > 1e-8 * sc),
                       default=0)
            cert.append(best)
        bad = [S for S, cc, tt in zip(pair_cuts, cert, (a, b, c)) if cc < min(tt, k)]
        verdict = "IMPOSSIBLE" if bad else "feasible"
        print(f"{(a,b,c)!s:>14}  {nc:>7}  {dimW:>6}  {tuple(cert)!s:>14}  {verdict}")


if __name__ == "__main__":
    if "--battery" in sys.argv:
        battery()
        raise SystemExit(0)
    raise SystemExit(main())
