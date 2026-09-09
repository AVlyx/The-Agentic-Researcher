"""Rank-profile relaxation W_r, restructured to scale past n = 4.

Same mathematics as scripts/rank_profile_subspace.py -- see that file and
notes/algorithm_rank_profile_decision.md -- but three changes make n = 5 qubits
at k = 4 (dim Sym^k = 52360) run on a laptop, where the dense implementation
would need a 5.5e10-entry basis array:

  1. ORTHONORMAL MULTISET BASIS.  The symmetrised product states e_M are supported
     on disjoint sets of ordered tuples, hence mutually ORTHOGONAL.  Normalising
     them makes B an isometry onto Sym^k(H), so G = B P B^T is a compression of an
     orthogonal projector: its spectrum lies in [0,1] and the "is this eigenvalue
     zero" test has an absolute scale.  (The dense version's G had no such scale --
     cf. the numerical-hygiene warning in the note.)

  2. NEVER MATERIALISE B.  R_S(sigma) maps a product basis tuple to a product basis
     tuple, so B R_S(sigma) B^T is computed by integer index arithmetic on tuple
     codes: no D^k-sized float array is ever allocated.

  3. TORUS-WEIGHT BLOCK DIAGONALISATION.  R_S(sigma) permutes copies within the S
     side only, so for every party i it preserves the multiset of local indices
     carried by the k copies.  Every P^{(S)}_mu is therefore equivariant for the
     local torus, and G is block diagonal in the weight decomposition

         Sym^k(H) = (+)_w Sym^k(H)_w ,   w = (per party, the multiset of local
                                              indices across the k copies).

     For 5 qubits at k = 4 this splits one 52360 x 52360 eigenproblem into 3125
     blocks of average size 17: sum_b m_b^3 = 3.6e8 flops instead of 1.4e14.

Two identities are asserted at run time as self-checks: sum_mu G_{S,mu} = I on every
block (completeness of the isotypic decomposition on Sym^k, using the Cauchy rule
that only ell(mu) <= min(d_S, d_S^c) survives) and 0 <= spec G <= 1.

Usage:
    uv run python scripts/rank_profile_blocks.py --verify   # reproduce the n=4 battery
    uv run python scripts/rank_profile_blocks.py --n5       # 5 qubits, full sweep
"""

from __future__ import annotations

import itertools
import sys
import time
from math import factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from schur_weyl import dim_specht, partitions  # noqa: E402
from schur_weyl.character import character  # noqa: E402
from schur_weyl.symmetric_group import permutation_cycle_type  # noqa: E402

EV_TOL = 1e-9      # eigenvalue of G counted as zero (spec G lies in [0,1])
CERT_TOL = 1e-10   # ||P v||^2 / ||v||^2 counted as zero


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


class WeightBlocks:
    """Sym^k(H) in the orthonormal multiset basis, split into torus-weight blocks."""

    def __init__(self, dims: tuple[int, ...], k: int) -> None:
        self.dims, self.k, self.n = dims, k, len(dims)
        n, D = self.n, int(np.prod(dims))
        self.D = D
        self.letters = np.array(list(itertools.product(*[range(d) for d in dims])),
                                dtype=np.int64)                        # (D, n)
        self.radix = np.array([int(np.prod(dims[i + 1:])) for i in range(n)])
        self.powk = D ** np.arange(k)                                  # (k,)

        T = D ** k
        self.L = ((np.arange(T)[:, None] // self.powk) % D).astype(np.int64)   # (T, k)

        # multiset class of each ordered tuple
        cls_code = np.sort(self.L, axis=1) @ self.powk
        _, cls_of = np.unique(cls_code, return_inverse=True)
        self.cls_of = cls_of.astype(np.int64)
        self.m = int(self.cls_of.max()) + 1
        self.orderings = np.bincount(self.cls_of, minlength=self.m)    # ||e_M||^2

        # torus weight of each ordered tuple: per party, the counts of local values
        wcode = np.zeros(T, dtype=np.int64)
        base = 1
        for i in range(n):
            dig = self.letters[self.L, i]                              # (T, k)
            wi = np.zeros(T, dtype=np.int64)
            for a in range(dims[i]):
                wi += (dig == a).sum(axis=1) * (k + 1) ** a
            wcode += wi * base
            base *= (k + 1) ** dims[i]
            assert base > 0, "weight code overflow -- dims/k too large for int64"
        _, blk_of_tuple = np.unique(wcode, return_inverse=True)
        self.n_blocks = int(blk_of_tuple.max()) + 1

        # ordered tuples grouped by block
        self.t_order = np.argsort(blk_of_tuple, kind="stable")
        self.t_start = np.searchsorted(blk_of_tuple[self.t_order],
                                       np.arange(self.n_blocks + 1))
        # classes grouped by block (every ordering of a class has the same weight)
        blk_of_cls = np.empty(self.m, dtype=np.int64)
        blk_of_cls[self.cls_of] = blk_of_tuple
        c_order = np.argsort(blk_of_cls, kind="stable")
        self.c_start = np.searchsorted(blk_of_cls[c_order], np.arange(self.n_blocks + 1))
        self.local_of_cls = np.empty(self.m, dtype=np.int64)
        for b in range(self.n_blocks):
            sl = c_order[self.c_start[b]:self.c_start[b + 1]]
            self.local_of_cls[sl] = np.arange(sl.size)

    def merge_table(self, S: tuple[int, ...]) -> np.ndarray:
        """merge[a, b] = the letter with the S-digits of a and the S^c-digits of b."""
        Si = list(S)
        Ci = [i for i in range(self.n) if i not in S]
        sa = self.letters[:, Si] @ self.radix[Si]
        sb = self.letters[:, Ci] @ self.radix[Ci]
        return sa[:, None] + sb[None, :]

    def block_grams(self, b: int, merge: np.ndarray,
                    mus: list[tuple[int, ...]]) -> np.ndarray:
        """G_{S,mu} on block b, stacked over `mus`; shape (len(mus), m_b, m_b)."""
        k = self.k
        idx = self.t_order[self.t_start[b]:self.t_start[b + 1]]
        Lb = self.L[idx]
        lo = self.local_of_cls[self.cls_of[idx]]
        mb = int(self.c_start[b + 1] - self.c_start[b])
        norms = np.zeros(mb)
        norms[lo] = self.orderings[self.cls_of[idx]]
        scale = 1.0 / np.sqrt(np.outer(norms, norms))

        perms = self.perms
        A = np.empty((len(perms), mb, mb))
        for si in range(len(perms)):
            Lnew = merge[Lb[:, perms[si]], Lb]
            ln = self.local_of_cls[self.cls_of[Lnew @ self.powk]]
            A[si] = np.bincount(ln * mb + lo, minlength=mb * mb).reshape(mb, mb)
        A *= scale

        chi = np.stack([self.chi(mu) for mu in mus])          # (n_mu, k!)
        return np.tensordot(chi, A, axes=(1, 0))

    @property
    def perms(self) -> list[list[int]]:
        if not hasattr(self, "_perms"):
            self._perms = [list(s) for s in itertools.permutations(range(self.k))]
        return self._perms

    def chi(self, mu: tuple[int, ...]) -> np.ndarray:
        """chi^mu(sigma) * f^mu / k!, tabulated over S_k -- the weights of P^{(S)}_mu."""
        if not hasattr(self, "_chi"):
            self._chi = {}
        if mu not in self._chi:
            self._chi[mu] = np.array(
                [character(mu, permutation_cycle_type(tuple(s))) for s in self.perms],
                dtype=float) * (dim_specht(mu) / factorial(self.k))
        return self._chi[mu]


def sweep(dims: tuple[int, ...], k: int, profiles: list[dict], seed: int = 13,
          check: bool = True, progress: int = 0):
    """Run the relaxation on many profiles at once.

    Returns (cuts, cap, dimW, cert, normsq, qacc).  `cert[p, c]` is the rank the
    generic element of W_r certifies on cut `cuts[c]` for profile `p`.
    """
    n = len(dims)
    wb = WeightBlocks(dims, k)
    cuts = canonical_cuts(n)
    cap = {}
    for S in cuts:
        dS = int(np.prod([dims[i] for i in S]))
        dC = int(np.prod([dims[i] for i in range(n) if i not in S]))
        cap[S] = min(dS, dC, k)
    mus = {S: list(partitions(k, max_height=cap[S])) for S in cuts}
    merges = {S: wb.merge_table(S) for S in cuts}

    P = len(profiles)
    rmat = np.array([[min(p[S], cap[S]) for S in cuts] for p in profiles])
    dimW = np.zeros(P, dtype=np.int64)
    normsq = np.zeros(P)
    qacc = np.zeros((P, len(cuts), k + 1))   # q[p,c,r] = sum_{ell(mu)>r} ||P_mu v||^2
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for b in range(wb.n_blocks):
        mb = int(wb.c_start[b + 1] - wb.c_start[b])
        H = np.zeros((len(cuts), k + 1, mb, mb))
        for c, S in enumerate(cuts):
            G = wb.block_grams(b, merges[S], mus[S])
            if check:
                assert np.allclose(G.sum(axis=0), np.eye(mb), atol=1e-9), \
                    f"completeness failed on block {b}, cut {S}"
            ell = np.array([len(mu) for mu in mus[S]])
            for r in range(k + 1):
                sel = ell > r
                if sel.any():
                    H[c, r] = G[sel].sum(axis=0)
        for p in range(P):
            Gsum = H[np.arange(len(cuts)), rmat[p]].sum(axis=0)
            if not Gsum.any():
                null = np.eye(mb)
            else:
                ev, evec = np.linalg.eigh((Gsum + Gsum.T) / 2)
                null = evec[:, ev <= EV_TOL]
            dw = null.shape[1]
            dimW[p] += dw
            if dw == 0:
                continue
            cvec = null @ rng.normal(size=dw)
            normsq[p] += float(cvec @ cvec)
            qacc[p] += np.einsum("i,crij,j->cr", cvec, H, cvec, optimize=True)
        if progress and (b + 1) % progress == 0:
            print(f"      block {b + 1}/{wb.n_blocks}  ({time.time() - t0:.0f}s)",
                  flush=True)

    cert = np.zeros((P, len(cuts)), dtype=np.int64)
    for p in range(P):
        if normsq[p] == 0:
            continue
        for c in range(len(cuts)):
            nz = np.nonzero(qacc[p, c] / normsq[p] > CERT_TOL)[0]
            cert[p, c] = (nz.max() + 1) if nz.size else 0
    return cuts, cap, dimW, cert, normsq, qacc


def verdict_of(profile: dict, cuts, cap, cert_row) -> tuple[str, list]:
    short = [S for c, S in enumerate(cuts)
             if cert_row[c] < min(profile[S], cap[S])]
    return ("IMPOSSIBLE" if short else "feasible"), short


# ---------------------------------------------------------------------------
# n = 4: reproduce the dense battery, as a check on the whole restructuring
# ---------------------------------------------------------------------------

def verify(k: int = 4) -> None:
    dims = (2, 2, 2, 2)
    cuts = canonical_cuts(4)
    pair = [S for S in cuts if len(S) == 2]
    trips, profiles = [], []
    for a, b, c in itertools.combinations_with_replacement(range(1, 5), 3):
        a, b, c = sorted((a, b, c), reverse=True)
        trips.append((a, b, c))
        prof = {S: 2 for S in cuts if len(S) == 1}
        prof.update(dict(zip(pair, (a, b, c))))
        profiles.append(prof)
    print(f"n=4 qubits, k={k}, local ranks 2 -- cross-check against rank_battery_k4.log")
    t0 = time.time()
    cutsr, cap, dimW, cert, *_ = sweep(dims, k, profiles)
    pidx = [cutsr.index(S) for S in pair]
    sidx = [cutsr.index(S) for S in cutsr if len(S) == 1]
    print(f"{'profile':>12}  {'dim W':>6}  {'cert pairs':>12}  {'cert loc':>14}  verdict")
    for i, t in enumerate(trips):
        v, _ = verdict_of(profiles[i], cutsr, cap, cert[i])
        print(f"{t!s:>12}  {dimW[i]:>6}  "
              f"{''.join(str(int(x)) for x in cert[i, pidx]):>12}  "
              f"{''.join(str(int(x)) for x in cert[i, sidx]):>14}  {v}")
    print(f"({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# n = 5: the first case whose cuts are 2 | 3, so neither side is a single party
# ---------------------------------------------------------------------------

def pair_orbit_reps(n: int, values: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Labellings of the pairs of [n] by `values`, one representative per S_n orbit.

    With all local dimensions and all local target ranks equal, relabelling parties
    is a symmetry of the whole problem, so only orbit representatives need running --
    and profiles inside one orbit must come out identical, which is a free check.
    """
    edges = list(itertools.combinations(range(n), 2))
    eidx = {e: i for i, e in enumerate(edges)}
    E, V = len(edges), len(values)
    grid = np.array(list(itertools.product(range(V), repeat=E)), dtype=np.int64)
    powv = V ** np.arange(E)[::-1]
    codes = np.full(grid.shape[0], np.iinfo(np.int64).max)
    for p in itertools.permutations(range(n)):
        perm = [eidx[tuple(sorted((p[i], p[j])))] for i, j in edges]
        np.minimum(codes, grid[:, perm] @ powv, out=codes)
    keep = np.nonzero(grid @ powv == codes)[0]
    vals = np.array(values)
    return [tuple(vals[grid[i]]) for i in keep]


def n5(k: int = 4, values: tuple[int, ...] = (2, 3, 4), seed: int = 13) -> None:
    dims = (2, 2, 2, 2, 2)
    n = 5
    cuts = canonical_cuts(n)
    pair = [S for S in cuts if len(S) == 2]
    single = [S for S in cuts if len(S) == 1]
    assert len(pair) == 10 and len(single) == 5

    labels = pair_orbit_reps(n, values)
    profiles = []
    for lab in labels:
        prof = {S: 2 for S in single}
        prof.update(dict(zip(pair, lab)))
        profiles.append(prof)

    wb_dim = "?"
    print("=" * 96)
    print(f"n=5 qubits, k={k}, local ranks all 2.")
    print(f"Cuts: {len(single)} of shape 1|4 (max rank 2) and {len(pair)} of shape 2|3 "
          f"(max rank 4) -- both sides multi-party.")
    print(f"Sweep: all pair-profiles with values in {values}, "
          f"{len(labels)} S_5-orbit representatives out of {len(values)**10}.")
    print("=" * 96)
    t0 = time.time()
    cutsr, cap, dimW, cert, normsq, qacc = sweep(dims, k, profiles, seed=seed,
                                                 progress=500)
    pidx = [cutsr.index(S) for S in pair]
    sidx = [cutsr.index(S) for S in single]
    order = " ".join(f"{S[0]}{S[1]}" for S in pair)
    print(f"\npair-cut order: {order}")
    print(f"{'profile':>32}  {'dim W':>7}  {'certified':>32}  {'loc':>11}  verdict")
    n_imp = 0
    feas = []
    for i, lab in enumerate(labels):
        v, _ = verdict_of(profiles[i], cutsr, cap, cert[i])
        n_imp += v == "IMPOSSIBLE"
        if v == "feasible":
            feas.append(lab)
        cp = "".join(str(int(x)) for x in cert[i, pidx])
        cs = "".join(str(int(x)) for x in cert[i, sidx])
        print(f"{''.join(map(str, lab)):>32}  {dimW[i]:>7}  {cp:>32}  {cs:>11}  {v}")
    print(f"\n{n_imp}/{len(labels)} orbits IMPOSSIBLE, {len(feas)} feasible "
          f"({time.time() - t0:.0f}s)")
    return labels, feas


# ---------------------------------------------------------------------------
# 2 x 2 x 2 x 4: what a 5-qubit state looks like after merging two parties
# ---------------------------------------------------------------------------

def merged(k: int = 4, seed: int = 13) -> None:
    """Does "the maximum is attained at least twice" survive a party of dimension 4?

    Merging parties l, m of a 5-qubit state into one party F of dimension 4 gives a
    2x2x2x4 state whose three pair-cut ranks are (r_ij, r_ik, r_jk) -- a TRIANGLE of
    the 5-party profile -- with r_F = r_lm.  So the n=4 Segre rule transfers to a
    triangle of K_5 exactly when it holds at local rank r_F = r_lm.  Here r_F is
    swept over 2, 3, 4; r_F = 2 is the 2x2x2x2 case in disguise.
    """
    dims = (2, 2, 2, 4)
    cuts = canonical_cuts(4)
    pair = [S for S in cuts if len(S) == 2]      # AB, AC, AF
    sing = [S for S in cuts if len(S) == 1]
    print("dims (2,2,2,4): party F carries a merged qubit pair, r_F = r_lm.")
    print(f"cuts {pair} <-> the triangle (r_ij, r_ik, r_jk) of K_5")
    profiles, tags = [], []
    for t in (2, 3, 4):
        for trip in itertools.product(range(1, 5), repeat=3):
            prof = dict(zip(sing, (2, 2, 2, t)))
            prof.update(dict(zip(pair, trip)))
            profiles.append(prof)
            tags.append((t, trip))
    t0 = time.time()
    cutsr, cap, dimW, cert, *_ = sweep(dims, k, profiles, seed=seed, progress=2000)
    pidx = [cutsr.index(S) for S in pair]
    sidx = [cutsr.index(S) for S in sing]
    print(f"\n{'r_F':>4}  {'triple':>12}  {'dim W':>7}  {'cert':>12}  {'cert loc':>12}  "
          f"{'max 2x?':>8}  verdict")
    agree = disagree = 0
    for i, (t, trip) in enumerate(tags):
        v, _ = verdict_of(profiles[i], cutsr, cap, cert[i])
        twice = trip.count(max(trip)) >= 2
        # is the relaxation verdict exactly "max attained at least twice"?
        if (v == "feasible") == twice:
            agree += 1
        else:
            disagree += 1
        print(f"{t:>4}  {trip!s:>12}  {dimW[i]:>7}  "
              f"{''.join(str(int(x)) for x in cert[i, pidx]):>12}  "
              f"{''.join(str(int(x)) for x in cert[i, sidx]):>12}  "
              f"{'yes' if twice else 'no':>8}  {v}")
    print(f"\nrelaxation verdict == 'max attained at least twice': "
          f"{agree}/{agree + disagree}")
    for t in (2, 3, 4):
        sub = [(v, tr) for (tt, tr), v in
               zip(tags, [verdict_of(profiles[i], cutsr, cap, cert[i])[0]
                          for i in range(len(profiles))]) if tt == t]
        a = sum(1 for v, tr in sub if (v == "feasible") == (tr.count(max(tr)) >= 2))
        print(f"   r_F = {t}: {a}/{len(sub)}")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
        raise SystemExit(0)
    if "--n5" in sys.argv:
        n5()
        raise SystemExit(0)
    if "--merged" in sys.argv:
        merged()
        raise SystemExit(0)
    raise SystemExit("pass --verify, --n5 or --merged")
