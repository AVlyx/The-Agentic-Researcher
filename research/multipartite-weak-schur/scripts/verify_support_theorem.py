"""Verify the support theorem for multipartite weak Schur sampling.

Theorem (support).  Let Pi_lam = (x)_i P^{(i)}_{lam^i} on (x)_i H_i^{(x)k}.  Then

    (a)  dim( Im Pi_lam  cap  Sym^k(H) ) = g(lam^1,...,lam^n) * prod_i dim V^{d_i}_{lam^i}
    (b)  there exists a pure state psi with p_psi(lam) > 0
             <=>  g(lam^1,...,lam^n) > 0  and  ell(lam^i) <= d_i for all i.

Three independent checks:
  T1  rank( Pi_lam restricted to Sym^k(H) ) computed by SVD, against the formula (a).
  T2  the support of a single Haar-random state, against the prediction (b).
  T3  the union of supports over many random states, against the prediction (b).

Usage:  uv run python scripts/verify_support_theorem.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mws  # noqa: E402


def check_dimension_formula(dims: tuple[int, ...], k: int) -> tuple[int, int]:
    """T1: rank of Pi_lam on Sym^k(H) by SVD vs g * prod dim V.  Returns (ok, total)."""
    n = len(dims)
    basis = mws.sym_basis(dims, k)  # (m,) + copy layout
    m = basis.shape[0]
    lam_choices = [mws._partitions(k, max_height=min(dims[i], k)) for i in range(n)]

    ok = total = 0
    for lams in itertools.product(*lam_choices):
        B = basis.astype(np.float64)
        B = B / np.linalg.norm(B.reshape(m, -1), axis=1).reshape((m,) + (1,) * (B.ndim - 1))
        T = B
        for i, lam in enumerate(lams):
            T = mws.apply_isotypic(T, lam, (i,), n, k, lead=1)
        rank = _numeric_rank(T.reshape(m, -1), in_scale=1.0)
        predicted = mws.block_dimension(dims, lams)
        total += 1
        ok += rank == predicted
        if rank != predicted:
            print(f"    MISMATCH {lams}: rank={rank} predicted={predicted}")
    return ok, total


def _numeric_rank(M: np.ndarray, in_scale: float = 1.0, atol: float = 1e-6) -> int:
    """Rank via the Gram matrix (M is m x N with m << N).

    `atol` is a tolerance on SINGULAR values, measured against `in_scale`, the norm
    of a typical row BEFORE projection.  It must be absolute, not relative to M: if
    the projector annihilates the whole input then M is pure roundoff, and a
    relative test would cheerfully report full rank.  Roundoff singular values sit
    around 1e-9 here, genuine ones at O(in_scale), so 1e-6 sits in the gap.

    Warns if the kept/discarded spectrum has no clear gap, rather than silently
    returning a threshold-dependent answer.
    """
    G = M @ M.conj().T
    ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)[::-1]
    thresh = (atol * max(in_scale, 1e-300)) ** 2
    r = int(np.sum(ev > thresh))
    if 0 < r < len(ev):
        lo, hi = ev[r - 1], ev[r]
        if hi > 0 and lo / hi < 1e4:
            print(f"    WARNING: weak rank gap {lo:.3e} -> {hi:.3e} (rank {r})")
    return r


def check_dimension_by_span(
    dims: tuple[int, ...], k: int, seed: int = 5, slack: int = 6
) -> tuple[int, int]:
    """T1b: dim span{ Pi_lam psi^{(x)k} : psi random } vs g * prod_i dim V^{d_i}.

    Cheaper than T1a for large D, and a sharper statement: it says exactly how much
    of the lambda-block the i.i.d. states psi^{(x)k} actually reach.
    """
    n = len(dims)
    rng = np.random.default_rng(seed)
    lam_choices = [mws._partitions(k, max_height=min(dims[i], k)) for i in range(n)]

    ok = total = 0
    for lams in itertools.product(*lam_choices):
        predicted = mws.block_dimension(dims, lams)
        vecs = []
        for _ in range(predicted + slack):
            T = mws.copy_tensor(mws.random_pure_state(dims, rng), k)
            for i, lam in enumerate(lams):
                T = mws.apply_isotypic(T, lam, (i,), n, k)
            vecs.append(T.reshape(-1))
        rank = _numeric_rank(np.stack(vecs), in_scale=1.0)
        total += 1
        ok += rank == predicted
        if rank != predicted:
            print(f"    MISMATCH {lams}: span rank={rank} predicted={predicted}")
    return ok, total


def check_support(dims: tuple[int, ...], k: int, n_states: int, seed: int) -> dict:
    """T2/T3: observed support of random states vs Kronecker-positivity prediction."""
    rng = np.random.default_rng(seed)
    predicted = mws.support_prediction(dims, k)
    union: set = set()
    single_ok = True
    for s in range(n_states):
        psi = mws.random_pure_state(dims, rng)
        supp = set(mws.mws_distribution(psi, k, tol=1e-11).keys())
        if s == 0:
            single_ok = supp == predicted
            single_extra = supp - predicted
            single_missing = predicted - supp
        union |= supp
    return {
        "predicted": predicted,
        "union": union,
        "single_ok": single_ok,
        "single_extra": single_extra,
        "single_missing": single_missing,
    }


def check_against_package() -> int:
    """T0: our primitives vs schur_weyl >= 0.2.0.  Returns number of mismatches."""
    from schur_weyl import apply_isotypic_proj, isotypic_proj, partitions

    bad = 0
    n_kron = 0
    for k in (2, 3, 4, 5):
        ps = list(partitions(k))
        for a in ps:
            for b in ps:
                n_kron += 1
                bad += mws.kronecker_ref(a, b) != mws.kronecker(a, b)
                for c in ps:
                    n_kron += 1
                    bad += mws.kronecker_ref(a, b, c) != mws.kronecker(a, b, c)
    print(f"  kronecker_ref vs schur_weyl.kronecker_coefficient: {n_kron} pairs/triples, {bad} bad")

    rng = np.random.default_rng(1)
    n_proj = 0
    for k in (2, 3, 4):
        for d in (2, 3):
            for lam in partitions(k, max_height=d):
                T = rng.normal(size=(d,) * k)
                n_proj += 1
                bad += not np.allclose(
                    mws.apply_isotypic(T, lam, (0,), 1, k), apply_isotypic_proj(T, lam)
                )
                bad += not np.allclose(
                    mws.apply_isotypic(T, lam, (0,), 1, k).reshape(-1),
                    isotypic_proj(lam, d) @ T.reshape(-1),
                )
    print(f"  apply_isotypic vs schur_weyl apply_isotypic_proj / isotypic_proj: {n_proj} cases")
    return bad


def main() -> int:
    failures = 0

    print("=" * 78)
    print("T0  cross-check against schur_weyl >= 0.2.0")
    print("=" * 78)
    bad = check_against_package()
    failures += bad > 0
    print(f"  {'PASS' if bad == 0 else 'FAIL'}  ({bad} mismatches)")
    print()

    print("=" * 78)
    print("T1a dim(Im Pi_lam cap Sym^k H) == g(lam) * prod_i dim V^{d_i}_{lam^i}")
    print("    (explicit symmetrised product basis of Sym^k H)")
    print("=" * 78)
    for dims, k in [
        ((2, 2), 3),
        ((2, 2, 2), 2),
        ((2, 2, 2), 3),
        ((2, 3, 2), 3),
        ((2, 2, 2), 4),
        ((2, 2, 2, 2), 3),
    ]:
        ok, total = check_dimension_formula(dims, k)
        status = "PASS" if ok == total else "FAIL"
        failures += ok != total
        print(f"  dims={dims!s:14s} k={k}   {ok}/{total} lambda-tuples   {status}")

    print()
    print("=" * 78)
    print("T1b dim span{ Pi_lam psi^(x)k : psi random } == g(lam) * prod_i dim V")
    print("=" * 78)
    for dims, k in [
        ((2, 2, 2), 3),
        ((2, 2, 2), 4),
        ((3, 3, 2), 3),
        ((3, 3, 3), 3),
        ((2, 2, 2, 2), 3),
        ((2, 2, 2, 2, 2), 3),
    ]:
        ok, total = check_dimension_by_span(dims, k)
        status = "PASS" if ok == total else "FAIL"
        failures += ok != total
        print(f"  dims={dims!s:14s} k={k}   {ok}/{total} lambda-tuples   {status}")

    print()
    print("=" * 78)
    print("T2/T3  support of weak Schur sampling == {lam : g(lam) > 0}")
    print("=" * 78)
    for dims, k, n_states in [
        ((2, 2, 2), 3, 5),
        ((2, 2, 2), 4, 5),
        ((2, 2, 2), 5, 3),
        ((2, 3, 2), 3, 5),
        ((2, 3, 4), 3, 5),
        ((3, 3, 3), 3, 3),
        ((3, 3, 3), 4, 2),
        ((2, 2, 2, 2), 3, 5),
        ((2, 2, 2, 2), 4, 2),
        ((2, 2, 2, 2, 2), 3, 3),
    ]:
        r = check_support(dims, k, n_states, seed=17)
        union_ok = r["union"] == r["predicted"]
        status = "PASS" if union_ok and r["single_ok"] else "FAIL"
        failures += not (union_ok and r["single_ok"])
        print(
            f"  dims={dims!s:16s} k={k}  |predicted|={len(r['predicted']):4d} "
            f" |union|={len(r['union']):4d}  single-state support == prediction: "
            f"{r['single_ok']}   {status}"
        )
        if not union_ok:
            print(f"    extra in union: {r['union'] - r['predicted']}")
            print(f"    missing:        {r['predicted'] - r['union']}")
        elif not r["single_ok"]:
            print(f"    single extra:   {r['single_extra']}")
            print(f"    single missing: {r['single_missing']}")

    print()
    print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
