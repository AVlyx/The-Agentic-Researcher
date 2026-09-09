"""Witness search for an arbitrary number of parties and an arbitrary cut family.

Same method as scripts/rank_witness_search.py (which is hard-wired to 4 qubits):
minimise the tail spectral mass

    f(psi) = sum_S ( 1 - sum_{j <= r_S} eig_j(rho_S) ) ,

which is >= 0 and vanishes exactly when rank rho_S <= r_S for every cut S; the lower
bounds are then checked by inspection at the minimiser.  Two changes make it usable
at n = 5, where the 4-qubit script would need 15 small eigendecompositions per
finite-difference component:

  * ANALYTIC GRADIENT.  By the envelope theorem the tail mass is <psi| Q_S (x) I |psi>
    with the tail projector Q_S held fixed, so d f / d psi-bar = sum_S (Q_S (x) I) psi,
    corrected for the normalisation psi = z / ||z||.  This replaces 2D + 1 objective
    evaluations per gradient by one, ~65x fewer at n = 5.
  * ARBITRARY dims / cuts, so the same code runs the n = 4 case as a regression check.

Usage:
    uv run python scripts/rank_witness_general.py --check   # reproduce the n=4 run
    uv run python scripts/rank_witness_general.py --n5      # profiles from the n=5 log
"""

from __future__ import annotations

import itertools
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from rank_profile_blocks import canonical_cuts  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def _flat_maps(dims: tuple[int, ...], cuts: list[tuple[int, ...]]):
    """Per cut: the axis permutation to (S, S^c), its inverse, and the two side dims."""
    n = len(dims)
    out = []
    for S in cuts:
        rest = tuple(i for i in range(n) if i not in S)
        perm = S + rest
        dS = int(np.prod([dims[i] for i in S]))
        out.append((perm, tuple(np.argsort(perm)), tuple(dims[i] for i in perm), dS))
    return out


def objective_and_grad(x: np.ndarray, dims, maps, ranks):
    """f(psi) and its gradient in the real coordinates x = (Re z, Im z)."""
    D = x.size // 2
    z = x[:D] + 1j * x[D:]
    nz2 = float(np.vdot(z, z).real)
    psi = z.reshape(dims)
    f_un = 0.0
    gz = np.zeros(D, dtype=complex)
    for (perm, inv, pdims, dS), r in zip(maps, ranks):
        keep = dS - r                      # eigenvalues below the target rank cut
        if keep <= 0:
            continue
        M = psi.transpose(perm).reshape(dS, -1)
        ev, U = np.linalg.eigh(M @ M.conj().T)          # ascending
        f_un += float(ev[:keep].sum())
        Q = U[:, :keep]
        QM = Q @ (Q.conj().T @ M)                        # (Q_S (x) I) psi, flattened
        gz += QM.reshape(pdims).transpose(inv).reshape(-1)
    f = f_un / nz2
    d = (gz - f * z) / nz2                               # d f / d z-bar
    return f, np.concatenate([2.0 * d.real, 2.0 * d.imag])


def profile_of(psi: np.ndarray, maps, tol: float = 1e-7) -> tuple[int, ...]:
    out = []
    for perm, inv, pdims, dS in maps:
        M = psi.transpose(perm).reshape(dS, -1)
        ev = np.linalg.eigvalsh(M @ M.conj().T)
        out.append(int((ev > tol * max(ev.max(), 1e-30)).sum()))
    return tuple(out)


def search(dims: tuple[int, ...], cuts: list[tuple[int, ...]], ranks: tuple[int, ...],
           restarts: int = 40, seed: int = 0, ftol: float = 1e-11):
    """Return (found, achieved_profile, best_f)."""
    D = int(np.prod(dims))
    maps = _flat_maps(dims, cuts)
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(restarts):
        x0 = rng.normal(size=2 * D)
        res = minimize(objective_and_grad, x0, args=(dims, maps, ranks), jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-18, "gtol": 1e-14})
        best = min(best, float(res.fun))
        if res.fun < ftol:
            z = res.x[:D] + 1j * res.x[D:]
            psi = (z / np.linalg.norm(z)).reshape(dims)
            prof = profile_of(psi, maps)
            if prof == tuple(ranks):
                return True, prof, float(res.fun)
    return False, None, best


# ---------------------------------------------------------------------------
# regression: the n = 4 case, against artifacts/rank_witness.log
# ---------------------------------------------------------------------------

def check() -> None:
    dims = (2, 2, 2, 2)
    cuts = canonical_cuts(4)
    pair = [S for S in cuts if len(S) == 2]
    print("n=4 qubits, local ranks 2 -- regression against rank_witness.log")
    print(f"{'profile':>12}  {'witness':>8}  {'best f':>10}")
    found = []
    for trip in sorted(itertools.combinations_with_replacement(range(1, 5), 3),
                       key=lambda t: tuple(sorted(t, reverse=True))):
        t = tuple(sorted(trip, reverse=True))
        prof = {S: 2 for S in cuts if len(S) == 1}
        prof.update(dict(zip(pair, t)))
        ranks = tuple(prof[S] for S in cuts)
        ok, _, f = search(dims, cuts, ranks, restarts=40, seed=1)
        if ok:
            found.append(t)
        print(f"{t!s:>12}  {'FOUND' if ok else 'none':>8}  {f:>10.2e}")
    print(f"witnesses: {sorted(found)}")
    print("expected  : [(2, 2, 2), (3, 3, 2), (3, 3, 3), (4, 4, 1), (4, 4, 2), "
          "(4, 4, 3), (4, 4, 4)]")


# ---------------------------------------------------------------------------
# n = 5: run on the profiles the relaxation could not rule out
# ---------------------------------------------------------------------------

def read_log(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        mo = re.match(r"^\s*(\d{10})\s+\d+\s+\S+\s+\d{5}\s+(feasible|IMPOSSIBLE)\s*$",
                      line)
        if mo:
            rows.append((mo.group(1), mo.group(2)))
    return rows


def n5(restarts: int = 100, seed: int = 1) -> None:
    """Run the witness search on EVERY orbit representative, not only the ones the
    relaxation admits: a witness for a profile the relaxation called IMPOSSIBLE would
    be a contradiction, so running both halves everywhere is a check on both."""
    dims = (2,) * 5
    cuts = canonical_cuts(5)
    pair = [S for S in cuts if len(S) == 2]
    rows = read_log(HERE / "artifacts" / "rank_blocks_n5.log")
    n_feas = sum(v == "feasible" for _, v in rows)
    print(f"n=5 qubits, local ranks 2, {restarts} restarts per profile.")
    print(f"{len(rows)} S_5-orbit representatives; the relaxation admits {n_feas}.")
    print(f"pair-cut order: {' '.join(f'{S[0]}{S[1]}' for S in pair)}")
    print(f"{'profile':>12}  {'relaxation':>11}  {'witness':>8}  {'best f':>10}  "
          f"{'achieved':>12}  verdict")
    t0 = time.time()
    settled = contra = 0
    open_cases = []
    for lab, relax in rows:
        prof = {S: 2 for S in cuts if len(S) == 1}
        prof.update({S: int(ch) for S, ch in zip(pair, lab)})
        ranks = tuple(prof[S] for S in cuts)
        ok, got, f = search(dims, cuts, ranks, restarts=restarts, seed=seed)
        if ok:
            verdict = "ACHIEVABLE (witness)"
        elif relax == "IMPOSSIBLE":
            verdict = "impossible (relaxation)"
        else:
            verdict = "OPEN"
        if relax == "IMPOSSIBLE" and ok:
            contra += 1
            verdict = "CONTRADICTION"
        elif relax == "IMPOSSIBLE" or ok:
            settled += 1
        else:
            open_cases.append(lab)
        gp = "".join(str(got[cuts.index(S)]) for S in pair) if got else "-"
        print(f"{lab:>12}  {relax:>11}  {'FOUND' if ok else 'none':>8}  {f:>10.2e}  "
              f"{gp:>12}  {verdict}", flush=True)
    print(f"\nsettled {settled}/{len(rows)}   open {len(open_cases)}   "
          f"contradictions {contra}   ({time.time() - t0:.0f}s)")
    print(f"open: {open_cases}")


# ---------------------------------------------------------------------------
# the rank-1 sector, which the sweep over values {2,3,4} deliberately omits
# ---------------------------------------------------------------------------

def product_forcing(trials: int = 2000, seed: int = 7) -> None:
    """r_ij = 1 forces psi = eta_ij (x) theta_klm, hence (with all local ranks 2)
    the whole profile: the 6 edges crossing {i,j} | {k,l,m} are 4, the 3 edges
    inside {k,l,m} are 2.  Sampled here to confirm no other profile is reachable."""
    dims = (2,) * 5
    cuts = canonical_cuts(5)
    pair = [S for S in cuts if len(S) == 2]
    maps = _flat_maps(dims, cuts)
    rng = np.random.default_rng(seed)
    seen = {}
    for _ in range(trials):
        eta = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        th = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
        psi = np.tensordot(eta, th, axes=0)
        psi /= np.linalg.norm(psi)
        prof = profile_of(psi, maps)
        loc = tuple(prof[cuts.index(S)] for S in cuts if len(S) == 1)
        key = ("".join(str(prof[cuts.index(S)]) for S in pair), loc)
        seen[key] = seen.get(key, 0) + 1
    print("psi = eta_{01} (x) theta_{234}, random; profiles reached")
    print(f"pair-cut order: {' '.join(f'{S[0]}{S[1]}' for S in pair)}")
    for (p, loc), c in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"   pairs {p}   local {''.join(map(str, loc))}   x{c}")


def n5_open(restarts: int = 600, seed: int = 11) -> None:
    """Re-run only the profiles left OPEN by --n5, with a bigger budget and a fresh
    seed.  A profile that survives this is much more likely to be genuinely
    unachievable than a bad local minimum."""
    dims = (2,) * 5
    cuts = canonical_cuts(5)
    pair = [S for S in cuts if len(S) == 2]
    text = (HERE / "artifacts" / "rank_witness_n5.log").read_text()
    mo = re.search(r"^open: \[(.*)\]$", text, re.M)
    todo = re.findall(r"'(\d{10})'", mo.group(1)) if mo else []
    print(f"{len(todo)} open profiles, {restarts} restarts each, seed {seed}")
    print(f"pair-cut order: {' '.join(f'{S[0]}{S[1]}' for S in pair)}")
    t0 = time.time()
    found = []
    for lab in todo:
        prof = {S: 2 for S in cuts if len(S) == 1}
        prof.update({S: int(ch) for S, ch in zip(pair, lab)})
        ranks = tuple(prof[S] for S in cuts)
        ok, _, f = search(dims, cuts, ranks, restarts=restarts, seed=seed)
        if ok:
            found.append(lab)
        print(f"{lab:>12}  {'FOUND' if ok else 'none':>8}  {f:>10.2e}", flush=True)
    print(f"\nnewly settled achievable: {len(found)}/{len(todo)}  "
          f"({time.time() - t0:.0f}s)")
    print(f"still open: {[l for l in todo if l not in found]}")


def merged_witness(restarts: int = 200, seed: int = 3) -> None:
    """Does Segre's 'max attained at least twice' survive a party of dimension 4?

    The relaxation on 2x2x2x4 (rank_profile_blocks.py --merged) calls several
    max-attained-once triples feasible once r_F >= 3.  Feasible is only a necessary
    condition, so it does not by itself refute the rule -- a witness does."""
    dims = (2, 2, 2, 4)
    cuts = canonical_cuts(4)
    sing = [S for S in cuts if len(S) == 1]
    pair = [S for S in cuts if len(S) == 2]
    print("dims (2,2,2,4), local ranks (2,2,2,r_F); triples on (AB, AC, AF)")
    print(f"{'r_F':>4}  {'triple':>10}  {'max 2x?':>8}  {'witness':>8}  {'best f':>10}")
    for rF, trip in [(2, (3, 3, 4)), (3, (3, 3, 4)), (4, (3, 3, 4)),
                     (3, (2, 2, 3)), (4, (2, 2, 3)), (4, (2, 2, 4)),
                     (3, (3, 4, 4)), (4, (3, 4, 4)), (3, (2, 3, 4))]:
        prof = dict(zip(sing, (2, 2, 2, rF)))
        prof.update(dict(zip(pair, trip)))
        ranks = tuple(prof[S] for S in cuts)
        ok, _, f = search(dims, cuts, ranks, restarts=restarts, seed=seed)
        twice = "yes" if trip.count(max(trip)) >= 2 else "no"
        print(f"{rF:>4}  {trip!s:>10}  {twice:>8}  {'FOUND' if ok else 'none':>8}  "
              f"{f:>10.2e}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--n5" in sys.argv:
        n5()
    elif "--product" in sys.argv:
        product_forcing()
    elif "--merged" in sys.argv:
        merged_witness()
    elif "--n5-open" in sys.argv:
        n5_open()
    else:
        raise SystemExit("pass --check, --n5, --n5-open, --product or --merged")
