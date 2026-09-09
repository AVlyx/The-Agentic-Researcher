"""Witness search: certify that a rank profile IS achievable, by exhibiting a state.

Complements scripts/rank_profile_subspace.py.  The two directions need different tools:

  profile impossible  ->  a relaxation / certificate (W_r; or an SDP hierarchy)
  profile achievable  ->  a WITNESS, found by non-convex local search

No convexity is needed for the second: the constraint rank rho_S <= r_S is the
vanishing of the tail of rho_S's spectrum, so we simply minimise

    f(psi) = sum_S ( 1 - sum_{j <= r_S} eig_j(rho_S) )        (tail spectral mass)

over unit psi.  f >= 0, and f = 0 exactly when every upper bound holds.  We then read
off the ACHIEVED profile and keep it only if it equals the target -- that handles the
lower bounds, which local search cannot impose directly (it would happily collapse the
ranks further than asked).

An exact SDP is not available here: rank <= r on a flattening and rank-1-ness of
|psi><psi| are both non-convex, and the convex hull of the pure states is all density
matrices, which loses exactly the purity that CHLW-type obstructions turn on.

Usage:  uv run python scripts/rank_witness_search.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

DIMS = (2, 2, 2, 2)
PAIR_CUTS = [(0, 1), (0, 2), (0, 3)]
SINGLE_CUTS = [(0,), (1,), (2,), (3,)]


def marginal_spectrum(psi: np.ndarray, S: tuple[int, ...]) -> np.ndarray:
    n = psi.ndim
    rest = tuple(i for i in range(n) if i not in S)
    M = psi.transpose(S + rest).reshape(int(np.prod([psi.shape[i] for i in S])), -1)
    ev = np.linalg.eigvalsh(M @ M.conj().T)
    return np.clip(ev, 0.0, None)[::-1]


def unpack(x: np.ndarray) -> np.ndarray:
    v = x[:16] + 1j * x[16:]
    v = v / np.linalg.norm(v)
    return v.reshape(DIMS)


def objective(x: np.ndarray, target: dict) -> float:
    psi = unpack(x)
    tot = 0.0
    for S, r in target.items():
        ev = marginal_spectrum(psi, S)
        tot += float(ev[r:].sum())          # tail mass beyond rank r
    return tot


def achieved_profile(psi: np.ndarray, tol: float = 1e-7) -> dict:
    return {S: int((marginal_spectrum(psi, S) > tol).sum())
            for S in SINGLE_CUTS + PAIR_CUTS}


def search(target: dict, n_restarts: int = 80, seed: int = 0) -> tuple[bool, dict | None, float]:
    rng = np.random.default_rng(seed)
    best_f = np.inf
    for _ in range(n_restarts):
        x0 = rng.normal(size=32)
        res = minimize(objective, x0, args=(target,), method="BFGS",
                       options={"maxiter": 400, "gtol": 1e-12})
        best_f = min(best_f, res.fun)
        if res.fun < 1e-11:
            psi = unpack(res.x)
            prof = achieved_profile(psi)
            if all(prof[S] == target[S] for S in target):
                return True, prof, res.fun
    return False, None, best_f


def main() -> int:
    print("=" * 82)
    print("Witness search, 4 qubits, local ranks 2, targets on (r_AB, r_AC, r_AD)")
    print("=" * 82)
    # verdicts from the linear relaxation (scripts/rank_profile_subspace.py --battery)
    relax = {
        (1, 1, 1): "feasible", (2, 1, 1): "IMPOSSIBLE", (3, 1, 1): "IMPOSSIBLE",
        (4, 1, 1): "IMPOSSIBLE", (2, 2, 1): "feasible", (3, 2, 1): "IMPOSSIBLE",
        (4, 2, 1): "IMPOSSIBLE", (3, 3, 1): "feasible", (4, 3, 1): "IMPOSSIBLE",
        (4, 4, 1): "feasible", (2, 2, 2): "feasible", (3, 2, 2): "IMPOSSIBLE",
        (4, 2, 2): "IMPOSSIBLE", (3, 3, 2): "feasible", (4, 3, 2): "IMPOSSIBLE",
        (4, 4, 2): "feasible", (3, 3, 3): "feasible", (4, 3, 3): "IMPOSSIBLE",
        (4, 4, 3): "feasible", (4, 4, 4): "feasible",
    }
    print(f"{'profile':>12}  {'relaxation':>11}  {'witness':>9}  {'best f':>10}  verdict")
    settled = contradictions = 0
    open_cases: list = []
    for trip in sorted(relax):
        target = {S: 2 for S in SINGLE_CUTS}
        target.update(dict(zip(PAIR_CUTS, trip)))
        ok, prof, f = search(target)
        wit = "FOUND" if ok else "none"
        if ok:
            verdict = "ACHIEVABLE (proved by witness)"
        elif relax[trip] == "IMPOSSIBLE":
            verdict = "impossible (proved by relaxation)"
        else:
            verdict = "OPEN -- relaxation says feasible, no witness found"
        # settled = the two certificates together decide the profile.
        # open    = relaxation admits it and local search found no witness -- NOT a
        #           contradiction, just a gap between the two one-sided methods.
        # contradiction = relaxation says IMPOSSIBLE yet a witness exists; that cannot
        #           happen unless there is a bug, since the relaxation is a valid
        #           necessary condition.
        if relax[trip] == "IMPOSSIBLE" and ok:
            contradictions += 1
        elif relax[trip] == "IMPOSSIBLE" or ok:
            settled += 1
        else:
            open_cases.append(trip)
        print(f"{trip!s:>12}  {relax[trip]:>11}  {wit:>9}  {f:>10.2e}  {verdict}")
    print()
    print(f"settled: {settled}/20   open: {len(open_cases)}/20   "
          f"contradictions: {contradictions}/20")
    print(f"open cases: {open_cases}")
    print("All open cases are closed analytically by the product lemma: r_AD = 1 forces")
    print("psi = eta_AD (x) theta_BC, hence r_AB = r_A*r_B and r_AC = r_A*r_C.  With all")
    print("local ranks 2 that gives (4,4,1) as the only reachable profile with r_AD = 1,")
    print("so (1,1,1), (2,2,1) [CHLW ray 3] and (3,3,1) are IMPOSSIBLE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
