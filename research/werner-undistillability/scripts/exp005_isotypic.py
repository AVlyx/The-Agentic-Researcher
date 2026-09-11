"""E005: Schur-Weyl structured search -- restrict psi and chi to S_k-isotypic sectors.

The form B_c(psi, chi) = <psi (x) chi| F_0 (x) (I - cF)^{(x)k} |psi (x) chi> is
invariant under the diagonal action of U(d) on the k qudits and under the simultaneous
permutation of the k copies.  Its extremisers therefore come in G-orbits, and highly
symmetric configurations are a natural place for a counterexample to hide -- exactly the
kind of structure ("uniform equal-norm tight frames") used by Tabia-Chen-Hsieh to break
the analogous conjecture for the wider DiVincenzo family at two copies.

Schur-Weyl duality decomposes (C^d)^{(x)k} = sum_lambda (S^lambda (x) V_lambda) into
S_k-isotypic components indexed by partitions lambda of k, of dimension
dim_specht(lambda) * dim_weyl(lambda, d).  We take the projectors P_lambda from the
schur-weyl package, restrict psi to C^2 (x) range(P_lambda_psi) and chi to
C^2 (x) range(P_lambda_chi), and minimise theta over every ordered pair of sectors.

This is a valid one-sided search: any negative value found is a genuine certificate of
k-copy distillability; finding none strengthens the evidence for the endpoint conjecture
in precisely the symmetric directions where it is most at risk.

Output: data/exp005.json
"""
from __future__ import annotations

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402
from bilinear import R_batch, _psi_from_stiefel, _select_verified  # noqa: E402
from schur_weyl import partitions, isotypic_proj, dim_specht, dim_weyl  # noqa: E402

torch.set_num_threads(8)
HERE = os.path.join(os.path.dirname(__file__), "..")


def sector_basis(lam, d):
    """Orthonormal basis (columns) of the range of the S_k-isotypic projector P_lam."""
    P = np.asarray(isotypic_proj(lam, d), dtype=float)
    r = int(round(np.trace(P)))
    w, V = np.linalg.eigh(P)
    B = V[:, -r:] if r > 0 else V[:, :0]
    return B, r


def minimise_in_sectors(d, k, c, Bp, Bq, batch=192, steps=2500, seed=3, lr=0.03):
    """theta restricted to psi in C^2 (x) range(Bp), chi in C^2 (x) range(Bq)."""
    Tp = torch.tensor(Bp, dtype=torch.complex128)
    Tq = torch.tensor(Bq, dtype=torch.complex128)
    np_, nq = Bp.shape[1], Bq.shape[1]
    g = torch.Generator().manual_seed(seed)
    xm = torch.randn((batch, 2, np_, 2), generator=g, dtype=torch.float64)
    xc = torch.randn((batch, 2, nq, 2), generator=g, dtype=torch.float64)
    xm.requires_grad_(True)
    xc.requires_grad_(True)
    opt = torch.optim.Adam([xm, xc], lr=lr)
    best, arg = np.inf, None
    for step in range(steps):
        opt.zero_grad()
        M = torch.view_as_complex(xm) @ Tp.T          # (batch,2,d^k)
        psi = _psi_from_stiefel(M, d, k)
        chi = torch.view_as_complex(xc) @ Tq.T
        chi = chi.reshape(batch, -1)
        chi = chi / chi.norm(dim=1, keepdim=True)
        chi = chi.reshape((batch, 2) + (d,) * k)
        Bv, A0 = R_batch(psi, chi, d, k, c)
        vals = 2.0 * Bv
        vals.sum().backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                best, arg = _select_verified(psi, chi, vals, A0, d, k, c, best, arg)
    return best, arg


CASES = [(3, 3), (4, 3), (3, 4), (2, 5), (3, 5)]
CS = [0.50, 0.48]
res = []
for (d, k) in CASES:
    if 2 * d**k > 1600:
        continue
    lams = [lam for lam in partitions(k) if dim_weyl(lam, d) > 0]
    bases = {}
    for lam in lams:
        Bmat, r = sector_basis(lam, d)
        assert r == dim_specht(lam) * dim_weyl(lam, d), (lam, r)
        bases[lam] = Bmat
    tot = sum(b.shape[1] for b in bases.values())
    print(f"# d={d} k={k}: sectors "
          f"{ {str(l): bases[l].shape[1] for l in lams} } total={tot} (= d^k = {d**k})",
          flush=True)
    assert tot == d**k
    for c in CS:
        for lp in lams:
            if bases[lp].shape[1] < 2:
                # psi needs a rank-2 qubit marginal for the normalisation
                # rho^psi_0 = I/2; a one-dimensional sector cannot supply one.
                continue
            for lq in lams:
                t0 = time.time()
                v, arg = minimise_in_sectors(d, k, c, bases[lp], bases[lq])
                psi, chi = arg
                p, q = psi.reshape(-1), chi.reshape(-1)
                Bv = W.B_form(p, q, d, k, c)
                A0 = float(np.real(q.conj() @ W.qubit_marginal_op(p, d, k) @ q))
                chk = Bv / A0
                rec = dict(d=d, k=k, c=c, lam_psi=str(lp), lam_chi=str(lq),
                           theta=v, verified=chk, secs=round(time.time() - t0, 1))
                res.append(rec)
                flag = "  *** NEGATIVE ***" if chk < -1e-9 else ""
                print(f"d={d} k={k} c={c:.2f} psi~{str(lp):10s} chi~{str(lq):10s} "
                      f"theta={v:+.8f} verified={chk:+.8f}{flag}", flush=True)
                if chk < -1e-9:
                    np.save(os.path.join(HERE, "data",
                            f"iso_witness_d{d}_k{k}_c{c}.npy"), np.stack([psi, chi]))
                with open(os.path.join(HERE, "data", "exp005.json"), "w") as f:
                    json.dump(res, f, indent=1)

print("\nwrote data/exp005.json")
