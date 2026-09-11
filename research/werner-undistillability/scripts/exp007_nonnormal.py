"""E007: search the sector left OPEN by Wu-Zou for a three-copy counterexample.

Wu and Zou (arXiv:2608.02647, 31 July 2026) prove the three-copy endpoint form
q_3(-1/2, C) >= 0 for all positive semidefinite rank-two C, all NORMAL rank-two C, and
some nonnormal cases with geometric constraints.  What they leave open is the genuinely
NONNORMAL rank-two sector with full local support at every site.

This matters for our earlier searches.  The equality cases we kept finding (padded
one-copy optimiser, GHZ) all have chi = psi, hence C = sum_a |psi_a><psi_a| >= 0 -- they
sit in the positive semidefinite sector, which is already proved.  A generic optimiser is
attracted there, so it may never probe the open sector at all.

Here we work directly with C = A B^dag (A, B of shape D x 2, D = d^3) and minimise
    theta(C) = q_3(-1/2, C)/||C||^2,   q_k(alpha,C) = sum_S alpha^{|S|} ||Tr_S C||^2,
subject to a lower bound on the normalised nonnormality
    nu(C) = || C C^dag - C^dag C ||^2 / ||C||^4 ,
enforced by a penalty.  nu = 0 exactly on normal C, so sweeping nu_0 upward walks the
search away from the proved sector and into the open one.

A verified negative value would refute the three-copy endpoint conjecture and prove
c_3 < 1/2.

Output: data/exp007.json
"""
from __future__ import annotations

import sys, os, json, time
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
HERE = os.path.join(os.path.dirname(__file__), "..")

K = 3


def ptrace_t(C, d, S):
    """Partial trace of a batch of operators on (C^d)^{(x)3} over the sites in S."""
    b = C.shape[0]
    T = C.reshape((b,) + (d,) * K + (d,) * K)
    dims = list(range(K))
    k = K
    for i in sorted(S, reverse=True):
        pos = dims.index(i)
        T = torch.diagonal(T, dim1=1 + pos, dim2=1 + k + pos).sum(-1)
        dims.pop(pos)
        k -= 1
    D = d ** len(dims) if dims else 1
    return T.reshape(b, D, D)


def q3(C, d, alpha):
    tot = 0.0
    for r in range(K + 1):
        for S in combinations(range(K), r):
            X = ptrace_t(C, d, S)
            tot = tot + (alpha ** r) * (X.abs() ** 2).sum((1, 2))
    return tot


def nonnormality(C):
    M = C @ C.conj().transpose(1, 2) - C.conj().transpose(1, 2) @ C
    return (M.abs() ** 2).sum((1, 2))


def run(d, nu0, batch=96, steps=2500, seed=0, lr=0.02, lam=30.0):
    D = d ** K
    g = torch.Generator().manual_seed(seed)
    xa = torch.randn((batch, D, 2, 2), generator=g)
    xb = torch.randn((batch, D, 2, 2), generator=g)
    xa.requires_grad_(True)
    xb.requires_grad_(True)
    opt = torch.optim.Adam([xa, xb], lr=lr)
    best, arg = np.inf, None
    for step in range(steps):
        opt.zero_grad()
        A = torch.view_as_complex(xa)
        B = torch.view_as_complex(xb)
        C = A @ B.conj().transpose(1, 2)
        n2 = (C.abs() ** 2).sum((1, 2)).clamp_min(1e-12)
        theta = q3(C, d, -0.5) / n2
        nu = nonnormality(C) / n2 ** 2
        pen = torch.clamp(nu0 - nu, min=0.0) ** 2
        (theta + lam * pen).sum().backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                okmask = torch.isfinite(theta) & (nu >= nu0 - 1e-6)
                if not okmask.any():
                    continue
                masked = torch.where(okmask, theta, torch.full_like(theta, np.inf))
                i = int(masked.argmin())
                if float(masked[i]) < best:
                    best = float(masked[i])
                    arg = (C[i].detach().numpy().copy(), float(nu[i]))
    return best, arg


def verify_numpy(C, d):
    """Independent re-evaluation of theta and the sector diagnostics."""
    def pt(M, S):
        T = M.reshape((d,) * K + (d,) * K)
        dims = list(range(K))
        k = K
        for i in sorted(S, reverse=True):
            pos = dims.index(i)
            T = np.trace(T, axis1=pos, axis2=k + pos)
            dims.pop(pos)
            k -= 1
        D = d ** len(dims) if dims else 1
        return T.reshape(D, D)
    n2 = np.linalg.norm(C) ** 2
    tot = 0.0
    for r in range(K + 1):
        for S in combinations(range(K), r):
            tot += ((-0.5) ** r) * np.linalg.norm(pt(C, S)) ** 2
    nu = np.linalg.norm(C @ C.conj().T - C.conj().T @ C) ** 2 / n2 ** 2
    herm = np.linalg.norm(C - C.conj().T) / np.sqrt(n2)
    ev = np.linalg.eigvals(C)
    psd_like = float(np.min(ev.real)) / (abs(ev).max() + 1e-15)
    # local support ranks: rank of the reduced "output" operator at each site
    ranks = []
    for i in range(K):
        R = pt(C @ C.conj().T, tuple(j for j in range(K) if j != i))
        ranks.append(int(np.linalg.matrix_rank(R, tol=1e-8)))
    return tot / n2, nu, herm, psd_like, ranks


CASES = [(3, nu0) for nu0 in (0.0, 0.2, 0.5, 1.0, 1.5)] + \
        [(4, nu0) for nu0 in (0.5, 1.0, 1.5)] + \
        [(2, nu0) for nu0 in (0.5, 1.0)]

res = []
for (d, nu0) in CASES:
    t0 = time.time()
    v, arg = run(d, nu0, seed=17 + int(10 * nu0))
    if arg is None:
        print(f"d={d} nu0={nu0}: no feasible point", flush=True)
        continue
    C, nu_t = arg
    theta, nu, herm, psd_like, ranks = verify_numpy(C, d)
    rec = dict(d=d, nu0=nu0, theta_torch=v, theta_verified=theta, nonnormality=nu,
               nonhermiticity=herm, min_eig_ratio=psd_like, local_ranks=ranks,
               secs=round(time.time() - t0, 1))
    res.append(rec)
    flag = "  *** NEGATIVE ***" if theta < -1e-9 else ""
    print(f"d={d} nu0={nu0:.1f}  theta={theta:+.9f}  nu={nu:.4f}  "
          f"||C-C^dag||/||C||={herm:.3f}  local ranks={ranks}{flag}", flush=True)
    if theta < -1e-9:
        np.save(os.path.join(HERE, "data", f"nonnormal_ce_d{d}_nu{nu0}.npy"), C)
    with open(os.path.join(HERE, "data", "exp007.json"), "w") as f:
        json.dump(res, f, indent=1)

print("\nwrote data/exp007.json")
print(f"verified negative values: {len([r for r in res if r['theta_verified'] < -1e-9])}")
