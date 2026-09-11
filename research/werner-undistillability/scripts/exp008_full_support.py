"""E008: is the three-copy endpoint form bounded AWAY from zero on the open sector?

E007 found something suggestive: every configuration achieving q_3(-1/2,C) = 0 had at least
one site whose local reduction has rank 2, while forcing genuine nonnormality pushed the
value strictly positive.  Wu-Zou (arXiv:2608.02647) prove the endpoint for positive
semidefinite and normal rank-two C, and additionally whenever the "local output-input
support overlap is <= 2 at some site".  So the equality set may lie entirely inside their
proved region, and the sector they leave open -- full local support at every site -- may
carry a strictly positive margin.

Here we test that.  For rank-two C = A B^dag on (C^d)^{(x)3} define the local reductions
    R_i^out = Tr_{j != i} (C C^dag),      R_i^in = Tr_{j != i} (C^dag C),
and impose the full-support margin
    lambda_min(R) / Tr(R) >= eps   for all six reductions
(the maximum possible value is 1/d, attained when a reduction is maximally mixed).  We then
minimise theta(C) = q_3(-1/2,C)/||C||^2 subject to that margin, sweeping eps.

Note this is a PROXY for Wu-Zou's condition, not a restatement of it: full local rank at
every site is necessary for their overlap to exceed 2, not equivalent to it.  A margin
delta(eps) > 0 here would say the open sector is quantitatively easy, which combined with
their result would be a route to closing k=3.  A verified negative value would instead
refute the endpoint conjecture.

Output: data/exp008.json
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


def support_margins(C, d):
    """min over the six local reductions of lambda_min(R)/Tr(R)."""
    out = []
    for M in (C @ C.conj().transpose(1, 2), C.conj().transpose(1, 2) @ C):
        for i in range(K):
            R = ptrace_t(M, d, tuple(j for j in range(K) if j != i))
            R = 0.5 * (R + R.conj().transpose(1, 2))
            w = torch.linalg.eigvalsh(R)
            out.append(w[:, 0] / w.sum(1).clamp_min(1e-18))
    return torch.stack(out, 1).min(1).values


def run(d, eps, batch=96, steps=3000, seed=0, lr=0.02, lam=200.0):
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
        m = support_margins(C, d)
        pen = torch.clamp(eps - m, min=0.0) ** 2
        (theta + lam * pen).sum().backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                feas = torch.isfinite(theta) & (m >= eps - 1e-7)
                if not feas.any():
                    continue
                masked = torch.where(feas, theta, torch.full_like(theta, np.inf))
                i = int(masked.argmin())
                if float(masked[i]) < best:
                    best = float(masked[i])
                    arg = (C[i].detach().numpy().copy(), float(m[i]))
    return best, arg


def verify_numpy(C, d):
    def pt(M, S):
        T = M.reshape((d,) * K + (d,) * K)
        dims = list(range(K))
        k = K
        for i in sorted(S, reverse=True):
            pos = dims.index(i)
            T = np.trace(T, axis1=pos, axis2=k + pos)
            dims.pop(pos)
            k -= 1
        return T.reshape(d ** len(dims) if dims else 1, -1)
    n2 = np.linalg.norm(C) ** 2
    tot = 0.0
    for r in range(K + 1):
        for S in combinations(range(K), r):
            tot += ((-0.5) ** r) * np.linalg.norm(pt(C, S)) ** 2
    margins = []
    ranks = []
    for M in (C @ C.conj().T, C.conj().T @ C):
        for i in range(K):
            R = pt(M, tuple(j for j in range(K) if j != i))
            R = 0.5 * (R + R.conj().T)
            w = np.linalg.eigvalsh(R)
            margins.append(w[0] / w.sum())
            ranks.append(int((w > 1e-9 * max(w.max(), 1e-30)).sum()))
    nu = np.linalg.norm(C @ C.conj().T - C.conj().T @ C) ** 2 / n2 ** 2
    return tot / n2, float(min(margins)), ranks, nu


CASES = [(3, e) for e in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)] + \
        [(4, e) for e in (0.05, 0.10, 0.15, 0.20)]

res = []
for (d, eps) in CASES:
    t0 = time.time()
    v, arg = run(d, eps, seed=23 + int(100 * eps))
    if arg is None:
        print(f"d={d} eps={eps}: no feasible point found", flush=True)
        continue
    C, m_t = arg
    theta, margin, ranks, nu = verify_numpy(C, d)
    rec = dict(d=d, eps=eps, theta=theta, margin=margin, local_ranks=ranks,
               nonnormality=nu, secs=round(time.time() - t0, 1))
    res.append(rec)
    flag = "  *** NEGATIVE ***" if theta < -1e-9 else ""
    print(f"d={d} eps={eps:.2f} (max {1/d:.3f})  min theta={theta:+.9f}  "
          f"margin={margin:.4f}  ranks={ranks}  nu={nu:.3f}{flag}", flush=True)
    if theta < -1e-9:
        np.save(os.path.join(HERE, "data", f"fs_ce_d{d}_eps{eps}.npy"), C)
    with open(os.path.join(HERE, "data", "exp008.json"), "w") as f:
        json.dump(res, f, indent=1)

print("\nwrote data/exp008.json")
print(f"verified negative values: {len([r for r in res if r['theta'] < -1e-9])}")
