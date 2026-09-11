"""Batched torch optimisation of the marginal-purity functional (R3).

For a unit vector psi in C^2 (x) (C^d)^{(x)k} (index 0 = qubit, 1..k = qudits),

    D_c(psi) = sum_{R subset [k]} (-c)^{k-|R|} Tr[(rho^psi_R)^2],

which by (R3) equals B_c(psi, psi), the diagonal of the biquadratic form.  D_c >= 0
for all psi is NECESSARY for k-copy undistillability of the Werner state rho_{-c},
so any psi with D_c(psi) < 0 is a genuine certificate of k-copy distillability.

The equivalent "rank-two operator" form used in the literature is
    q_k(alpha, C) = sum_{S subset [k]} alpha^{|S|} ||Tr_S C||_HS^2,   rank C <= 2,
and D_c is the sub-case C = C^dagger-like coming from chi = psi.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch


def subsets(k):
    out = []
    for r in range(k + 1):
        out.extend(combinations(range(1, k + 1), r))
    return out


def purities(psi: torch.Tensor, d: int, k: int) -> dict[tuple, torch.Tensor]:
    """Tr[rho_R^2] for every R subset of the k qudits.  psi: (batch, 2, d,...,d)."""
    batch = psi.shape[0]
    out = {}
    for R in subsets(k):
        rest = [a for a in range(1, k + 2) if (a - 1) not in [r - 1 for r in R]]
        # axes of psi: 0=batch, 1=qubit, 2..k+1 = qudits 1..k
        Rax = [r + 1 for r in R]
        restax = [a for a in range(1, k + 2) if a not in Rax]
        M = psi.permute([0] + Rax + restax).reshape(batch, d ** len(R), -1)
        G = M @ M.conj().transpose(1, 2)          # rho_R
        out[R] = torch.einsum("bij,bji->b", G, G).real
    return out


def D_c(psi: torch.Tensor, d: int, k: int, c: float) -> torch.Tensor:
    P = purities(psi, d, k)
    tot = 0.0
    for R, p in P.items():
        tot = tot + ((-c) ** (k - len(R))) * p
    return tot


def minimise_D(d: int, k: int, c: float, batch: int = 512, steps: int = 4000,
               seed: int = 0, lr: float = 0.05, device: str = "cpu",
               init: np.ndarray | None = None):
    """Batched Adam on D_c over unit psi.  Returns (best value, best psi)."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, 2) + (d,) * k
    if init is None:
        x = torch.randn(shape + (2,), generator=g, device=device, dtype=torch.float64)
    else:
        x = torch.tensor(init, device=device, dtype=torch.float64)
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best, best_psi = np.inf, None
    for step in range(steps):
        opt.zero_grad()
        psi = torch.view_as_complex(x)
        psi = psi / psi.reshape(batch, -1).norm(dim=1).reshape((batch,) + (1,) * (k + 1))
        vals = D_c(psi, d, k, c)
        loss = vals.sum()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                finite = torch.isfinite(vals)
                if not finite.any():
                    continue
                masked = torch.where(finite, vals, torch.full_like(vals, np.inf))
                i = int(masked.argmin())
                v = float(masked[i])
                cand = psi[i].detach().cpu().numpy().copy()
                # Re-evaluate the extracted state on its own: guards against a
                # reported minimum that does not belong to the returned vector
                # (degenerate/near-zero-norm batch elements).
                nrm = np.linalg.norm(cand)
                if not np.isfinite(nrm) or nrm < 1e-8:
                    continue
                cand = cand / nrm
                recheck = float(D_c(torch.tensor(cand)[None], d, k, c).item())
                if abs(recheck - v) > 1e-8 * max(1.0, abs(v)):
                    v = recheck  # trust the independently re-evaluated state
                if v < best:
                    best, best_psi = v, cand
    return best, best_psi
