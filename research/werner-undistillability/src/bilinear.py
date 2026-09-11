"""Batched global minimisation of the scale-invariant ratio

    R_c(psi, chi) = B_c(psi, chi) / A_0(psi, chi),
    B_c(psi, chi) = sum_{S subset [k]} (-c)^{|S|} Tr[rho^psi_{0S} rho^chi_{0S}],
    A_0(psi, chi) = Tr[rho^psi_0 rho^chi_0],

over psi, chi in C^2 (x) (C^d)^{(x)k}.  Under the correspondence
C = sum_a |chi_a><psi_a| this is exactly q_k(-c, C)/||C||^2 on rank-<=2 operators,
so theta_k(c) := min R_c is the quantity whose sign decides k-copy distillability
of the Werner state rho_{-c}:  theta_k(c) < 0  <=>  k-copy distillable.

The alternating (see-saw) iteration in werner.py is a strict descent method but
gets trapped: at d=3, k=2, c=0.4 it returns 0.20 from random starts while the
padded one-copy optimiser already gives 0.12.  Joint gradient descent on both
arguments, batched over many restarts and seeded with the known structured
extremisers, is far more reliable.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch


def _subsets(k):
    out = []
    for r in range(k + 1):
        out.extend(combinations(range(1, k + 1), r))
    return out


def R_batch(psi: torch.Tensor, chi: torch.Tensor, d: int, k: int, c: float):
    """psi, chi: (batch, 2, d, ..., d) complex.  Returns (B, A0) per batch item."""
    batch = psi.shape[0]
    B = psi.new_zeros(batch, dtype=torch.float64)
    A0 = None
    for S in _subsets(k):
        if len(S) == k:
            # Nothing is traced out, so rho_{0S} = |psi><psi| and the term is just
            # |<psi|chi>|^2.  Computing it as a (2 d^k) x (2 d^k) matrix product, as the
            # generic branch would, dominates the whole cost for larger d and k.
            ov = (psi.reshape(batch, -1).conj() * chi.reshape(batch, -1)).sum(1)
            term = (ov.conj() * ov).real
        else:
            Sax = [s + 1 for s in S]
            rest = [a for a in range(2, k + 2) if a not in Sax]
            pv = psi.permute([0, 1] + Sax + rest).reshape(batch, 2 * d ** len(S), -1)
            cv = chi.permute([0, 1] + Sax + rest).reshape(batch, 2 * d ** len(S), -1)
            rp = pv @ pv.conj().transpose(1, 2)
            rc = cv @ cv.conj().transpose(1, 2)
            term = torch.einsum("bij,bji->b", rp, rc).real
        B = B + ((-c) ** len(S)) * term
        if len(S) == 0:
            A0 = term
    return B, A0


def minimise_R(d: int, k: int, c: float, batch: int = 256, steps: int = 4000,
               seed: int = 0, lr: float = 0.03, inits: list[np.ndarray] | None = None,
               diagonal: bool = False):
    """Adam on R_c over (psi, chi).  `inits` are extra structured starting psi
    (each used with chi = psi).  If diagonal, force chi = psi."""
    g = torch.Generator().manual_seed(seed)
    shape = (batch, 2) + (d,) * k
    xp = torch.randn(shape + (2,), generator=g, dtype=torch.float64)
    xc = torch.randn(shape + (2,), generator=g, dtype=torch.float64)
    if inits:
        for i, v in enumerate(inits[:batch]):
            t = torch.tensor(np.stack([v.real, v.imag], -1).reshape(shape[1:] + (2,)))
            xp[i] = t
            xc[i] = t
    params = [xp] if diagonal else [xp, xc]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    best, arg = np.inf, None
    for step in range(steps):
        opt.zero_grad()
        psi = torch.view_as_complex(xp)
        chi = psi if diagonal else torch.view_as_complex(xc)
        B, A0 = R_batch(psi, chi, d, k, c)
        vals = B / A0.clamp_min(1e-12)
        vals.sum().backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                i = int(vals.argmin())
                if vals[i].item() < best:
                    best = vals[i].item()
                    arg = (psi[i].detach().numpy().copy(),
                           chi[i].detach().numpy().copy())
    return best, arg


# ---------------------------------------------------------- structured states


def ghz(d, k):
    v = np.zeros((2,) + (d,) * k, dtype=complex)
    v[(0,) + (0,) * k] = 1 / np.sqrt(2)
    v[(1,) + (1,) * k] = 1 / np.sqrt(2)
    return v


def padded(d, k, m=1):
    """One-copy optimiser entangled with the first m qudits, product on the rest."""
    v = np.zeros((2,) + (d,) * k, dtype=complex)
    v[(0,) + (0,) * k] = 1 / np.sqrt(2)
    v[(1,) + (1,) * m + (0,) * (k - m)] = 1 / np.sqrt(2)
    return v


def structured(d, k):
    out = [ghz(d, k)] + [padded(d, k, m) for m in range(1, k + 1)]
    return out


# --------------------------------------------------- degeneracy-free formulation
#
# B_c is invariant under the filtering pair (psi, chi) -> ((A (x) I) psi,
# (A^{-dag} (x) I) chi) for any invertible 2x2 A: the marginals transform as
# rho^psi_{0S} -> A rho^psi_{0S} A^dag and rho^chi_{0S} -> A^{-dag} rho^chi_{0S} A^{-1},
# so every trace Tr[rho^psi_{0S} rho^chi_{0S}] is unchanged.  Whenever rho^psi_0 has
# full rank we may therefore normalise
#
#     rho^psi_0 = I/2,   ||chi|| = 1     =>     A_0 = Tr[rho^psi_0 rho^chi_0] = 1/2,
#
# and then theta = R_c = B_c / A_0 = 2 B_c.  This matters: minimising the raw ratio
# B_c/A_0 lets the optimiser drive A_0 -> 0, where B_c -> 0 too, and the 0/0 produces
# spurious large negative values.  Fixing rho^psi_0 removes that failure mode.


def _inv_sqrt_2x2(G: torch.Tensor) -> torch.Tensor:
    """G^{-1/2} for a batch of 2x2 Hermitian positive definite matrices.

    Closed form, so it is smooth and has no eigenvector phase ambiguity (which
    makes torch.linalg.eigh non-differentiable in the complex case):
    with t = tr G and s = sqrt(det G),  sqrt(G) = (G + s I)/sqrt(t + 2s), and
    det(G + s I) = s(t + 2s), hence  G^{-1/2} = adj(G + s I) / (s sqrt(t + 2s)).
    """
    a, b = G[:, 0, 0], G[:, 0, 1]
    cc, dd = G[:, 1, 0], G[:, 1, 1]
    t = (a + dd).real
    s = (a * dd - b * cc).real.clamp_min(1e-24).sqrt()
    A = a + s.to(G.dtype)
    D = dd + s.to(G.dtype)
    adj = torch.stack([torch.stack([D, -b], -1), torch.stack([-cc, A], -1)], -2)
    denom = (s * (t + 2 * s).clamp_min(1e-24).sqrt()).to(G.dtype)
    return adj / denom[:, None, None]


def _psi_from_stiefel(M: torch.Tensor, d: int, k: int) -> torch.Tensor:
    """M: (batch, 2, d^k) complex -> psi with rho^psi_0 = I/2."""
    G = M @ M.conj().transpose(1, 2)                       # (batch,2,2) Hermitian PD
    V = _inv_sqrt_2x2(G) @ M                               # V V^dag = I
    batch = M.shape[0]
    return (V / np.sqrt(2.0)).reshape((batch, 2) + (d,) * k)


def minimise_theta(d: int, k: int, c: float, batch: int = 256, steps: int = 3000,
                   seed: int = 0, lr: float = 0.03,
                   inits: list[np.ndarray] | None = None):
    """theta_k(c) = min over rank-<=2 C of q_k(-c,C)/||C||^2, computed in the
    degeneracy-free parametrisation.  Returns (theta, (psi, chi))."""
    g = torch.Generator().manual_seed(seed)
    D = d**k
    xm = torch.randn((batch, 2, D, 2), generator=g, dtype=torch.float64)
    xc = torch.randn((batch, 2, D, 2), generator=g, dtype=torch.float64)
    if inits:
        for i, v in enumerate(inits[:batch]):
            t = torch.tensor(np.stack([v.real, v.imag], -1).reshape(2, D, 2))
            xm[i] = t * np.sqrt(2.0)
            xc[i] = t
    xm.requires_grad_(True)
    xc.requires_grad_(True)
    opt = torch.optim.Adam([xm, xc], lr=lr)
    best, arg = np.inf, None
    for step in range(steps):
        opt.zero_grad()
        psi = _psi_from_stiefel(torch.view_as_complex(xm), d, k)
        chi = torch.view_as_complex(xc)
        chi = chi / chi.reshape(batch, -1).norm(dim=1).reshape(batch, 1, 1)
        chi = chi.reshape((batch, 2) + (d,) * k)
        B, A0 = R_batch(psi, chi, d, k, c)
        vals = 2.0 * B                       # = B / A_0 since A_0 == 1/2
        vals.sum().backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                best, arg = _select_verified(psi, chi, vals, A0, d, k, c, best, arg)
    return best, arg


def verified_value(psi: np.ndarray, chi: np.ndarray, d: int, k: int, c: float):
    """theta for one (psi, chi) pair, computed with the independent numpy code in
    werner.py.  Returns None if the pair is degenerate (A_0 ~ 0), where theta is
    undefined and the ratio would be a meaningless 0/0."""
    import werner as W
    p, q = np.asarray(psi).reshape(-1), np.asarray(chi).reshape(-1)
    A0 = float(np.real(q.conj() @ W.qubit_marginal_op(p, d, k) @ q))
    if not np.isfinite(A0) or A0 < 1e-6:
        return None
    B = W.B_form(p, q, d, k, c)
    if not np.isfinite(B):
        return None
    return B / A0


def _select_verified(psi, chi, vals, A0, d, k, c, best, arg, top=5):
    """Keep the best candidate as measured by the INDEPENDENT numpy evaluation.

    The torch value is only used to shortlist.  It cannot be trusted on its own:
    the Stiefel construction psi -> rho^psi_0 = I/2 becomes ill-conditioned when the
    two rows of M are nearly dependent, and a single ill-conditioned step can
    otherwise be latched in as a spurious record minimum.
    """
    ok = torch.isfinite(vals)
    if not ok.any():
        return best, arg
    masked = torch.where(ok, vals, torch.full_like(vals, np.inf))
    order = torch.argsort(masked)[:top]
    for i in order.tolist():
        p = psi[i].detach().cpu().numpy().copy()
        q = chi[i].detach().cpu().numpy().copy()
        v = verified_value(p, q, d, k, c)
        if v is not None and v < best:
            best, arg = v, (p, q)
    return best, arg
