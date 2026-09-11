"""E003: local analysis at the equality points of the endpoint conjecture.

At c = 1/2 the biquadratic form B_{1/2}(psi, chi) vanishes on a whole manifold of
configurations.  Two explicit families:

  * padded one-copy optimiser  psi_pad = (|0 e_0> + |1 e_1>)/sqrt2 (x) |e_0>^{(x)(k-1)}
    -- B_c = (1-2c)(1-c)^{k-1}, which vanishes at c = 1/2 for every k;
  * GHZ-like  psi_ghz = (|0>|e_0>^{(x)k} + |1>|e_1>^{(x)k})/sqrt2
    -- B_c = ((1-c)^k + (-c)^k)/2, which vanishes at c = 1/2 for every ODD k.

B_{1/2} >= 0 everywhere is exactly the endpoint conjecture (open for k >= 3).  At
any zero the gradient must vanish and the Hessian must be PSD.  A strictly
negative Hessian eigenvalue would be a counterexample, proving c_k < 1/2.

B is a quartic polynomial in the real coordinates, so a PSD Hessian is not the end
of the story: along the Hessian null space the quartic term decides.  We therefore
also minimise B restricted to (equality point + null space) -- a direct search in
exactly the flat directions where the inequality could fail.

Output: data/exp003.json
"""
from __future__ import annotations

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
HERE = os.path.join(os.path.dirname(__file__), "..")


def B_torch(psi_r, chi_r, d, k, c):
    """B_c(psi, chi) from real parameter vectors (homogeneous, unnormalised)."""
    shape = (2,) + (d,) * k
    psi = torch.view_as_complex(psi_r.reshape(-1, 2)).reshape(shape)
    chi = torch.view_as_complex(chi_r.reshape(-1, 2)).reshape(shape)
    tot = psi.new_zeros((), dtype=torch.float64)
    for mask in range(1 << k):
        S = [i + 1 for i in range(k) if (mask >> i) & 1]
        keep = [0] + S
        rest = [a for a in range(k + 1) if a not in keep]
        pv = psi.permute(keep + rest).reshape(2 * d ** len(S), -1)
        cv = chi.permute(keep + rest).reshape(2 * d ** len(S), -1)
        rp = pv @ pv.conj().T
        rc = cv @ cv.conj().T
        tot = tot + ((-c) ** len(S)) * torch.einsum("ij,ji->", rp, rc).real
    return tot


def to_real(v: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.stack([v.real.ravel(), v.imag.ravel()], axis=-1).ravel())


def ghz(d, k):
    v = np.zeros((2,) + (d,) * k, dtype=complex)
    v[(0,) + (0,) * k] = 1 / np.sqrt(2)
    v[(1,) + (1,) * k] = 1 / np.sqrt(2)
    return v.reshape(-1)


def padded(d, k):
    v = np.zeros((2,) + (d,) * k, dtype=complex)
    v[(0, 0) + (0,) * (k - 1)] = 1 / np.sqrt(2)
    v[(1, 1) + (0,) * (k - 1)] = 1 / np.sqrt(2)
    return v.reshape(-1)


def analyse(psi, chi, d, k, c, null_tol=1e-7, null_steps=1500, seed=0):
    x0 = torch.cat([to_real(psi), to_real(chi)])
    n = x0.numel() // 2
    f = lambda x: B_torch(x[:n], x[n:], d, k, c)  # noqa: E731

    val = f(x0).item()
    grad = torch.autograd.functional.jacobian(f, x0).numpy()
    H = torch.autograd.functional.hessian(f, x0).numpy()
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    nneg = int((w < -1e-8).sum())

    # --- quartic probe inside the Hessian null space --------------------------
    scale = np.abs(w).max()
    null = V[:, w < null_tol * max(scale, 1.0)]
    best_null = 0.0
    if null.shape[1] > 0 and nneg == 0:
        rng = np.random.default_rng(seed)
        Nt = torch.tensor(null)
        coef = torch.tensor(rng.normal(size=(null.shape[1],)) * 0.1, requires_grad=True)
        opt = torch.optim.Adam([coef], lr=0.02)
        for _ in range(null_steps):
            opt.zero_grad()
            x = x0 + Nt @ coef
            v = f(x)
            v.backward()
            opt.step()
            best_null = min(best_null, v.item())
    return val, float(np.abs(grad).max()), w, nneg, null.shape[1], best_null


CASES = [(2, 3), (3, 3), (4, 3), (5, 3), (2, 4), (3, 4), (4, 4),
         (2, 5), (3, 5), (2, 6), (2, 7)]
res = []
for (d, k) in CASES:
    for name, mk in (("padded", padded), ("ghz", ghz)):
        t0 = time.time()
        psi = mk(d, k)
        val, gmax, w, nneg, ndim, bn = analyse(psi, psi, d, k, 0.5)
        rec = dict(d=d, k=k, point=name, B=val, gradmax=gmax, hess_min=float(w[0]),
                   n_neg=nneg, npar=len(w), null_dim=ndim, best_in_null=bn,
                   secs=round(time.time() - t0, 1))
        res.append(rec)
        print(f"d={d} k={k} {name:7s} B={val:+.2e} |g|={gmax:.1e} "
              f"lam_min(H)={w[0]:+.3e} #neg={nneg} null={ndim}/{len(w)} "
              f"minB|null={bn:+.3e} ({rec['secs']}s)", flush=True)
        with open(os.path.join(HERE, "data", "exp003.json"), "w") as f:
            json.dump(res, f, indent=1)

print("\nwrote data/exp003.json")
bad = [r for r in res if r["B"] < 1e-10 and (r["n_neg"] > 0 or r["best_in_null"] < -1e-9)]
print(f"equality points with a descent direction: {len(bad)}")
for r in bad:
    print("  ", r)
