"""E004: measure theta_k(c) = min over rank-<=2 C of q_k(-c, C)/||C||^2.

theta_k(c) < 0  <=>  the Werner state rho_alpha, alpha = -c, is k-copy distillable.
theta_1(c) = 1 - 2c exactly, so c_1 = 1/2 in every dimension; Fraser-Huber-Pozsgay-Vona
and Fu-Gao-Park proved c_2 = 1/2 as well (July 2026).  c_k for k >= 3 is open.

Padding the one-copy optimiser with product qudits multiplies B_c by (1-c) per added
factor and leaves A_0 unchanged, so

    theta_k(c) <= (1 - 2c)(1 - c)^{k-1}                       (product padding)

and padding instead with a traceless rank-one factor kills every partial trace that
touches it, giving theta_k(c) <= theta_1(c) = 1 - 2c.  For c <= 1/2 the first bound
is the stronger one.

H1 (sharp endpoint conjecture): theta_k(c) = (1-2c)(1-c)^{k-1} for 0 <= c <= 1/2.
H0 (endpoint conjecture, open for k >= 3): theta_k(1/2) = 0.

Every reported minimum is re-evaluated with the independent numpy implementation in
werner.py before being recorded -- an earlier torch-only run reported a spurious
negative that its own witness did not reproduce.

Output: data/exp004.json
"""
from __future__ import annotations

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402
from bilinear import minimise_theta, structured  # noqa: E402

torch.set_num_threads(8)
HERE = os.path.join(os.path.dirname(__file__), "..")

CS = [0.30, 0.40, 0.45, 0.48, 0.495, 0.50]
CASES = [(3, 1), (3, 2), (4, 2), (3, 3), (4, 3), (5, 3), (6, 3), (3, 4), (4, 4), (3, 5)]

res = []
for (d, k) in CASES:
    dim = 2 * d**k
    batch = 256 if dim <= 200 else (96 if dim <= 700 else 24)
    steps = 3000 if dim <= 700 else 1500
    for c in CS:
        t0 = time.time()
        v, (psi, chi) = minimise_theta(d, k, c, batch=batch, steps=steps, seed=17,
                                       inits=structured(d, k))
        # independent re-evaluation of the returned witness
        p, q = psi.reshape(-1), chi.reshape(-1)
        B = W.B_form(p, q, d, k, c)
        A0 = float(np.real(q.conj() @ W.qubit_marginal_op(p, d, k) @ q))
        check = B / A0
        pred = (1 - 2 * c) * (1 - c) ** (k - 1)
        rec = dict(d=d, k=k, c=c, dim=dim, theta=v, verified=check, A0=A0,
                   padded_bound=pred, gap=check - pred, secs=round(time.time() - t0, 1))
        res.append(rec)
        flag = ""
        if check < pred - 1e-7:
            flag += "  <-- BELOW padded bound"
        if check < -1e-9:
            flag += "  *** NEGATIVE: k-copy DISTILLABLE ***"
            np.save(os.path.join(HERE, "data", f"witness_d{d}_k{k}_c{c}.npy"),
                    np.stack([psi, chi]))
        print(f"d={d} k={k} dim={dim:5d} c={c:.3f}  theta={v:+.8f} verified={check:+.8f}"
              f"  padded={pred:+.8f}{flag}", flush=True)
        with open(os.path.join(HERE, "data", "exp004.json"), "w") as f:
            json.dump(res, f, indent=1)

print("\nwrote data/exp004.json")
