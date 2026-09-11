"""E006: endpoint-focused sweep -- the sign of theta_k at c = 1/2 (and just below).

E004 measured theta_k(c) on a full c-grid and matched the padded bound
(1-2c)(1-c)^{k-1} to 1e-15 everywhere it completed, but the full grid over all (d,k) is
hours of compute.  The decisive question is only the SIGN at the endpoint: theta_k(1/2) = 0
is the endpoint conjecture, theta_k(1/2) < 0 refutes it and proves c_k < 1/2.  So this
sweep spends the budget on c in {0.48, 0.50} across the widest (d,k) range instead.

This is a scope reduction for compute, not a change of metric: the quantity, the optimiser
and the verification are identical to E004, and no completed E004 row is discarded.

Output: data/exp006.json
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

CASES = [(5, 3), (6, 3), (8, 3), (3, 4), (4, 4), (5, 4), (3, 5), (4, 5), (2, 6), (3, 6)]
CS = [0.50, 0.48]

res = []
for (d, k) in CASES:
    dim = 2 * d**k
    if dim > 1600:
        print(f"# skip d={d} k={k} (dim {dim} too large for this budget)", flush=True)
        continue
    batch = 64 if dim <= 300 else (32 if dim <= 700 else 16)
    steps = 600 if dim <= 700 else 400
    for c in CS:
        t0 = time.time()
        v, (psi, chi) = minimise_theta(d, k, c, batch=batch, steps=steps, seed=101,
                                       inits=structured(d, k))
        p, q = psi.reshape(-1), chi.reshape(-1)
        A0 = float(np.real(q.conj() @ W.qubit_marginal_op(p, d, k) @ q))
        check = W.B_form(p, q, d, k, c) / A0
        pred = (1 - 2 * c) * (1 - c) ** (k - 1)
        rec = dict(d=d, k=k, c=c, dim=dim, theta=v, verified=check,
                   padded_bound=pred, secs=round(time.time() - t0, 1))
        res.append(rec)
        flag = "  *** NEGATIVE: k-copy DISTILLABLE ***" if check < -1e-9 else ""
        print(f"d={d} k={k} dim={dim:5d} c={c:.2f}  theta={check:+.9f}  "
              f"padded={pred:+.9f}  ({rec['secs']}s){flag}", flush=True)
        if check < -1e-9:
            np.save(os.path.join(HERE, "data", f"w6_d{d}_k{k}_c{c}.npy"),
                    np.stack([psi, chi]))
        with open(os.path.join(HERE, "data", "exp006.json"), "w") as f:
            json.dump(res, f, indent=1)

print("\nwrote data/exp006.json")
neg = [r for r in res if r["verified"] < -1e-9]
print(f"verified negative values (counterexamples): {len(neg)}")
