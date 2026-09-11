"""E002: minimise the diagonal functional D_c(psi) at the endpoint c = 1/2.

D_{1/2}(psi) >= 0 for all psi is necessary for k-copy undistillability of the
Werner state rho_{-1/2}.  The endpoint conjecture (open for k >= 3) says it holds.
Any strictly negative value is a counterexample and would prove c_k < 1/2.

Output: data/exp002.json
"""
from __future__ import annotations

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from purity import minimise_D  # noqa: E402

torch.set_num_threads(8)
HERE = os.path.join(os.path.dirname(__file__), "..")

CASES = [(d, k) for k in (2, 3, 4) for d in (2, 3, 4, 5, 6, 8)] + \
        [(d, 5) for d in (2, 3, 4)] + [(d, 6) for d in (2, 3)]

res = []
for (d, k) in CASES:
    dim = 2 * d**k
    if dim > 40000:
        continue
    batch = 512 if dim <= 500 else (128 if dim <= 5000 else 32)
    t0 = time.time()
    v, psi = minimise_D(d, k, 0.5, batch=batch, steps=3000, seed=11, lr=0.05)
    rec = dict(d=d, k=k, dim=dim, batch=batch, minD=v, secs=round(time.time() - t0, 1))
    res.append(rec)
    print(f"d={d} k={k} dim={dim:6d} batch={batch:4d}  min D_(1/2) = {v:+.6e}   "
          f"({rec['secs']}s)", flush=True)
    if v < -1e-9:
        np.save(os.path.join(HERE, "data", f"counterexample_d{d}_k{k}.npy"), psi)
        print("   *** NEGATIVE -- counterexample saved ***", flush=True)

with open(os.path.join(HERE, "data", "exp002.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nwrote data/exp002.json")
