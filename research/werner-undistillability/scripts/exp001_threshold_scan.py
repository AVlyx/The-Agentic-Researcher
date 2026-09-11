"""E001: does the n-copy criterion improve on the one-copy criterion?

c_1(d) = 1/2 for every d.  If c_n(d) < 1/2 for some n >= 2, then at c = 1/2 the
minimum of the scale-invariant ratio R_c(psi, chi) = B_c(psi, chi)/A_0(psi, chi)
must be strictly negative.  Search for that, with many random restarts, and also
report the minimum just below 1/2.

Output: data/exp001.json
"""
from __future__ import annotations

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")

CASES = [(d, n) for d in (3, 4, 5, 6) for n in (2, 3)] + [(3, 4), (4, 4)]
CS = [0.50, 0.499, 0.49]

results = []
for (d, n) in CASES:
    dim = 2 * d**n
    for c in CS:
        t0 = time.time()
        restarts = 60 if dim <= 60 else (30 if dim <= 200 else 12)
        v, arg = W.gt_n(d, n, c, restarts=restarts, seed=2024, iters=500)
        psi, chi = arg
        rec = dict(d=d, n=n, c=c, dim=dim, restarts=restarts, minR=v,
                   overlap=float(abs(np.vdot(psi, chi))),
                   B=W.B_form(psi, chi, d, n, c),
                   secs=round(time.time() - t0, 1))
        results.append(rec)
        print(f"d={d} n={n} dim={dim:5d} c={c:.3f}  minR={v:+.3e}  "
              f"|<psi|chi>|={rec['overlap']:.3f}  ({rec['secs']}s)", flush=True)

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
with open(os.path.join(HERE, "data", "exp001.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nwrote data/exp001.json")

neg = [r for r in results if r["minR"] < -1e-9]
print(f"strictly negative minima found: {len(neg)}")
for r in neg:
    print("  ", r)
