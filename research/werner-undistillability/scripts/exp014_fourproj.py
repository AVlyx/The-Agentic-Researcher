"""E014: the four-projector lemma (per-site).  Unit a1,a2,b1,b2 in C^d,
  p_i = 1 - c|<a_i|b_i>|^2 ,  z = <a2|a1><b1|b2> - c<a2|b2><b1|a1>.
Claim: p1 p2 >= |z|^2  for 0 <= c <= 1/2  (and FALSE for c>1/2).
Equivalent geometric form with A=|<a1|a2>|,B=|<b1|b2>|,u=|<a1|b1>|,v=|<a2|b2>|,
Delta = <a2|a1><a1|b1><b1|b2><b2|a2>:      A^2B^2 - 1 + c(u^2+v^2) <= 2c Re Delta.
"""
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(4242)

def qty(v, c):
    a1, a2, b1, b2 = v
    ip = np.vdot
    p1 = 1 - c*abs(ip(a1, b1))**2
    p2 = 1 - c*abs(ip(a2, b2))**2
    z = ip(a2, a1)*ip(b1, b2) - c*ip(a2, b2)*ip(b1, a1)
    return p1, p2, z

def norm4(p, d):
    v = (p[:4*d] + 1j*p[4*d:]).reshape(4, d)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

print("min of p1*p2-|z|^2 over four unit vectors  (negative => FALSE)", flush=True)
print(f"{'c':>6s} " + "".join(f"{'d='+str(d):>13s}" for d in (2,3,4,6)), flush=True)
for c in (0.3, 0.45, 0.5, 0.52, 0.6):
    row = []
    for d in (2, 3, 4, 6):
        worst, warg = np.inf, None
        for _ in range(120):
            r = minimize(lambda p: (lambda t: t[0]*t[1]-abs(t[2])**2)(qty(norm4(p, d), c)),
                         rng.normal(size=8*d), method="L-BFGS-B",
                         options=dict(maxiter=1500, ftol=1e-15))
            v = norm4(r.x, d); p1, p2, z = qty(v, c)      # numpy re-evaluation
            if p1*p2-abs(z)**2 < worst:
                worst, warg = p1*p2-abs(z)**2, v
        row.append(worst)
        if c == 0.5 and d == 4:
            a1, a2, b1, b2 = warg; ip = np.vdot
            print(f"   argmin(c=.5,d=4): A={abs(ip(a1,a2)):.4f} B={abs(ip(b1,b2)):.4f} "
                  f"u={abs(ip(a1,b1)):.4f} v={abs(ip(a2,b2)):.4f} "
                  f"ReDelta={(ip(a2,a1)*ip(a1,b1)*ip(b1,b2)*ip(b2,a2)).real:.4f}", flush=True)
    print(f"{c:6.2f} " + "".join(f"{x:13.8f}" for x in row), flush=True)
