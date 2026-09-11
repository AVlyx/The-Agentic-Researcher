"""E013: per-site lemma.  For unit a1,a2,b1,b2 in C^d and 0<=c<=1/2,
   p1 p2 >= |z|^2,   p_i = 1 - c|<a_i|b_i>|^2,
   z = <a2|a1><b1|b2> - c<a2|b2><b1|a1>   ( = <a2 b1|(I-cF)|a1 b2> ).
If true, multiplicativity over sites proves theta_k(c)>=0 whenever the Schmidt vectors
a_i, b_i are PRODUCT across the k copies.  Also: does it survive c>1/2 (should fail)?
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(99)


def quantities(a1, a2, b1, b2, c):
    ip = lambda x, y: np.vdot(x, y)
    p1 = 1 - c * abs(ip(a1, b1)) ** 2
    p2 = 1 - c * abs(ip(a2, b2)) ** 2
    z = ip(a2, a1) * ip(b1, b2) - c * ip(a2, b2) * ip(b1, a1)
    return p1, p2, z


def scan(c, d, ortho, restarts=300):
    def unpack(p):
        v = (p[:4*d] + 1j * p[4*d:]).reshape(4, d)
        if ortho:
            Qa, _ = np.linalg.qr(v[:2].T); Qb, _ = np.linalg.qr(v[2:].T)
            return Qa[:, 0], Qa[:, 1], Qb[:, 0], Qb[:, 1]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return v[0], v[1], v[2], v[3]
    def f(p):
        p1, p2, z = quantities(*unpack(p), c)
        return p1 * p2 - abs(z) ** 2
    worst = np.inf
    for _ in range(restarts):
        r = minimize(f, rng.normal(size=8*d), method="L-BFGS-B",
                     options=dict(maxiter=2000, ftol=1e-15))
        p1, p2, z = quantities(*unpack(r.x), c)      # numpy re-evaluation
        worst = min(worst, p1 * p2 - abs(z) ** 2)
    return worst


print("min of p1*p2 - |z|^2   (negative => lemma FALSE)")
print(f"{'c':>6s} {'d':>3s} {'free':>13s} {'a1|_a2, b1|_b2':>16s}")
for c in (0.25, 0.4, 0.5, 0.55, 0.6):
    for d in (2, 3, 5):
        print(f"{c:6.2f} {d:3d} {scan(c,d,False):13.8f} {scan(c,d,True):16.8f}")
