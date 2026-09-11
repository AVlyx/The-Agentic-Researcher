"""E015: end-to-end check of the N-form reduction and the multiplicativity theorem.
(1) q_k(-c,C) = l1^2 P1 + l2^2 P2 + 2 l1 l2 Z   [N-form identity]
(2) theta_k(c) >= 0  <=>  P1 P2 >= Z^2
(3) THEOREM: if the Schmidt vectors factor across the k copies, P1P2 >= Z^2 for c<=1/2,
    since P_i = prod_j p_i^(j), Z_C = prod_j z^(j), and p1p2 >= |z|^2 per site.
"""
import numpy as np
from itertools import combinations
from functools import reduce
rng = np.random.default_rng(1234)

def ptrace(C, dims, S):
    k = len(dims); T = C.reshape(tuple(dims)+tuple(dims)); dims = list(dims)
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k+i); k -= 1; dims = dims[:i]+dims[i+1:]
    return T.reshape(int(np.prod(dims)) if dims else 1, -1)

def q_sub(C, dims, c):
    return sum((-c)**len(S)*np.linalg.norm(ptrace(C, dims, S))**2
               for r in range(len(dims)+1) for S in combinations(range(len(dims)), r))

def site_pz(a1, a2, b1, b2, c):
    ip = np.vdot
    return (1-c*abs(ip(a1,b1))**2, 1-c*abs(ip(a2,b2))**2,
            ip(a2,a1)*ip(b1,b2)-c*ip(a2,b2)*ip(b1,a1))

def rvec(d):
    v = rng.normal(size=d)+1j*rng.normal(size=d); return v/np.linalg.norm(v)

print("product-Schmidt-vector configurations: per-site vs global", flush=True)
print(f"{'k':>2s} {'d':>2s} {'q_sub':>12s} {'Nform':>12s} {'P1P2-Z^2':>12s} "
      f"{'prod|z_j|^2 bnd':>16s} {'q>=0':>6s}", flush=True)
c = 0.5
bad = 0
for k in (1, 2, 3, 4):
    for d in (2, 3, 4):
        if d**k > 300: continue
        for trial in range(60):
            # product Schmidt vectors, with one site forced orthogonal on each side
            jA, jB = rng.integers(k), rng.integers(k)
            A1, A2, B1, B2 = [], [], [], []
            for j in range(k):
                a1 = rvec(d); b1 = rvec(d)
                if j == jA:
                    Q, _ = np.linalg.qr(np.stack([a1, rvec(d)], 1)); a1, a2 = Q[:,0], Q[:,1]
                else:
                    a2 = rvec(d)
                if j == jB:
                    Q, _ = np.linalg.qr(np.stack([b1, rvec(d)], 1)); b1, b2 = Q[:,0], Q[:,1]
                else:
                    b2 = rvec(d)
                A1.append(a1); A2.append(a2); B1.append(b1); B2.append(b2)
            kr = lambda L: reduce(np.kron, L)
            a1, a2, b1, b2 = kr(A1), kr(A2), kr(B1), kr(B2)
            l1, l2 = (lambda t: (t[0]/np.linalg.norm(t), t[1]/np.linalg.norm(t)))(
                rng.random(2)+0.2)
            C = l1*np.outer(a1, b1.conj()) + l2*np.outer(a2, b2.conj())
            ps = [site_pz(A1[j], A2[j], B1[j], B2[j], c) for j in range(k)]
            P1 = np.prod([p[0] for p in ps]); P2 = np.prod([p[1] for p in ps])
            Zc = np.prod([p[2] for p in ps]); Z = Zc.real
            nform = l1**2*P1 + l2**2*P2 + 2*l1*l2*Z
            qs = q_sub(C, [d]*k, c)
            bound = np.prod([p[0]*p[1]-abs(p[2])**2 for p in ps])
            if trial == 0:
                print(f"{k:2d} {d:2d} {qs:12.8f} {nform:12.8f} {P1*P2-Z**2:12.8f} "
                      f"{bound:16.8f} {str(qs>=-1e-12):>6s}", flush=True)
            if abs(qs-nform) > 1e-9*max(1, abs(qs)) or qs < -1e-12 or P1*P2-Z**2 < -1e-12:
                bad += 1
print(f"\nviolations across all trials: {bad}", flush=True)

# sanity: NON-product Schmidt vectors are NOT covered -- confirm q can approach 0
print("\nentangled-across-copies Schmidt vectors (not covered by the theorem):", flush=True)
for k in (2, 3):
    d = 3; D = d**k; lo = np.inf
    for _ in range(4000):
        Qa, _ = np.linalg.qr(rng.normal(size=(D,2))+1j*rng.normal(size=(D,2)))
        Qb, _ = np.linalg.qr(rng.normal(size=(D,2))+1j*rng.normal(size=(D,2)))
        C = np.outer(Qa[:,0], Qb[:,0].conj())/np.sqrt(2) + np.outer(Qa[:,1], Qb[:,1].conj())/np.sqrt(2)
        lo = min(lo, q_sub(C, [d]*k, c))
    print(f"  k={k} d={d}: min q over 4000 random rank-2 = {lo:.6f}", flush=True)
