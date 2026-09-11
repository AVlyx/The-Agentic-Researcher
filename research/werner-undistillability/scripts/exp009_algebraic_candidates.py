"""E009: hunt for dimension-free algebraic inequalities on x_S = ||Tr_S C||^2/||C||^2
for rank-<=2 C, aimed at killing the LP vertex (x_j)=(1,1,1), x_{jl}=0, x_{123}=2.

Pure numpy.  Random + structured rank-2 C, several dimensions, unequal local dims.
"""
import numpy as np
from itertools import combinations

rng = np.random.default_rng(20260910)


def ptrace(C, dims, S):
    k = len(dims); T = C.reshape(tuple(dims) + tuple(dims)); dims = list(dims)
    for i in sorted(S, reverse=True):
        T = np.trace(T, axis1=i, axis2=k + i); k -= 1; dims = dims[:i] + dims[i+1:]
    D = int(np.prod(dims)) if dims else 1
    return T.reshape(D, D)


def xvec(C, dims):
    k = len(dims); n2 = np.linalg.norm(C) ** 2
    x = {}
    for r in range(k + 1):
        for S in combinations(range(k), r):
            x[S] = np.linalg.norm(ptrace(C, dims, S)) ** 2 / n2
    return x


def rand_rank2(dims, kind):
    D = int(np.prod(dims))
    def g(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if kind == "generic":
        return g((D, 2)) @ g((D, 2)).conj().T
    if kind == "proj":                      # rank-2 projector: saturates x_[k] = 2
        Q, _ = np.linalg.qr(g((D, 2)))
        return Q @ Q.conj().T
    if kind == "proj_prod":                 # rank-2 projector onto product vectors
        vs = []
        for _ in range(2):
            v = np.array([1.0])
            for dd in dims:
                w = g(dd); w /= np.linalg.norm(w); v = np.kron(v, w)
            vs.append(v)
        Q, _ = np.linalg.qr(np.stack(vs, 1))
        return Q @ Q.conj().T
    if kind == "sv":                        # random singular values
        Qu, _ = np.linalg.qr(g((D, 2))); Qv, _ = np.linalg.qr(g((D, 2)))
        s = rng.random(2)
        return Qu @ np.diag(s) @ Qv.conj().T
    if kind == "lowent":                    # rank-2, low-entanglement factors
        vs = []
        for _ in range(2):
            v = np.array([1.0])
            for dd in dims:
                w = g(dd); w[2:] *= 0.05; w /= np.linalg.norm(w); v = np.kron(v, w)
            vs.append(v)
        A = np.stack(vs, 1); B = A.conj() if rng.random() < .5 else np.stack(
            [np.roll(v, 1) for v in vs], 1)
        return A @ B.conj().T
    raise ValueError(kind)


# candidate inequalities: name -> (lhs, rhs) with claim lhs <= rhs
def candidates(x):
    s1 = x[(0,)] + x[(1,)] + x[(2,)]
    s2 = x[(0, 1)] + x[(0, 2)] + x[(1, 2)]
    t = x[(0, 1, 2)]
    return {
        "A: |TrC|^2 <= 1 + sum x_jl":            (t, 1 + s2),
        "B: |TrC|^2 <= 1 + (1/2) sum x_jl":      (t, 1 + 0.5 * s2),
        "C: sum x_j <= 2 + (1/2) sum x_jl":      (s1, 2 + 0.5 * s2),
        "D: sum x_j <= 2 + sum x_jl":            (s1, 2 + s2),
        "E: sum x_j <= 1 + s2 + t/2":            (s1, 1 + s2 + 0.5 * t),
        "F: t <= 1 + s2 + (1/2)(2-s1)":          (t, 1 + s2 + 0.5 * (2 - s1)),
        "G: sum x_j + t <= 3 + (3/2) s2":        (s1 + t, 3 + 1.5 * s2),
        "H: 2t <= s1 + 2 + s2":                  (2 * t, s1 + 2 + s2),
        "I: t <= s1 + s2 - 1":                   (t, s1 + s2 - 1),
        "J: t^2 <= 2 * s2 (per-pair CS)":        (t ** 2, 2 * s2),
    }


kinds = ["generic", "proj", "proj_prod", "sv", "lowent"]
dimsets = [[2,2,2],[3,3,3],[4,4,4],[5,5,5],[2,3,4],[3,2,5],[6,6,6],[2,2,5],[7,3,2]]
worst = {}
argworst = {}
for dims in dimsets:
    for kind in kinds:
        for _ in range(400):
            C = rand_rank2(dims, kind)
            if np.linalg.norm(C) < 1e-9:
                continue
            x = xvec(C, dims)
            for name, (l, r) in candidates(x).items():
                v = l - r
                if v > worst.get(name, -np.inf):
                    worst[name] = v
                    argworst[name] = (dims, kind, x)

print(f"{'candidate':42s} {'max(lhs-rhs)':>13s}  verdict")
for name in sorted(worst):
    v = worst[name]
    print(f"{name:42s} {v:13.6f}  {'VALID?' if v <= 1e-9 else 'FALSE'}")
print()
for name in sorted(worst):
    if worst[name] > 1e-9:
        dims, kind, x = argworst[name]
        print(f"  violator {name[:1]}: dims={dims} kind={kind} "
              f"x1..3={[round(x[(i,)],3) for i in range(3)]} "
              f"x_jl={[round(x[s],3) for s in [(0,1),(0,2),(1,2)]]} "
              f"t={round(x[(0,1,2)],3)}")
