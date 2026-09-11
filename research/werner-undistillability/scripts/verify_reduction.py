"""Verify the reduction chain (R1)-(R3) in src/werner.py.

Run:  python scripts/verify_reduction.py
"""
from __future__ import annotations

import sys, os
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import werner as W  # noqa: E402

rng = np.random.default_rng(12345)
FAILS = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------- 1. Werner basics
for d in (2, 3, 4):
    for t in (-1.0, -0.7, -1 / d, 0.0, 0.5, 1.0):
        r = W.werner(t, d)
        ok = abs(np.trace(r) - 1) < 1e-12 and np.min(sla.eigvalsh(r)) > -1e-12
        check(f"werner d={d} t={t:+.3f} is a state", ok)
        ppt = np.min(sla.eigvalsh(W.partial_transpose(r, d))) > -1e-12
        check(f"werner d={d} t={t:+.3f} PPT<=>t>=-1/d", ppt == (t >= -1 / d - 1e-12),
              f"(PPT={ppt})")

# ------------------------------------- 2. Choi matrix of Lambda_c vs rho^Gamma
for d in (2, 3, 5):
    for c in (0.1, 0.4, 0.9):
        J = W.choi(c, d)
        pt = W.partial_transpose(W.werner(-c, d), d)
        scale = d * d - c * d  # rho_{-c}^Gamma = J / (d^2 - c d)
        check(f"J(Lambda_c) = (d^2-cd) rho_-c^Gamma  d={d} c={c}",
              np.allclose(J, scale * pt))

# ---------------- 3. Lambda_c is k-positive iff c <= 1/k  (k=1,2 and CP at 1/d)
for d in (3, 4):
    for c in (0.9, 1.01):
        # positivity of Lambda_c: I - c|v><v| psd for unit v
        ok = (c <= 1.0)
        val = 1 - c
        check(f"Lambda_c positive iff c<=1  d={d} c={c}", (val >= -1e-12) == ok)
    for c in (1 / d - 1e-6, 1 / d + 1e-6):
        cp = np.min(sla.eigvalsh(W.choi(c, d))) > -1e-9
        check(f"Lambda_c CP iff c<=1/d  d={d} c={c:.6f}", cp == (c <= 1 / d))

# -------- 4. exact correspondence  <phi|J(Lambda)^{(x)n}|phi> = B_c(psi,chi)
# phi = sum_a |chi_a>_out (x) |conj(psi_a)>_in  has Schmidt rank <= 2.
def choi_n(c, d, n):
    """J(Lambda_c^{(x)n}) on H_out (x) H_in with H = (C^d)^{(x)n}.

    Take the n-fold Kronecker power of J(Lambda_c) -- whose ket index is
    (o_1,i_1,...,o_n,i_n) -- and reorder to (o_1..o_n, i_1..i_n).
    """
    K = np.array([[1.0]])
    J1 = W.choi(c, d)
    for _ in range(n):
        K = np.kron(K, J1)
    K = K.reshape((d,) * (2 * n) + (d,) * (2 * n))
    outs = [2 * j for j in range(n)]
    ins = [2 * j + 1 for j in range(n)]
    perm = outs + ins + [2 * n + a for a in outs] + [2 * n + a for a in ins]
    K = np.transpose(K, perm)
    D = d**n
    return K.reshape(D * D, D * D)


for (d, n) in [(2, 1), (3, 1), (2, 2), (3, 2), (2, 3)]:
    D = d**n
    c = 0.37
    Jn = choi_n(c, d, n)
    for _ in range(5):
        psi = rng.normal(size=2 * D) + 1j * rng.normal(size=2 * D)
        chi = rng.normal(size=2 * D) + 1j * rng.normal(size=2 * D)
        psi /= np.linalg.norm(psi)
        chi /= np.linalg.norm(chi)
        psi_a = psi.reshape(2, D)
        chi_a = chi.reshape(2, D)
        phi = np.zeros(D * D, dtype=complex)
        for a in range(2):
            phi += np.kron(chi_a[a], psi_a[a].conj())
        lhs = np.real(phi.conj() @ Jn @ phi)
        rhs = W.B_form(psi, chi, d, n, c)
        if not np.isclose(lhs, rhs, atol=1e-9):
            check(f"Choi<->B correspondence d={d} n={n}", False, f"{lhs} vs {rhs}")
            break
    else:
        check(f"Choi<->B correspondence d={d} n={n}", True)

# --------------------------- 5. apply_T agrees with the 2^n subset sum, and (R3)
for (d, n) in [(2, 1), (3, 2), (2, 3), (3, 3), (2, 4)]:
    D = 2 * d**n
    c = 0.41
    ok_cross, ok_diag = True, True
    for _ in range(4):
        psi = rng.normal(size=D) + 1j * rng.normal(size=D)
        chi = rng.normal(size=D) + 1j * rng.normal(size=D)
        psi /= np.linalg.norm(psi)
        chi /= np.linalg.norm(chi)
        ok_cross &= np.isclose(W.B_form(psi, chi, d, n, c),
                               W.B_form_marginals(psi, chi, d, n, c), atol=1e-10)
        ok_diag &= np.isclose(W.B_form(psi, psi, d, n, c),
                              W.B_diagonal_purities(psi, d, n, c), atol=1e-10)
    check(f"apply_T == subset sum (R2)  d={d} n={n}", ok_cross)
    check(f"diagonal == marginal purities (R3)  d={d} n={n}", ok_diag)

# ---------------------- 6. B_c is symmetric in its two arguments
for (d, n) in [(3, 2), (2, 3)]:
    D = 2 * d**n
    c = 0.44
    ok = True
    for _ in range(4):
        psi = rng.normal(size=D) + 1j * rng.normal(size=D)
        chi = rng.normal(size=D) + 1j * rng.normal(size=D)
        ok &= np.isclose(W.B_form(psi, chi, d, n, c), W.B_form(chi, psi, d, n, c))
    check(f"B_c symmetric  d={d} n={n}", ok)

# ---------------------- 7. n = 1 threshold is exactly 1/2 for every d
for d in (2, 3, 4, 5):
    for c, expect_neg in [(0.49, False), (0.51, True)]:
        v, _ = W.g_n(d, 1, c, restarts=6, seed=1)
        check(f"g_1 sign at d={d} c={c}", (v < -1e-10) == expect_neg, f"g_1={v:+.6f}")
    v, _ = W.g_n(d, 1, 0.5, restarts=8, seed=2)
    check(f"g_1(1/2)=0 at d={d}", abs(v) < 1e-8, f"g_1={v:+.2e}")

# ---------------------- 8. CP region: c <= 1/d must give g_n >= 0 for all n
for (d, n) in [(3, 2), (3, 3), (4, 2), (2, 4)]:
    c = 1 / d - 1e-3
    v, _ = W.g_n(d, n, c, restarts=8, seed=3)
    check(f"g_n >= 0 in the PPT region d={d} n={n} c={c:.4f}", v > -1e-10, f"g_n={v:+.3e}")

# ---------------------- 9. product ansatz identity  B_n(psi (x) f) = (1-c) B_{n-1}(psi)
d, n, c = 3, 3, 0.45
D = 2 * d ** (n - 1)
psi = rng.normal(size=D) + 1j * rng.normal(size=D)
psi /= np.linalg.norm(psi)
f = np.zeros(d)
f[0] = 1.0
big = np.kron(psi.reshape(2, d ** (n - 1)), f).reshape(-1)
check("appending a product qudit multiplies B by (1-c)",
      np.isclose(W.B_form(big, big, d, n, c),
                 (1 - c) * W.B_form(psi, psi, d, n - 1, c)))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:", FAILS)
    sys.exit(1)
print("all checks passed")
