"""Build tables.tex and images/theta.pdf|png from the experiment JSON files."""
from __future__ import annotations

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(HERE, "data")


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else []


e3, e4, e5 = load("exp003.json"), load("exp004.json"), load("exp005.json")
e6 = load("exp006.json")
e4 = e4 + e6          # E006 is the same measurement, endpoint-focused
rows = []

# ---------------------------------------------------------------- tables.tex
out = []
out.append(r"\begin{table}[h]\centering")
out.append(r"\caption{E003: exact second-order analysis at the equality points of "
           r"$B_{1/2}$. Only genuine equality points ($B=0$) are informative; GHZ at even "
           r"$k$ has $B=2^{-k}>0$ and is listed for completeness. "
           r"$n_-$ counts negative Hessian eigenvalues; the last column minimises "
           r"$B_{1/2}$ over the Hessian null space.}\label{tab:hess}")
out.append(r"{\setlength{\tabcolsep}{8pt}\renewcommand{\arraystretch}{1.2}")
out.append(r"\begin{tabular}{llrrrrr}\toprule")
out.append(r"$d$ & point & $k$ & $B_{1/2}$ & $\lambda_{\min}(H)$ & $n_-$ & "
           r"$\min B|_{\ker H}$ \\\midrule")
for r in e3:
    if r["B"] > 1e-10:
        continue
    out.append(f"{r['d']} & {r['point']} & {r['k']} & ${r['B']:+.1e}$ & "
               f"${r['hess_min']:+.1e}$ & {r['n_neg']} & ${r['best_in_null']:+.1e}$ \\\\")
out.append(r"\bottomrule\end{tabular}}")
out.append(r"\end{table}")
out.append("")

out.append(r"\begin{table}[h]\centering")
out.append(r"\caption{E004: $\theta_k(c)$ from the global search (column "
           r"``measured'' is the independently re-verified value), against the padded "
           r"upper bound $(1-2c)(1-c)^{k-1}$. $\theta_k(c)<0$ would certify $k$-copy "
           r"distillability.}\label{tab:theta}")
out.append(r"{\setlength{\tabcolsep}{8pt}\renewcommand{\arraystretch}{1.2}")
out.append(r"\begin{tabular}{rrrrrr}\toprule")
out.append(r"$d$ & $k$ & $c$ & measured $\theta_k$ & padded bound & gap \\\midrule")
for r in e4:
    out.append(f"{r['d']} & {r['k']} & {r['c']:.3f} & ${r['verified']:+.8f}$ & "
               f"${r['padded_bound']:+.8f}$ & ${r['verified']-r['padded_bound']:+.1e}$ \\\\")
out.append(r"\bottomrule\end{tabular}}")
out.append(r"\end{table}")
out.append("")

if e5:
    out.append(r"\begin{table}[h]\centering")
    out.append(r"\caption{E005: $\theta$ restricted to $S_k$-isotypic sectors "
               r"(Schur--Weyl), at the endpoint $c=1/2$ and just below. A negative entry "
               r"would be a certificate of $k$-copy distillability.}\label{tab:iso}")
    out.append(r"{\setlength{\tabcolsep}{8pt}\renewcommand{\arraystretch}{1.2}")
    out.append(r"\begin{tabular}{rrrllr}\toprule")
    out.append(r"$d$ & $k$ & $c$ & $\lambda_\psi$ & $\lambda_\chi$ & $\theta$ \\\midrule")
    for r in e5:
        lp = r["lam_psi"].replace("(", "").replace(")", "").replace(",", " ").strip()
        lq = r["lam_chi"].replace("(", "").replace(")", "").replace(",", " ").strip()
        out.append(f"{r['d']} & {r['k']} & {r['c']:.2f} & ${lp}$ & ${lq}$ & "
                   f"${r['verified']:+.8f}$ \\\\")
    out.append(r"\bottomrule\end{tabular}}")
    out.append(r"\end{table}")

open(os.path.join(HERE, "tables.tex"), "w").write("\n".join(out) + "\n")
print("wrote tables.tex")

# ------------------------------------------------------------------- figure
os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
cs = np.linspace(0, 0.5, 200)
for k, col in zip((1, 2, 3, 4, 5), ["C0", "C1", "C2", "C3", "C4"]):
    ax.plot(cs, (1 - 2 * cs) * (1 - cs) ** (k - 1), col, lw=1.4,
            label=f"$(1-2c)(1-c)^{{{k-1}}}$  ($k={k}$)")
    pts = [(r["c"], r["verified"]) for r in e4 if r["k"] == k and r["c"] <= 0.5]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], col + "o", ms=4)
ax.axhline(0, color="k", lw=0.8)
ax.axvline(0.5, color="grey", ls=":", lw=1)
ax.set_xlabel("$c=-\\alpha$")
ax.set_ylabel(r"$\theta_k(c)$")
ax.set_title("Measured $\\theta_k$ vs the padded bound\n(markers: global search, all $d$)")
ax.legend(fontsize=7.5)

ax = axes[1]
ks = np.arange(2, 10)


def bound_root(k):
    def f(c):
        v = (1 - c) * (1 - 2 * c)
        for j in range(3, k + 1):
            v -= c * ((1 + c) ** (j - 1) + (1 - c) ** (j - 1))
        return v
    lo, hi = 1e-6, 0.5
    if f(hi) > 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


ours = [bound_root(int(k)) for k in ks]
pub = {2: 0.5, 3: 1 / 6, 4: 0.1241036, 5: 0.0981112}
lp = load("lp_gamma_extended.json")
ax.plot([r["k"] for r in lp], [r["gamma"] for r in lp], "^-", color="C2", lw=2,
        label="this work: LP bound")
ax.plot(ks, ours, "o-", color="C0", alpha=0.65, label="this work: recursion only")
ax.plot(sorted(pub), [pub[k] for k in sorted(pub)], "s--", color="C1",
        label="published $\\gamma_k$")
ax.plot(ks, 0.5 * np.ones_like(ks, dtype=float), "k:", label="endpoint $c=1/2$ (conjectured $c_k$)")
for dd, st in ((3, "-."), (4, "--"), (5, ":")):
    ax.axhline(1 / dd, color="grey", ls=st, lw=0.8)
    ax.text(9.1, 1 / dd, f"$1/{dd}$", fontsize=7, va="center")
ax.set_xlabel("$k$ (number of copies)")
ax.set_ylabel(r"$\gamma_k$ (proved undistillable for $c\leq\gamma_k$)")
ax.set_title("Rigorous constants: $c\\leq\\gamma_k$ $\\Rightarrow$ $k$-copy undistillable")
ax.set_ylim(0, 0.55)
ax.legend(fontsize=8)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(HERE, "images", f"theta.{ext}"), dpi=160)
print("wrote images/theta.pdf and .png")
