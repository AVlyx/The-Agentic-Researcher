"""How often does a tuple of isotypic projectors annihilate EVERY state?

Panel (a): for n = 3, 4, 5 parties, the fraction of partition-tuples (lam^1,...,lam^n)
with g(lam) = 0 -- i.e. the fraction of projector tuples Pi_lam that kill the whole
state space -- as a function of the number of copies k.

Panel (b): for n = 3, k = 7, the support restricted to row lengths, showing that the
rich per-k vanishing pattern collapses onto exactly the polygon-admissible rank
triples r_i <= prod_{j != i} r_j when projected to (ell lam, ell mu, ell nu).

Outputs images/vanishing.pdf and images/vanishing.png.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from schur_weyl import kronecker_coefficient, partitions  # noqa: E402

KMAX = {3: 9, 4: 7, 5: 6}
OUT = Path(__file__).resolve().parent.parent / "images"


def fraction_vanishing(n: int, k: int) -> tuple[int, int]:
    parts = list(partitions(k))
    total = zero = 0
    for lams in itertools.product(parts, repeat=n):
        total += 1
        zero += kronecker_coefficient(*lams) == 0
    return zero, total


def polygon(r: tuple[int, ...]) -> bool:
    prod = 1
    for x in r:
        prod *= x
    return all(x * x <= prod for x in r)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- panel (a) -------------------------------------------------------
    for n, marker in zip((3, 4, 5), ("o", "s", "^")):
        ks, fracs = [], []
        for k in range(2, KMAX[n] + 1):
            zero, total = fraction_vanishing(n, k)
            ks.append(k)
            fracs.append(zero / total)
            print(f"n={n} k={k}: {zero}/{total} vanish ({zero / total:.3f})")
        ax1.plot(ks, fracs, marker=marker, label=f"$n={n}$ parties")
    ax1.set_xlabel("copies $k$")
    ax1.set_ylabel(r"fraction of tuples with $g(\vec\lambda)=0$")
    ax1.set_title(r"(a) projector tuples that annihilate every state")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ---- panel (b) -------------------------------------------------------
    k, rmax = 7, 4
    parts = list(partitions(k))
    reach = set()
    for lams in itertools.product(parts, repeat=3):
        ells = tuple(len(x) for x in lams)
        if max(ells) <= rmax and ells not in reach:
            if kronecker_coefficient(*lams) > 0:
                reach.add(ells)
    grid_r3 = range(1, rmax + 1)
    # show slices r_C = 1..4 side by side as a single (r_A, r_B) x r_C image
    img = np.zeros((rmax, rmax * rmax))
    for ic, rc in enumerate(grid_r3):
        for ia, ra in enumerate(grid_r3):
            for ib, rb in enumerate(grid_r3):
                t = (ra, rb, rc)
                val = 2 if t in reach else (1 if polygon(t) else 0)
                img[ia, ic * rmax + ib] = val
    ax2.imshow(img, origin="lower", cmap="Blues", vmin=0, vmax=2, aspect="auto")
    for ic in range(1, rmax):
        ax2.axvline(ic * rmax - 0.5, color="k", lw=1.2)
    ax2.set_xticks([ic * rmax + (rmax - 1) / 2 for ic in range(rmax)])
    ax2.set_xticklabels([f"$r_C={rc}$" for rc in grid_r3])
    ax2.set_yticks(range(rmax))
    ax2.set_yticklabels([f"$r_A={ra}$" for ra in grid_r3])
    ax2.set_title(rf"(b) $n=3$, $k={k}$: reachable rank triples vs polygon")
    mismatch = {t for t in itertools.product(grid_r3, repeat=3) if (t in reach) != polygon(t)}
    ax2.set_xlabel(
        f"dark = reachable, mid = polygon-only ({len(mismatch)} mismatches), "
        f"light = excluded\n(within each block $r_B$ runs 1..{rmax})",
        fontsize=8,
    )
    print(f"panel (b): {len(mismatch)} mismatches between reachable and polygon")

    fig.tight_layout()
    fig.savefig(OUT / "vanishing.pdf")
    fig.savefig(OUT / "vanishing.png", dpi=160)
    print(f"wrote {OUT / 'vanishing.pdf'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
