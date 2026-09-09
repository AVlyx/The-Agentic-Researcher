# `schur-weyl` API wishlist

Written from friction actually hit while building multipartite weak Schur sampling.
Each item gives a signature, semantics, the concrete before/after, and a **gating
identity** in the style of the existing test suite (exact identities, not hand-computed
examples).

Verified already in 0.2.0, cross-checked against independent reimplementations:
`kronecker_coefficient` (590 pairs/triples, k ≤ 5, 0 mismatches),
`apply_isotypic_proj` (16 cases vs both my transpose-sum version and
`isotypic_proj(lam,d) @ v`, 0 mismatches). `dim_weyl(lam, d) == 0` for ℓ(λ) > d
confirmed — several things below rely on it.

---

## 1. `apply_isotypic_proj` should accept axis *groups*

### Signature

```python
apply_isotypic_proj(tensor, lam, axes=None)
```

with `axes` allowed to be **either**

- a flat sequence of `k` ints — today's behaviour, unchanged; or
- a sequence of `k` equal-length tuples of ints, where `axes[c]` lists the axes that
  together carry copy `c`.

### Semantics

`R(σ)` sends copy `c` to copy `σ(c)`, moving each axis group as a unit:
position `j` within group `c` maps to position `j` within group `σ(c)`. Validate that
all groups have the same length `m`, that the `k*m` axes are distinct, and that
`tensor.shape[axes[c][j]]` is independent of `c` (otherwise the action is not defined).

### Why

This is the one thing that makes multipartite work awkward today. For a **block**
`B ⊆ [n]` of parties, the relevant `S_k` action permutes the copies of *all* parties in
`B` simultaneously. In the natural layout — one axis per `(party, copy)` pair, axis
`i*k + c` being party `i` copy `c` — copy `c` of block `B` is a *group* of `|B|` axes,
never a single axis. So today I must transpose + reshape to merge each group into one
axis, call the function, then unmerge. Instead I wrote my own:

```python
# scripts/mws.py -- what I need, and what the package can't express
def permute_copies(T, sigma, block, n, k, lead=0):
    axes = list(range(lead + n * k))
    for i in block:                       # every party in the block ...
        for c in range(k):                # ... permuted by the SAME sigma
            axes[lead + i * k + c] = lead + i * k + sigma[c]
    return T.transpose(axes)
```

With grouped axes this becomes a one-liner against the package:

```python
groups = tuple(tuple(i * k + c for i in block) for c in range(k))
T = apply_isotypic_proj(T, lam, axes=groups)
```

Singleton blocks (`|B| = 1`) are exactly today's flat case, so this is strictly a
generalisation. It is what turns bipartition/subset projectors from "user reimplements
the permutation action" into a first-class package feature — and bipartitions are the
whole point of the multipartite setting.

### Gating identities

```python
# (i) consistency with the flat case: merging groups into single axes must agree
merged = T.transpose(flatten(groups)).reshape(merged_shape)
assert allclose(unmerge(apply_isotypic_proj(merged, lam)),
                apply_isotypic_proj(T, lam, axes=groups))

# (ii) idempotence and resolution of identity, for grouped axes
P = lambda X: apply_isotypic_proj(X, lam, axes=groups)
assert allclose(P(P(T)), P(T))
assert allclose(sum(apply_isotypic_proj(T, l, axes=groups)
                    for l in partitions(k)), T)
```

A third, physically meaningful one (I verified it to ~1e-17, it is a good regression
test): for a **pure** state's copy tensor, complementary blocks agree,
`P^(S)_lam psi^{⊗k} == P^(S^c)_lam psi^{⊗k}`, because
`R_S(σ) R_{S^c}(σ) = R_[n](σ)` acts trivially on `Sym^k`.

### Also worth a `lead=` / batch dimension

I added a `lead` parameter (number of leading axes to ignore) so I could project a whole
*stack* of tensors at once — essential for computing `dim span{Π ψ^{⊗k}}` over hundreds
of random states without a Python loop. With explicit `axes` this is already expressible
(just offset the indices), so it needs no new parameter — but a note in the docstring
that `axes` makes batching free would help.

---

## 2. `permutation_action(tensor, sigma, axes=None)`

### Signature and semantics

Apply `R(σ)` alone — a pure `transpose`, no character weighting. Same `axes` convention
as item 1 (flat or grouped).

### Why

It is the atom underneath `isotypic_proj`, `apply_isotypic_proj`, and the Schur–Weyl
measure, and it is currently private (inlined in `isotypic`, which your README notes as
"phase 6, currently inlined"). Exposing it lets users build **any** element of the group
algebra, not only the central isotypic projectors. Concretely I needed Jucys–Murphy-like
and non-central elements to test commutation; with only `apply_isotypic_proj` available
there is no way to get at a single `R(σ)`.

### Convention note — worth documenting explicitly

There are two conventions, `(R(σ)T)[a_1..a_k] = T[a_{σ(1)}..a_{σ(k)}]` and the one with
`σ^{-1}`. **For `P_λ` the choice is invisible**: the sum runs over all of `S_k`, which is
closed under inversion, and `χ^λ(σ) = χ^λ(σ^{-1})`, so both conventions give the same
operator. That is why `isotypic_proj` can stay silent about it. The moment you expose a
bare `permutation_action` the convention becomes observable and must be pinned down in
the docstring.

### Gating identity

Homomorphism, which catches convention slips immediately:

```python
assert allclose(permutation_action(permutation_action(T, tau, axes), sigma, axes),
                permutation_action(T, permutation_compose(sigma, tau), axes))
assert allclose(permutation_action(T, permutation_identity(k), axes), T)
```

plus reconstruction of the existing function:

```python
P = sum(character(lam, permutation_cycle_type(s)) * permutation_action(T, s, axes)
        for s in permutations(range(k))) * dim_specht(lam) / factorial(k)
assert allclose(P, apply_isotypic_proj(T, lam, axes))
```

---

## 3. `class_sum_coefficients(lam)` — the projector as a group-algebra element

### Signature

```python
class_sum_coefficients(lam) -> dict[tuple[int, ...], Fraction]
```

mapping cycle type `μ` to `c_μ = f^λ · χ^λ(μ) / k!`, so that
`P_λ = Σ_μ c_μ Σ_{σ ∈ C_μ} R(σ)`. `Fraction` (not float) so the result stays exact.

### Why — this is the item I'd argue hardest for

My cleanest result in this project was proved and verified **entirely in the group
algebra**, never touching a Hilbert space:

> `P^(S)_λ` and `P^(T)_μ` commute **iff** the blocks `S`, `T` are nested or disjoint.

The check is exact integer arithmetic on `dict[(σ_1,...,σ_n) -> int]`, and because it
never picks a representation it settles **all local dimensions at once** — no `d`
appears anywhere. That is far stronger than any numerical check on `(C^d)^{⊗k}`, and
much cheaper: 0 violations across 6,619 nested/disjoint instances for n ≤ 5, k ≤ 4, in
seconds. The package currently pushes users into the operator picture immediately, so I
had to rebuild `z_λ` from `character` and `dim_specht` by hand:

```python
# scripts/verify_commutation.py -- rebuilding what the package already knows
def delta(lam, block, n, k):
    e = permutation_identity(k)
    out = {}
    for sigma in itertools.permutations(range(k)):
        chi = character(lam, permutation_cycle_type(sigma))
        if chi == 0: continue
        key = tuple(sigma if i in block else e for i in range(n))
        out[key] = out.get(key, 0) + chi
    return out
```

### Gating identity

The centre of `C[S_k]` has the `z_λ` as orthogonal idempotents summing to 1 — checkable
in exact arithmetic with no numpy at all, very much in the existing test style:

```python
z = {lam: central_element(lam, k) for lam in partitions(k)}
assert mul(z[lam], z[mu]) == (z[lam] if lam == mu else {})   # orthogonal idempotents
assert sum_over(z.values()) == {identity: 1}                  # resolution of identity
```

If you want to go one step further, a tiny `GroupAlgebraElement` type (a dict wrapper
with `__mul__`, `__add__`, exact coefficients) would make the above two lines literal.
That may be more than you want in scope; the `dict` return alone already unlocks it.

---

## 4. `partitions(..., height=r)` — exactly `r` rows

### Signature

Add `min_height` alongside the existing `max_part` / `max_height`, or a single
`height=r` meaning exactly `r`.

### Why

`ℓ(λ)` is precisely what tracks Schmidt rank — the whole rank half of this project is
indexed by row counts, not by partitions. I wrote
`[l for l in partitions(k) if len(l) == r]` in three different scripts. Generating
directly is also strictly cheaper than filtering, and for the reachability sweep
(k ≤ 9, all triples) that inner loop is the hot path.

### Gating identity

```python
assert set(partitions(k, height=r)) == {l for l in partitions(k) if len(l) == r}
assert sum(len(list(partitions(k, height=r))) for r in range(1, k+1)) \
       == len(list(partitions(k)))
```

---

## 5. `schur_weyl_measure` — accept a density matrix, and go multipartite

### 5a. Accept `rho`

`schur_weyl_measure(spectrum: list[float], k)` currently forces the caller to
diagonalise first. Accepting an ndarray and taking `eigvalsh` internally removes a step
every caller performs. Gate: `measure(rho, k) == measure(sorted(eigvalsh(rho))[::-1], k)`.

### 5b. The multipartite entry point

```python
multipartite_schur_weyl_measure(psi, k) -> dict[tuple[partition, ...], float]
```

`psi` an ndarray with one axis per party. This is ~40 lines on top of item 1, and it is
the natural headline feature for the multipartite case — it is exactly what I built in
`scripts/mws.py::mws_distribution`.

### Gating identities — unusually strong here

```python
# (i) normalisation
assert isclose(sum(p.values()), 1.0)

# (ii) product states factorise into independent per-party Schur-Weyl measures
#      p(lam^1,...,lam^n) == prod_i schur_weyl_measure(spec(rho_i), k)[lam^i]

# (iii) n = 1 reduces to the existing function

# (iv) SUPPORT == KRONECKER POSITIVITY
assert {lams for lams, q in p.items() if q > tol} \
    == {lams for lams in candidates
        if kronecker_coefficient(*lams) > 0
        and all(len(l) <= d for l, d in zip(lams, psi.shape))}
```

Identity (iv) is the theorem this project proved and verified (n = 2..5, k = 2..5, all
local dims up to (3,3,3) and (2,3,4) — exact match every time, including for a *single*
Haar-random state). It ties `kronecker_coefficient` and the isotypic projectors together
in one executable assertion, which is a nice property for a package whose whole design
philosophy is identity-gated tests. Happy to hand over my implementation and test cases.

---

## 6. Documentation only: `kronecker_coefficient(*lams) instead of kronecker_coefficient(\*partitions) 

`g(λ¹,…,λᵐ,ν)` = multiplicity of `S^ν` in `S^{λ¹} ⊗ … ⊗ S^{λᵐ}`, because the characters
of `S_k` are real, so trivial-multiplicity and Hom formulations agree. Same function, but
that reading is what you need for recursive / tree constructions — the tree
factorisation

```
dim = prod over internal nodes v of  g(lam^{children(v)}, ..., lam^{v})
```

is only obvious once you see it. One sentence and one doctest would do it.

---

## Deliberately out of scope

- Laminar-family / commutation helpers — application-level, belongs in my code.
- Anything referring to states, entanglement or Schmidt rank — keep the package
  representation-theoretic. Item 5b is the one borderline case; it is justified because
  the Schur–Weyl measure is already in the package.
