# Deciding whether a Schmidt-rank profile across **all** bipartitions is realisable

**Problem.** Given local dimensions $(d_1,\dots,d_n)$ and a target
$r = (r_S)_{S}$ indexed by bipartitions $S \mid S^c$ (one representative per complementary
pair; note $r_S = r_{S^c}$ always), decide whether there is a pure state
$\psi \in \mathcal{H} = \bigotimes_i \mathbb{C}^{d_i}$ with

$$\text{rank}\,\rho_S = r_S \quad \text{for every bipartition } S .$$

For single-party cuts this is settled exactly by Kronecker positivity — see
[`proof_kronecker_rank_obstruction.md`](proof_kronecker_rank_obstruction.md). This note
handles the general case, where that argument fails because isotypic projectors for
**overlapping** blocks do not commute, so there is no joint measurement to constrain.

The algorithm below sidesteps that entirely: it never asks for a joint measurement, only
for **one state on which each cut's projections behave as prescribed**.

---

## 1. Why there is no exact SDP

Worth stating up front, because it is the natural first instinct.

The feasible set is $\{\psi : \text{rank}\,M_S(\psi) \le r_S\ \forall S\}$ where $M_S(\psi)$
is the flattening of $\psi$ across $S \mid S^c$. Two obstructions to a convex encoding:

1. **Rank constraints are non-convex.** $\text{rank}\,M_S \le r_S$ is the vanishing of all
   $(r_S+1)\times(r_S+1)$ minors — polynomial equations of degree $r_S+1$, not linear matrix
   inequalities.
2. **Purity is non-convex, and it is essential.** Lifting to $\sigma \succeq 0$,
   $\text{Tr}\,\sigma = 1$ and dropping rank-1 replaces the pure states by their convex
   hull, which is *all* density matrices. That discards exactly the structure the hard
   obstructions use: CHLW ray 3 turns on "$r_{AD} = 1 \Rightarrow \psi = \eta_{AD}\otimes\theta_{BC}$",
   a statement about a **pure** state.

Moreover an exact convex method here would be too strong: deciding Kronecker positivity is
NP-hard [IMW17], and the criterion of the companion note ties rank realisability to it.
So **every SDP in this setting is a relaxation by necessity, not by choice** — and it is
only useful on the *impossibility* side.

---

## 2. The two certificates

The two directions are not symmetric and want different tools:

| direction | tool | why it is a proof | why it can fail |
|---|---|---|---|
| $r$ is **impossible** | linear subspace $W_r$ (this is level 1 of a hierarchy) | every valid $\psi$ would give an element of $W_r$; if none exists, none does | $W_r \neq 0$ proves nothing |
| $r$ is **achievable** | witness search (non-convex local optimisation) | exhibits an explicit $\psi$ | failure could be a bad local minimum |

Where the two agree, the profile is **settled**. A profile that is $W_r$-feasible but
witness-less is genuinely open, and is where a higher SDP level would earn its keep.

---

## 3. Part I — the impossibility certificate

### 3.1 The upper bounds are linear

**Proposition 1.** Let $k \ge \max_S \min(d_S, d_{S^c})$, where $d_S = \prod_{i \in S} d_i$.
For a pure $\psi$ and any bipartition $S$,

$$\text{rank}\,\rho_S \le r_S
\quad\Longleftrightarrow\quad
P^{(S)}_\mu\, \psi^{\otimes k} = 0 \ \ \text{for every } \mu \vdash k \text{ with } \ell(\mu) > r_S .$$

*Proof.* ($\Rightarrow$) Let $t = \text{rank}\,\rho_S \le r_S$. Then $\psi^{\otimes k}$ lies
in $V^{\otimes k} \otimes (\cdots)$ with $V = \text{supp}\,\rho_S \cong \mathbb{C}^t$, and
$P^{(S)}_\mu$ restricted to $V^{\otimes k}$ projects onto $S^\mu \otimes V^t_\mu$, which is
zero when $\ell(\mu) > t$. Since $\ell(\mu) > r_S \ge t$, it vanishes.

($\Leftarrow$) Contrapositive: let $t = \text{rank}\,\rho_S > r_S$. Note
$t \le \min(d_S,d_{S^c}) \le k$, so there is a partition $\mu \vdash k$ with
$\ell(\mu) = t > r_S$. By Lemma 2 of the companion note,
$\|P^{(S)}_\mu \psi^{\otimes k}\|^2 = \text{Tr}[P^{(S)}_\mu \rho_S^{\otimes k}]
= f^\mu s_\mu(\text{spec}\,\rho_S) > 0$ because $\ell(\mu) = t$. $\square$

The bound on $k$ matters: with $k$ too small there are no partitions tall enough to detect
the rank, and the test is vacuous. Taking $k = \max_S \min(d_S,d_{S^c})$ is enough; larger
$k$ is safe but costs more.

### 3.2 The feasible set is a subspace

**Definition.**
$$W_r \;=\; \text{Sym}^k(\mathcal{H}) \ \cap \bigcap_{S}\ \bigcap_{\ell(\mu) > r_S} \ker P^{(S)}_\mu .$$

By Proposition 1, **if $\psi$ satisfies all the upper bounds then $\psi^{\otimes k} \in W_r$.**
$W_r$ is a linear subspace because each condition is the kernel of a linear map, and
$\text{Sym}^k(\mathcal{H})$ is a subspace.

### 3.3 A generic element realises every maximum at once

Define, for $v \in W_r$, the *certified rank*
$$\text{cert}_S(v) \;=\; \max\{\ell(\mu) \;:\; P^{(S)}_\mu v \neq 0\} .$$

**Lemma 2 (genericity).** There is a single $v^\star \in W_r$ with
$\text{cert}_S(v^\star) = \max_{v \in W_r} \text{cert}_S(v)$ **for every $S$
simultaneously**, and a Haar-random $v \in W_r$ has this property with probability 1.

*Proof.* For each pair $(S,\mu)$ the set $U_{S,\mu} = \{v \in W_r : P^{(S)}_\mu v \neq 0\}$
is the complement of the proper linear subspace $\ker P^{(S)}_\mu \cap W_r$, hence Zariski-open
in $W_r$, and nonempty exactly when $P^{(S)}_\mu\big|_{W_r} \neq 0$. A finite intersection of
nonempty Zariski-open subsets of an irreducible variety (a vector space is irreducible) is
nonempty and dense, and its complement has measure zero. Intersect over the finitely many
$(S,\mu)$ with $P^{(S)}_\mu|_{W_r} \neq 0$. $\square$

So one random draw computes all the maxima — no search over $W_r$ is needed.

### 3.4 The obstruction theorem

**Theorem 3 (impossibility certificate).** Let $v$ be generic in $W_r$. If
$\text{cert}_S(v) < r_S$ for some $S$ — in particular if $W_r = 0$ — then **no** pure state
$\psi \in \mathcal{H}$ has $\text{rank}\,\rho_S = r_S$ for all $S$.

*Proof.* Suppose such a $\psi$ existed. Its ranks satisfy all the upper bounds, so
$\psi^{\otimes k} \in W_r$ by Proposition 1. By Lemma 2, $\text{cert}_S$ is maximised at the
generic $v$, so $\text{cert}_S(\psi^{\otimes k}) \le \text{cert}_S(v) < r_S$. But by
Proposition 1 ($\Leftarrow$ direction, applied with $r_S$ replaced by $\text{cert}_S$),
$\text{rank}\,\rho_S \le \text{cert}_S(\psi^{\otimes k}) < r_S$, contradicting
$\text{rank}\,\rho_S = r_S$. $\square$

### 3.5 Why it is only a relaxation

$W_r$ contains $\psi^{\otimes k}$ for every good $\psi$, but also many symmetric tensors that
are **not** $k$-th powers. The set $\{\psi^{\otimes k}\}$ is the Veronese variety, which is
not linear, and $W_r$ is its linear relaxation intersected with the constraints. Hence
$W_r \neq 0$ with $\text{cert} = r$ shows only that *some symmetric tensor* meets the
bounds — never that a *product* one does. This is precisely why the method misses CHLW
ray 3, whose obstruction ($r_{AD}=1$ forcing a product structure) is nonlinear in
$\psi^{\otimes k}$.

---

## 4. Part II — the achievability certificate

No convexity is needed here. Minimise the **tail spectral mass**

$$f(\psi) \;=\; \sum_S \ \sum_{j > r_S} \sigma_j\big(M_S(\psi)\big)^2
 \;=\; \sum_S \Big(1 - \sum_{j \le r_S} \text{eig}_j(\rho_S)\Big)$$

over unit vectors $\psi$, where $\text{eig}_j$ are the eigenvalues of $\rho_S$ in decreasing
order.

**Proposition 4.** $f \ge 0$, and $f(\psi) = 0$ **iff** $\text{rank}\,\rho_S \le r_S$ for
every $S$.

*Proof.* Each term is a sum of eigenvalues of a PSD matrix, hence $\ge 0$, and vanishes iff
$\rho_S$ has at most $r_S$ nonzero eigenvalues. $\square$

**Handling the lower bounds.** Local search cannot impose $\text{rank}\,\rho_S \ge r_S$ — it
would happily collapse ranks further than asked, since that also gives $f = 0$. So the lower
bounds are handled by *inspection*: at a minimiser with $f \approx 0$, compute the achieved
profile and keep the run only if it equals $r$ exactly. Restart until success or budget
exhaustion.

**Why a witness is a proof.** Exhibiting $\psi$ settles achievability outright — no
certificate needed. Caveat: the found $\psi$ is numerical, and "rank" is decided by a
singular-value threshold. For a rigorous claim, either (i) round to exact rationals/algebraic
numbers and verify the minors symbolically, or (ii) bound the relevant singular values away
from zero and the tail below the perturbation. In practice the gap is large (the runs here
separate $\sigma \sim 10^{-1}$ from $\sim 10^{-8}$), but the rounding step is what turns it
into a theorem.

---

## 5. The algorithm, step by step

**Input:** dimensions $(d_1,\dots,d_n)$; target $r = (r_S)_S$.

1. **Choose $k$.** Set $k \ge \max_S \min(d_S, d_{S^c})$ so that partitions tall enough to
   detect every rank exist (Proposition 1). Larger $k$ is safe but the cost grows fast.

2. **Build a basis of $\text{Sym}^k(\mathcal{H})$.** Symmetrise product basis states: for
   each multiset of $k$ elements of the product basis, sum over its distinct permutations.
   These are linearly independent, giving $m = \binom{D + k - 1}{k}$ vectors, $D = \prod_i d_i$.
   Stack them as $B \in \mathbb{R}^{m \times D^k}$.

3. **Assemble the constraint Gram matrix.** For each bipartition $S$ and each $\mu \vdash k$
   with $r_S < \ell(\mu) \le \min(d_S, d_{S^c})$, form $G_{S,\mu} = B\,P^{(S)}_\mu\,B^{\!\top}$
   and set $G = \sum_{S,\mu} G_{S,\mu}$.
   *Note:* each $P^{(S)}_\mu$ is PSD, so $G$ is PSD and
   $\ker G = \bigcap_{S,\mu}\ker\big(P^{(S)}_\mu B^\top\big)$ — the intersection comes for
   free from a single kernel computation, no stacking of huge matrices.
   *Implementation:* apply $P^{(S)}_\mu$ as a weighted sum of **axis transposes** on a
   $(\text{party},\text{copy})$ tensor layout — never materialise a $d_S^k \times d_S^k$
   matrix.

4. **Compute $W_r = \ker G$** by symmetric eigendecomposition; keep eigenvectors with
   eigenvalue below a threshold relative to $\|G\|$. If $\dim W_r = 0$ → **IMPOSSIBLE**, stop.

5. **Draw a generic $v \in W_r$** (random combination of the kernel basis) and compute
   $\text{cert}_S(v)$ for every $S$ by testing $\|P^{(S)}_\mu v\|$ against a threshold.

6. **If $\text{cert}_S(v) < r_S$ for any $S$** → **IMPOSSIBLE** (Theorem 3), stop.

7. **Otherwise run the witness search** (§4): minimise $f$ from many random starts; on each
   run reaching $f \approx 0$, check the achieved profile equals $r$.
   If found → **ACHIEVABLE**, return $\psi$.

8. **Else → OPEN.** The relaxation admits the profile but no witness was found. Either
   increase the restart budget, or go to a higher relaxation level (§7).

**Numerical hygiene.** Thresholds for "is this zero" must be **absolute against the
pre-projection scale**, never relative to the projected object. When a projector annihilates
its input the output is pure roundoff, and a relative test reports full rank. This produced
116 spurious failures in an earlier run of this project before being fixed; always check that
the kept/discarded spectrum has a clear gap.

---

## 6. Cost

Let $D = \prod_i d_i$ and $m = \binom{D+k-1}{k} = \dim\text{Sym}^k(\mathcal{H})$.

| step | cost |
|---|---|
| basis $B$ | $O(m \cdot k! )$ nonzeros; storage $m D^k$ |
| each $G_{S,\mu}$ | $k!$ tensor transposes of size $m D^k$, then an $m \times D^k$ by $D^k \times m$ product |
| $\ker G$ | $O(m^3)$ |
| $\text{cert}_S(v)$ | $O(\#\text{cuts} \cdot p(k) \cdot k! \cdot D^k)$ |
| witness search | $O(\text{restarts} \times \text{iters} \times \#\text{cuts})$ small eigendecompositions |

Concretely for four qubits at $k = 4$: $D = 16$, $m = 3876$, $D^k = 65536$; the basis stack is
$\approx 2$ GB in float64, which is the binding constraint. The distinct $(S,\mu)$ Gram blocks
should be cached — there are at most $\#\text{cuts} \times p(k)$ of them.

The witness search is cheap by comparison: $2D$ real parameters (32 for four qubits).

### 6.1 Making it scale: three changes that unblock $n \ge 5$

The table above is the cost of the *dense* implementation
(`scripts/rank_profile_subspace.py`), and it stops at four qubits: five qubits at $k=4$
would need a basis stack of $m D^k = 52360 \times 32^4 \approx 5.5\times10^{10}$ doubles.
`scripts/rank_profile_blocks.py` reorganises the same computation and runs it in minutes.

1. **The multiset basis is orthogonal.** The symmetrised product states $e_M$ are supported
   on disjoint sets of ordered $k$-tuples, so $\langle e_M, e_N\rangle = 0$ for $M \neq N$.
   Normalising them makes $B$ an isometry onto $\text{Sym}^k(\mathcal{H})$, so
   $G_{S,\mu} = B P^{(S)}_\mu B^{\!\top}$ is a *compression of an orthogonal projector*:
   $\text{spec}\,G \subseteq [0,1]$, and the threshold for "is this eigenvalue zero" finally
   has an absolute scale. This is the clean fix for the numerical-hygiene problem in §5.

2. **$B$ never needs to exist.** $R_S(\sigma)$ maps a product basis tuple to a product basis
   tuple, so $B R_S(\sigma) B^{\!\top}$ is pure integer index arithmetic on tuple codes. No
   $D^k$-sized float array is ever allocated; the $2$ GB stack disappears entirely.

3. **Block-diagonalise by the local torus.** $R_S(\sigma)$ permutes copies *within the $S$
   side*, so for every party $i$ it preserves the multiset of local indices carried by the
   $k$ copies. Every $P^{(S)}_\mu$ is therefore equivariant for the local torus
   $\prod_i (\mathbb{C}^*)^{d_i}$, and $G$ is block diagonal in the weight decomposition
   $$\text{Sym}^k(\mathcal{H}) \;=\; \bigoplus_w \text{Sym}^k(\mathcal{H})_w .$$
   For five qubits at $k=4$ this splits one $52360 \times 52360$ eigenproblem into $3125$
   blocks of mean size $16.8$ and max size $336$: $\sum_b m_b^3 = 4.3\times10^8$ flops
   instead of $m^3 = 1.4\times10^{14}$, a factor $3\times10^5$.

Two identities are asserted per block at run time: $\sum_\mu G_{S,\mu} = I$ (completeness of
the isotypic decomposition on $\text{Sym}^k$, using the Cauchy rule that only
$\ell(\mu) \le \min(d_S,d_{S^c})$ survives) and $0 \le \text{spec}\,G \le 1$.

Two further savings come from the block form. $\sum_{\ell(\mu)>r} G_{S,\mu}$ is precomputed
once per block for each $r$, so a profile costs one sum of $\#\text{cuts}$ matrices plus one
`eigh`; and since
$q_S(r) := \sum_{\ell(\mu)>r}\|P^{(S)}_\mu v\|^2 = c^{\!\top}\!\big(\sum_{\ell(\mu)>r}G_{S,\mu}\big)c$
is exactly that same matrix, the whole certificate $\text{cert}_S = \min\{r : q_S(r) = 0\}$
is read off the objects already built — one `einsum` per profile per block, no separate
projector applications.

**Validation.** Re-running the four-qubit battery through the block code reproduces
`artifacts/rank_battery_k4.log` exactly: all 20 values of $\dim W_r$ and all 20 certified
pair-triples agree. One verdict changes, and it is a fix, not a discrepancy: $(1,1,1)$ was
reported feasible because the dense `battery()` never certified the *singleton* cuts (a known
gap, recorded in `TODO.md`). The block code certifies all 15 cuts, finds
$\text{cert}_A = 1 < 2$, and calls it IMPOSSIBLE — which is what Lemma 5 says independently.

---

## 7. Results so far, and how to close the gap

Four qubits, $k = 4$, local ranks all 2, sweeping $(r_{AB}, r_{AC}, r_{AD})$ over all 20
triples with $r_{AB} \ge r_{AC} \ge r_{AD}$ (`scripts/rank_profile_subspace.py --battery`,
log `artifacts/rank_battery_k4.log`):

| verdict | profiles |
|---|---|
| feasible | (1,1,1) (2,2,1) (3,3,1) (4,4,1) (2,2,2) (3,3,2) (4,4,2) (3,3,3) (4,4,3) (4,4,4) |
| **IMPOSSIBLE** | (2,1,1) (3,1,1) (4,1,1) (3,2,1) (4,2,1) (4,3,1) (3,2,2) (4,2,2) (4,3,2) (4,3,3) |

The split is exactly **"the maximum must be attained at least twice"**, 20/20, and in every
impossible case $\text{cert}$ equals the target with the maximum pulled down to the second
largest — $(4,3,3)\to(3,3,3)$, $(3,2,1)\to(2,2,1)$, $(2,1,1)\to(1,1,1)$.

At $\ell = 4$ the projector $P^{(S)}_{(1,1,1,1)}$ on a four-dimensional cut *is* the
flattening determinant, and the obstruction is **Segre's identity**, verified here
independently on random $2\times2\times2\times2$ tensors:
$\det_{AB} - \det_{AC} + \det_{AD} = 0$, residual $10^{-15}$, the $6\times3$ matrix of
determinants having rank 2. The sweep shows the same pattern also at $\ell = 2$ and $\ell = 3$,
so there is a **ladder** of Segre-type relations, one per rank level, captured automatically.

**This beats the polygon inequalities** — the first method in this project to do so — and
per the literature review these constraints are invisible to entropy inequalities.

**What it still misses.** CHLW ray 3, $(2,2,2,2 \mid 2,2,1)$, is reported feasible
($\dim W_r = 1125$, $\text{cert}$ equal to the target including local ranks 2), yet it is
genuinely unachievable [CHLW14]. Its obstruction is nonlinear (§3.5). This is the canonical
target for a stronger method.

**Closing the gap.** Tighten $W_r$ towards the Veronese variety. The present computation is
**level 1**: linear conditions on $\text{Sym}^k(\mathcal{H})$. A Lasserre/moment-SOS or
DPS-style hierarchy adds, at level 2 and beyond, positivity and symmetric-extension
constraints on a moment matrix built from $\psi$, which restricts the feasible set closer to
genuine $k$-th powers. That is the *only* place an SDP earns its keep here — on the
impossibility side, for profiles that level 1 admits and witness search cannot realise.

### 7.1 Running both halves: the problem is completely solved for this case

Pairing the relaxation with the witness search (`scripts/rank_witness_search.py`, log
`artifacts/rank_witness.log`; 80 restarts per profile) gives, over all 20 triples:

* **10 settled impossible** by the relaxation, no witness found — consistent.
* **7 settled achievable** by explicit witness: (2,2,2) (3,3,2) (3,3,3) (4,4,1) (4,4,2) (4,4,3) (4,4,4).
* **3 open**: (1,1,1), (2,2,1), (3,3,1) — the relaxation admits them, local search finds no
  witness.
* **0 contradictions** (relaxation IMPOSSIBLE together with a witness would be a bug; it
  never occurred).

**The three open cases close analytically, all by one lemma.**

> **Lemma 5 (product forcing).** If $r_{AD} = 1$ then $\rho_{AD}$ is pure, so
> $\psi = \eta_{AD} \otimes \theta_{BC}$. Hence
> $\rho_{AB} = \rho_A^{(\eta)} \otimes \rho_B^{(\theta)}$ and
> $r_{AB} = r_A \cdot r_B$, likewise $r_{AC} = r_A \cdot r_C$.

With all local ranks equal to 2 this forces $(r_{AB}, r_{AC}) = (4,4)$. So among the four
profiles with $r_{AD}=1$ — namely (1,1,1), (2,2,1), (3,3,1), (4,4,1) — **only (4,4,1) is
achievable**, and the other three are impossible. (2,2,1) is exactly CHLW ray 3.

Verified numerically by sampling $\eta_{AD}\otimes\theta_{BC}$ states: with local ranks
$(2,2,2,2)$ the only pair-profile reached is $(4,4,1)$; the profile $(2,2,1)$ *is* reached,
but only with local ranks $(1,2,2,1)$ or $(2,1,1,2)$ — which is precisely why the
triple-only rule of [GMO20] permits $(2,2,1)$ while CHLW rule it out once the local ranks
are pinned to 2. The apparent tension in the literature is resolved.

### 7.2 Final answer for $2\times2\times2\times2$ with all local ranks 2

Writing the triple sorted as $a \ge b \ge c$, the achievable profiles are exactly

$$a = b \quad\textbf{and}\quad \big(c \ge 2 \ \text{ or } \ (a,b) = (4,4)\big),$$

i.e. the 7 profiles (2,2,2), (3,3,2), (3,3,3), (4,4,1), (4,4,2), (4,4,3), (4,4,4). The first
clause is the Segre ladder, caught by the relaxation; the second is the product-forcing
Lemma 5, which the relaxation cannot see because it is nonlinear.

**Methodological summary.** The relaxation caught 10 of the 13 impossible profiles — all the
determinantal ones — and the remaining 3 needed one line of reasoning about purity. That is
the exact shape of the gap predicted in §3.5, and it says where an SDP hierarchy would have
to bite: at profiles whose obstruction is a *product-structure* forcing, not a determinantal
identity.

---

## 8. Five parties: where the cut structure stops being degenerate

Everything above is $n = 4$, and the profile there is a *triple* $(r_{AB},r_{AC},r_{AD})$.
That is small enough to be misleading. The hierarchy of degeneracy is:

* $n = 3$: every bipartition is *single party vs. the rest*, so the profile is
  $(r_A,r_B,r_C)$ and the whole question is settled by Kronecker positivity
  ([`proof_kronecker_rank_obstruction.md`](proof_kronecker_rank_obstruction.md)).
  There is nothing for §3 to do.
* $n = 4$: the three $2|2$ cuts are the three *perfect matchings* of $K_4$. Both sides
  are multi-party, but there are only three of them and they are permuted transitively.
* $n = 5$: a $2|3$ cut is an **edge** of $K_5$ — its complement is the opposite triangle —
  so a profile is a 10-edge labelling, together with 5 single-party cuts. This is the
  first case where the cuts have a nontrivial incidence structure rather than being a
  single orbit of three, and it is where the interesting question lives.

### 8.1 The sweep

Five qubits, $k = 4$ (which is $\max_S \min(d_S,d_{S^c}) = \min(4,8)$),
$\dim\text{Sym}^k = 52360$, all local ranks 2. The 10 pair-cut ranks are swept over
$\{2,3,4\}$: all $3^{10} = 59049$ labellings, reduced to **792 $S_5$-orbit
representatives** (relabelling parties is an exact symmetry, since all local dimensions
and all local target ranks are equal). `scripts/rank_profile_blocks.py --n5`, log
`artifacts/rank_blocks_n5.log`, 24 minutes.

**634 of the 792 orbits are IMPOSSIBLE; 158 survive.** The local ranks come back
certified at 2 on all five single-party cuts in all 792 (so the profiles really are the
local-rank-2 ones), and $\dim W_r$ ranges from 35539 to 52360.

The value 1 is left out of the sweep because it is settled outright:

> **Lemma 6 (product forcing at $n=5$).** If $r_{ij} = 1$ then $\rho_{ij}$ is pure, so
> $\psi = \eta_{ij}\otimes\theta_{klm}$. With all local ranks 2 this fixes the *entire*
> profile: each of the six edges crossing $\{i,j\}\,|\,\{k,l,m\}$ has
> $r = r^{(\eta)}\cdot r^{(\theta)} = 2\cdot 2 = 4$, and each of the three edges inside
> $\{k,l,m\}$ has $\rho_{kl} = \rho^{(\theta)}_{kl}$, whose rank is $r_m = 2$.

So exactly one profile with a rank-1 edge is achievable, and every other one is
impossible — the $n=5$ analogue of Lemma 5. Sampling $\eta\otimes\theta$ confirms it:
2000/2000 draws give pair-profile `1444444222` with local ranks `22222`
(`scripts/rank_witness_general.py --product`, log `artifacts/n5_product_forcing.log`).

### 8.2 Which rules are inherited from four parties, and which are not

Coarse-graining is the obvious source of $n=5$ constraints, and it is worth being exact
about what it can give. Merging 5 parties into 4 groups means merging **one pair**
$\{l,m\}$ into a party $F$ with $d_F = 4$ and $r_F = r_{lm}$; the three $2|2$ cuts of the
merged system are $ij|kF$, $ik|jF$, $iF|jk$, i.e. exactly the **triangle**
$(r_{ij}, r_{ik}, r_{jk})$ of $K_5$ on the complementary triple. All $\binom{5}{2} = 10$
merges give all $\binom{5}{3} = 10$ triangles, so *triangles are the complete list of
rules inherited from four parties*. (Merging into 3 groups gives only polygon
inequalities, and at these dimensions every one of them is vacuous: the sharpest is
$r_{ij} \le 2\,r_{kl}$ for disjoint pairs, and $4 \le 2\cdot 2$ already.)

Does the $n=4$ "max attained at least twice" rule survive the merge? Only at $r_F = 2$.
Running the relaxation directly on $2\times2\times2\times4$ with local ranks $(2,2,2,r_F)$,
all 64 triples, for each $r_F$ (`scripts/rank_profile_blocks.py --merged`, log
`artifacts/rank_blocks_2224.log`):

| $r_F$ | verdict $=$ "max attained at least twice" |
|---|---|
| 2 | **63/64** — the only exception is $(1,1,1)$, killed by Lemma 5, not by Segre |
| 3 | 42/64 |
| 4 | 39/64 |

At $r_F = 2$ the merged party is effectively a qubit and the $2^{\times 4}$ rule is
reproduced exactly. At $r_F \ge 3$ it fails, and the failure is **proved by explicit
states**, not merely undetected by the relaxation: $(3,3,4)$ — maximum attained once —
has no witness at $r_F = 2$ but has one at $r_F = 3$ and at $r_F = 4$
(`scripts/rank_witness_general.py --merged`, log `artifacts/rank_witness_2224.log`,
200 restarts). **Segre's identity is a statement about local rank 2 on all four parties,
not about four parties.**

That predicts precisely one inherited rule at $n=5$, and the sweep confirms it with no
exceptions:

> **Conditional triangle rule.** For every triangle $\{i,j,k\}$ whose opposite edge has
> $r_{lm} = 2$, the maximum of $(r_{ij}, r_{ik}, r_{jk})$ is attained at least twice.

All 158 relaxation-feasible profiles satisfy it; **0 of the 792 violate it while being
called feasible** (`scripts/analyse_n5_rules.py`, log `artifacts/n5_rules.log`). The
*unconditional* triangle rule is strictly stronger than feasibility and is not a valid
constraint — 131 relaxation-feasible profiles violate it, as the $r_F\ge3$ rows above
predict. The global rule "the maximum over all 10 edges is attained at least twice"
is badly wrong (it admits 562 impossible profiles); at $n=4$ it looked like the right
statement only because there the three cuts form a single triangle.

### 8.3 The relaxation sees obstructions that no coarse-graining does

**59 of the 634 impossible profiles satisfy the conditional triangle rule** — and hence
every rule inherited from a 4-party or 3-party merge. These are genuinely 5-partite
obstructions. Examples (pair-cut order `01 02 03 04 12 13 14 23 24 34`):

| profile | certified | falls short on |
|---|---|---|
| `2224333333` | `2223333333` | $04$ |
| `2233433333` | `2233333333` | $12$ |
| `2222344444` | `2222344443` | $34$ |
| `2223334444` | `2223333333` | $14, 23, 24, 34$ |
| `2224243434` | `2223233333` | $04, 13, 23, 34$ |

This is the point of going to five parties. At $n=4$ the relaxation reproduced a rule that
was already known (Segre, via [GMO20]); at $n=5$ it produces constraints with no
lower-party explanation, on a cut structure rich enough that a hand-written rule was
never going to be guessed from three numbers.

### 8.4 Running both halves at $n=5$

The witness search (§4), generalised to arbitrary $n$ with an analytic gradient, was run
on **all 792** orbit representatives — not only the 158 the relaxation admits, since a
witness for a profile called IMPOSSIBLE would be a bug and running both halves everywhere
tests both (`scripts/rank_witness_general.py --n5`, 100 restarts, log
`artifacts/rank_witness_n5.log`; then `--n5-open`, 600 restarts and a fresh seed on the
survivors, log `artifacts/rank_witness_n5_open.log`).

| | count |
|---|---|
| settled IMPOSSIBLE by the relaxation, no witness | 634 |
| settled ACHIEVABLE by explicit witness | 34 |
| **open** — relaxation admits, no witness found | **124** |
| **contradictions** | **0** |

$668/792$ settled. The deeper second pass converted only 2 of 126 open profiles, so the
remaining 124 are unlikely to be local-minimum artefacts.

**The $n=4$ intuition does not survive.** Sixteen of the 34 achievable profiles violate the
unconditional triangle rule, and five violate the vertex-star version — both refuted by
explicit states. `3333333334` is the cleanest: nine edges of rank 3 and one of rank 4, so
the maximum is attained exactly once, globally and in several triangles, and a witness
exists. The only surviving max-attained-twice statement at $n=5$ is the conditional one of
§8.2, which every achievable *and* every relaxation-feasible profile obeys.

**Where the gap is now.** At $n=4$ the gap between the two certificates was 3 profiles and
all three closed by one lemma about purity (Lemma 5). At $n=5$ it is 124, and no analogous
lemma is available: product forcing (Lemma 6) only bites on rank-1 cuts, which are settled
separately and excluded from the sweep. So the open set is not a residue of one special
case — it is the generic situation, and it is a much sharper target for a level-2
Lasserre/DPS relaxation (§7) than CHLW ray 3 was: 124 profiles, on a cut structure where
no coarse-graining argument reaches.

Fitting a *local* rule to the 34 witnesses (allowed patterns on a vertex star plus the
opposite triangle) gives 0 false negatives and only 11 false positives, so a clean
combinatorial characterisation may well exist. It cannot be settled from this data, though:
with 124 profiles undecided, every one of those "false positives" is a profile whose true
status is unknown.

---

## 9. References

| Key | Reference | Link |
|---|---|---|
| [CHLW14] | J. Cadney, M. Huber, N. Linden, A. Winter, *Inequalities for the ranks of multipartite quantum states*, Linear Algebra Appl. **452**:153–171 (2014) | [arXiv:1308.0539](https://arxiv.org/abs/1308.0539) |
| [IMW17] | C. Ikenmeyer, K. D. Mulmuley, M. Walter, *On vanishing of Kronecker coefficients*, comput. complex. (2017) | [arXiv:1507.02955](https://arxiv.org/abs/1507.02955) |
| [GMO20] | M. Gharahi, S. Mancini, G. Ottaviani, *Fine-structure classification of multiqubit entanglement by algebraic geometry*, Phys. Rev. Research (2020) | [arXiv:1910.09665](https://arxiv.org/abs/1910.09665) |
| [JSV25] | J. Jovcheva, T. Seynnaeve, N. Vannieuwenhoven, *Minimality of Tree Tensor Network Ranks* (2025) | [arXiv:2509.09463](https://arxiv.org/abs/2509.09463) |
| [SCSH23] | Z. Song, L. Chen, Y. Sun, M. Hu, *A complete picture of the four-party linear inequalities in terms of the 0-entropy*, IEEE Trans. Inf. Theory **69**(4):2385–2399 (2023) | [arXiv:2105.07679](https://arxiv.org/abs/2105.07679) |
| [KW01] | M. Keyl, R. F. Werner, *Estimating the spectrum of a density operator*, Phys. Rev. A **64**, 052311 (2001) | [arXiv:quant-ph/0102027](https://arxiv.org/abs/quant-ph/0102027) |
| [H13] | A. W. Harrow, *The Church of the Symmetric Subspace* (2013) | [arXiv:1308.6595](https://arxiv.org/abs/1308.6595) |

The "max attained at least twice" rule for $2\times2\times2\times2$ is attributed to
[GMO20] via Segre's 1920 identity. **Caveat:** my notes record that rule as *reported*, and it
concerns the triple $(r_{AB},r_{AC},r_{AD})$ alone, without pinning local ranks. It therefore
permits $(2,2,1)$ — realisable with $(r_A,r_B,r_C,r_D) = (1,2,2,1)$ — which is consistent with
CHLW ray 3 being impossible only once all local ranks are fixed to 2. Segre (1920) and the GMO
theorem statement are both on the unverified list in `notes/lit_rank_profiles.md`; verify
before citing as primary.
