# Kronecker coefficients decide which Schmidt-rank vectors exist along the local cuts

**Scope.** This note proves that the $n$-fold Kronecker coefficients decide exactly which
vectors of Schmidt ranks $(r_1,\dots,r_n)$ across the $n$ **single-party cuts**
$\{i\} \mid [n]\setminus\{i\}$ are realised by some multipartite pure state. These are the
cuts the polygon inequalities $r_i \le \prod_{j\neq i} r_j$ refer to.

**Warning about scope.** This does *not* extend to all $2^{n-1}-1$ bipartitions. The
criterion below is exact for single-party cuts only; for general bipartitions the isotypic
projectors of overlapping blocks do not commute and the argument in Step 5 breaks. See
[`algorithm_rank_profile_decision.md`](algorithm_rank_profile_decision.md) for what can be
done there.

**Novelty.** The statement is assembled from known pieces and is best described as
folklore-adjacent; see §7. Cite it as an observation, not a new theorem, until the reads
listed in §7 are done.

---

## 1. Setup and notation

Parties $i \in [n]$, local spaces $\mathcal{H}_i = \mathbb{C}^{d_i}$,
$\mathcal{H} = \mathcal{H}_1 \otimes \cdots \otimes \mathcal{H}_n$. Fix $k$ copies. Regroup
copies by party,

$$\mathcal{H}^{\otimes k} \;\cong\; \bigotimes_{i=1}^n \mathcal{H}_i^{\otimes k}. \tag{1}$$

For $\sigma \in S_k$ let $R_i(\sigma)$ permute the $k$ copies of party $i$, and let
$R_{[n]}(\sigma) = \bigotimes_i R_i(\sigma)$ be the diagonal action (permuting whole copies
of $\mathcal{H}$). For $\lambda \vdash k$ write $f^\lambda = \text{dim}\,S^\lambda$ for the
Specht module dimension, $\chi^\lambda$ for its character, $\ell(\lambda)$ for the number of
rows, and

$$P^{(i)}_\lambda \;=\; \frac{f^\lambda}{k!}\sum_{\sigma\in S_k}\chi^\lambda(\sigma)\,R_i(\sigma)$$

for the isotypic projector on $\mathcal{H}_i^{\otimes k}$. Let
$V^{d}_\lambda = S^\lambda(\mathbb{C}^d)$ be the Weyl ($GL_d$) module, so
$\text{dim}\,V^d_\lambda = 0 \iff \ell(\lambda) > d$.

For a tuple $\vec\lambda = (\lambda^{(1)},\dots,\lambda^{(n)})$ of partitions of $k$ set

$$\Pi_{\vec\lambda} = \bigotimes_{i=1}^n P^{(i)}_{\lambda^{(i)}}, \qquad
p_\psi(\vec\lambda) = \big\|\, \Pi_{\vec\lambda}\, \psi^{\otimes k} \,\big\|^2 .$$

The $\Pi_{\vec\lambda}$ are mutually orthogonal projectors summing to the identity, so
$\{p_\psi(\vec\lambda)\}$ is a probability distribution: **multipartite weak Schur sampling**.

The **$n$-fold (generalized) Kronecker coefficient** is

$$g(\lambda^{(1)},\dots,\lambda^{(n)})
 \;=\; \text{dim}\Big(S^{\lambda^{(1)}}\otimes\cdots\otimes S^{\lambda^{(n)}}\Big)^{S_k}
 \;=\; \frac{1}{k!}\sum_{\sigma\in S_k}\prod_{i=1}^n \chi^{\lambda^{(i)}}(\sigma). \tag{2}$$

For $n=3$ this is the classical Kronecker coefficient. Note $g$ depends only on the
partitions — **not** on the local dimensions.

---

## 2. Two standard lemmas

**Lemma 1 (symmetric subspace is spanned by $k$-th powers).**
$\text{span}\{\psi^{\otimes k} : \psi \in \mathcal{H}\} = \text{Sym}^k(\mathcal{H})$.

*Proof.* Harrow, *The Church of the Symmetric Subspace*, Theorem 3 [H13]. $\square$

**Lemma 2 (rank bound and its sharpness).** Let $\rho$ be a state on $\mathbb{C}^d$ with
$t = \text{rank}\,\rho$. Then for $\lambda \vdash k$,

$$\text{Tr}\big[P_\lambda\,\rho^{\otimes k}\big] \;=\; f^\lambda\, s_\lambda(\text{spec}\,\rho)
\qquad\text{and}\qquad
\text{Tr}\big[P_\lambda\,\rho^{\otimes k}\big] > 0 \iff \ell(\lambda) \le t .$$

*Proof.* The first identity is the Schur–Weyl measure [KW01, CM06]. For the second, write
$\text{spec}\,\rho = (p_1,\dots,p_t,0,\dots,0)$ with all $p_j > 0$. The Schur polynomial
$s_\lambda$ is a sum, over semistandard Young tableaux of shape $\lambda$ with entries in
$\{1,\dots,d\}$, of monomials in the $p_j$ — a sum of *nonnegative* terms. A tableau
contributes a nonzero monomial iff all its entries lie in $\{1,\dots,t\}$, and such a
tableau exists iff $\ell(\lambda) \le t$ (columns are strictly increasing). Hence
$s_\lambda(\text{spec}\,\rho) > 0 \iff \ell(\lambda) \le t$. Since $f^\lambda > 0$, done.

Equivalently and more structurally: $\rho^{\otimes k}$ is supported on $V^{\otimes k}$ where
$V = \text{supp}\,\rho \cong \mathbb{C}^t$, and $P_\lambda$ restricted to $V^{\otimes k}$
projects onto $S^\lambda \otimes V^t_\lambda$, which vanishes iff $\ell(\lambda) > t$. $\square$

---

## 3. The support theorem

**Theorem A.** With the notation above:

**(a)**
$$\text{dim}\Big(\text{Im}\,\Pi_{\vec\lambda} \,\cap\, \text{Sym}^k(\mathcal{H})\Big)
 \;=\; g(\lambda^{(1)},\dots,\lambda^{(n)}) \cdot \prod_{i=1}^n \text{dim}\,V^{d_i}_{\lambda^{(i)}} .$$

**(b)** There exists a pure $\psi \in \mathcal{H}$ with $p_\psi(\vec\lambda) > 0$ **iff**
$g(\lambda^{(1)},\dots,\lambda^{(n)}) > 0$ and $\ell(\lambda^{(i)}) \le d_i$ for every $i$.

Equivalently: **the projector tuple $\Pi_{\vec\lambda}$ annihilates every state of
$\mathcal{H}$ precisely when the $n$-fold Kronecker coefficient vanishes** (or some
$\lambda^{(i)}$ is too tall to fit in $\mathcal{H}_i$).

*Proof.*

**(b) reduces to (a).** $\Pi_{\vec\lambda}$ is an orthogonal projector, so
$p_\psi(\vec\lambda) = \|\Pi_{\vec\lambda}\psi^{\otimes k}\|^2$ vanishes for *every* $\psi$
iff $\Pi_{\vec\lambda}$ annihilates $\text{span}\{\psi^{\otimes k}\}$, which by Lemma 1 is
$\text{Sym}^k(\mathcal{H})$. So some $\psi$ has $p_\psi > 0$ iff
$\Pi_{\vec\lambda}\big|_{\text{Sym}^k(\mathcal{H})} \neq 0$, iff the dimension in (a) is
nonzero, iff $g > 0$ and every $\text{dim}\,V^{d_i}_{\lambda^{(i)}} > 0$, i.e.
$\ell(\lambda^{(i)}) \le d_i$.

**(a).** Apply Schur–Weyl duality to each factor of (1):
$\mathcal{H}_i^{\otimes k} \cong \bigoplus_{\lambda} S^\lambda \otimes V^{d_i}_\lambda$,
where $S_k$ acts on the first tensorand and $GL(\mathcal{H}_i)$ on the second. Therefore

$$\text{Im}\,\Pi_{\vec\lambda}
 \;=\; \Big(\bigotimes_i S^{\lambda^{(i)}}\Big) \otimes \Big(\bigotimes_i V^{d_i}_{\lambda^{(i)}}\Big). \tag{3}$$

Under (1), $\text{Sym}^k(\mathcal{H})$ is precisely the fixed-point set of the **diagonal**
action $R_{[n]}$, because permuting whole copies of $\mathcal{H}$ is the same as permuting
each party's copies simultaneously. In the decomposition (3), $R_{[n]}(\sigma)$ acts as
$\bigotimes_i \rho^{\lambda^{(i)}}(\sigma)$ on the Specht tensorands and as the identity on
the Weyl tensorands (the $GL$ and $S_k$ actions commute).

$\Pi_{\vec\lambda}$ commutes with $R_{[n]}(\sigma)$ for every $\sigma$: each
$P^{(i)}_{\lambda^{(i)}}$ is a central element of $\mathbb{C}[S_k]$ acting on factor $i$
only, hence commutes with the diagonal image. Consequently $\Pi_{\vec\lambda}$ and the
symmetriser $\Pi_{\text{Sym}} = \frac{1}{k!}\sum_\sigma R_{[n]}(\sigma)$ commute, so their
product is the orthogonal projector onto the intersection of their images. Taking $S_k$
invariants in (3),

$$\text{Im}\,\Pi_{\vec\lambda} \cap \text{Sym}^k(\mathcal{H})
 \;=\; \Big(\bigotimes_i S^{\lambda^{(i)}}\Big)^{S_k} \otimes \Big(\bigotimes_i V^{d_i}_{\lambda^{(i)}}\Big),$$

whose dimension is $g(\vec\lambda)\prod_i \text{dim}\,V^{d_i}_{\lambda^{(i)}}$ by
definition (2). $\square$

**Remark (why this is not circular).** Taking traces reproduces (a) in one line: since
$\text{Tr}[P_\lambda R(\tau)] = \chi^\lambda(\tau)\,\text{dim}\,V^d_\lambda$ on
$(\mathbb{C}^d)^{\otimes k}$,
$\text{Tr}[\Pi_{\vec\lambda}\Pi_{\text{Sym}}] = \frac{1}{k!}\sum_\tau \prod_i \chi^{\lambda^{(i)}}(\tau)\text{dim}\,V^{d_i}_{\lambda^{(i)}}$,
which is the right-hand side. That is the *same* computation, not an independent check —
which is why the numerical verification computes ranks by linear algebra instead
(`scripts/verify_support_theorem.py`, T1a/T1b).

---

## 4. From projector supports to Schmidt ranks

Let $r_i = \text{rank}\,\rho_i$ be the Schmidt rank of $\psi$ across the cut
$\{i\} \mid [n]\setminus\{i\}$.

**Theorem B (rank reachability).** Let $r = (r_1,\dots,r_n)$ with $1 \le r_i \le d_i$. There
exists a pure state $\psi \in \mathcal{H}$ with $\text{rank}\,\rho_i = r_i$ for all $i$
**iff** there exist $k$ and partitions $\lambda^{(i)} \vdash k$ with

$$\ell(\lambda^{(i)}) = r_i \ \ \text{for all } i,
\qquad\text{and}\qquad g(\lambda^{(1)},\dots,\lambda^{(n)}) > 0 .$$

*Proof.*

**($\Leftarrow$).** Suppose such $k, \vec\lambda$ exist. Work inside the subspace
$\mathcal{K} = \bigotimes_i \mathbb{C}^{r_i} \subseteq \mathcal{H}$, i.e. apply Theorem A
with local dimensions $r_i$ in place of $d_i$. Since $\ell(\lambda^{(i)}) = r_i$ we have
$\text{dim}\,V^{r_i}_{\lambda^{(i)}} > 0$, and $g(\vec\lambda) > 0$ by hypothesis, so
Theorem A(b) supplies a pure $\psi \in \mathcal{K}$ with $p_\psi(\vec\lambda) > 0$.

For that $\psi$: the $i$-th marginal of the outcome distribution is
$\text{Tr}[P^{(i)}_{\lambda^{(i)}}\rho_i^{\otimes k}] \ge p_\psi(\vec\lambda) > 0$, so by
Lemma 2, $\ell(\lambda^{(i)}) \le \text{rank}\,\rho_i$, i.e. $\text{rank}\,\rho_i \ge r_i$.
Conversely $\rho_i$ is a state on $\mathbb{C}^{r_i}$, so $\text{rank}\,\rho_i \le r_i$.
Hence $\text{rank}\,\rho_i = r_i$ exactly, and $\psi \in \mathcal{K} \subseteq \mathcal{H}$.

**($\Rightarrow$).** Suppose $\psi$ has $\text{rank}\,\rho_i = r_i$ for all $i$. By Lemma 2
every outcome in the support satisfies $\ell(\lambda^{(i)}) \le r_i$; we must produce one
achieving equality *simultaneously* for all $i$.

The $i$-th marginal of $p_\psi$ is exactly the bipartite Schur–Weyl measure of $\rho_i$
(sum over the other indices, using $\sum_\mu P^{(j)}_\mu = \mathbb{1}$). By the
Keyl–Werner concentration theorem [KW01], $\lambda^{(i)}/k \to \text{spec}\,\rho_i$ in
probability (exponentially fast). Let $p_{\min} > 0$ be the smallest nonzero eigenvalue over
all $\rho_i$. Choosing $\varepsilon < p_{\min}$, for $k$ large enough each event
$E_i = \{\ell(\lambda^{(i)}) < r_i\}$ has $\Pr[E_i] < 1/n$, since $\ell(\lambda^{(i)}) < r_i$
forces $\lambda^{(i)}_{r_i}/k = 0$ while $(\text{spec}\,\rho_i)_{r_i} \ge p_{\min}$. By the
union bound $\Pr[\bigcup_i E_i] < 1$, so with positive probability the outcome $\vec\lambda$
has $\ell(\lambda^{(i)}) = r_i$ for every $i$. That outcome has $p_\psi(\vec\lambda) > 0$, so
$g(\vec\lambda) > 0$ by Theorem A(b). $\square$

**Corollary C (the non-existence criterion).** Fix $r = (r_1,\dots,r_n)$. If

$$g(\lambda^{(1)},\dots,\lambda^{(n)}) = 0
\quad\text{for every } k \text{ and every } \vec\lambda \text{ with } \ell(\lambda^{(i)}) = r_i,$$

then **no** multipartite pure state — in any local dimensions — has Schmidt-rank vector $r$
across the $n$ local cuts.

Note the criterion is dimension-free: $g$ does not see $d_i$, matching the fact that
achievability of $r$ depends only on $r$ (given $r_i \le d_i$).

---

## 5. What the criterion actually yields

Computationally (`scripts/rank_reachability.py`, experiment E004), enumerating Kronecker
positivity by row lengths gives, for $n = 3$ and $n = 4$, $r_i \le 4$, $k \le 9$:

$$\{r : \exists k, \vec\lambda,\ \ell(\lambda^{(i)}) = r_i,\ g(\vec\lambda) > 0\}
\;=\; \Big\{r : r_i \le \prod_{j \ne i} r_j \ \ \forall i\Big\}$$

with zero discrepancies (37/37 triples, 208/208 quadruples; the set saturates by $k = 6$).
The right-hand side is the **polygon-admissible** set, which by Carlini–Kleppe [CK11] is
exactly the achievable set for every $n$.

So Corollary C is *correct and exact*, but the obstruction it delivers is the polygon
inequality and nothing more. The vanishing pattern of $g$ is far richer than its shadow on
row lengths — about half of all $\vec\lambda$ have $g = 0$, and deciding positivity is
NP-hard [IMW17] — but ranks only see $\ell(\lambda^{(i)})$, and that projection collapses to
the polygon.

**This is the honest headline:** Kronecker vanishing *does* certify non-existence of rank
vectors, exactly; it just does not certify anything the polygon inequalities did not
already give.

---

## 6. Why this stops at single-party cuts

To constrain a general bipartition $S \mid S^c$ one needs the isotypic projector
$P^{(S)}_\lambda$ built from the $S_k$ action permuting the copies of the whole block
$\mathcal{H}_S$. Then:

* $P^{(S)}_\lambda$ and $P^{(T)}_\mu$ **commute iff $S \subseteq T$, $T \subseteq S$, or
  $S \cap T = \emptyset$** — verified exactly in $\mathbb{C}[S_k]^{\otimes n}$ (0 violations
  in 6,619 nested/disjoint instances; 2,649 overlapping instances fail).
* So jointly measurable families are exactly **laminar families** (trees), and Theorem A
  generalises to a product of Kronecker coefficients over tree nodes.
* But on a single tree the reachable row-length data decouples into the node-local polygon
  conditions (E004/R3: 379 reachable = 379 predicted), so no new obstruction appears.
* Genuinely overlapping cuts — which is what CHLW-type obstructions need — are not jointly
  measurable at all, and both workarounds (independent copy-groups; sequential measurement)
  fail (E006).

The escape from this is *not* to measure but to impose the vanishing conditions as **linear
constraints on one state**; that is the subject of the companion note.

---

## 7. Prior work and what is genuinely new

**Already in the literature.**

* The multipartite weak Schur sampling POVM $\bigotimes_i P^{(i)}_{\lambda^{(i)}}$, the
  outcome probability $\|\Pi_{\vec\lambda}\psi^{\otimes k}\|^2$, and the generalized
  Kronecker coefficient (2) for arbitrary $n$: **Botero–Mejía 2018, Eqs. (4)–(8)** [BM18].
  Their Eq. (5) is exactly the "only if" half of Theorem A.
* The algebraic identity behind Theorem A(a): $n = 3$ in **Christandl–Doran–Walter** [CDW12]
  §VI.1; general $n$ in **Amanov–Yeliussizov** [AY25] §2.10.
* $\text{span}\{\psi^{\otimes k}\} = \text{Sym}^k$: **Harrow** [H13] Thm 3.
* Asymptotic spectrum/Kronecker dictionary (bipartite, stretched form):
  **Christandl–Mitchison** [CM06]; **Christandl–Harrow–Mitchison** [CHM07];
  **Klyachko** [K04].
* Concentration of $\lambda/k$ on $\text{spec}\,\rho$: **Keyl–Werner** [KW01].
* Row-length necessary condition $g_{\lambda\mu\nu} = 0$ if
  $\ell(\lambda) > \ell(\mu)\ell(\nu)$: **Bürgisser–Christandl–Ikenmeyer** [BCI11] §2.1.
* Achievable local-rank vectors are the polygon-admissible ones, all $n$:
  **Carlini–Kleppe** [CK11].
* NP-hardness of deciding $g > 0$, and failure of saturation ("holes"):
  **Ikenmeyer–Mulmuley–Walter** [IMW17].

**Apparently not written down.** The exact, finite-$k$, **two-way** statement of Theorem A;
the operational reading "these projector tuples annihilate every state"; Theorem B with
$\text{rank}\,\rho_i$ in place of $d_i$; and essentially anything for $n \ge 4$ on the
quantum side.

**Before claiming novelty, read:** Botero–Mejía [BM18] §II–III in full (highest risk of an
unstated version of Theorem A); Gonzalez [G25] (body could not be fetched); Michael Walter's
thesis (arXiv:1410.6820).

---

## 8. References

Verified against fetched arXiv abstract pages unless marked otherwise.

| Key | Reference | Link |
|---|---|---|
| [BM18] | A. Botero, J. Mejía, *Universal and distorsion-free entanglement concentration of multiqubit quantum states in the W class*, Phys. Rev. A **98**, 032326 (2018) | [arXiv:1712.09174](https://arxiv.org/abs/1712.09174) |
| [CM06] | M. Christandl, G. Mitchison, *The Spectra of Quantum States and the Kronecker Coefficients of the Symmetric Group*, Commun. Math. Phys. **261**(3):789–797 (2006) | [arXiv:quant-ph/0409016](https://arxiv.org/abs/quant-ph/0409016) |
| [CHM07] | M. Christandl, A. W. Harrow, G. Mitchison, *Nonzero Kronecker Coefficients and What They Tell Us about Spectra*, Commun. Math. Phys. **270** (2007) | [arXiv:quant-ph/0511029](https://arxiv.org/abs/quant-ph/0511029) |
| [K04] | A. Klyachko, *Quantum marginal problem and representations of the symmetric group* (2004) | [arXiv:quant-ph/0409113](https://arxiv.org/abs/quant-ph/0409113) |
| [KW01] | M. Keyl, R. F. Werner, *Estimating the spectrum of a density operator*, Phys. Rev. A **64**, 052311 (2001) | [arXiv:quant-ph/0102027](https://arxiv.org/abs/quant-ph/0102027) |
| [H13] | A. W. Harrow, *The Church of the Symmetric Subspace* (2013) | [arXiv:1308.6595](https://arxiv.org/abs/1308.6595) |
| [CDW12] | M. Christandl, B. Doran, M. Walter, *Computing Multiplicities of Lie Group Representations*, FOCS 2012 | [arXiv:1204.4379](https://arxiv.org/abs/1204.4379) |
| [AY25] | A. Amanov, D. Yeliussizov, *Highest weight vectors of tensors* (2025) | [arXiv:2504.15413](https://arxiv.org/abs/2504.15413) |
| [BCI11] | P. Bürgisser, M. Christandl, C. Ikenmeyer, *Nonvanishing of Kronecker coefficients for rectangular shapes*, Adv. Math. **227** (2011) | [arXiv:0910.4512](https://arxiv.org/abs/0910.4512) |
| [IMW17] | C. Ikenmeyer, K. D. Mulmuley, M. Walter, *On vanishing of Kronecker coefficients*, comput. complex. (2017) | [arXiv:1507.02955](https://arxiv.org/abs/1507.02955) |
| [G25] | W. Gonzalez, *Kronecker states: a powerful source of multipartite maximally entangled states in quantum information* (2025) | [arXiv:2504.16256](https://arxiv.org/abs/2504.16256) |
| [CHLW14] | J. Cadney, M. Huber, N. Linden, A. Winter, *Inequalities for the ranks of multipartite quantum states*, Linear Algebra Appl. **452**:153–171 (2014) | [arXiv:1308.0539](https://arxiv.org/abs/1308.0539) |
| [CK11] | E. Carlini, J. Kleppe, *Ranks derived from multilinear maps*, J. Pure Appl. Algebra **215**(8):1999–2004 (2011) | **no arXiv preprint** — see caveat below |

**Citation caveat for [CK11].** The ScienceDirect page returns HTTP 403 and there is no
arXiv preprint. Title/authors/journal/volume/pages/year were verified from the Monash
research portal; the *statement* of the theorem is verified only **indirectly**, from two
papers that recall it verbatim: Jovcheva–Seynnaeve–Vannieuwenhoven
[arXiv:2509.09463](https://arxiv.org/abs/2509.09463) Eq. (3), and Gharahi–Mancini–Ottaviani
[arXiv:1910.09665](https://arxiv.org/abs/1910.09665) App. B. **Obtain the original PDF before
citing it as a primary reference.**

---

## 9. Computational verification

| Claim | Script | Status |
|---|---|---|
| Theorem A(a), dimension formula | `scripts/verify_support_theorem.py` (T1a, T1b) | 203/203 $\vec\lambda$-tuples, $n \le 5$, $k \le 5$ |
| Theorem A(b), support | same (T2, T3) | 10/10 settings, exact match |
| Theorem B / Corollary C vs polygon | `scripts/rank_reachability.py` | 0 discrepancies, $k \le 9$ |
| Kronecker routine | cross-checked vs independent implementation | 590 pairs/triples, 0 mismatches |

Theorem B direction ($\Rightarrow$) relies on Keyl–Werner concentration and is **stated, not
independently verified numerically** — see `TODO.md`.
