# Literature review: multipartite weak Schur sampling and Kronecker positivity

Compiled 2026-09-02. Every entry in Section (A) was verified by fetching the arXiv abstract page
(and, where noted, the arXiv HTML / ar5iv full text). Anything I could not verify is in Section (D).

## Notation used throughout

- `H = H_1 ⊗ ... ⊗ H_n`, `d_i = dim H_i`, `|psi> in H` pure, `k` copies.
- `P_{lambda}^{(i)}` = projector onto the `lambda`-isotypic component of the `S_k` action on `H_i^{⊗k}`.
- `bold-lambda = (lambda^(1),...,lambda^(n))`, each `lambda^(i) |- k`.
- `P_bold-lambda = P_{lambda^(1)}^{(1)} ⊗ ... ⊗ P_{lambda^(n)}^{(n)}` acting on `H^{⊗k} ≅ (H_1^{⊗k}) ⊗ ... ⊗ (H_n^{⊗k})`.
- `p_psi(bold-lambda) = || P_bold-lambda |psi>^{⊗k} ||^2`.
- `g(lambda^(1),...,lambda^(n)) = dim ( S^{lambda^(1)} ⊗ ... ⊗ S^{lambda^(n)} )^{S_k}`  (n-fold / "generalized" Kronecker coefficient).
- `V_lambda^{(d)} = S^lambda(C^d)` the Schur (GL_d) module; `V_lambda^{(d)} = 0` iff `ell(lambda) > d`.

**The conjecture under study (C):**
`p_psi(bold-lambda) > 0` for some `psi` in `H`  <=>  `g(lambda^(1),...,lambda^(n)) > 0` AND `ell(lambda^(i)) <= d_i` for all `i`.

---

## (A) Verified bibliography

Each entry: verified against the fetched arXiv abstract page (title, full author list, year, venue,
arXiv ID, DOI where shown). "HTML-verified" means I additionally read the arXiv-HTML / ar5iv full text
and the quoted claim is from that text.

### A.1 Directly on the multipartite weak-Schur-sampling setup

```bibtex
@article{BoteroMejia2018,
  author  = {Alonso Botero and Jos{\'e} Mej{\'i}a},
  title   = {Universal and distorsion-free entanglement concentration of multiqubit
             quantum states in the {W} class},
  journal = {Physical Review A},
  volume  = {98},
  pages   = {032326},
  year    = {2018},
  eprint  = {1712.09174},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note    = {arXiv v1 26 Dec 2017, v2 19 Jun 2018}
}
```
**What it proves / contains (HTML-verified via ar5iv).** This is the closest thing in the literature to
our exact setup. They construct precisely the multipartite weak Schur sampling POVM: "each party then
performs a measurement of the set of projectors `{P_{lambda(i)} | lambda(i) |- n}` onto the subspaces
`V_{lambda(i)} ⊗ [lambda(i)]`". Their Eq. (4) is the multipartite Schur–Weyl decomposition
`H^{⊗n} = ⊕_{bold-lambda} V_bold-lambda ⊗ [bold-lambda]` as a `GL_d^{×N} × S_n^{×N}` module, and their
**Eq. (5) is exactly the "only if" half of our conjecture**:
`|psi>^{⊗n} ∈ ⊕_{bold-lambda} V_bold-lambda ⊗ [bold-lambda]^{S_n}`,
where `[bold-lambda]^{S_n}` is the invariants under the *coordinated* (diagonal) `S_n` action. Their
Eq. (6) defines the **generalized Kronecker coefficient**
`k_bold-lambda = (1/n!) Σ_pi χ_{lambda(1)}(pi) ... χ_{lambda(N)}(pi) = dim [bold-lambda]^{S_n}`,
for arbitrary `N` (i.e. n-fold, not just triples). Their Eq. (8) gives the post-measurement state
`|psi>^{⊗n} --P_bold-lambda--> Σ_{s=1}^{k_bold-lambda} |Phi_{bold-lambda,s}(psi)> ⊗ |K_{bold-lambda,s}>`,
and their outcome probability is `p(bold-lambda | psi) = || P_bold-lambda |psi>^{⊗n} ||^2`. They define
**Kronecker states**: "Any normalized state `|K_bold-lambda> ∈ [bold-lambda]^{S_n}` is called a Kronecker
state", and note these have maximally mixed marginals. They do **not** state the support theorem (C).

```bibtex
@article{Mejia2016,
  author  = {Jose Mejia},
  title   = {Entanglement distillation using {S}chur-{W}eyl decomposition for three qubits},
  year    = {2016},
  eprint  = {1610.09552},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi     = {10.48550/arXiv.1610.09552}
}
```
**What it proves.** Three-qubit special case of the above: asymptotic rates for the probability of landing
in an invariant subspace of the Wedderburn decomposition with "effective Kronecker coefficient
`g_{alpha beta gamma} = 1`", plus a combinatorial formula for two-row Young diagrams. Verified abstract
only; I could not extract the body (PDF text layer unreadable).

```bibtex
@article{Gonzalez2025,
  author  = {Walther Gonzalez},
  title   = {Kronecker states: a powerful source of multipartite maximally entangled states
             in quantum information},
  year    = {2025},
  eprint  = {2504.16256},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi     = {10.48550/arXiv.2504.16256}
}
```
**What it proves.** Construction methods ("W-state stitching", tensor-network representations) for
Kronecker states — locally maximally entangled states spanning Kronecker subspaces `[bold-lambda]^{S_k}`,
whose dimensions are Kronecker coefficients. Constructive, not a support/positivity theorem.
(Abstract page verified; full HTML exceeded the fetch size limit, so body claims are unverified.)

### A.2 The bipartite/tripartite spectral theory (Christandl–Mitchison / CHM / Klyachko / Keyl–Werner)

```bibtex
@article{ChristandlMitchison2006,
  author  = {Matthias Christandl and Graeme Mitchison},
  title   = {The Spectra of Quantum States and the Kronecker Coefficients of the Symmetric Group},
  journal = {Communications in Mathematical Physics},
  volume  = {261},
  number  = {3},
  pages   = {789--797},
  year    = {2006},
  doi     = {10.1007/s00220-005-1435-1},
  eprint  = {quant-ph/0409016},
  archivePrefix = {arXiv},
  note    = {arXiv title is "The Spectra of Density Operators and the Kronecker
             Coefficients of the Symmetric Group"; v1 2 Sep 2004, v2 20 Feb 2006}
}
```
**What it proves.** For a bipartite state `rho^AB` with spectra `(r^A, r^B, r^AB)`, one can attach Young
diagrams whose normalised row lengths approximate the spectra, and **for allowed spectra the composite
diagram's representation is contained in the tensor product of the two subsystem representations**, i.e.
`g_{mu nu lambda} != 0` for a sequence of diagrams converging to the spectra. This is the "forward"
(necessary) direction, in *asymptotic* form, for `n = 2` marginals + the joint state. Note: the arXiv
title differs from the published CMP title — cite carefully.

```bibtex
@article{ChristandlHarrowMitchison2007,
  author  = {Matthias Christandl and Aram W. Harrow and Graeme Mitchison},
  title   = {Nonzero {K}ronecker Coefficients and What They Tell Us about Spectra},
  journal = {Communications in Mathematical Physics},
  volume  = {270},
  pages   = {575--585},
  year    = {2007},
  doi     = {10.1007/s00220-006-0157-3},
  eprint  = {quant-ph/0511029},
  archivePrefix = {arXiv},
  note    = {arXiv title: "On Nonzero Kronecker Coefficients and their
             Consequences for Spectra"; submitted 3 Nov 2005}
}
```
**What it proves (HTML-verified).** Four results in our notation:
- **Thm 2.1 (estimation / Keyl–Werner).** `Tr[P_lambda rho^{⊗k}] <= (k+1)^{d(d-1)/2} exp(-k D(bar-lambda || r))`.
- **Thm 2.2.** For every `rho^AB` there is a sequence `(mu^(j), nu^(j), lambda^(j))` of partitions with
  `g_{mu nu lambda} != 0` converging to `(Spec rho^A, Spec rho^B, Spec rho^AB)`.
- **Thm 2.4 (Klyachko's converse).** "Let `mu, nu` and `lambda` be diagrams with `k` boxes and at most
  `m`, `n` and `mn` rows, respectively. If `g_{mu nu lambda} != 0`, then there exists a density operator
  `rho^AB` on `C^m ⊗ C^n`" with (approximately) those spectra. **The row bounds `m, n, mn` are an explicit
  hypothesis** — this is the local-dimension half of our conjecture, stated for the bipartite case.
- **Thm 3.1 (semigroup property, settling a Klyachko conjecture).** "`Kron` is a semigroup with respect to
  row-wise addition, i.e. `g_{mu nu lambda} != 0` and `g_{mu' nu' lambda'} != 0` implies
  `g_{mu+mu', nu+nu', lambda+lambda'} != 0`."
- **Thm 3.4.** `SPEC`, the set of admissible spectral triples, is a convex polytope.
- The proof of Thm 2.4 goes through **Lemma 5.1**, which *constructs a pure state `|psi>` from a nonzero
  invariant vector in `(V_mu ⊗ V_nu ⊗ V_lambda)^{S_k}`* and then applies the estimation theorem. This is
  the same mechanism our conjecture uses, but deployed asymptotically for spectra rather than as an exact
  finite-`k` support statement.

```bibtex
@article{Klyachko2004,
  author  = {Alexander Klyachko},
  title   = {Quantum marginal problem and representations of the symmetric group},
  year    = {2004},
  eprint  = {quant-ph/0409113},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi     = {10.48550/arXiv.quant-ph/0409113},
  note    = {submitted 17 Sep 2004}
}
```
**What it proves (HTML-verified via ar5iv).** Complete solution of the mixed-state marginal problem by
linear spectral inequalities, with tables up to 4 qubits; then a representation-theoretic route.
**Theorem 5.3.1** is the key statement: "The following conditions are equivalent... (1)
`g(m lambda, m nu, m mu) != 0` for some `m > 0` ... (2) There exists mixed state `rho_AB` of two component
system ... with spectrum `nu` and margins of spectra `lambda, mu`." Section 5.3, Eq. (5.7), identifies
restriction multiplicities with Kronecker coefficients. **Bipartite only** — I found no `n >= 3`
Kronecker statement and no `Sym^k(H_1 ⊗ ... ⊗ H_n)` decomposition in this paper.

```bibtex
@article{KeylWerner2001,
  author  = {M. Keyl and R. F. Werner},
  title   = {Estimating the spectrum of a density operator},
  journal = {Physical Review A},
  volume  = {64},
  pages   = {052311},
  year    = {2001},
  doi     = {10.1103/PhysRevA.64.052311},
  eprint  = {quant-ph/0102027},
  archivePrefix = {arXiv},
  note    = {submitted 5 Feb 2001}
}
```
**What it proves.** *Single-system* weak Schur sampling: measuring `{P_lambda}` on `rho^{⊗N}`, the
normalised row lengths `bar-lambda` converge to `Spec rho` as `N -> infinity`, with error probability
decaying exponentially at an explicitly computed (relative-entropy) rate. This is the analytic engine
behind everything above. It is a *concentration* statement; the exact support statement
(`Tr[P_lambda rho^{⊗k}] = 0` iff `ell(lambda) > rank rho`) is the trivial companion fact.

### A.3 Kronecker = multiplicity in a symmetric power (the algebraic core of our conjecture)

```bibtex
@inproceedings{ChristandlDoranWalter2012,
  author    = {Matthias Christandl and Brent Doran and Michael Walter},
  title     = {Computing Multiplicities of {L}ie Group Representations},
  booktitle = {Proceedings of the 2012 IEEE 53rd Annual Symposium on Foundations of
               Computer Science (FOCS'12)},
  pages     = {639--648},
  year      = {2012},
  eprint    = {1204.4379},
  archivePrefix = {arXiv},
  note      = {v1 19 Apr 2012, v2 30 Oct 2012}
}
```
**What it proves (HTML-verified) — THE key algebraic identity, stated for n = 3.** Their §VI.1:
"Let `H = U(a) x U(b) x U(c)` and `G = U(abc)` ... the Kronecker coefficient `g_{lambda,mu,nu}` is then
given by the multiplicity of the irreducible `H`-representation
`V_{H,(lambda,mu,nu)} = V_{U(a),lambda} ⊗ V_{U(b),mu} ⊗ V_{U(c),nu}` in the restriction of the symmetric
power `Sym^k(C^{abc})`."
That is: `Sym^k(C^a ⊗ C^b ⊗ C^c) = ⊕_{lambda,mu,nu} g(lambda,mu,nu) · S^lambda(C^a) ⊗ S^mu(C^b) ⊗ S^nu(C^c)`.
They do **not** state the `n >= 4` version. Their algorithm is polynomial time for a bounded number of rows.

```bibtex
@article{AmanovYeliussizov2025,
  author  = {Alimzhan Amanov and Damir Yeliussizov},
  title   = {Highest weight vectors of tensors},
  year    = {2025},
  eprint  = {2504.15413},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2504.15413},
  note    = {v1 21 Apr 2025, revised 27 Jan 2026}
}
```
**What it proves (HTML-verified) — the n-fold version of the same identity.** §2.10: "The generalized
Kronecker coefficient `g(bold-lambda)`, indexed by a `d`-tuple `bold-lambda` of partitions of `m`, is the
multiplicity of the trivial `S_m`-irreducible representation `[1^m]` in the tensor product
`[lambda^(1)] ⊗ ... ⊗ [lambda^(d)]`." And, crucially,
`dim HWV_bold-lambda ( Sym^m (C^n)^{⊗d} ) = g(1^m, bold-lambda) = g(bold-lambda)`, for any `d >= 2`
(with the alternating analogue `dim HWV_bold-lambda ( Λ^m (C^n)^{⊗d} ) = g(m·1, bold-lambda)`).
This is *exactly* the algebraic content of our conjecture's `n`-fold statement, phrased via highest-weight
vectors. It is stated for equal local dimensions `n` and does not discuss the `ell(lambda^(i)) <= d_i`
vanishing condition explicitly (it is implicit in `HWV` being taken inside `(C^n)^{⊗d}`).

```bibtex
@article{AmanovYeliussizov2023,
  author  = {Alimzhan Amanov and Damir Yeliussizov},
  title   = {Some unimodal sequences of {K}ronecker coefficients},
  year    = {2023},
  eprint  = {2312.17054},
  archivePrefix = {arXiv},
  primaryClass = {math.CO},
  doi     = {10.48550/arXiv.2312.17054}
}
```
**What it proves (HTML-verified).** Defines `g(lambda^(1),...,lambda^(d))` as the multiplicity of `[(m)]`
in `[lambda^(1)] ⊗ ... ⊗ [lambda^(d)]`, identifies it with `dim HWV_{bold-lambda'} Λ^m V` for
`V = (C^k)^{⊗d}` under `GL(k)^{×d}`; **Conjecture 1.2** (unimodality of `{g(rho_k^n bold-lambda)}` for odd
`d >= 3`, even `k`), proven for `k = 2` (**Theorem 1.6**). One of the few papers doing genuine `n >= 4`
generalized-Kronecker work.

```bibtex
@article{Harrow2013,
  author  = {Aram W. Harrow},
  title   = {The Church of the Symmetric Subspace},
  year    = {2013},
  eprint  = {1308.6595},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi     = {10.48550/arXiv.1308.6595},
  note    = {submitted 29 Aug 2013}
}
```
**What it proves (HTML-verified).** Review of the symmetric subspace. **Theorem 3** is the spanning fact
our conjecture needs: `Sym^n(C^d) = span{ |phi>^{⊗n} : |phi> in C^d }`.

### A.4 Quantum algorithms / complexity for Kronecker coefficients

```bibtex
@article{BravyiChowdhuryGossetHavlicekZhu2024,
  author  = {Sergey Bravyi and Anirban Chowdhury and David Gosset and Vojtech Havlicek and Guanyu Zhu},
  title   = {Quantum complexity of the {K}ronecker coefficients},
  journal = {PRX Quantum},
  volume  = {5},
  number  = {1},
  pages   = {010329},
  year    = {2024},
  eprint  = {2302.11454},
  archivePrefix = {arXiv},
  note    = {v1 22 Feb 2023, v3 7 May 2024}
}
```
**What it proves (HTML-verified).** **Lemma 1:** `g_{mu nu lambda} = (1/(d_mu d_nu d_lambda)) Tr(P_{mu nu lambda})`
where `P_{mu nu lambda} = Q (Pi_mu^L ⊗ Pi_nu^L ⊗ Pi_lambda^L)` and `Q = (1/n!) Σ_{sigma in S_n} sigma ⊗ sigma ⊗ sigma`,
acting on `(C S_n)^{⊗3}` (three copies of the left regular representation). Also
`Tr(P_{mu nu lambda}) = (1/n!) Σ_sigma χ^mu χ^nu χ^lambda`. They prove **positivity of Kronecker
coefficients is in QMA** and give an efficient quantum algorithm approximating normalised Kronecker
coefficients to inverse-polynomial additive error. **Structurally this is our POVM** — the diagonal `S_n`
symmetriser `Q` composed with a tensor product of isotypic projectors — but carried out on the regular
representation rather than on `k` copies of a physical state, and only for three partitions.

```bibtex
@article{IkenmeyerSubramanian2023,
  author  = {Christian Ikenmeyer and Sathyawageeswar Subramanian},
  title   = {A remark on the quantum complexity of the {K}ronecker coefficients},
  year    = {2023},
  eprint  = {2307.02389},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi     = {10.48550/arXiv.2307.02389},
  note    = {submitted 5 Jul 2023; journal version: Kronecker Coefficients in \#BQP,
             ACM Trans. Quantum Computing, doi:10.1145/3762673}
}
```
**What it proves.** Kronecker coefficients (and plethysm coefficients) are in `#BQP`, strengthening
Bravyi et al. Abstract verified; the ACM journal title/DOI is from a search result, not a fetched page —
treat the journal metadata as tentative.

```bibtex
@article{LaroccaHavlicek2024,
  author  = {Martin Larocca and Vojtech Havlicek},
  title   = {Quantum Algorithms for Representation-Theoretic Multiplicities},
  year    = {2024},
  eprint  = {2407.17649},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi     = {10.48550/arXiv.2407.17649},
  note    = {v1 24 Jul 2024, revised 25 Mar 2025}
}
```
**What it proves (HTML-verified).** Quantum algorithms for Kostka, Littlewood–Richardson, plethysm and
Kronecker coefficients via weak Fourier sampling of a subgroup restriction: `p_alpha(beta) =
mult(r^H_beta, r^G_alpha ↓_H) · d_beta / d_alpha`. Their Kronecker definition is
`g_{lambda mu nu} = mult(r^{S_n}_nu, r^{S_n}_lambda ⊗ r^{S_n}_mu)`. **They do not use the `Sym^k` /
tensor-product-of-isotypic-projectors picture, and do not treat n-fold coefficients.**

```bibtex
@article{Panova2025,
  author  = {Greta Panova},
  title   = {Polynomial time classical versus quantum algorithms for representation
             theoretic multiplicities},
  year    = {2025},
  eprint  = {2502.20253},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2502.20253},
  note    = {v1 27 Feb 2025, revised 18 Oct 2025; TQC 2025 / Computational Complexity}
}
```
**What it proves.** Classical polynomial-time algorithms for many of the Kronecker/plethysm families that
Larocca–Havlicek handled quantumly, refuting their conjecture and limiting the claimed quantum speedup.

### A.5 Vanishing / positivity of Kronecker coefficients

```bibtex
@article{IkenmeyerMulmuleyWalter2017,
  author  = {Christian Ikenmeyer and Ketan D. Mulmuley and Michael Walter},
  title   = {On vanishing of {K}ronecker coefficients},
  journal = {computational complexity},
  volume  = {26},
  number  = {4},
  pages   = {949--992},
  year    = {2017},
  doi     = {10.1007/s00037-017-0158-y},
  eprint  = {1507.02955},
  archivePrefix = {arXiv},
  note    = {v1 10 Jul 2015, last revised 21 Jul 2017}
}
```
**What it proves.** **Deciding positivity of Kronecker coefficients is NP-hard** (so Kronecker is harder
than Littlewood–Richardson unless P = NP; this refuted Mulmuley's earlier "in P" conjecture). Also: a
`#P` formula for a restricted subclass; and **"holes" in the saturation/stretching property** — triples
with `g(lambda,mu,nu) = 0` but `g(N lambda, N mu, N nu) > 0` for some `N` — are abundant and efficiently
findable. This is exactly the obstruction that makes the *finite-`k`* support of multipartite weak Schur
sampling strictly smaller than the *asymptotic* (spectral / polytope) picture.

```bibtex
@article{BurgisserChristandlIkenmeyer2011,
  author  = {Peter B{\"u}rgisser and Matthias Christandl and Christian Ikenmeyer},
  title   = {Nonvanishing of {K}ronecker coefficients for rectangular shapes},
  journal = {Advances in Mathematics},
  volume  = {227},
  pages   = {2082--2091},
  year    = {2011},
  doi     = {10.1016/j.aim.2011.04.012},
  eprint  = {0910.4512},
  archivePrefix = {arXiv},
  note    = {v1 23 Oct 2009, v4 11 Jun 2012}
}
```
**What it proves (HTML-verified).** For any partition `lambda` of size `ell d` with at most `d^2` parts
there is a stretching factor `k >= 1` with `g(k lambda, k □, k □) != 0` (`□` = the `d x ell` rectangle),
plus an effective bound `k = O(d^4 eps^{-2} log(d/eps))` in the approximate version, and a transfer to
symmetric Kronecker coefficients (`g(lambda,mu,mu) != 0 ⟹ sg_{2 lambda} != 0`).
**Also §2.1 records the row-length vanishing criterion verbatim: "It is known that `g_{lambda,mu,nu}`
vanishes if `ell(lambda) > ell(mu) ell(nu)`."** and gives the `GL(d_1 d_2) ↓ GL(d_1) x GL(d_2)`
restriction interpretation (their Eq. (1)).

```bibtex
@article{Ressayre2019,
  author  = {Nicolas Ressayre},
  title   = {Horn inequalities for nonzero {K}ronecker coefficients},
  journal = {Advances in Mathematics},
  year    = {2019},
  eprint  = {1907.07931},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.1907.07931},
  note    = {submitted 18 Jul 2019}
}
```
**What it proves.** "We extend the essential Horn inequalities to the triples of partitions corresponding
to a nonzero Kronecker coefficient" — i.e. explicit linear necessary conditions on `(lambda,mu,nu)` for
`g != 0` beyond the row bounds. (Abstract-level verification; I did not confirm the exact inequality list
in the body, and could not confirm whether row-length conditions are stated there.)

```bibtex
@article{PakPanova2014,
  author  = {Igor Pak and Greta Panova},
  title   = {Bounds on the {K}ronecker coefficients},
  year    = {2014},
  eprint  = {1406.2988},
  archivePrefix = {arXiv},
  primaryClass = {math.CO},
  doi     = {10.48550/arXiv.1406.2988},
  note    = {v1 11 Jun 2014, v2 16 Jun 2014}
}
```
**What it proves.** Upper and lower bounds on `g(lambda,mu,nu)`; introduces a "k-stability" notion
generalising ordinary stability; lower bound via character estimates; applications to unimodality of
q-binomials. Relevant quantitative fact reported in secondary sources: for `ell(lambda) <= a`,
`ell(mu) <= b`, `ell(nu) <= c` one has `g <= 2^{abc}` — I did **not** verify this specific inequality
against the paper body.

```bibtex
@article{IkenmeyerPanova2024,
  author  = {Christian Ikenmeyer and Greta Panova},
  title   = {All {K}ronecker coefficients are reduced {K}ronecker coefficients},
  journal = {Forum of Mathematics, Pi},
  volume  = {12},
  pages   = {e22},
  year    = {2024},
  doi     = {10.1017/fmp.2024.23},
  eprint  = {2305.03003},
  archivePrefix = {arXiv},
  note    = {submitted 4 May 2023}
}
```
**What it proves.** Every Kronecker coefficient equals a reduced Kronecker coefficient by explicit
construction; hence Stanley's (2000) and Kirillov's (2004) combinatorial-interpretation questions are
equivalent, positivity of reduced Kronecker coefficients is NP-hard, and computing them is #P-hard under
parsimonious reductions.

```bibtex
@article{BriandOrellanaRosas2011,
  author  = {Emmanuel Briand and Rosa Orellana and Mercedes Rosas},
  title   = {The stability of the {K}ronecker products of {S}chur functions},
  journal = {Journal of Algebra},
  year    = {2011},
  doi     = {10.1016/j.jalgebra.2010.12.026},
  eprint  = {0907.4652},
  archivePrefix = {arXiv},
  note    = {v1 27 Jul 2009, v2 6 Aug 2009}
}
```
**What it proves.** Determines the exact `n` at which *all* coefficients of a Kronecker product of Schur
functions stabilise (Murnaghan stability), improving Brion's and Vallejo's bounds.

```bibtex
@article{SamSnowden2016,
  author  = {Steven V Sam and Andrew Snowden},
  title   = {Proof of {S}tembridge's conjecture on stability of {K}ronecker coefficients},
  journal = {Journal of Algebraic Combinatorics},
  volume  = {43},
  number  = {1},
  pages   = {1--10},
  year    = {2016},
  doi     = {10.1007/s10801-015-0622-1},
  eprint  = {1501.00333},
  archivePrefix = {arXiv},
  note    = {v1 2 Jan 2015, v2 11 Jul 2015}
}
```
**What it proves.** Stembridge's conjecture on stable Kronecker triples (a vast generalisation of
Murnaghan stability): the sequence `g(lambda + N alpha, mu + N beta, nu + N gamma)` stabilises for stable
triples. Proof by identifying the sequences with Hilbert functions of modules over finitely generated
algebras; uses only Schur–Weyl duality and Borel–Weil. (Journal metadata from search; arXiv abstract page
was not directly fetched — see (D).)

### A.6 Multipartite marginals, polytopes and local ranks

```bibtex
@article{WalterDoranGrossChristandl2013,
  author  = {Michael Walter and Brent Doran and David Gross and Matthias Christandl},
  title   = {Entanglement Polytopes: Multiparticle Entanglement from Single-Particle Information},
  journal = {Science},
  volume  = {340},
  number  = {6137},
  pages   = {1205--1208},
  year    = {2013},
  doi     = {10.1126/science.1232957},
  eprint  = {1208.0365},
  archivePrefix = {arXiv}
}
```
**What it proves.** Each SLOCC entanglement class of a multipartite pure state has an associated
*entanglement polytope* of compatible one-body spectra; membership certificates detect entanglement class
from local data alone. The relevant multiplicities are (asymptotically) the `n`-fold Kronecker
coefficients / moment-polytope data. (Journal metadata verified via search; arXiv abs page not fetched —
see (D).)

```bibtex
@article{ChristandlDoranKousidisWalter2014,
  author  = {Matthias Christandl and Brent Doran and Stavros Kousidis and Michael Walter},
  title   = {Eigenvalue Distributions of Reduced Density Matrices},
  journal = {Communications in Mathematical Physics},
  volume  = {332},
  pages   = {1--52},
  year    = {2014},
  doi     = {10.1007/s00220-014-2144-4},
  eprint  = {1204.0741},
  archivePrefix = {arXiv},
  note    = {v1 3 Apr 2012, revised 20 Oct 2014}
}
```
**What it proves (HTML-verified in part).** Symplectic-geometry method for the joint eigenvalue
distribution of the one-body reduced density matrices of a random multipartite pure state; solves the
one-body quantum marginal problem by reduction to a classical marginal problem; efficient computation of
Kronecker and plethysm coefficients. States in the introduction: "the existence of a pure tripartite
quantum state with given marginal eigenvalue spectra is equivalent to the asymptotic non-vanishing of an
associated sequence of Kronecker coefficients". Handles `n` parties for the *polytope*, but does not write
the `n`-fold Kronecker `Sym^k` decomposition.

```bibtex
@phdthesis{Walter2014,
  author  = {Michael Walter},
  title   = {Multipartite Quantum States and their Marginals},
  school  = {ETH Zurich},
  year    = {2014},
  note    = {Diss. ETH No. 22051},
  eprint  = {1410.6820},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.1410.6820}
}
```
**What it proves.** Consolidates arXiv:1208.0365, 1204.0741, 1204.4379, 1302.6990, 1210.0463: part I on
one-body marginals of multipartite states (entanglement polytopes, moment polytopes, Kronecker/plethysm
multiplicities), part II on general quantum marginals from an entropy perspective. Best single
"background text" for the multipartite moment-polytope side. (Abstract page verified; I did not verify
individual theorem statements inside the thesis.)

```bibtex
@article{VergneWalter2017,
  author  = {Mich{\`e}le Vergne and Michael Walter},
  title   = {Inequalities for Moment Cones of Finite-Dimensional Representations},
  journal = {Journal of Symplectic Geometry},
  volume  = {15},
  number  = {4},
  pages   = {1209--1250},
  year    = {2017},
  doi     = {10.4310/JSG.2017.v15.n4.a8},
  eprint  = {1410.8144},
  archivePrefix = {arXiv},
  note    = {v1 29 Oct 2014, v3 29 Jan 2016}
}
```
**What it proves.** Finitely many explicit linear inequalities describing the moment cone of an arbitrary
finite-dimensional unitary representation of a compact connected Lie group. Applications: generalised Horn
inequalities, new inequalities for the one-body quantum marginal problem, i.e. **for the asymptotic
support of the (n-fold) Kronecker coefficients of the symmetric group**. This is the sharpest general
`n`-party result on the *asymptotic* support.

```bibtex
@article{BryanReichsteinVanRaamsdonk2018,
  author  = {Jim Bryan and Zinovy Reichstein and Mark Van Raamsdonk},
  title   = {Existence of locally maximally entangled quantum states via geometric invariant theory},
  journal = {Annales Henri Poincar{\'e}},
  volume  = {19},
  pages   = {2491--2511},
  year    = {2018},
  doi     = {10.1007/s00023-018-0682-6},
  eprint  = {1708.01645},
  archivePrefix = {arXiv},
  note    = {v1 4 Aug 2017, v2 28 Sep 2017}
}
```
**What it proves.** For `V_1 ⊗ ... ⊗ V_n`, locally maximally entangled (LME) states — every one-body
marginal proportional to the identity — exist **iff an explicit function `R(d_1,...,d_n) >= 0`**, computed
as a product / alternating sum in gcds of the `d_i`. A recursive algorithm decides the question and
computes the quotient dimension. This is the cleanest existing "local dimensions ⟹ existence of a
multipartite state with prescribed local behaviour" result.

```bibtex
@article{BryanLeutheusserReichsteinVanRaamsdonk2019,
  author  = {Jim Bryan and Samuel Leutheusser and Zinovy Reichstein and Mark Van Raamsdonk},
  title   = {Locally Maximally Entangled States of Multipart Quantum Systems},
  journal = {Quantum},
  volume  = {3},
  pages   = {115},
  year    = {2019},
  eprint  = {1801.03508},
  archivePrefix = {arXiv},
  note    = {v1 10 Jan 2018, v3 29 Dec 2018}
}
```
**What it proves.** Pedagogical companion: conditions for LME-state existence in terms of subsystem
dimensions, dimensions of the space of LME states modulo local unitaries, explicit `(2, A, B)`
constructions, a representation-theoretic family of stabilizer LME states, dimensions of SLOCC classes,
and when stabilizers are trivial.

```bibtex
@article{HuberWyderka2025,
  author  = {Felix Huber and Nikolai Wyderka},
  title   = {Refuting spectral compatibility of quantum marginals},
  journal = {Quantum},
  volume  = {9},
  pages   = {1918},
  year    = {2025},
  doi     = {10.22331/q-2025-11-20-1918},
  eprint  = {2211.06349},
  archivePrefix = {arXiv},
  note    = {submitted 11 Nov 2022}
}
```
**What it proves.** A symmetry-reduced SDP hierarchy, complete for detecting incompatibility of prescribed
overlapping-marginal spectra, producing dimension-free refutation certificates; extended to Hermitian sum
problems and **to certifying vanishing of Kronecker coefficients**. Practically the best computational tool
for refuting membership in the support.

### A.7 Schur transform / weak Schur sampling machinery

```bibtex
@article{BaconChuangHarrow2006,
  author  = {Dave Bacon and Isaac Chuang and Aram Harrow},
  title   = {Efficient Quantum Circuits for {S}chur and {C}lebsch-{G}ordan Transforms},
  journal = {Physical Review Letters},
  volume  = {97},
  pages   = {170502},
  year    = {2006},
  eprint  = {quant-ph/0407082},
  archivePrefix = {arXiv},
  note    = {v1 12 Jul 2004, v4 5 Aug 2004}
}
```
**What it proves.** `poly(n, d, log(1/eps))`-size circuits for the Schur transform on `n` qudits, built
from Clebsch–Gordan transforms; **and an efficient circuit for the restricted "project onto Schur
subspaces" task — i.e. weak Schur sampling — via nonabelian generalized phase estimation.** Makes the
`{P_lambda}` POVM (and hence `P_bold-lambda`, applied locally by each party) efficiently implementable.

```bibtex
@article{CerveroMancinska2023,
  author  = {Enrique Cervero and Laura Man{\v c}inska},
  title   = {Weak {S}chur sampling with logarithmic quantum memory},
  year    = {2023},
  eprint  = {2309.11947},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi     = {10.48550/arXiv.2309.11947},
  note    = {submitted 21 Sep 2023}
}
```
**What it proves.** Streaming weak-Schur-sampling algorithm returning both the Young label and the
multiplicity label, using `O(log_2 n)` qubits of quantum memory and `O(n^3 log_2(n/eps))` Clifford+T gates
for `n` qubits (qudit version: `O(d n^{2d} log_4(n^{2d}/eps))` gates, `O(log_d n)` qudits).

```bibtex
@article{ChenWangZhang2024,
  author  = {Kean Chen and Qisheng Wang and Zhicheng Zhang},
  title   = {Local Test for Unitarily Invariant Properties of Bipartite Quantum States},
  year    = {2024},
  eprint  = {2404.04599},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi     = {10.48550/arXiv.2404.04599},
  note    = {v1 6 Apr 2024, last revised 30 May 2025}
}
```
**What it proves.** For bipartite pure states, unitary invariance on one part implies an **optimal tester
acting only on the other part**; the optimal canonical tester for entanglement spectra is "locally
performing weak Schur sampling and classically postprocessing the results" (Cor. 1.2). Matching lower
bounds for **Schmidt-rank testing** (Cor. 1.4); improved bounds for testing MPS; optimal bounds for
maximal entanglement; extension to mixed states with one-way LOCC. **This is the bipartite operational
use of exactly the "one party Schur-samples" primitive**; it does not mention Kronecker coefficients.

```bibtex
@article{RicoGrinkoKrebsZaw2025,
  author  = {Albert Rico and Dmitry Grinko and Robin Krebs and Lin Htoo Zaw},
  title   = {Detection of many-body entanglement partitions in a quantum computer},
  year    = {2025},
  eprint  = {2511.13822},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi     = {10.48550/arXiv.2511.13822},
  note    = {submitted 17 Nov 2025}
}
```
**What it proves (HTML-verified).** Entanglement-partition detection (GME, m-separability, entanglement
depth) via witnesses built from Young projectors, implementable by weak Schur sampling; complete
characterisation for 3- and 4-partite unitarily+permutation-symmetric witnesses; a family of analytical
witnesses for arbitrary size; new immanant inequalities. **Important caveat: their weak Schur sampling is
applied to the `n` particles of a single copy of the state, yielding a single `lambda` — it is NOT `k`
copies with a tuple `bold-lambda`. Different setup from ours; no Kronecker coefficients appear.**

```bibtex
@article{MarvianSpekkens2014,
  author  = {Iman Marvian and Robert W. Spekkens},
  title   = {A Generalization of {S}chur--{W}eyl Duality with Applications in Quantum Estimation},
  journal = {Communications in Mathematical Physics},
  volume  = {331},
  pages   = {431--475},
  year    = {2014},
  eprint  = {1112.0638},
  archivePrefix = {arXiv}
}
```
**What it proves.** A framework generalising Schur–Weyl duality for estimation problems on `n` copies of
an unknown state; usefulness of collective measurements; upper bounds on entanglement needed for optimal
estimation. Background rather than directly on-point. (Metadata verified via search results, not a direct
abs fetch — see (D).)

```bibtex
@phdthesis{Harrow2005,
  author  = {Aram W. Harrow},
  title   = {Applications of coherent classical communication and the {S}chur transform
             to quantum information theory},
  school  = {Massachusetts Institute of Technology},
  year    = {2005},
  eprint  = {quant-ph/0512255},
  archivePrefix = {arXiv}
}
```
**What it proves.** Standard reference for the Schur transform, Schur duality in quantum information, and
its applications. (Verified only at the level of title/author/venue via arXiv listing + MIT DSpace; I did
not fetch a chapter to confirm any multipartite Kronecker statement.)

---

## (B) Prose synthesis

### B.1 Point 1 — what the primary literature actually proves, in our notation

**Christandl–Mitchison (CMP 261:789, 2006).** Bipartite, asymptotic, "only if" direction. For an admissible
`(r^A, r^B, r^{AB})` the composite diagram's `S_k`-irrep sits inside the tensor product of the marginals'
irreps, i.e. `g_{mu nu lambda} != 0` along a sequence approximating the spectra. Relative to (C): this is
**neither** the finite-`k` statement nor multipartite. Status: **(c) different statement**, but it is the
conceptual ancestor.

**Christandl–Harrow–Mitchison (CMP 270:575, 2007).** Bipartite. Thm 2.2 = necessity (asymptotic),
Thm 2.4 = Klyachko's converse *with the row hypotheses `ell(mu) <= m`, `ell(nu) <= n`, `ell(lambda) <= mn`*,
Thm 3.1 = semigroup property, Thm 3.4 = polytope. The proof of Thm 2.4 (Lemma 5.1) is exactly the
"take a nonzero `S_k`-invariant in `V_mu ⊗ V_nu ⊗ V_lambda`, read it as a quantum state, then apply the
Keyl–Werner estimation theorem" argument. Relative to (C): **(b) an easy corollary in the bipartite case,
and the proof technique is the same**, but they never state the exact finite-`k` support characterisation;
they always pass through spectra and stretching (`g(m lambda, m mu, m nu) != 0` for *some* `m`), which is a
strictly weaker/coarser statement than (C) because of the "holes" phenomenon (see B.3).

**Klyachko (quant-ph/0409113, 2004).** Thm 5.3.1 is the equivalence
`g(m lambda, m nu, m mu) != 0 for some m > 0`  <=>  `exists rho_AB` with spectrum `nu` and marginal spectra
`lambda, mu`. **Bipartite only**; ar5iv full-text search found no `n >= 3` Kronecker statement, no
`Sym^k(H_1 ⊗ ... ⊗ H_n)` decomposition and no semigroup theorem in that paper. Status relative to (C):
**(c)**, the asymptotic bipartite shadow of (C).

**Keyl–Werner (PRA 64:052311, 2001).** The single-system weak-Schur-sampling concentration theorem. It is
the analytic input to all of the above but says nothing about tuples or Kronecker coefficients. **(c)**.

**Walter's thesis / Walter–Doran–Gross–Christandl / Christandl–Doran–Kousidis–Walter / Vergne–Walter.**
These give the multipartite picture at the level of **moment polytopes**: which tuples of one-body spectra
`(Spec rho_1, ..., Spec rho_n)` are achievable, with explicit linear inequalities (Vergne–Walter), and the
statement that this is the *asymptotic support of the n-fold Kronecker coefficients*. Relative to (C):
**(c)** — they characterise the asymptotic/cone version, i.e. the closure of `{bar-bold-lambda : g > 0}`
after normalising and stretching. (C) is the exact, non-asymptotic, per-`k` version.

**Ikenmeyer–Mulmuley–Walter (comput. complex. 26:949, 2017).** NP-hardness of deciding `g > 0`. Combined
with (C), this immediately says: **deciding whether a given tuple of isotypic projectors annihilates every
multipartite state is NP-hard** (for `n = 3` and, a fortiori, `n >= 3`). Their "holes" results say the
finite-`k` support is genuinely different from (and strictly inside) the stretched/asymptotic support.
Relative to (C): **(c)**, but it supplies the complexity corollary.

**Bacon–Chuang–Harrow (PRL 97:170502, 2006).** The efficient implementability of `{P_lambda}` and hence of
`P_bold-lambda` by `n` parties acting locally. Purely algorithmic; **(c)**.

**Christandl–Doran–Walter (FOCS'12), §VI.1.** The **algebraic identity** for `n = 3`:
`Sym^k(C^a ⊗ C^b ⊗ C^c) ↓_{U(a) x U(b) x U(c)} = ⊕ g(lambda,mu,nu) V_lambda ⊗ V_mu ⊗ V_nu`.
Relative to (C): this **is** the `n = 3` algebraic content, stated cleanly. Together with the standard
spanning fact it makes (C) a two-line corollary for `n = 3`. Status: **(b)** for `n = 3`.

**Amanov–Yeliussizov (arXiv:2504.15413, §2.10).** The **`n`-fold algebraic identity**:
`dim HWV_bold-lambda ( Sym^m (C^n)^{⊗d} ) = g(bold-lambda)` for any `d >= 2`, with `g(bold-lambda)` the
multiplicity of the trivial `S_m`-irrep in `[lambda^(1)] ⊗ ... ⊗ [lambda^(d)]`. Relative to (C): this **is**
the algebraic core for arbitrary `n`, phrased for highest-weight vectors. Status: **(b)** — (C) follows from
it plus the spanning fact; but the paper is pure combinatorics/representation theory and does not mention
weak Schur sampling, POVMs, or quantum states.

**Botero–Mejía (PRA 98:032326, 2018).** The **operational** multipartite weak Schur sampling setup, with
`P_bold-lambda = ⊗_i P_{lambda^(i)}`, `p(bold-lambda | psi) = || P_bold-lambda |psi>^{⊗n} ||^2`, their
Eq. (5) `|psi>^{⊗n} ∈ ⊕_bold-lambda V_bold-lambda ⊗ [bold-lambda]^{S_n}`, and the generalized Kronecker
coefficient `k_bold-lambda = dim [bold-lambda]^{S_n}` for arbitrary `N` parties. Relative to (C): this
**states the "=>" direction implicitly** (Eq. 5 immediately gives `p = 0` unless `g > 0` and
`ell(lambda^(i)) <= d_i`), and supplies all the machinery for "<=", but **never writes down (C)**. Status:
**(b), and the closest existing work by a wide margin.**

### B.2 Point 2 — is the exact support statement written down anywhere?

**Short answer: not as an explicit "support = Kronecker positivity" theorem, as far as I can find.**
What exists is:

1. The **"only if"** half, essentially verbatim, as Botero–Mejía Eq. (5):
   `|psi>^{⊗k}` lies in `⊕_bold-lambda V_bold-lambda ⊗ [bold-lambda]^{S_k}`.
   Since `V_{lambda^(i)}(H_i) = 0` when `ell(lambda^(i)) > d_i` and `[bold-lambda]^{S_k} = 0` when
   `g(bold-lambda) = 0`, this gives `p_psi(bold-lambda) = 0` for every `psi` outside the claimed support.
2. The **algebraic identity** underlying the "if" half: Christandl–Doran–Walter §VI.1 (n = 3) and
   Amanov–Yeliussizov §2.10 (general `n`).
3. The **spanning fact** you cite, with a clean citation: Harrow, "The Church of the Symmetric Subspace",
   **Theorem 3**: `Sym^k(C^d) = span{|phi>^{⊗k}}`.

The two-line proof you have in mind is confirmed correct by these sources and goes:
`(H^{⊗k})^{S_k, diagonal} = Sym^k(H) = ⊕_bold-lambda V_bold-lambda ⊗ [bold-lambda]^{S_k}`
(the coordinated `S_k` acts only on the multiplicity factors of the local Schur–Weyl decompositions), the
`bold-lambda` block is nonzero iff `g(bold-lambda) > 0` and all `ell(lambda^(i)) <= d_i`, and
`span{psi^{⊗k}} = Sym^k(H)` means some `psi` has nonzero component in any nonzero block. The two nontrivial
inputs are Botero–Mejía Eq. (4)–(5) (or Amanov–Yeliussizov §2.10) and Harrow Thm 3.

I searched for the explicit statement under many phrasings ("multipartite weak Schur sampling", "support of
the distribution over partition tuples", "Schur sampling and Kronecker coefficients", "nonzero probability
iff Kronecker coefficient nonzero", plus author-directed searches on Christandl, Harrow, Klyachko, Walter,
Botero, Mejía, Bravyi, Ikenmeyer) and found no paper asserting it as a theorem. **There is no paper titled
or explicitly about "multipartite weak Schur sampling."**

Closest near-misses worth reading in full before claiming novelty:
- Botero & Mejía 2018 (arXiv:1712.09174), §II–III. If (C) is anywhere, it is here as an unstated remark.
- Gonzalez 2025 (arXiv:2504.16256) — I could not fetch the body (size limit); it is the one paper I would
  check by hand.
- Michael Walter's thesis (arXiv:1410.6820) — I verified the abstract only; the multipartite background
  chapter may contain the `Sym^k(H_1 ⊗ ... ⊗ H_n)` decomposition for general `n`.

### B.3 Point 3 — what is known about vanishing of Kronecker coefficients

**Row-length necessary conditions.**
- Verified verbatim in Bürgisser–Christandl–Ikenmeyer (arXiv:0910.4512, §2.1):
  **"It is known that `g_{lambda,mu,nu}` vanishes if `ell(lambda) > ell(mu) ell(nu)`."**
  By the `S_3` symmetry of `g`, all three cyclic versions hold. Quantum reading:
  `rank rho_{AB} <= rank rho_A · rank rho_B`.
- Same content appears as an explicit *hypothesis* in Christandl–Harrow–Mitchison Thm 2.4 ("diagrams with
  `k` boxes and at most `m`, `n` and `mn` rows").
- `n`-fold version (which is what (C) needs): the constraint is simply that the tuple is realisable inside
  `H_1 ⊗ ... ⊗ H_n`, i.e. `ell(lambda^(i)) <= d_i`, together with each `ell(lambda^(i)) <= prod_{j != i} d_j`
  forced by the tensor structure. I did **not** find an `n`-party statement of this written anywhere.

**Depth condition (James–Kerber), verified via Lee (arXiv:2310.17906) Thm 2.2:**
if `g^nu_{lambda,mu} != 0` then `|d_lambda - d_mu| <= d_nu <= d_lambda + d_mu` where `d_lambda = k - lambda_1`.

**Dvir (J. Algebra 154:125–140, 1993).** The maximal first part `rho_1` over `rho` occurring in
`chi^lambda ⊗ chi^mu` equals `|lambda ∩ mu| = Σ_i min(lambda_i, mu_i)`, with an analogous statement for the
maximal first column (hence for the maximal number of rows). This is the sharpest classical "extremal shape"
result. **Metadata from search results only — not fetched; see (D).**

**Horn-type inequalities.** Ressayre (arXiv:1907.07931, Adv. Math. 2019) extends the essential Horn
inequalities to triples with nonzero Kronecker coefficient. Vergne–Walter (JSG 15:1209, 2017) give finitely
many explicit linear inequalities cutting out the moment cone of an arbitrary representation, which for
`H_1 ⊗ ... ⊗ H_n` under `U(d_1) x ... x U(d_n)` is exactly the asymptotic support of the `n`-fold Kronecker
coefficients.

**Semigroup and saturation.**
- **Semigroup:** CHM Thm 3.1 — `Kron` closed under row-wise addition. Hence the support of (C), for a fixed
  `n` and fixed local dimensions, is a **finitely generated semigroup**, and after normalisation a convex
  polytope (CHM Thm 3.4). Klyachko conjectured this; CHM proved it.
- **Saturation FAILS:** `g(lambda,mu,nu) = 0` does not follow from `g(N lambda, N mu, N nu) = 0`.
  Ikenmeyer–Mulmuley–Walter construct abundant, efficiently findable **"holes"** — triples vanishing at
  `N = 1` but positive after stretching. **This is the single most important caveat for (C):** the exact
  per-`k` support is a proper (and combinatorially wild) subset of the rational points of the polytope.
- **Rectangular positivity:** Bürgisser–Christandl–Ikenmeyer (Adv. Math. 227:2082, 2011) — for any
  `lambda` of size `ell d` with at most `d^2` parts there is `k` with `g(k lambda, k □, k □) != 0`,
  with `k = O(d^4 eps^{-2} log(d/eps))` in the approximate version.

**Stability.** Murnaghan stability (reduced Kronecker coefficients); Briand–Orellana–Rosas (J. Algebra 2011)
pin down the exact stabilisation point, improving Brion and Vallejo; Stembridge's general stability
conjecture proved by Sam–Snowden (JACO 43:1, 2016); Pak–Panova introduce "k-stability" and prove upper and
lower bounds. Ikenmeyer–Panova (Forum Math. Pi 12:e22, 2024): every Kronecker coefficient *is* a reduced
Kronecker coefficient, so positivity of reduced coefficients is also NP-hard.

**Complexity.** Positivity is NP-hard (IMW 2017); computing is #P-hard (Bürgisser–Ikenmeyer) and in GapP;
positivity is in QMA (Bravyi–Chowdhury–Gosset–Havlicek–Zhu, PRX Quantum 5:010329, 2024); the coefficients
are in #BQP (Ikenmeyer–Subramanian); Panova (2025) gives classical polynomial-time algorithms for several
families previously conjectured to admit quantum speedup. Huber–Wyderka (Quantum 9:1918, 2025) give a
complete SDP hierarchy for *refuting* positivity.

**`n >= 4` Kronecker coefficients.** Thin. The only substantive works I verified that genuinely handle
`d`-tuples for `d >= 4`:
- Amanov–Yeliussizov, arXiv:2312.17054 (unimodality of generalized Kronecker sequences; proved for `k = 2`,
  conjectured in general; connections to a higher-dimensional Alon–Tarsi conjecture).
- Amanov–Yeliussizov, arXiv:2504.15413 (highest weight vectors of tensors; the general-`d`
  `Sym^m` / `Λ^m` identity).
- Botero–Mejía, arXiv:1712.09174 (their `k_bold-lambda` is defined for arbitrary `N` parties).
- Bryan–Reichstein–Van Raamsdonk / Bryan–Leutheusser–Reichstein–Van Raamsdonk on LME states (the `n`-party
  "all marginals maximally mixed" case, which is the `bold-lambda = (rectangle, ..., rectangle)` corner).
Everything else (Horn inequalities, stability, complexity, bounds) is **stated only for triples**. This is a
genuine gap in the literature and a place where our project can contribute.

### B.4 Point 4 — Kronecker positivity vs. constraints on local ranks

- **Bipartite / tripartite.** `g(lambda,mu,nu) = 0` when `ell(nu) > ell(lambda) ell(mu)` (BCI §2.1) is
  exactly `rank rho_{AB} <= rank rho_A rank rho_B`. Conversely CHM Thm 2.4 says a nonzero `g` with
  `ell(mu) <= m`, `ell(nu) <= n`, `ell(lambda) <= mn` is *realised* by a density operator on `C^m ⊗ C^n`.
  So in the bipartite setting the dictionary "Kronecker positivity ↔ realisability at given local ranks"
  is established, in asymptotic/stretched form.
- **Multipartite, asymptotic.** Entanglement polytopes (Walter–Doran–Gross–Christandl, Science 340:1205)
  and the moment-cone inequalities (Vergne–Walter) describe which one-body spectra — hence which local
  ranks, as a degenerate limit — are achievable by a pure state in `H_1 ⊗ ... ⊗ H_n`; the underlying
  multiplicities are the `n`-fold Kronecker coefficients.
- **Multipartite, exact, extreme case.** Bryan–Reichstein–Van Raamsdonk (Ann. Henri Poincaré 19:2491) give
  a **necessary and sufficient, exactly computable criterion `R(d_1,...,d_n) >= 0`** for the existence of a
  locally maximally entangled state — i.e. for the corner of the support where every `lambda^(i)` is a
  rectangle with `d_i` rows. Bryan–Leutheusser–Reichstein–Van Raamsdonk (Quantum 3:115) extend and
  compute dimensions. This is the sharpest existing "local ranks ⟹ existence" theorem for `n` parties.
- **Kronecker states.** Botero–Mejía define `[bold-lambda]^{S_k}` (dimension `= g(bold-lambda)`) as the
  space of *Kronecker states*, which are LME by symmetry. Gonzalez (arXiv:2504.16256) develops explicit
  constructions. So `g(bold-lambda) > 0` is literally the existence criterion for a maximally entangled
  state of that "shape" — the cleanest existing bridge between Kronecker positivity and multipartite
  local structure.
- **What is missing.** I found **no** paper stating a general rank-constraint theorem of the form
  "for a pure `|psi> ∈ H_1 ⊗ ... ⊗ H_n` with local ranks `r_i`, the achievable `bold-lambda` are exactly
  those with `g(bold-lambda) > 0` and `ell(lambda^(i)) <= r_i`". That is essentially (C) restated with
  `r_i` in place of `d_i`, and it appears to be open/unwritten.

---

## (C) Novelty assessment

**Already known / literally in the literature:**
1. `Tr[P_lambda rho^{⊗k}] = 0` when `ell(lambda) > rank rho` — folklore, and the concentration refinement is
   Keyl–Werner (2001).
2. The multipartite weak Schur sampling POVM `P_bold-lambda = ⊗_i P_{lambda^(i)}` on `k` copies of an
   `N`-partite pure state, with `p(bold-lambda|psi) = ||P_bold-lambda psi^{⊗k}||^2` and the `n`-fold
   Kronecker coefficient `k_bold-lambda = dim[bold-lambda]^{S_k}`: **Botero–Mejía 2018, Eqs. (4)–(8).**
3. The **"only if"** direction of (C) — i.e. that tuples with `g = 0` or `ell(lambda^(i)) > d_i` have zero
   probability for *every* state — is an immediate reading of **Botero–Mejía Eq. (5)**, though they do not
   phrase it that way.
4. The algebraic identity `Sym^k(H_1 ⊗ ... ⊗ H_n) = ⊕ g(bold-lambda) · S^{lambda^(1)}H_1 ⊗ ... ⊗ S^{lambda^(n)}H_n`:
   **`n = 3` explicitly in Christandl–Doran–Walter (FOCS'12) §VI.1; general `n` in Amanov–Yeliussizov
   (arXiv:2504.15413) §2.10** (as `dim HWV_bold-lambda Sym^m (C^n)^{⊗d} = g(bold-lambda)`).
5. `span{psi^{⊗k}} = Sym^k(H)`: **Harrow, arXiv:1308.6595, Theorem 3.**
6. The *asymptotic / stretched* bipartite equivalence "nonzero Kronecker ⟺ realisable spectra with the
   stated row bounds": **Klyachko Thm 5.3.1; Christandl–Mitchison; CHM Thms 2.2 & 2.4.** Multipartite
   asymptotic version: entanglement polytopes / Vergne–Walter moment cones.
7. Semigroup property (CHM Thm 3.1), polytope structure (CHM Thm 3.4), NP-hardness of positivity and the
   existence of saturation "holes" (Ikenmeyer–Mulmuley–Walter 2017), QMA / #BQP membership (Bravyi et al.;
   Ikenmeyer–Subramanian).

**Not stated anywhere I could find (candidate novelty):**
1. **(C) itself, as a theorem.** The exact, finite-`k`, `n`-partite statement
   "`supp` of multipartite weak Schur sampling over `lambda`-tuples `=` `{ bold-lambda : g(bold-lambda) > 0
   and ell(lambda^(i)) <= d_i for all i }`" is, to the best of my search, **not written down**. Its two
   halves exist in two disjoint literatures (quantum: Botero–Mejía; algebraic: Christandl–Doran–Walter /
   Amanov–Yeliussizov), and the bridging fact is Harrow Thm 3. **Realistic characterisation: a short,
   correct, previously-unstated proposition assembled from known pieces — "folklore-adjacent" rather than
   deep.** Expect referees to call the proof easy; the value is in the statement, the operational reading,
   and the corollaries.
2. **The operational reading**: "certain tuples of isotypic projectors annihilate *every* multipartite
   state." I found no paper phrasing Kronecker vanishing this way. This gives, via
   Ikenmeyer–Mulmuley–Walter, the corollary that **deciding whether a given tuple of local isotypic
   projectors annihilates the whole state space is NP-hard**, and via Bravyi et al. that the complementary
   problem is in QMA. I did not find either corollary stated.
3. **The exact (non-stretched) `n`-party local-rank characterisation** with `rank rho_i = r_i` in place of
   `d_i` — see B.4, last bullet.
4. **Anything about `n >= 4` on the quantum side.** The generalized-Kronecker `n >= 4` literature is
   essentially two Amanov–Yeliussizov papers plus Botero–Mejía's definition; no Horn inequalities, no
   semigroup theorem, no stability, no complexity results are on record for `n >= 4`. Extending CHM's
   semigroup property (Thm 3.1) to `n`-fold Kronecker coefficients, or the row-length necessary condition
   `ell(lambda^(i)) <= prod_{j != i} ell(lambda^(j))`, looks both doable and unwritten.

**Recommended framing.** Do not claim (C) as a new hard theorem. Claim: (i) a clean, exact, finite-copy,
`n`-partite statement of the support of multipartite weak Schur sampling, with a short proof, filling a gap
between the quantum (Botero–Mejía, CHM) and algebraic (Christandl–Doran–Walter, Amanov–Yeliussizov)
literatures; (ii) the complexity corollaries; (iii) whatever genuinely new `n >= 4` structural results you
can prove on top of it.

**Verification to do by hand before publishing a novelty claim** (I could not complete these):
- Read Botero–Mejía arXiv:1712.09174 §II–III in full and check for an unstated version of (C).
- Read Gonzalez arXiv:2504.16256 in full (I could not fetch the body).
- Read Michael Walter's thesis arXiv:1410.6820, the multipartite Schur–Weyl background chapter.
- Read Christandl's thesis arXiv:quant-ph/0604183 Ch. on spectra (I found no such statement in the
  fetched portion, but the fetch was partial).

---

## (D) Unverified / could not confirm

Listed so nothing here gets cited as verified.

1. **Dvir, "On the Kronecker product of `S_n` characters", J. Algebra 154:125–140 (1993).** Title, journal,
   volume, pages and the `rho_1 = Σ_i min(lambda_i, mu_i)` result come from search-result summaries only.
   No arXiv version; I did not fetch a page of the journal or a reliable copy. **Verify before citing.**
2. **Pak–Panova (arXiv:1406.2988) bound `g <= 2^{abc}` for `ell(lambda) <= a, ell(mu) <= b, ell(nu) <= c`.**
   The abstract page was fetched and verified; this specific inequality came from a search snippet, not the
   paper body.
3. **Sam–Snowden (arXiv:1501.00333), JACO 43(1):1–10 (2016).** Journal metadata and the statement of
   Stembridge's conjecture come from search results; I did not fetch the arXiv abs page directly.
4. **Walter–Doran–Gross–Christandl, Science 340(6137):1205–1208 (2013), arXiv:1208.0365.** Verified via
   multiple consistent search results (Science, PubMed, KU research portal) but I did not fetch the arXiv
   abs page. DOI 10.1126/science.1232957.
5. **Marvian–Spekkens, CMP 331:431–475 (2014), arXiv:1112.0638.** Metadata from search results only.
6. **Harrow PhD thesis, arXiv:quant-ph/0512255 (MIT, 2005).** Title/author/venue only; no chapter fetched,
   so **no claim about its contents regarding multipartite Kronecker coefficients should be made.**
7. **Ikenmeyer–Subramanian journal version** "Kronecker Coefficients in #BQP", ACM Trans. Quantum Computing,
   doi:10.1145/3762673 — from a search result; the arXiv abs page (2307.02389) was fetched and is verified,
   the ACM metadata is not.
8. **Gonzalez, arXiv:2504.16256 body.** Abstract page verified; the HTML exceeded the fetch size limit, so
   all body-level claims (definitions of Kronecker subspaces, W-state stitching, any support statement) are
   **unverified**.
9. **Mejía, arXiv:1610.09552 body.** Abstract verified; PDF text layer unreadable. Whether it defines
   `p(alpha,beta,gamma|psi)` explicitly is **unverified** (a search snippet suggested it does).
10. **Klyachko's tables and multipartite claims beyond Thm 5.3.1.** The ar5iv rendering was partial; I
    verified Thm 5.3.1 and §5.3/Eq. (5.7) only. My statement that the paper contains no `n >= 3` Kronecker
    result is a negative finding from a partial fetch, not a certainty.
11. **Whether Ressayre (arXiv:1907.07931) states row-length necessary conditions.** Abstract only; body not
    fetched.
12. **Christandl PhD thesis arXiv:quant-ph/0604183.** Title, author, institution and two general quotes
    verified; my negative finding ("does not contain the multipartite probability formulation") is from a
    partial fetch.
13. **Michael Walter thesis arXiv:1410.6820 contents.** Abstract verified; **no theorem inside it verified.**
14. **No paper titled or explicitly about "multipartite weak Schur sampling" exists.** This is a negative
    search result across many phrasings, not a proof of absence.
