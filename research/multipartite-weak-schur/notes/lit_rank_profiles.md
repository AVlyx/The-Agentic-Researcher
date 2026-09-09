# Achievable flattening-rank profiles of multipartite pure states / tensors

Literature review, compiled 2026-09-02.

**Setup.** `|psi>` in `H_1 (x) ... (x) H_n` with `dim H_i = d_i`; equivalently an order-`n`
tensor `T in V_1 (x) ... (x) V_n`. For `S subset [n]`, `r_S := rank(rho_S) = rank` of the
flattening `T^(S)` (Schmidt rank across `S | S^c`). Always `r_S = r_{S^c}`, so there are
`2^{n-1} - 1` independent numbers. Following Cadney–Huber–Linden–Winter (CHLW) I write
`S_0(S) := log r_S` (the 0-Rényi entropy / "0-entropy"), `Sigma_n` for the set of achievable
integer rank vectors and `Omega_n = log Sigma_n` for the set of 0-entropy vectors.

Two vocabularies describe the same object:

| quantum info | multilinear algebra |
|---|---|
| local rank `r_i` (singleton `S`) | mode-`i` flattening rank = `i`-th multilinear rank |
| rank vector / Schmidt-rank vector, `l = 1` | 1-multirank / multilinear rank (Tucker rank) |
| full rank vector over all bipartitions | `l`-multiranks for all `l`; "rank profile" |
| Schmidt rank across a tree cut | minimal bond dimension of a tree tensor network edge |

Every entry in section (A) was checked against a fetched arXiv abstract page and, where a
theorem statement is quoted, against the extracted full text of the PDF.

---

## (A) Verified bibliography

```bibtex
@article{CarliniKleppe2011,
  author  = {Carlini, Enrico and Kleppe, Johannes},
  title   = {Ranks derived from multilinear maps},
  journal = {Journal of Pure and Applied Algebra},
  volume  = {215},
  number  = {8},
  pages   = {1999--2004},
  year    = {2011},
  doi     = {10.1016/j.jpaa.2010.11.010}
}
% What it proves: characterises exactly which tuples of flattening ranks of the SINGLETON
% modes (the multilinear rank) are attained by some tensor. Attained iff
% r_i <= prod_{j != i} r_j and r_i <= dim V_i, for every i -- this is the "polygon"
% answer, and it holds for EVERY order d, not just d = 3. For d > 3 the paper also gives
% necessary conditions on the ranks of all induced (non-singleton) flattenings.
% (Statement as quoted in Jovcheva-Seynnaeve-Vannieuwenhoven eq. (3), and attributed the
% same way by Gharahi-Mancini-Ottaviani ref. [57]; I could NOT open the ScienceDirect page
% itself -- HTTP 403 -- so the abstract is unverified, see section (D).)

@article{HuberDeVicente2013,
  author  = {Huber, Marcus and de Vicente, Julio I.},
  title   = {Structure of Multidimensional Entanglement in Multipartite Systems},
  journal = {Physical Review Letters},
  volume  = {110},
  pages   = {030501},
  year    = {2013},
  doi     = {10.1103/PhysRevLett.110.030501},
  eprint  = {1210.6876},
  archivePrefix = {arXiv},
  note    = {arXiv title: "The structure of multidimensional entanglement in multipartite systems"}
}
% What it proves: introduces the Schmidt-rank (entanglement-dimensionality) vector, extends
% it to mixed states via a Schmidt-number vector, and gives multi-level witnesses. Endnote
% [31] states + proves the tripartite achievability result (submultiplicativity is necessary
% AND sufficient for (r_A,r_B,r_C)); endnote [32] gives the first explicit 4-party rank
% vector obeying submultiplicativity that is nevertheless impossible; endnote [33]
% conjectures r_AB r_AC r_BC >= r_A r_B r_C (later REFUTED by CHLW).

@article{HuberPerarnauDeVicente2013,
  author  = {Huber, Marcus and Perarnau-Llobet, Mart\'i and de Vicente, Julio I.},
  title   = {Entropy vector formalism and the structure of multidimensional entanglement
             in multipartite systems},
  journal = {Physical Review A},
  volume  = {88},
  pages   = {042328},
  year    = {2013},
  doi     = {10.1103/PhysRevA.88.042328},
  eprint  = {1307.3541},
  archivePrefix = {arXiv}
}
% What it proves: extends the rank/entropy-vector framework to k-separability, k-partite
% entanglement and partition-wise separability, and to assessing entanglement dimensionality
% in all those cases.

@article{CadneyHuberLindenWinter2014,
  author  = {Cadney, Josh and Huber, Marcus and Linden, Noah and Winter, Andreas},
  title   = {Inequalities for the ranks of multipartite quantum states},
  journal = {Linear Algebra and its Applications},
  volume  = {452},
  pages   = {153--171},
  year    = {2014},
  doi     = {10.1016/j.laa.2014.03.035},
  eprint  = {1308.0539},
  archivePrefix = {arXiv},
  note    = {arXiv title: "Inequalities for the Ranks of Quantum States"}
}
% What it proves: THE key paper. Defines Sigma_n / Omega_n; records Omega_3 = C ∩ log N^3
% (tripartite = polygon inequalities, exactly); proves two genuinely new 4-party rank
% inequalities r_A <= r_AB r_AC (Thm 1) and r_A^2 <= r_AB r_AC r_BC (Thm 2) using strong
% subadditivity of von Neumann entropy; computes the 8 extremal-ray families of the
% resulting cone, realises 6 of them asymptotically, PROVES rays 3 and 6 are not attained
% by any state, and leaves rays 4,5,7,8 open; conjectures Hypothesis 1 (r_BC <= r_AB r_AC),
% proved for r_AB <= 2; refutes the Huber-de Vicente conjecture r_A r_B r_C <= r_AB r_AC r_AD.

@article{SongChenSunHu2023,
  author  = {Song, Zhiwei and Chen, Lin and Sun, Yize and Hu, Mengyao},
  title   = {A complete picture of the four-party linear inequalities in terms of
             the 0-entropy},
  journal = {IEEE Transactions on Information Theory},
  volume  = {69},
  number  = {4},
  pages   = {2385--2399},
  year    = {2023},
  eprint  = {2105.07679},
  archivePrefix = {arXiv}
}
% What it proves: proves CHLW's Hypothesis 1, i.e. r(rho_AB) r(rho_AC) >= r(rho_BC) for any
% tripartite mixed state (= IQOQI Vienna Open Problem 41). Consequently the CLOSED CONE of
% four-party 0-entropy vectors is completely determined by: S_0(A)+S_0(B) >= S_0(AB);
% S_0(AB)+S_0(AC)+S_0(BC) >= 2 S_0(A); S_0(AB)+S_0(AC) >= S_0(BC). (CHLW's Thm 1 is a
% corollary of the last two.) Lemma 10 extends several of these to n parties.

@article{WangChen2025,
  author  = {Wang, Nalan and Chen, Lin},
  title   = {Equality condition for a matrix inequality by partial transpose},
  year    = {2025},
  eprint  = {2508.18644},
  archivePrefix = {arXiv}
}
% What it proves: studies when rank(sum_j A_j^T (x) B_j) <= K rank(sum_j A_j (x) B_j)
% (the matrix form of CHLW Hypothesis 2 / Song et al.) is SATURATED. Directly relevant to
% deciding which points on the boundary of the four-party cone are attained.

@article{GharahiManciniOttaviani2020,
  author  = {Gharahi, Masoud and Mancini, Stefano and Ottaviani, Giorgio},
  title   = {Fine-structure classification of multiqubit entanglement by algebraic geometry},
  journal = {Physical Review Research},
  volume  = {2},
  pages   = {043003},
  year    = {2020},
  doi     = {10.1103/PhysRevResearch.2.043003},
  eprint  = {1910.09665},
  archivePrefix = {arXiv}
}
% What it proves: SLOCC fine-structure classification of n-qubit states by secant varieties
% plus l-multiranks. Appendix B, Theorem 2 is the complete answer for 2x2x2x2:
% (i) among the three 2-multiranks (r_AB, r_AC, r_AD) the maximum is attained at least twice;
% (ii) that is the ONLY constraint, with the single exception of the triple (1,3,3), which
% cannot be achieved. Part (i) follows from a 1920 identity of Segre: the three 4x4
% determinants of the three flattenings of a 2x2x2x2 tensor sum to zero. Also states that
% Carlini-Kleppe classified all 1-multiranks for any number of qudits.

@article{Gharahi2025multiranks,
  author  = {Gharahi, Masoud},
  title   = {$\ell$-Multiranks of Multipartite Quantum States via Tensor Flattening:
             A Mathematica Codebase},
  year    = {2025},
  eprint  = {2601.11551},
  archivePrefix = {arXiv},
  note    = {submitted 9 Dec 2025; code at https://github.com/mathoud/Flattening}
}
% What it provides: Mathematica code computing the full l-multirank profile (ranks of all
% bipartition flattenings) of multiqudit states. Directly usable for the numerics in (C).

@article{JovchevaSeynnaeveVannieuwenhoven2025,
  author  = {Jovcheva, Jana and Seynnaeve, Tim and Vannieuwenhoven, Nick},
  title   = {Minimality of Tree Tensor Network Ranks},
  year    = {2025},
  eprint  = {2509.09463},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2509.09463}
}
% What it proves: necessary and sufficient conditions for a tuple of bond dimensions of a
% TREE tensor network to be a tree tensor network rank, generalising Carlini-Kleppe from
% the star graph (Tucker) to arbitrary trees. Admissibility (their eq. (4)):
% r_ij <= dim V_i * prod_{k in nb(i)\{j}} r_ik for every vertex i and neighbour j.
% Thm 3.6: TN^o(G,r) is nonempty iff r is admissible, and then it is Zariski (hence
% Euclidean) open and dense in TN(G,r), so minimality is generic.

@article{LandsbergQiYe2012,
  author  = {Landsberg, J. M. and Qi, Yang and Ye, Ke},
  title   = {On the geometry of tensor network states},
  journal = {Quantum Information \& Computation},
  volume  = {12},
  number  = {3--4},
  pages   = {346--354},
  year    = {2012},
  eprint  = {1105.4449},
  archivePrefix = {arXiv},
  doi     = {10.5555/2230976.2230988}
}
% What it proves: answers a question of Grasedyck -- the limit of tensors in a space of
% tensor network states need not be a tensor network state (non-closedness); geometric
% descriptions of tree and loop tensor network state spaces.

@article{YeLim2018,
  author  = {Ye, Ke and Lim, Lek-Heng},
  title   = {Tensor network ranks},
  year    = {2018},
  eprint  = {1801.02662},
  archivePrefix = {arXiv},
  note    = {v2 2019, 37 pp.}
}
% What it proves: defines G-rank for an arbitrary undirected graph G (matrix rank, tensor
% rank, multilinear rank, MPS/TTNS/PEPS bond dimensions are special cases); G-rank is
% polynomial-time computable and best low-G-rank approximation exists when G is acyclic;
% gaps between G-rank and classical ranks can be arbitrarily large.

@article{BuczynskaBuczynskiMichalek2015,
  author  = {Buczy\'nska, Weronika and Buczy\'nski, Jaros\l{}aw and Micha\l{}ek, Mateusz},
  title   = {The Hackbusch conjecture on tensor formats},
  journal = {Journal de Math\'ematiques Pures et Appliqu\'ees},
  volume  = {104},
  number  = {4},
  pages   = {749--761},
  year    = {2015},
  doi     = {10.1016/j.matpur.2015.05.002},
  eprint  = {1501.01120},
  archivePrefix = {arXiv}
}
% What it proves: compares the bond dimensions needed to represent the same tensor on the
% perfect binary tree (hierarchical format) vs the caterpillar/train-track tree (tensor
% train). i.e. quantitative comparison of two different partial rank profiles of one tensor.
% Part two: arXiv:1802.00222.

@article{CuiFreedmanSattathStongMinton2016,
  author  = {Cui, Shawn X. and Freedman, Michael H. and Sattath, Or and
             Stong, Richard and Minton, Greg},
  title   = {Quantum Max-flow/Min-cut},
  journal = {Journal of Mathematical Physics},
  volume  = {57},
  pages   = {062206},
  year    = {2016},
  doi     = {10.1063/1.4954231},
  eprint  = {1508.04644},
  archivePrefix = {arXiv}
}
% What it proves: quantum max-flow (max rank over all choices of vertex tensors) vs quantum
% min-cut (min product of edge capacities). Equality when every capacity is a power of a
% fixed integer; but the conjecture is FALSE in general, with concrete counterexamples.
% This is the tensor-network shadow of the same phenomenon: the naive submultiplicative
% (min-cut) bound on a flattening rank is not always attainable.

@article{Hastings2017,
  author  = {Hastings, Matthew B.},
  title   = {The Asymptotics of Quantum Max-Flow Min-Cut},
  journal = {Communications in Mathematical Physics},
  volume  = {351},
  pages   = {387--418},
  year    = {2017},
  eprint  = {1603.03717},
  archivePrefix = {arXiv}
}
% What it proves: for networks of identical tensors the ratio quantum-max-flow /
% quantum-min-cut tends to 1 as the edge dimension N -> infinity. So the obstructions are
% a small-dimension / low-rank phenomenon -- exactly the regime of interest here.

@article{GesmundoLandsbergWalter2018,
  author  = {Gesmundo, Fulvio and Landsberg, J. M. and Walter, Michael},
  title   = {Matrix product states and the quantum max-flow/min-cut conjectures},
  journal = {Journal of Mathematical Physics},
  volume  = {59},
  number  = {10},
  pages   = {102205},
  year    = {2018},
  doi     = {10.1063/1.5026985},
  eprint  = {1801.09106},
  archivePrefix = {arXiv}
}
% What it proves: three infinite sequences of examples where quantum max-flow < quantum
% min-cut, including a 4-cycle (periodic MPS) for infinitely many bond dimensions, verifying
% a prediction of Hastings, and 2d-cycles with all bond dimensions equal to two for all d.

@article{BernardiDeLazzariGesmundo2023,
  author  = {Bernardi, Alessandra and De Lazzari, Claudia and Gesmundo, Fulvio},
  title   = {Dimension of Tensor Network varieties},
  journal = {Communications in Contemporary Mathematics},
  volume  = {25},
  number  = {10},
  pages   = {2250059},
  year    = {2023},
  doi     = {10.1142/S0219199722500596},
  eprint  = {2101.03148},
  archivePrefix = {arXiv}
}
% What it proves: upper bounds on the dimension of the tensor network variety attached to a
% graph with given bond dimensions, refined for MPS and PEPS. Gives the "how big is the set
% of tensors reachable with this bond-dimension profile" side of the question.

@article{BarthelLuFriesecke2022,
  author  = {Barthel, Thomas and Lu, Jianfeng and Friesecke, Gero},
  title   = {On the closedness and geometry of tensor network state sets},
  journal = {Letters in Mathematical Physics},
  volume  = {112},
  pages   = {72},
  year    = {2022},
  eprint  = {2108.00031},
  archivePrefix = {arXiv}
}
% What it proves: MPS with open boundary conditions, TTNS and MERA sets are closed;
% translation-invariant periodic MPS, heterogeneous MPS and PEPS sets are generally NOT
% closed. Non-closedness is exactly why "asymptotically achievable" rank profiles (CHLW's
% rays 3-6) need not be achievable.

@article{ChristandlLysikovSteffanWernerWitteveen2024,
  author  = {Christandl, Matthias and Lysikov, Vladimir and Steffan, Vincent and
             Werner, Albert H. and Witteveen, Freek},
  title   = {The resource theory of tensor networks},
  journal = {Quantum},
  volume  = {8},
  pages   = {1560},
  year    = {2024},
  eprint  = {2307.07394},
  archivePrefix = {arXiv}
}
% What it proves: resource theory generalising bond dimension to arbitrary multipartite
% "entanglement structures", with transformations between entanglement structures that beat
% edge-by-edge conversion, plus obstructions imported from algebraic complexity theory.

@article{WalterDoranGrossChristandl2013,
  author  = {Walter, Michael and Doran, Brent and Gross, David and Christandl, Matthias},
  title   = {Entanglement Polytopes: Multiparticle Entanglement from Single-Particle
             Information},
  journal = {Science},
  volume  = {340},
  number  = {6137},
  pages   = {1205--1208},
  year    = {2013},
  doi     = {10.1126/science.1232957},
  eprint  = {1208.0365},
  archivePrefix = {arXiv}
}
% What it proves: each SLOCC class has an entanglement polytope of compatible single-particle
% SPECTRA. Related but strictly different from the rank question (spectra vs supports);
% see synthesis note in (B4).

@article{DurVidalCirac2000,
  author  = {D\"ur, W. and Vidal, G. and Cirac, J. I.},
  title   = {Three qubits can be entangled in two inequivalent ways},
  journal = {Physical Review A},
  volume  = {62},
  pages   = {062314},
  year    = {2000},
  eprint  = {quant-ph/0005115},
  archivePrefix = {arXiv}
}
% What it proves: GHZ and W are SLOCC-inequivalent -- the canonical demonstration that the
% local-rank vector (2,2,2) for both) does NOT determine the SLOCC class.

@article{VerstraeteDehaeneDeMoorVerschelde2002,
  author  = {Verstraete, F. and Dehaene, J. and De Moor, B. and Verschelde, H.},
  title   = {Four qubits can be entangled in nine different ways},
  journal = {Physical Review A},
  volume  = {65},
  pages   = {052112},
  year    = {2002},
  eprint  = {quant-ph/0109033},
  archivePrefix = {arXiv}
}
% What it proves: nine SLOCC families for four qubits (continuous families inside).
% Gharahi-Mancini-Ottaviani refine these by l-multiranks.
```

Secondary sources cited *inside* CHLW that matter for the rank story (verified only as
reference-list entries of CHLW, not fetched independently -- see (D)):

* N. Linden, M. Mosonyi, A. Winter, *The structure of Rényi entropic inequalities*,
  arXiv:1212.0248 — shows the alpha-Rényi entropy vectors for alpha in (0,1) u (1,inf) obey
  only non-negativity; poses alpha = 0 as the open case that CHLW then answers.
* A. J. Schwenk, J. I. Munro, *How small can the mean shadow of a set be?*,
  Amer. Math. Monthly 90(5):325–329, 1983 — the classical "uniform cover" / shadow
  inequality `s_J^{C(|J|-1,k-1)} <= prod_{|I|=k, I subset J} s_I`, of which CHLW Thm 2 is
  the quantum `|J|=3, k=2` case.
* C. Segre, Ann. Math. Pura Appl. Ser. III 29:105, 1920 — the identity behind
  Gharahi–Mancini–Ottaviani Thm 2(i).

---

## (B) Synthesis

### B1. Local ranks only (Question 1)

**Answer: your guess is right, and it is right for every `n`, not just `n = 3`.**

Theorem (Carlini–Kleppe 2011). There is a tensor `T in V_1 (x) ... (x) V_n` with
multilinear rank exactly `(r_1, ..., r_n)` if and only if

```
r_i <= prod_{j != i} r_j        and        r_i <= dim V_i        for all i.
```

This is quoted verbatim as eq. (3) of Jovcheva–Seynnaeve–Vannieuwenhoven (2025) and is
attributed to Carlini–Kleppe as "classified all possible one-multiranks for any number of
qudits" by Gharahi–Mancini–Ottaviani (2020, Appendix B).

Independently, the quantum-information literature has the `n = 3` case: Huber–de Vicente
(PRL 110:030501, endnote [31]) prove necessity from `r_A = r_BC <= r_B r_C` and sufficiency
by exhibiting
`|psi> = (r_B r_C)^{-1/2} sum_{m<r_B} sum_{n<r_C} |m(r_C-1)+n>_A |m>_B |n>_C`
(which attains `r_A = r_B r_C`) and then deleting terms. CHLW restate this as
`Omega_3 = C ∩ log N^3` with `C = {(x,y,z) >= 0 : x <= y+z, y <= x+z, z <= x+y}`, i.e. for
three parties **every** integer triple obeying the polygon inequalities is realised.

Caveat that matters a lot for your project: "local ranks" is *not* the same as "the rank
vector" once `n >= 4`. For `n = 3` the singletons already exhaust all bipartitions
(`A|BC`, `B|AC`, `C|AB`), so `n = 3` is a degenerate case. For `n >= 4` the local ranks are
only `n` of the `2^{n-1} - 1` entries of the rank vector, and Carlini–Kleppe say nothing
about the rest. **The interesting question begins at `n = 4`.**

### B2. Full rank profiles over all bipartitions (Question 2)

**Obvious constraints.** `r_S = r_{S^c}`, `r_S <= prod_{i in S} d_i`, and submultiplicativity
`r_{S u T} <= r_S r_T` for disjoint `S, T` (from `supp(psi_{ST}) subset supp(psi_S) (x) supp(psi_T)`).
In 0-entropy language submultiplicativity is subadditivity `S_0(S) + S_0(T) >= S_0(S u T)`.

**Non-obvious constraints exist, starting at `n = 4`.** Note that `S_0` does *not* satisfy
strong subadditivity: CHLW's explicit counterexample is the purification of
`rho_ABC = (1/5)(|000><000| + |100><100| + |101><101| + |110><110| + |111><111|)`, which has
`r_A = 2`, `r_ABC = 5`, `r_AB = r_AC = 3`.

Nevertheless CHLW prove, for any four-party pure `|psi>_ABCD` (so `r_BC = r_AD` etc.):

* **Theorem 1**: `r_A <= r_AB r_AC`.
* **Theorem 2**: `r_A^2 <= r_AB r_AC r_BC`.

Both are proved by first applying an invertible local filter on `A` (which preserves all
ranks) to make `psi_A` maximally mixed, and then applying *von Neumann* strong subadditivity
/ weak monotonicity together with `S(rho) <= log rank(rho)`. Theorem 1 also has a
self-contained algebraic proof (their Lemma 3). Neither is an instance of submultiplicativity:
in `r_A <= r_AB r_AC` the two sets `AB` and `AC` are not disjoint, and the submultiplicative
bounds available are only `r_A = r_BCD <= min(r_AB r_B, r_AC r_C, r_AD r_D, r_B r_C r_D)`.

* **CHLW Hypothesis 1**, proved by **Song–Chen–Sun–Hu (2021/2023)**: `r_BC <= r_AB r_AC`
  (equivalently `r(rho_AB) r(rho_AC) >= r(rho_BC)` for any *tripartite mixed* state; IQOQI
  Vienna Open Problem 41). In pure-4-party notation with `p = r_AB, q = r_AC, s = r_AD`
  this says the triple `(p, q, s)` itself obeys the polygon inequalities
  `s <= pq`, `p <= qs`, `q <= ps`.

**Status for `n = 4`: the closed cone is now completely known.** Song et al. state the full
list as (their (79)–(81), plus non-negativity):

```
S_0(A) + S_0(B) >= S_0(AB)                       (subadditivity)
S_0(AB) + S_0(AC) + S_0(BC) >= 2 S_0(A)          (CHLW Thm 2)
S_0(AB) + S_0(AC) >= S_0(BC)                     (CHLW Hyp 1 = Song et al. Thm 9)
```

with CHLW Thm 1 `S_0(AB) + S_0(AC) >= S_0(A)` a *corollary* of the last two. Six extremal-ray
families of this cone are (asymptotically) realised by explicit states; the two remaining
LRS rays (7 and 8) of the weaker cone are cut off by the newly proved inequality.

**But the cone is not the answer.** `Omega_n` is a *discrete* set (a subset of
`log N^{2^{n-1}-1}`), closed under addition (tensoring states adds 0-entropy vectors) but
not a cone. CHLW explicitly flag this in their conclusions: `Omega_3 = C_3 ∩ log N^3`, but
"the analogue of this is not true for general `n`", and they pose as an open problem whether
every log-integer point in the *interior* of `C_n` lies in `Omega_n`, or at least every
interior point of sufficiently large norm. **As of this review that question is still open**
— I found no paper resolving it. This is exactly the gap your numerics could probe.

**Tensor-network side.** For a *tree* tensor network, the minimal bond dimension of an edge
equals the flattening rank across the cut that removing the edge induces. Jovcheva–Seynnaeve–
Vannieuwenhoven (2025) Thm 3.6 gives the complete answer for the `n - 1` cuts of a fixed
tree: the tuple is realised (and generically so) iff it is **admissible**,
`r_ij <= dim V_i * prod_{k in nb(i)\{j}} r_ik`. So *any single tree's* worth of cuts is
governed by nothing but the obvious local polygon/dimension conditions. All the difficulty
in the full profile question comes from imposing several *incompatible* trees at once
(e.g. for `n = 4`, `r_AB` belongs to the caterpillar A-B-C-D while `r_AC` belongs to
A-C-B-D). This is a clean way to state what is genuinely new about CHLW's inequalities.

For graphs *with loops* the analogous question is the quantum max-flow/min-cut problem
(Cui–Freedman–Sattath–Stong–Minton 2016): the min-cut value (a product of capacities, i.e.
the submultiplicative bound) is *not* always attained by the max rank; equality holds when
all capacities are powers of a fixed integer, but concrete counterexamples exist otherwise.
Gesmundo–Landsberg–Walter (2018) give three infinite families of strict inequality, including
the periodic 4-cycle MPS. Hastings (2017) shows the ratio -> 1 as edge dimension -> infinity,
so the obstructions are a small-dimension phenomenon.

**Matroid / polymatroid remark.** For *classical* joint distributions the analogue of `r_S`
is the support size `s_S`, and CHLW note this is the special case of the quantum problem for
purifications of classical distributions. Classically one has *extra* inequalities the
quantum ranks lack: monotonicity `s_I <= s_J` for `I subset J` (false for quantum ranks!)
and the uniform-cover / shadow inequality of Schwenk–Munro,
`s_J^{C(|J|-1, k-1)} <= prod_{|I| = k, I subset J} s_I` (Loomis–Whitney / Bollobás–Thomason
box theorem territory). CHLW Theorem 2 is exactly the quantum case `|J| = 3, k = 2` of that
inequality; **the general `|J|, k` quantum version is open** and is a natural place to look
for either new inequalities or counterexamples. So `log r_S` is *not* a polymatroid rank
function (it is not monotone), but it is subadditive and, at least in the known cases,
obeys the uniform-cover family.

Song et al. Lemma 10 also gives verified multipartite extensions, e.g. (their numbering)
`sum_{j=1}^{n} S_0(A_j A_{j+1}) >= 2 S_0(A_1)` (cyclic),
`sum_{j=1}^{n-1} S_0(A_j A_{j+1}) >= S_0(A_1 A_n)`,
`S_0(A_1A_2A_3) + S_0(A_1A_2A_4) + S_0(A_1A_3A_4) >= S_0(A_2A_3A_4)`,
`sum_k S_0(A_1 ... \hat{A_k} ... A_n) >= S_0(A_1)`,
and `sum_j S_0(A_j) >= S_0(A_1...A_n)`. **No new independent inequality for `n >= 5` is known**;
whether the 5-party cone needs more than the 4-party ones (lifted) is open, in exact analogy
with the von Neumann case.

**Order-4 / `2x2x2x2` specifically.** Gharahi–Mancini–Ottaviani (2020) Appendix B, Thm 2
settles the two-multiranks of four qubits completely:
(i) among `(r_AB, r_AC, r_AD)` the maximum is attained at least twice; (ii) that is the only
constraint, except that `(1,3,3)` is not achievable. Constraint (i) comes from Segre's 1920
identity that the three `4x4` determinants of the three flattenings of a `2x2x2x2` tensor
sum to zero — a genuinely *algebraic* (not information-theoretic) obstruction, and one that
is invisible to all of the inequalities above. They also note that the classical partial
classification in Segre's paper forgot the case `(4,4,2)` and its permutations.

### B3. Is there a profile obeying all obvious constraints yet unachievable? (Question 3)

**Yes — several, at three different levels of subtlety.**

1. *Ruled out by the known non-obvious inequalities.* Huber–de Vicente endnote [32]:
   `r_A = r_B = r_C = r_D = 2`, `r_AB = r_AC = r_BC = 1`. It obeys submultiplicativity but
   is clearly impossible (three simultaneous product cuts force a fully product state).
   CHLW Thm 1 excludes it (`2 > 1*1`). Good sanity check, not a real counterexample.

2. *Obeys ALL currently known inequalities and is still provably impossible.*
   CHLW's **ray 3**, integer point `(r_A, r_B, r_C, r_D, r_AB, r_AC, r_AD) = (2,2,2,2,2,2,1)`.
   Every constraint above is satisfied (two of them with equality). Yet `r_AD = 1` forces
   `|psi>_ABCD = |eta>_AD |theta>_BC`, hence `r_AB = r_A r_B = 4 != 2`. This is CHLW's own
   argument (they use it to explain why ray 3 is not attained), and it is *the* clean answer
   to your question 3. Same for **ray 6**, `(8,8,8,2,4,4,4)` — CHLW assert "similarly, one
   can show that it is not possible to find states on ray 6" but give no proof.

3. *Dimension-specific algebraic obstructions.* For `2x2x2x2`, `(r_AB, r_AC, r_AD) = (4,3,3)`
   (with all local ranks 2) obeys everything above but violates GMO's "max attained twice"
   and so is impossible for four qubits (it *is* possible for larger local dimensions).
   The exceptional `(1,3,3)` is impossible in *any* local dimensions once you also pin
   `r_A = 2`, because `r_AB = 1` forces `r_AC = r_A r_C`, and `3` is not a multiple of `2`.

So: the linear-inequality / cone picture, even though it is now *complete* for `n = 4`, is
strictly weaker than the true achievability question. The failures come from (a) boundary
degeneracies where some `r_S = 1` collapses the state into a product, and (b) determinantal
identities among flattenings that have no information-theoretic shadow.

### B4. Related: SLOCC classification, Schmidt-rank vectors, entanglement polytopes

* Each `r_S` is an SLOCC monotone (invariant under invertible local operations), which is
  why the rank vector is the coarsest useful SLOCC invariant. Dür–Vidal–Cirac (2000) already
  show it is not complete: GHZ and W both have local rank vector `(2,2,2)` but are
  SLOCC-inequivalent (tensor rank 2 vs 3).
* Huber–de Vicente (2013) define the Schmidt-rank vector for pure states and a Schmidt-number
  vector for mixed states via `r_j = min_{D(rho)} max_{psi_i} r_j^{psi_i}`; each entry is an
  entanglement monotone, and the induced order is only *partial* (e.g. `(4,2,2)` and `(3,3,2)`
  are incomparable). Huber–Perarnau-Llobet–de Vicente (2013) extend it to `k`-separability.
* Verstraete–Dehaene–De Moor–Verschelde (2002): nine SLOCC families of four qubits.
  Gharahi–Mancini–Ottaviani (2020) refine to `ceil(2^n/(n+1))` families by secant varieties
  and then split each by `l`-multiranks — this is the modern "classify by rank profile" work,
  and Gharahi's 2025 Mathematica codebase (arXiv:2601.11551) computes those profiles.
* Entanglement polytopes (Walter–Doran–Gross–Christandl 2013) constrain the *local spectra*
  compatible with an SLOCC class. That is a finer but different object from the rank vector:
  ranks see only supports, spectra see eigenvalues. A rank constraint is not a corollary of a
  polytope, and vice versa. Worth checking whether entanglement-polytope data for a small
  system (4 qubits) can certify some of the unachievable profiles in (C).
* CHLW's own suggested extension: replace bipartition Schmidt ranks by *tensor ranks* over
  all set partitions of `[n]` (Bell-number many); they call the universal inequalities among
  those "very interesting, yet wide-open".

---

## (C) Candidate rank profiles worth testing numerically

Convention for `n = 4`: `(r_A, r_B, r_C, r_D | r_AB, r_AC, r_AD)`, with `r_AB = r_CD`,
`r_AC = r_BD`, `r_AD = r_BC`. "All known inequalities" = submultiplicativity for disjoint
sets + `r_A^2 <= r_AB r_AC r_AD` (CHLW Thm 2) + polygon among `(r_AB, r_AC, r_AD)`
(Song et al.) + `r_S <= prod_{i in S} d_i`.

| # | profile | min local dims | passes all known ineqs? | expected status | why interesting |
|---|---|---|---|---|---|
| C1 | `(2,2,2,2 \| 2,2,1)` | 2,2,2,2 | yes (two tight) | **provably unachievable** | CHLW ray 3. `r_AD=1` ⇒ `|psi>=|eta>_AD|theta>_BC` ⇒ `r_AB = r_A r_B = 4`. The headline counterexample; smallest possible system; use as the correctness test of any search code. |
| C1' | `(d,d,d,d \| d,d,1)`, `d>=2` | `d` each | yes | unachievable, same proof | infinite family; also check the "fixed" version `(d,d,d,d \| d^2,d^2,1)` which IS achievable. |
| C2 | `(8,8,8,2 \| 4,4,4)` | 8,8,8,2 | yes (two tight) | asserted unachievable, **no published proof** | CHLW ray 6. They say "one can show" but do not. Worth either finding the proof or a state. Family: `(d^3,d^3,d^3,d \| d^2,d^2,d^2)`. |
| C3 | `(4,4,2,2 \| 4,2,2)` | 4,4,2,2 | yes (four tight) | **OPEN** in CHLW | ray 4. Their `psi_4` gives `(d^2+1,d^2+1,2d,2d \| 2d^2,2d,2d)` — on the ray only asymptotically. Family `(d^2,d^2,d,d \| d^2,d,d)`. Highest-value target: an actual state would settle a 12-year-old open case. |
| C4 | `(4,4,4,2 \| 4,2,2)` | 4,4,4,2 | yes (three tight) | **OPEN** in CHLW | ray 5. Family `(d^2,d^2,d^2,d \| d^2,d,d)`; CHLW's `psi_5` gives `(d^2+d+1, d^2+d+1, d^2+2d, 2d+1 \| 3d^2, 3d, 3d)`. |
| C5 | `(2,2,2,2 \| 4,3,3)` | 2,2,2,2 | yes | **unachievable for qubits** (achievable for larger `d_i`) | GMO Thm 2(i) via Segre's determinant identity. A purely algebraic obstruction that no entropy inequality sees. Cheap to test exhaustively/numerically. |
| C5' | `(2,2,2,2 \| 4,2,2)`, `(2,2,2,2 \| 3,2,2)`, `(2,2,2,2 \| 4,4,3)` | 2,2,2,2 | yes | unachievable for qubits (max attained once) | same mechanism; `(4,4,2)` and `(3,3,2)` and `(4,4,4)` ARE achievable — good positive controls. |
| C6 | `(2,2,2,2 \| 1,3,3)` | any | yes | **unachievable in every dimension** | GMO's exceptional triple. `r_AB=1` ⇒ product across `AB|CD` ⇒ `r_AC = r_A r_C`, and `3` is not divisible by `r_A = 2`. Generalises: whenever some `r_S = 1`, all ranks factor, and *that* factorisation constraint is not implied by submultiplicativity. |
| C7 | `(2,2,2,2 \| 1,1,1)` | 2,2,2,2 | **no** (fails CHLW Thm 1) | unachievable | Huber–de Vicente endnote [32]. Negative control: your checker should reject this from the inequalities alone. |
| C8 | `(2,2,2,2 \| 2,2,2)` | 2,2,2,2 | yes | achievable (GHZ_4) | positive control. |
| C9 | `(3,3,3,3 \| 9,9,9)` | 3,3,3,3 | yes | achievable | CHLW `psi_2 = sum_{i,j=0}^{2} \|i>_A\|j>_B\|i+j>_C\|i+2j>_D` (mod 3). Positive control on ray 2. |
| C10 | interior sweep | small | yes | ??? | CHLW's actual open question: enumerate ALL log-integer points inside `C_4` up to some bound (say all `r_S <= 8`), decide achievability numerically, and see whether every *interior* point is achievable. If yes for a large window, that is real evidence for their conjecture; a single interior failure would be a genuinely new result. |
| C11 | `n = 5`: quantum uniform-cover, `\|J\|=4, k=2` | any | — | ??? | Is `r_A^3 <= r_BC r_BD r_BE r_CD r_CE r_DE` true for 5-party pure states? Classically true (Schwenk–Munro); quantum version unproved. CHLW only prove `\|J\|=3, k=2`. A random-search counterexample here would be a new inequality *failure*, i.e. news. |
| C12 | `n = 5` cone | any | — | ??? | Are the lifted 4-party inequalities complete for 5 parties? Direct analogue of the famous open von Neumann problem. Generate random states / structured `⊕`-sums (CHLW's orthogonal-direct-sum trick adds rank vectors) and look for points outside the lifted cone. |
| C13 | periodic 4-cycle MPS | bond dim 2,3,... | — | max-flow < min-cut | Cui et al. / Gesmundo–Landsberg–Walter. Not a rank *vector* question but the same phenomenon; a good source of concrete tensors whose flattening ranks are strictly below the submultiplicative bound. |

**Numerical method notes.**
* Achievability is a *generic* question: sample random tensors inside a parameterised
  family (e.g. random `T` with prescribed `r_AB` by construction, then measure `r_AC`,
  `r_AD`) and take the *maximum* observed rank; ranks are lower semicontinuous, so random
  sampling gives the generic (= maximal) value on any irreducible family.
* Rank profiles are only *upper* semicontinuous-limitable: the set of tensors with rank
  profile `<= r` is closed, but the set with profile *exactly* `r` is only locally closed.
  Because MPS-with-PBC/PEPS sets are not closed (Barthel–Lu–Friesecke; Landsberg–Qi–Ye),
  "approached in the limit" is genuinely weaker than "attained" — precisely CHLW rays 3–6.
* CHLW's two constructions are the workhorses for building profiles: (a) tensor product of
  states *adds* 0-entropy vectors; (b) orthogonal direct sum `|psi> ⊕ |eta>` (embedding in
  orthogonal local subspaces) *adds* rank vectors.
* Gharahi's `mathoud/Flattening` Mathematica code computes full `l`-multirank profiles and
  is the obvious off-the-shelf checker.

---

## (D) Unverified / could not confirm

* **Carlini–Kleppe abstract.** The ScienceDirect page
  `https://www.sciencedirect.com/science/article/pii/S0022404910002616` returns HTTP 403,
  and I found no arXiv preprint. Title/authors/journal/volume/pages/year/DOI are verified
  from the Monash University research portal record. The *statement* of the theorem
  (`r_i <= prod_{j!=i} r_j` and `r_i <= dim V_i`, all orders `d`) is verified only
  **indirectly**, from two independent papers that recall it: Jovcheva–Seynnaeve–
  Vannieuwenhoven arXiv:2509.09463 (eq. (3), "The main result of [CK11] states that the set
  TN°(G,r) is nonempty if and only if the bond dimensions satisfy ...") and
  Gharahi–Mancini–Ottaviani arXiv:1910.09665 App. B ("Carlini and Kleppe have classified all
  possible one-multiranks for any number of qudits"). **Get the original PDF before citing
  the theorem as your own primary reference.**
* **CHLW ray 6 unachievability.** The paper writes "Similarly, one can show that it is not
  possible to find states on ray 6, but we do not know this for rays 4 and 5" — no proof is
  given. Treat as asserted-not-proved.
* **Gesmundo–Landsberg–Walter gap size.** Two independent fetches of the same abstract
  rendered the max-flow/min-cut gap for the `2d`-cycle at bond dimension two as "at least
  `2d - 2`" and as "at least `2^(d-2)`". I could not disambiguate. Read the paper before
  quoting a number.
* **Song–Chen–Sun–Hu journal metadata.** IEEE Trans. Inf. Theory 69(4):2385–2399 (2023) came
  from a WebFetch of the arXiv abstract page's journal-ref field; I did not open the IEEE
  page. DOI not captured.
* **Huber–Perarnau-Llobet–de Vicente abstract** was summarised, not quoted verbatim; the
  PRA 88, 042328 (2013) reference is from the arXiv journal-ref field.
* **Linden–Mosonyi–Winter (arXiv:1212.0248)**, **Schwenk–Munro (1983)**, **Segre (1920)**,
  **Briand–Luque–Thibon (2003)**: recorded only as reference-list entries inside CHLW /
  GMO. Not independently fetched.
* **Buczyńska–Buczyński–Michałek "part two" (arXiv:1802.00222)**: existence and title
  confirmed by search results only; abstract not fetched.
* **Ye–Lim, "Tensor network ranks"**: no journal publication found; the paper appears to
  exist only as arXiv:1801.02662 (v2, 2019) plus a copy on Lek-Heng Lim's page. If a
  published version exists I did not find it.
* **"Quantum Max-Flow Min-Cut theorem", arXiv:2110.00905**, and **"Quantum Max-flow in the
  Bridge Graph" (Transformation Groups, 2024)** appeared in searches and look relevant to
  the achievability-of-min-cut question, but I did not fetch or verify them.
* **No paper found** that (i) resolves CHLW's interior-log-integer-point question, (ii)
  gives any 5-party rank inequality independent of the lifted 4-party ones, or (iii) treats
  the full `2^{n-1}-1` rank profile for `n >= 5`. If such a paper exists my searches missed it.
* I did **not** find a dedicated "rank profile of a tensor" literature under that exact
  phrase; the relevant work is split between the quantum-info "rank vector / 0-entropy"
  thread (CHLW → Song et al.) and the algebraic-geometry "multilinear rank / tensor network
  rank" thread (Carlini–Kleppe → Ye–Lim → Jovcheva et al.). Connecting those two threads
  explicitly looks like an open niche.
