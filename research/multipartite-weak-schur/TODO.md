# TODO

## Open research questions

- [ ] **TOP PRIORITY: close the two-sided characterisation of rank profiles.** The pair
      is (a) the linear relaxation $W_r$ — cheap, exact, certifies **impossible**; and
      (b) witness search by non-convex local search on $\psi$ — certifies **achievable**.
      Where they meet, a profile is settled. On $2\times2\times2\times2$ at $k=4$ the
      relaxation already reproduces the full "max of $(r_{AB},r_{AC},r_{AD})$ attained at
      least twice" rule, 20/20 triples, beating the polygon inequalities. Remaining work
      is the gap where the relaxation says feasible but no witness exists.
- [ ] **Do NOT expect an exact SDP.** rank $\le r_S$ on a flattening and rank-1-ness of
      $|\psi\rangle\langle\psi|$ are both non-convex, and the convex hull of the pure
      states is all density matrices — which discards precisely the purity that CHLW
      ray 3 turns on. An exact convex encoding would decide NP-hard problems (Kronecker
      positivity is NP-hard, Ikenmeyer–Mulmuley–Walter). So SDP here is only ever a
      relaxation, and it is only useful on the *impossibility* side; the feasibility side
      wants a witness, not convexity.
- [ ] **If a Lasserre/DPS level-2 is attempted**, its only job is to close cases where
      $W_r$ says feasible and witness search finds nothing — e.g. CHLW ray 3
      $(2,2,2,2 \mid 2,2,1)$. Level 1 is the current linear computation.
- [x] ~~**E007 at larger $n$ / larger local dimensions.**~~ **DONE.**
      `scripts/rank_profile_blocks.py` removes the 2 GB basis array: orthonormal multiset
      basis, index arithmetic instead of a materialised $B$, and block diagonalisation by
      the local torus ($\sum_b m_b^3 = 4.3\times10^8$ instead of $m^3 = 1.4\times10^{14}$).
      Swept $2^{\times 5}$ at $k=4$ — all 792 $S_5$-orbit representatives of pair-rank
      profiles in $\{2,3,4\}$ — and $(2,2,2,4)$ at all 64 triples for each $r_F$. See §8
      of `notes/algorithm_rank_profile_decision.md`.
- [ ] **Is the "ladder" of Segre relations a theorem?** The max-attained-twice pattern
      held at $\ell = 2, 3$ and $4$, not only at $\ell=4$ where it is Segre's determinant
      identity. Is there a uniform statement across rank levels? Looks provable, and I
      did not find it stated. **Now known to be a local-rank-2 statement, not a
      four-party one**: on $2\times2\times2\times4$ with $r_F = 3$ the triple $(3,3,4)$
      has an explicit witness even though its maximum is attained once
      (`artifacts/rank_witness_2224.log`).
- [x] ~~**`battery()` does not check singleton certification.**~~ **FIXED** in
      `rank_profile_blocks.py`, which certifies every cut. It reproduces all 20 values of
      $\dim W_r$ and all 20 certified triples of `rank_battery_k4.log`, and flips
      $(1,1,1)$ to IMPOSSIBLE — the verdict Lemma 5 gives independently.

## New from the $n=5$ sweep

- [ ] **TOP PRIORITY: 124 open profiles at $n=5$.** Two-sided run over all 792
      $S_5$-orbit representatives ($2^{\times5}$, $k=4$, local ranks 2, pair ranks in
      $\{2,3,4\}$): 634 IMPOSSIBLE by the relaxation, 34 ACHIEVABLE by witness, **124
      open**, 0 contradictions. A second pass at 6x the restart budget converted only 2
      of 126, so these are probably not local-minimum artefacts. This is a much sharper
      target for a level-2 Lasserre/DPS relaxation than CHLW ray 3, which was one profile
      closed by one purity lemma.
- [ ] **Prove the conditional triangle rule.** For every triangle $\{i,j,k\}$ of $K_5$
      whose opposite edge has $r_{lm} = 2$, the maximum of $(r_{ij},r_{ik},r_{jk})$ is
      attained at least twice. It should follow from the $2^{\times4}$ rule by merging
      $l,m$ into one party of dimension 4 and local rank 2; the sweep confirms it with 0
      exceptions in 792. The *unconditional* version is false — 16 of the 34 witnesses
      violate it (e.g. `3333333334`) — as is the vertex-star version (5 witnesses).
- [ ] **Characterise the 59 genuinely 5-partite obstructions.** 59 of the 634 impossible
      profiles satisfy every rule inherited from a coarse-graining. (All 4-group merges
      of a 5-set are pair merges, giving exactly the 10 triangles; 3-group merges give
      polygon inequalities, every one of which is vacuous at these dimensions.) A local
      rule fitted on (vertex star + opposite triangle) has 0 false negatives against the
      witnesses, so a clean combinatorial statement may exist — but it cannot be settled
      while 124 profiles are undecided.
- [ ] **`report.tex` §E007 is now out of date.** It reports only the $n=4$ sweep and
      states the max-attained-twice rule without the local-rank-2 caveat. Extend it with
      §8 of the note, or scope its claims explicitly to $2^{\times4}$.

- [ ] **TOP PRIORITY: tighten the E007 relaxation towards the Veronese variety.** The
      linear relaxation ($W_r$ = subspace of $\text{Sym}^k$ cut out by the vanishing
      conditions) reproduces the whole $2\times2\times2\times2$ "max attained at least
      twice" rule, 20/20 triples, and beats the polygon inequalities. It is level 1 of
      what should be a hierarchy. It misses CHLW ray 3 because $W_r \neq 0$ only says
      some *symmetric tensor* meets the bounds, not some *product* tensor. Add
      Lasserre/DPS-style constraints (symmetric extensions, PPT) to force $\sigma$
      closer to a mixture of genuine $\psi^{\otimes k}$ — **this is where an SDP earns
      its place**, unlike level 1 which is pure linear algebra. Does level 2 reach
      ray 3? Start from `scripts/rank_profile_subspace.py`.
      (These three items were duplicated here; see the merged versions above.)

- [x] ~~**Several incompatible trees**, one per copy-group.~~ **CLOSED, negative** (E006).
      $\psi^{\otimes mk} = (\psi^{\otimes k})^{\otimes m}$, so the joint distribution
      factorises into the per-group marginals and one learns only the conjunction of the
      per-tree conditions. CHLW ray 3 passes all six node-local polygon conditions and
      survives.
- [x] ~~**Sequential / order-dependent measurement.**~~ **CLOSED, negative** (E006) — but
      with a genuinely interesting by-product, see below. The sequential support IS
      strictly smaller than the product of marginals ($n=4$, $k=4$, $S=AB$, $T=AC$: 4
      coupled pairs killed, $10^{-17}$ vs $10^{-2}$). However it certifies nothing about
      rank: on an explicit state with $r_{AB}=4$, $r_{AC}=3$, the outcome
      $(\ell^{AB},\ell^{AC}) = (4,3)$ has probability zero, and sequential outcomes with
      $\ell^{AC} = 4 > r_{AC} = 3$ occur with probability $\sim 10^{-4}$. The first
      projection destroys the bound $\ell(\lambda^{(T)}) \le r_T$.

- [ ] **NEW: characterise $\rank\bigl(P^{(T)}_\mu P^{(S)}_\lambda|_{\Sym^k}\bigr)$ for
      overlapping $S,T$.** This is the one genuinely coupled structure found in the whole
      project, it is not explained by any factorisation identity I know, and it first
      appears at $k=4$ (nothing at $k=3$). Questions: is there a closed form? Does the
      coupled set grow with $k$? Is it some kind of "Kronecker coefficient for a
      non-commuting pair"? Not a route to rank obstructions, but well-posed and possibly
      the most novel thing here. Start from `scripts/sequential_support.py`.
- [ ] **Extend R3 to deeper/wider trees.** Only the $n=4$ tree with one internal node
      was tested. Check a balanced $n=8$ tree and a caterpillar, to be sure the
      decoupling is not an artefact of having a single internal node.
- [ ] **$r_{\max} > 4$ in E004.** Reachability saturated by $k=6$ for $r \le 4$; confirm
      the decoupling persists at $r_{\max} = 5,6$ (cost grows as $p(k)^3$, still cheap).
- [ ] **$n$-fold semigroup property.** CHM prove the Kronecker semigroup property for
      triples. Lit review found no $n \ge 4$ version on record. Row-wise addition of
      $\bmlam$ preserving $g > 0$ looks provable by the same argument and would be a
      small genuinely new result. Test numerically first.
- [ ] **Row-length necessary condition for $n \ge 4$.** Is
      $\ell(\lambda^{(i)}) \le \prod_{j \ne i} \ell(\lambda^{(j)})$ the *only* row-length
      constraint for the $n$-fold coefficient? E004/R2 says yes up to $k=9$, $r\le4$ ---
      worth proving.

## Unverified claims in report.tex

- [ ] **Sufficiency for TREE rank profiles is not established here.** Corollary
      (rank reachability) is an equivalence only for *local* ranks, where the
      ($\Leftarrow$) direction can set $d_i := r_i$. At an internal node that squeeze
      fails: $d_A = r_A, d_B = r_B$ gives $\text{rank}\,\rho_{AB} \le r_A r_B$, not
      $\le r_{AB}$. So R3 proves only the ($\Rightarrow$) direction. The converse holds
      by Jovcheva--Seynnaeve--Vannieuwenhoven, from the literature. Either prove it
      directly (construct a state realising a prescribed tree profile from a
      Kronecker-positive assignment) or cite it explicitly. See Remark (Scope of R3)
      in report.tex.
- [ ] Corollary (rank reachability), direction ($\Rightarrow$), uses Keyl--Werner
      concentration plus a union bound. Stated but **not** independently verified
      numerically. Add a check: sample states with prescribed ranks, confirm
      $\ell(\lambda^{(i)}) = r_i$ occurs jointly for moderate $k$.
- [ ] Proposition (tree factorisation) is verified only for the $n=4$ one-internal-node
      tree. The general statement is proved but only spot-checked.

## Citations to verify before any writeup leaves the workspace

- [ ] **Carlini--Kleppe 2011** --- ScienceDirect returned 403; no arXiv preprint. The
      statement was confirmed only *indirectly* (arXiv:2509.09463 Eq. 3,
      arXiv:1910.09665 App. B). Marked `% TODO: verify` in report.tex. Get the actual
      paper.
- [ ] **Botero--Mej\'ia arXiv:1712.09174 §II--III** --- read in full and check whether the
      support theorem appears as an unstated remark. This is the main novelty risk.
- [ ] **Gonzalez arXiv:2504.16256** --- body could not be fetched (size limit). Read by
      hand; it is about Kronecker states and could contain the support statement.
- [ ] Items 1--14 in `notes/lit_kronecker.md` section (D) and the section (D) list in
      `notes/lit_rank_profiles.md` are all unverified. Do not cite any of them as
      established.

## Engineering

- [ ] `mws_distribution` is $O(k! \cdot |\text{tuples}| \cdot D^k)$. Fine to $k=5$,
      $n=5$. If deeper $k$ is needed, sum over conjugacy classes instead of individual
      permutations ($p(k)$ terms instead of $k!$).
- [ ] Package requests are in `notes/package_wishlist.md`. The one that would most
      simplify this code: let `apply_isotypic_proj` take k axis-*groups* so block
      projectors are first-class.

## Done

- [x] Support theorem stated and verified (E001)
- [x] Laminar commutation established exactly in the group algebra (E002)
- [x] Tree factorisation (E003)
- [x] Rank reachability vs polygon inequalities (E004)
- [x] Vanishing statistics + figure (E005)
- [x] Literature review: `notes/lit_kronecker.md`, `notes/lit_rank_profiles.md`
