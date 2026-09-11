# TODO

## Highest-value next step
- [ ] **Close the constant factor in the spread bound to settle $k=3$.**
  Proved (verify_spread_lemma.py): for rank-two $C$ with right singular vectors $b_a$ and
  $L=\max_i\max_a\lambda_{\max}(\rho^{b_a}_i)$,
  $q_3(-1/2,C) \ge (3/4 - 3L)\|C\|^2$, so the endpoint holds whenever $L<1/4$.
  The most spread case is $L=1/d$, so this settles $d\ge5$ but not $d=3,4$.
  The bound throws away the non-negative even terms $\|\mathrm{Tr}_{ij}C\|^2$, and E008
  shows those are NOT negligible on that sector. Keeping them should close the factor.
  Combined with Wu-Zou (arXiv:2608.02647), who cover the low-overlap / normal / PSD cases,
  this would settle the three-copy endpoint.
- [ ] E008 measured the margin on the full-support sector: $\theta_3 \gtrsim 0.025$ at
  $d=3$, growing roughly linearly in the support margin $\varepsilon$. Turn that into a
  proved lower bound $\delta(\varepsilon)$.
- [ ] **Push the LP bound past $\gamma_3 > 1/3$.** Currently $\gamma_3 \ge 0.312908$, which
  gives NPT + provably 3-copy-undistillable Werner states from $d\ge4$. Crossing $1/3$
  would give them in the *minimal* nontrivial dimension $d=3$. The gap is small.
  Diagnosis of where the LP is loose: at the optimum it sets
  $x_{12}=x_{13}=x_{23}=0$ while $x_{123}=|\mathrm{Tr}\,C|^2=2$, which is *physically
  impossible* — $\mathrm{Tr}_{12}C = 0$ forces $\mathrm{Tr}\,C = 0$. So a valid
  dimension-free inequality bounding $|\mathrm{Tr}\,C|^2$ in terms of
  $\|\mathrm{Tr}_S C\|^2$ would tighten it immediately. The obvious route
  ($|\mathrm{Tr}\,C|^2 \le \mathrm{rank}(\mathrm{Tr}_S C)\,\|\mathrm{Tr}_S C\|^2$) is
  dimension-dependent and therefore unusable as-is; a dimension-free substitute is the
  thing to find. Note the obvious candidate is now exhausted: the BGH difference
  inequality (their Eq. 82), lifted to spectator form, is non-binding at $k=3$.
- [ ] Search for further valid dimension-free inequalities on
  $x_S = \|\mathrm{Tr}_S C\|^2/\|C\|^2$ for rank-$\le2$ $C$ (three-term analogues of
  FHPV; anything constraining $x_S$ across nested subsets). Test candidates numerically
  against random rank-2 $C$ before adding them to the LP.

## Open
- [ ] **Prove the $k=3$ endpoint conjecture** ($\theta_3(1/2)=0$, equivalently $c_3=1/2$).
  Equivalent form (recursion, verified): $\sum_{m,m'} q_2(C^{(mm')}) \ge \tfrac12 q_2(\mathrm{Tr}_3 C)$.
  The obstruction is that $\mathrm{Tr}_3 C$ has rank up to $2d$ while the blocks
  $C^{(mm')}$ have rank $\le2$.
  - [x] Ruled out: dropping the off-diagonal blocks. The resulting statement
    $\sum_m q_2(D_m)\ge\tfrac12 q_2(\sum_m D_m)$ is **false** (take all $D_m$ equal).
- [ ] Prove or refute H1: $\theta_k(c)=(1-2c)(1-c)^{k-1}$ for $0\le c\le1/2$ (the padded
  one-copy optimiser is globally optimal). Verified numerically at $k\le3$, $d\le4$.
  **False for $c>1/2$** — a traceless rank-one pad gives $\theta_k\le1-2c$, which is lower.
- [ ] Complete the E004 full-$c$ grid at larger $(d,k)$ — stopped for compute; E006 covers
  the decisive endpoint $c\in\{0.48,0.5\}$ instead.
- [ ] $k=6,7$ at $d=3$ in the full space needs a matrix-free Lanczos for $T_c$
  ($4374^2$ dense at $k=7$). Not implemented.
- [ ] Does $\theta_k(c)$ saturate in $d$? Dimension-free for $k=1,2$ (proved); data covers
  $d\le6$ at $k=3$.

## Claim status
- [x] Reduction chain R1--R3: **verified** (`scripts/verify_reduction.py`, 60/60).
- [x] $\gamma_k$ LP bound and the spectator lemma: **verified**
  (`scripts/verify_lp_gamma.py`, 5/5), including two sanity checks that the bound never
  exceeds the padded upper bound nor any sampled rank-2 value.
- [x] $\gamma_k$ recursion bound: **verified** (`scripts/verify_gamma.py`, 8/8). Superseded
  by the LP bound but kept — it is the simpler statement.
- [x] $\theta_1(c)=1-2c$, $\theta_2(c)=(1-2c)(1-c)$ for $c\le1/2$: **verified** to $10^{-15}$;
  $\theta_2$ also follows rigorously from FHPV and agrees with the July 2026 theorem.
- [x] No descent direction at 22 equality points, $k=3..7$: **verified**
  (`scripts/exp003_hessian.py`).
- [x] Novelty of the improved $\gamma_k$ relative to arXiv:2607.24479: **checked against
  the full text** (v1). Their Thm 1.1 / 6.1 / 9.1 and Eqs (81)-(82) are all stated for
  *complementary* bipartitions only; no spectator version, no optimisation over
  $\|\mathrm{Tr}_S C\|^2$. Table 1 confirms $\gamma_3=1/6$, $\gamma_4=0.124104$,
  $\gamma_5=0.098111$. The delta is the spectator lift + the LP. See report Sec.
  "Relation to Bharti-Gajjala-Haug".
- [x] Their Eq (82) (difference inequality) lifted to spectator form and added to the LP:
  valid, but **non-binding** for $k\le4$. $\gamma_3$ unchanged at $0.312908$.
- [ ] Still unchecked: whether Fraser-Huber-Pozsgay-Vona (arXiv:2607.24309) or
  Tabia-Chen-Hsieh (arXiv:2608.08836) contain a spectator-type inequality. Only 2607.24479
  was read in full.

## Resolved
- [x] Spurious "counterexample" at $d=3,k=3$ ($-6.87\times10^{-2}$): numerical artifact.
  Cause and fix in report.tex; all minima are now selected *and* reported via independent
  numpy re-evaluation (`bilinear.verified_value`).
- [x] The see-saw in `werner.py` is unreliable (misses the padded optimiser at $d=3,k=2$).
  Superseded by the batched degeneracy-free optimiser.
