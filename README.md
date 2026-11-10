# Gadget reductions and NP-hardness of MAX-k-CUT for small k

## Headline results (smaller ratio = stronger hardness)

Paper draft: `paper/main.tex` (compiled `paper/main.pdf`). Numbers below are the paper's.

| problem | previously stated best NP-hardness | here | how |
|---|---|---|---|
| MAX-3-CUT | 16/17 ≈ 0.941176 (AOTW; perfect completeness) | **27/29 ≈ 0.931034** (gap 29/30 vs 9/10) | negation merge of Håstad's 3LIN(3), gadgets E (NRT's type-0 gadget) and N (triangle + one aux) |
| MAX-(3+L)-CUT | 1 − 1/(17(k+c_k)+c_k) (AOTW Thm 1.2): 85/86, 120/121, 101/102, 136/137 | **1 − 2/Γ_L**, Γ_L = 29+18L+3·C(L,2): **45/47** (k=4), **33/34** (k=5), 45/46, 117/119 | family (2A_L, N_L, N_L): E/N plus L universal globals inside the gadgets, analysed with the soundness criterion (paper Thm 4.2); proof by hand for all L |
| k-colourable graphs, k = 3+L ≥ 4 | same formula | 1 − 3/(51+34L+3·C(L,2)): 82/85, 119/122, … | universal-vertex reduction of KKLP/GS applied directly to AOTW's 16/17 |

The new bounds are the best known for 3 ≤ k ≤ 14 (family) and 15 ≤ k ≤ 20 (universal vertices); AOTW's
formula is better for k ≥ 21.

Status (not peer reviewed):
* All three positive results have proofs by hand; gadget values were also confirmed by exhaustive
  enumeration (`scripts/verify_sum_criterion.py` on `results/unified_L*.json`, L ≤ 3) and by an
  independent dynamic program (`scripts/independent/family_dp.py`, L ≤ 6).
* Independent adversarial reviews: 27/29 (Håstad's paper re-read, fresh brute force, end-to-end tests),
  the soundness criterion (local LP over all decoding maps), the universal-vertex lemma, and four referee
  passes over the paper draft (two mathematical, citations, presentation).
* Literature search (NRT v7 Mar 2026, Heilman Jul 2026, AOTW, GS, KKLP, HHMOW17, ECCC/arXiv 2025–26): no
  prior bound at or below 27/29, no use of the negation trick, no statement of 82/85-type constants.
* Barrier (computer-assisted): under the soundness criterion no 3LIN(3) → MAX-3-CUT gadget family of any
  size beats 27/29 — rational LP-dual certificates `results/barrier_k3/cert_t{0,1,2}.json`, checked exactly
  by `check_exact.py` and by an independent pairwise checker `check_exact_pairwise.py`.
* Superseded files: `results/sumS_k*.json` (earlier LP-found k = 4..7 families with the same ratios; the
  paper uses the symmetric unified family) and all k = 4, 5 gadget families further below.

## Reproducing the paper's claims

Requirements: Python 3 with NumPy (all checks below); HiGHS (`highspy`), OR-Tools and CVXPY are only needed
for the original LP searches and the superseded k = 4, 5 certification pipeline.

```bash
python3 scripts/verify_sym3.py                     # Thm 1.1: gadget values (Lemma 3.1), identity (3), 27/29
python3 scripts/verify_sum_criterion.py results/unified_L0.json results/unified_L1.json results/unified_L2.json
                                                   # Lemma 5.2 / Thm 1.2 for k = 3, 4, 5 (exhaustive enumeration)
python3 scripts/make_unified_family.py 3 /tmp/L3.json && python3 scripts/verify_sum_criterion.py /tmp/L3.json
(cd scripts/independent && python3 family_dp.py 6) # independent dynamic program, L <= 6
python3 scripts/universal_vertex.py                # Lemma 6.1 (brute force) and the small-k table
cd results/barrier_k3 && python3 check_exact.py cert_t0.json 8/9 && python3 check_exact.py cert_t1.json 9/10 \
  && python3 check_exact.py cert_t2.json 9/10       # Thm 7.1: exact check of the barrier certificates (~10 min each)
```

`results/barrier_k3/check_exact_pairwise.py` is a second, independently written exact checker of the barrier
certificates (it reads the `mudual_*.pkl` files and rationalises them).

## MAX-3-CUT below 16/17: 27/29 by merging t and −t (negation trick)

**Theorem.** For every ε > 0 it is NP-hard, given a weighted graph, to distinguish 3-cut value
≥ (29/30 − ε)·(total weight) from ≤ (9/10 + ε)·(total weight). Hence MAX-3-CUT is NP-hard to
approximate within 27/29 + ε. (AOTW's 16/17 has perfect completeness, i.e. holds on 3-colourable
graphs; there it still stands.)

*Source.* Håstad's Max-E3-Lin over Z_3 (JACM 2001, Thm 5.9 / test L_Γ(u)): equations
x_U + x_W1 + x_W2 = t, all coefficients +1, one variable from a U-table and two from a W-table;
soundness 1/3 + ε holds for *every* assignment of the (folded) tables. Types are balanced (weight 1/3
per t) by taking three copies and shifting the U-variables of copy a by a (needs U- and W-role
tables to be disjoint: give each prover its own tables; without balancing see below).

*Negation trick.* If X is an assignment then so is −X, and −X satisfies an equation iff X has sum
s = −t. Hence in the NO case Σ_eq ([s = t] + [s = −t]) ≤ 2(1/3 + ε)m. For t = 0 the summand is 2[s = 0];
for t ≠ 0 it is [s ≠ 0]. Both "s = 0" (all equal or all distinct) and "s ≠ 0" (exactly two equal) are
invariant under all colour permutations, so no global vertices (and no chirality) are needed.

*Gadgets* (`results/sym3_k3.json`):
* type 0 → E-gadget: NRT's 18-edge type-0 gadget with weights doubled; value 36 if s = 0, else 32 (α = 9);
* types 1, 2 → N-gadget: weight-1 triangle on the primaries plus one auxiliary vertex joined to all three
  with weight 3; value 11 if exactly two are equal, else 9 (α = 5.5).

Per equation the gadget value is exactly Base_t + 2([s = t] + [s = −t]) (Base_0 = 32, Base_1 = Base_2 = 9).
Soundness: J ≤ (32/3 + 18/3)m + 2·(2/3 + 2ε)m = (18 + 4ε)m. Completeness: ≥ (36/3 + 22/3)m − O(εm) = (58/3)m − O(εm)
(per-equation weight 20). Ratio 54/58 = 27/29. Without balancing (type-0 fraction w0 arbitrary) the ratio is
(23w0 + 31/3)/(25w0 + 11) ≤ 31/33 ≈ 0.9394, still below 16/17 (balance is in fact automatic in the NO case,
see Theorem S). Equations whose two W-variables fall in the
same folded class (negligible weight) are discarded; weighted → unweighted by Crescenzi–Silvestri–Trevisan.

In NRT's per-type framework the N-gadget is useless (it does not separate s = 1 from s = 2, so S2 = C);
the gain comes entirely from pairing X with −X in the soundness analysis.

*Checks.* `scripts/verify_sym3.py` (exact brute force of both gadgets, the identity, both ratios);
`scripts/check_sym3.py` (identity re-checked on 200 random 3LIN(3) instances over all assignments);
independent referee (fresh code, end-to-end: OPT of the produced graph = Base + 2·max_X(#sat(X) + #sat(−X))
on 40 random instances). *Limits of this route:* α = 9 is optimal for type-0 gadgets (LP-dual lower bound,
track L), column generation with global vertices finds no N-gadget below α = 5.5, and for these two gadgets
the uniformly random assignment already meets the soundness bound — so 27/29 is essentially the limit of
"E/N" gadgets from 3LIN(3).

## Soundness criterion (paper Thm 4.2; called "Theorem S" in scratch notes)

Let k ≥ 3 and let I_0, I_1, I_2 be gadgets (MAX-k-CUT instances on 3 primaries, k globals, private
auxiliaries) for the three types of Håstad's 3LIN(3). Write V_t(x, y) = max over auxiliary colourings,
for x ∈ Z_k^3 and y ∈ Y_k (restricted growth strings), and C_t = min{V_t(x, y0) : x ∈ {0,1,2}^3, x1+x2+x3 = t}.
If every V_t is symmetric in the two W-primaries (or each gadget is placed in both W-orientations), then
MAX-k-CUT is NP-hard to approximate within

    ρ = max_{y ∈ Y_k} max_{x ∈ Z_k^3} Σ_t V_t(x, y)  /  Σ_t C_t   (+ε).

*Proof sketch.* Normalise the global colouring to y ∈ Y_k and let p(t, x) be the law of (type, colours of
(U, W1, W2)) over the equations. For all maps a, b: Z_k → Z_3, decoding U-variables by a and W-variables by b
is an assignment of the 3LIN instance (U- and W-role variables are disjoint in Håstad's verifier), so
⟨p, g_ab⟩ ≤ 1/3 + ε with g_ab(t, x) = [a(x1) + b(x2) + b(x3) = t]; since Σ_c g_{a+c,b} = 1, also ≥ 1/3 − 2ε.
Combining shifts of a isolates the Fourier components ω^{±t}·A(x1)B(x2)B(x3) with A, B arbitrary
cube-root-valued vectors; these span ω^{±t}·(C^k ⊗ Sym² C^k), i.e. every W-symmetric function with zero mean over t.
Hence V_t(x) = (1/3)Σ_s V_s(x) + Σ μ_ab (g_ab − 1/3) for real μ, and J/m ≤ (1/3) max_x Σ_t V_t(x, y) + O(ε).
Completeness is (1/3)Σ_t C_t after balancing the types (three copies, U-variables shifted by c). For per-type
flat gadgets this is NRT's 1 − (k−1)/Σα; for the E/N family it gives 27/29. The bound is exactly the value of
the black-box local LP (attained by p = δ_{x*} ⊗ uniform(t)), and ρ ≥ α_k (Frieze–Jerrum / de Klerk et al.)
for every family, by gluing the canonical SDP solutions of the three gadgets.

*Families* (`results/sumS_k4.json`, `results/sumS_k5.json`; u = the global vertex of colour 3 (and 4) at y0):
* k = 4, type 0: NRT's 18-edge E-gadget with unit weights, plus a unit edge from each of its 6 auxiliaries to u
  (C = 24 = total weight); type 1: one auxiliary joined to x1, x2, x3 and u with weight 3 (C = 12);
  type 2: unit triangle on the primaries plus weight-3 edges from every primary to u (C = 11).
  max Σ_t V_t = 45, Σ C = 47: **ρ = 45/47 ≈ 0.957447**.
* k = 5: same pattern with two universal globals u, v (joined by weight 3 inside the type-0 gadget):
  C = (33, 15, 20), max Σ_t V_t = 66: **ρ = 33/34 ≈ 0.970588**.

`scripts/verify_sum_criterion.py` evaluates any family exactly (all x ∈ Z_k^3, all y ∈ Y_k, brute force over
auxiliaries, W-symmetry checked): 27/29, 45/47 and 33/34. The families were found by a joint LP over the three
gadgets with column and "motif" generation (scratch code); column generation found nothing better.

*Barrier for k = 3.* An LP-dual certificate (adversary μ* at y0: mass 5/29 on all-equal, 6/29 on all-distinct,
18/29 on two-equal tuples) shows E_μ* V_t(x, y0) ≥ (27/29)·C_t for every t-gadget of any size, so no 3LIN(3)
gadget family beats 27/29 under Theorem S (`results/barrier_k3/`, exhaustive check over all pairs of the 3^9
columns, exact integer arithmetic: `check_exact.py`). Going below 27/29 needs a different source or a non-black-box use of Håstad's test.

## MAX-k-CUT for k ≥ 4: universal vertices

*Lemma* (Kann–Khanna–Lagergren–Panconesi 1997 for L = 1; Guruswami–Sinop 2009, App. D, for L ≤ 2).
Add to a weighted graph G (total weight M, weighted degrees d_v) L new vertices u_1..u_L, each joined to
every v with weight d_v/3, and pairwise joined with weight w. Then
min(w, uncut_3(G)) ≤ uncut_{3+L}(H) ≤ uncut_3(G). (If two u's share a colour we lose w; otherwise a vertex
coloured like some u_i pays d_v/3 on that edge and can be recoloured with the best of the 3 remaining colours
at cost ≤ d_v/3.) The weight d_v/3 is the least uniform weight that works. Perfect completeness is preserved.

From a MAX-3-CUT gap with uncut fractions δ_c (YES) and δ_s (NO), w = δ_s M gives the ratio
(W/M − δ_s)/(W/M − δ_c) with W/M = 1 + 2L/3 + C(L,2)δ_s (`scripts/universal_vertex.py`, which also
brute-forces the lemma):

| k | stated as best known | from AOTW's 16/17 (perfect completeness) | from 27/29 (29/30 vs 9/10) |
|---|---|---|---|
| 4 | 85/86 ≈ 0.98837 | 82/85 ≈ 0.96471 | **47/49 ≈ 0.95918** |
| 5 | 120/121 ≈ 0.99174 | 119/122 ≈ 0.97541 | **35/36 ≈ 0.97222** |
| 6 | 101/102 ≈ 0.99020 | 53/54 ≈ 0.98148 | **48/49 ≈ 0.97959** |
| 7 | 136/137 ≈ 0.99270 | 202/205 ≈ 0.98537 | **125/127 ≈ 0.98425** |

The published constants (AOTW Thm 1.2: 1 − 1/(17(k + c_k) + c_k), c_k = k mod 3, via Guruswami–Sinop's
general-k reduction) use the base bound δ_3/3 for K = 3 colours; at k = 3 the same formula gives only 50/51.
Using δ_3 directly beats the formula for 4 ≤ k ≤ 20 (from k = 21 on the 1 − Θ(1/k) formula is better).
Even GS's own 32/33 fed through their App. D gives 54/55 ≈ 0.9818 for MAX-4-CUT.

## Framework (NRT, Appendix A)

An *i-gadget* is a weighted MAX-k-CUT instance on 3 primary, k global and auxiliary
vertices, replacing an equation x1+x2+x3 ≡ i (mod k). With y0 = (0,1,…,k−1) and Y the
restricted-growth strings of length k:

* C  = min over satisfying x of max_z I(x, y0, z)
* S1 = max over satisfying x, y ∈ Y, z;  S2 = same over unsatisfying x;  t = total weight

NRT Thm A.4: NP-hard to approximate within  `[Σ_i S2_i + max_i (S1_i − S2_i)] / Σ_i C_i`.

## What we do differently

1. **Exact LP for the weights** (TSSW-style, Trevisan–Sorkin–Sudan–Williamson 2000):
   fix the auxiliary vertices by their *columns* (their canonical witness value at each
   satisfying assignment); then finding optimal weights is an LP with exponentially many
   soundness constraints, solved by row generation (heuristic separation, then exact).
2. **Independent gadgets per predicate.** NRT obtain I_1..I_3 from I_0 by cyclically
   renaming the global vertices. But in the framework global vertex 0 is always colored 0
   (Y = restricted growth strings), so it is a true reference vertex; renaming moves that
   role around. For k=2 the shared construction gives only 17/18 while independent gadgets
   recover TSSW's optimal 16/17. Re-optimizing NRT's *own* witness columns with independent
   gadgets already gives 0.986404.
3. **S_3 symmetry.** The i-gadget problem is invariant under permuting the 3 primaries
   (this does not touch Y), so for symmetric column sets one may assume a symmetric gadget:
   one LP variable per edge orbit; separation over x up to permutation.
4. **Dual-priced column generation.** A new auxiliary vertex joined to existing vertices
   with weights d has as canonical column the best response
   f(x) ∈ argmax_c Σ_u d_u [c ≠ f_u(x)]; its first-order reduced cost is
   RC(d) = Σ_x π_x max_c Σ_u d_u[c ≠ f_u(x)] − Σ_r y_r max_c Σ_u d_u [c ≠ col_r(u)].
   We sample sparse d, hill-climb on RC, and add positive-RC orbits.
5. **Shared-global refinement of Thm A.4** (see below).

## Refined reduction theorem (shared global assignment)

For y ∈ Y let S1_i(y) = max_{x∈C_i, z} I_i(x,y,z), S2_i(y) = max_{x∈S_i, z} I_i(x,y,z),
d_i(y) = S1_i(y) − S2_i(y). Then MAX-k-CUT is NP-hard to approximate within

    b'/a,   b' Σt = max_{y∈Y} [ Σ_i S2_i(y) + max(0, max_r d_r(y)) ],   a Σt = Σ_i C_i .

*Proof.* Construction and completeness are exactly as in NRT. Soundness: take any
assignment (x, y, z) of the output instance J. Apply one permutation σ of Z_k to all
variables so that σ(y) ∈ Y (value unchanged); let x' = σ(x), y' = σ(y). All gadget copies
share y'. If α_i is the fraction of predicate-i equations satisfied by x' then
Σ_i α_i ≤ 1 + kε, and

    J ≤ (m/k) Σ_i [α_i S1_i(y') + (1−α_i) S2_i(y')]
      = (m/k) [Σ_i S2_i(y') + Σ_i α_i d_i(y')]
      ≤ (m/k) [Σ_i S2_i(y') + (1+kε) max(0, max_r d_r(y'))]  ≤ (b' + O(ε)) M,

with M = (m/k) Σ_i t_i. The max(0, ·) is necessary (Σ_i α_i may be close to 0). Since
S_i ≥ S_i(y) for every y, b' ≤ b whenever max_i (S1_i − S2_i) ≥ 0 — the condition under which
NRT's own normalization α ↦ α/Σα is valid — so in that regime this is never worse than
Thm A.4. Nothing requires S2_i(y) ≤ C_i: one gadget may be weak at some y as long as the
sum over the gadgets at that y is small, and this is exactly the slack the optimized
families exploit. ∎

(Balanced right-hand sides, needed because our gadgets differ: for k = 4 take 4 copies of
a hard instance, shifting all variables of copy a by a; since 3 is invertible mod 4 the
right-hand sides become uniformly distributed. This trick needs gcd(3, k) = 1; for other k
one relies on the balanced form of Håstad's theorem stated by NRT, which follows from the
structure of Håstad's test by shifting only the U-table variables.)

## Trust model of the certificates

Lower bounds (completeness) are explicit colorings stored in the `.cert.json` files and
re-evaluated in exact integer arithmetic. Upper bounds (soundness) are INFEASIBLE answers of
CP-SAT for "some auxiliary coloring has value > threshold" (strong model with pairwise
same-colour variables and transitivity); CP-SAT does not emit a checkable proof log, so these
rely on solver correctness. They were cross-checked on the binding instances by an
independent HiGHS MIP encoding written by a separate reviewer, which reproduced the ratio
exactly. Symmetry reduction (x up to permutation) is used only after checking six explicit
automorphisms of every gadget.

## Results (all ratios exact rationals; smaller is better)

| family | theorem | ratio | verification |
|---|---|---|---|
| NRT (AlphaEvolve), reproduced | A.4 | 195581/198140 ≈ 0.987085 | ours + 2 independent |
| NRT witness columns, LP weights, independent gadgets | A.4 | 2278661/2310069 ≈ 0.986404 | ours + 2 independent (brute force; CP-SAT/MIP) |
| column generation (cap 28 aux/gadget), joint refined LP — `results/fam_k4_v1.json` | refined | 4859448/4939823 ≈ 0.983729 | `verify_cert2.py`: CP-SAT proofs of every soundness bound, explicit witnesses for C |
| same columns, exact-polished joint LP for **NRT Thm A.4** — `results/fam_k4_v2o.json` | A.4 (original) | **13232277/13447232 ≈ 0.984015** | `verify_cert2.py` (CP-SAT strong model), cert in `fam_k4_v2o.cert.json`; **independently re-verified** (fresh code: CP-SAT + HiGHS MIP, all values agree) |
| same columns, exact-polished joint LP for the refined theorem — `results/fam_k4_v2r.json` | refined | **603609/613685 ≈ 0.983581** | `verify_cert2.py`, cert in `fam_k4_v2r.cert.json`; **independently re-verified** (fresh code: CP-SAT, HiGHS MIP on binding instances) |

## Layout

* `gadgets/core.py` — exact parameters (meet-in-the-middle brute force), CP-SAT variant
* `gadgets/search.py`, `symlp.py`, `jointlp.py` — LPs with row generation
* `gadgets/colgen.py`, `driver.py`, `scripts/run_search2.py` — column generation
* `scripts/reproduce_nrt.py` — reproduces NRT Lemmas A.5/A.6 exactly
* `scripts/verify_family.py family.json` — exact verification incl. refined ratio
* `scripts/verify_sym3.py`, `scripts/check_sym3.py` — the 27/29 MAX-3-CUT reduction (`results/sym3_k3.json`)
* `scripts/universal_vertex.py` — universal-vertex lemma (brute force) and the small-k table
* `scripts/verify_sum_criterion.py` — exact Theorem-S ratio of a 3LIN(3) → MAX-k-CUT family (`results/sumS_k*.json`)
* `results/barrier_k3/` — k = 3 barrier certificates and their exhaustive checkers (`check_exact.py`: exact)
* `paper/` — LaTeX draft of the paper (`main.tex`, `refs.bib`); `scripts/make_unified_family.py` builds
  the paper's family (2A_L, N_L, N_L) for MAX-(3+L)-CUT (`results/unified_L*.json`)
* `talk/` — slides of the talk about the paper (Beamer; `talk/slides.pdf`, build with `talk/build.sh`)
* `gadgets/symlp.py` — `SymDynLP(k, i, ...)` also accepts a set of residues `i` (predicate "sum ∈ i")
* `results/*.json` — gadget families (NRT edge-list format, 1-indexed)

## Status / caveats

* **Superseded as hardness bounds**: the k = 4 and k = 5 gadget families in this and the next section
  improve NRT's AlphaEvolve gadgets, but the bounds at the top (45/47 and 33/34 under Theorem S; 82/85 and
  119/122 from AOTW via universal vertices) are much stronger. They remain of interest as LP/certification methodology.
* Like NRT, the reduction produces weighted (multigraph) MAX-4-CUT instances; passing to simple
  graphs is the standard blow-up argument.
* The 0.984015 family uses only NRT's Theorem A.4. The 0.983581 family additionally uses the
  refined theorem above (reviewed independently; the proof is a short modification of NRT's).
* Our gadgets differ per predicate, so the balanced-RHS form of Håstad's theorem is needed;
  for k = 4 it follows from the shift argument above.
* Soundness bounds are proved by CP-SAT (and cross-checked with HiGHS by independent code);
  no external proof certificate (e.g. VeriPB) has been produced yet.
* Gadgets were capped at 28 auxiliary vertices to keep exact verification cheap; uncapped
  search reached smaller α per gadget, so further improvement is likely.

## k = 5 (MAX-5-CUT)

Best known NP-hardness for MAX-5-CUT: 120/121 ≈ 0.991736 (AOTW's general-k bound
1 − 1/(17(k + c_k) + c_k), c_k = k mod 3). In our framework this needs Σ_i α_i < 484.

**Result: `results/fam_k5_c32.json` certifies, via NRT's Theorem A.4 (original theorem),
NP-hardness of MAX-5-CUT within 441001222/444830061 ≈ 0.9913926 < 120/121.**
Certified α_i = 93.78, 93.62, 90.86, 92.95, 92.89 (30–32 auxiliary vertices per gadget).
Certificate: `results/fam_k5_c32.a4cert.json` (explicit completeness witnesses, per-gadget
bounds), proof cache `results/fam_k5_c32.a4cache.jsonl`.

How it was found:
* From-scratch column generation does not bootstrap (pairwise independence of the satisfying
  set: useful auxiliary vertices must see all three primaries and be colour-restricted via
  the globals).
* Structured seeds: restricted best responses f(x) = argmin_{c∈S} #{j : x_j = c}; "Pair/PP"
  columns on arithmetic progressions T = c + {−d, 0, d} around c = i/3 (they decode one primary
  from the other two on permutations of T — the certified k=4 gadgets contain the same
  structure); transplanted k=3 6-cycle gadgets on 3-colour sets; dual-priced level-2 columns;
  union with column-generation output. 53 aux/gadget: converged LP Σα = 447.0.
* Greedy orbit pruning with full re-convergence down to 30–32 aux (LP Σα = 461.5).

How it was certified (`scripts/verify_a4.py`):
* instances (x, y) up to checked S_3 automorphisms and permutations of colours unused by y
  (1202 orbits per gadget instead of 6500);
* upper bounds: a rigorous SDP bound (`gadgets/sdpbound.py`: Frieze–Jerrum relaxation with the
  fixed vertices as simplex anchors; the bound c0 + Σy + ⟨M,G⟩ + ΣΛ/(k−1) + N·λ_max(S)⁺ is valid
  for *any* multipliers, which come from an interior-point solve) closes ~78% of the orbits
  against the global class thresholds of Thm A.4; the rest are proved by CP-SAT (strong model
  with value-precedence symmetry breaking);
* the same pipeline reproduces the independently verified k=4 ratio 13232277/13447232 exactly.
* **Independently verified.** (1) A separate reviewer re-derived the SDP relaxation and the
  rigorous-bound formula, tested the bound against brute force on 1650 random and 120
  gadget-derived instances (never below the optimum; smallest margin among the 4760 SDP
  closures is 819 units vs ~1e-4 floating-point error), validated value precedence on 1750
  instances, re-implemented the certifier's bookkeeping, and re-proved the binding CP-SAT
  instances with its own model. (2) An independent verifier wrote an exact Russian-doll
  branch-and-bound solver in C (64-bit integer arithmetic, brute-force-tested) and solved all
  6500 (x, y) instances of every gadget with no symmetry assumptions: it reproduces
  C, S1, S2 of all five gadgets and the ratio 441001222/444830061 exactly.

## License

The code and data in this repository are released under the MIT License (`LICENSE`). The paper in `paper/`
(`main.tex`, `refs.bib`, `main.pdf`) is licensed under the Creative Commons Attribution 4.0 International
License, CC BY 4.0 (`paper/LICENSE`). The slides in `talk/` are derived from the paper and are under the same
CC BY 4.0 licence.
