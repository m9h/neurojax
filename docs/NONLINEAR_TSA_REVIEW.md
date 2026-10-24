# Nonlinear time-series analysis for the WAND dynamics pipeline — multichannel forms, JAX feasibility, and the log-signature unification (research review, 2026-06-27)

A review of the nonlinear-TSA tools that cross-check or extend the three-leg
dynamics analysis (irreversibility / Langevin–Fokker–Planck / SINDy–DMD–DYSCO /
CEBRA / TINDA), with three questions in view:

1. the **best multichannel** formulation of each (our data are rank-~18
   multivariate MEG envelope embeddings, ~180 k samples/record);
2. which admit a **JAX / differentiable / GPU** version, and which stay CPU oracles;
3. how all of it **connects to Lyons log-signatures** — which turns out to be the
   unifying thread.

Confidence tags follow the source briefs: **[EST]** well-established;
**[BRIDGE]** rigorous synthesis across two literatures, not stated verbatim in one
paper; **[GAP]** appears genuinely unpublished (a citable contribution, not prior art).

---

## 0. The one-paragraph thesis

The irreversibility half of the analysis is, in rough-path language, the **Lévy
area**. The antisymmetric part of the depth-2 path signature,
`A^{ij} = ½(∫X^i dX^j − ∫X^j dX^i)`, is the signed enclosed area / net circulation
of the trajectory in the `(i,j)` plane — the same object the physics literature
calls the *irreversible circulation* / *areal velocity* / *area-production rate*
and uses as the marker of broken detailed balance. For a linear Langevin
`dz = Az dt + √(2D) dW` the **expected Lévy-area production rate** is

```
α* = ½(A Σ − Σ Aᵀ)            (Tomita–Tomita 1974; Risken 1989; Strang 2024)
```

and — verified algebraically and numerically (`scripts`/derivation in §3) —

```
α*  =  A_sol · Σ              with  A_sol = A + D Σ⁻¹   (our langevin_solenoidal_part)
EPR =  tr(A_sol Σ A_solᵀ D⁻¹) =  tr(α* Σ⁻¹ α*ᵀ D⁻¹)    (a weighted ‖α*‖²)
```

So **the band-resolved Langevin we already committed (`wand_band_langevin.py`) is
a band-resolved expected-Lévy-area spectrum**, and the model-free
lagged-covariance asymmetry (`wand_timefreq_irrev.py`) is the *non-parametric*
estimator of the *same* antisymmetric level-2 signature term. The log-signature
(via `signax`, already a dependency) is the JAX-native, differentiable,
intrinsically-multichannel framework that subsumes this irreversibility axis and
extends it beyond level 2. Everything else (RQA, visibility graphs, CCM, transfer
entropy, persistent homology) becomes a model-free cross-check arranged around it.

---

## 1. The conceptual frame: two orthogonal axes

A load-bearing correction surfaced by the recurrence brief: **time-irreversibility
≠ determinism** [EST]. A linear Gaussian process is time-*reversible* whether or
not it is "stochastic"; a *dissipative* deterministic chaotic system is
irreversible; a *conservative* one is reversible (Lacasa et al. 2012). So the
"stochastic cycle, not a limit cycle" conclusion needs **two** independent
measurements, not one:

| axis | question | tools |
|---|---|---|
| **Determinism / effective dimension** | is there a low-dim deterministic skeleton? | RQA determinism (DET), L_max, recurrence-network **transitivity dimension**, CCM/EDM **S-map θ test** |
| **Arrow of time / circulation** | is detailed balance broken (net current)? | Lévy area / log-sig odd part, Langevin EPR, model-free lagged-cov asymmetry, DHVG-KLD irreversibility |

Our existing result lives on the arrow-of-time axis (theta-dominant
irreversibility, A_sol). It says nothing direct about determinism — which is
exactly what the DMD/SINDy/DYSCO ≈ 0 *tried* to say but only as absence-of-evidence.
The determinism axis (RQA DET, S-map θ) is the positive test we are missing.

---

## 2. Per-tool verdict table

"Role": **C** = confound check on the existing result, **M** = mechanistic
addition / triangulation. "JAX" = port candidate vs keep-as-CPU-oracle.

| tool | best multichannel form | role | JAX verdict |
|---|---|---|---|
| **Log-signature / Lévy area** | native — built from cross-integrals; `signax` on the rank-18 path | C+M | **port (have it)** — differentiable, GPU |
| **Signature-kernel MMD(X, X̄)** | native multichannel; Goursat-PDE kernel | C+M (the null) | **port** — `sigkerax` (JAX) or `lax.scan` anti-diagonal sweep |
| **Multivariate phase-randomization surrogate** | shared phase across channels (preserves cross-spectrum) | C (the correct null) | **port** — `rfft → shared phase → irfft`, vmap-batched, differentiable in x |
| **Multivariate IAAFT surrogate** | joint iteration, per-channel marginals + cross-spectrum | C (stronger null) | partial — `fori_loop`; rank step non-differentiable |
| **CCM / EDM (S-map, simplex)** | multiview embedding (Ye–Sugihara 2016) | C+M | **port (strong)** — soft-kNN + ridge S-map; DeepEDM (ICML 2025) is the proof |
| **RQA (DET/LAM/L_max/ENTR)** | single joint RP on the 18-dim vector, fixed-RR + Theiler window | C+M | soft/windowed **port** (novel) + exact CPU oracle |
| **Recurrence-network transitivity dimension** | ε-recurrence network on the joint embedding | M | CPU oracle (pyunicorn) |
| **Directed HVG irreversibility (KLD)** | per-coordinate DHVG-KLD, aggregated over channels | M | CPU oracle (ts2vg/pyunicorn) — combinatorial |
| **Transfer entropy (bits)** | greedy multivariate conditional TE (IDTxl) | M | CPU/OpenCL oracle |
| **Directed information (differentiable)** | DV-bound critic (DINE/TREET) | M | **port** — not InfoNCE (log-N ceiling) |
| **Persistent homology (ring test)** | VR/DTM on a diffusion/spectral distance + Fasy bootstrap | M | CPU/GPU oracle (ripser/giotto-ph) |

---

## 3. The log-signature unification (the crux)

### 3.1 Lévy area = circulation = the irreversibility object [EST]

The depth-2 signature splits into a symmetric part (`S^{ii}=½(ΔX^i)²`, products of
increments) and an antisymmetric part = the Lévy area
`A^{ij}=½(S^{ij}−S^{ji})=½∮(x dy − y dx)` — signed enclosed area, `+` =
anticlockwise (Chevyrev–Kormilitzin 2016; Lévy 1940; Friz–Victoir 2010). The same
`½∮(x dy − y dx)` is the physics "areal velocity / area-production rate" used as a
detailed-balance-violation detector (González–Neu–Teitsworth 2019; du Buisson et
al. 2023; Gnesotto et al. 2018; Battle et al. 2016). The discrete antisymmetric
level-2 signature `Σ_t (x^i_t Δx^j_t − x^j_t Δx^i_t)` estimates, in expectation,
the antisymmetric one-lag cross-covariance `C_ij(τ)−C_ji(τ)` — i.e. the
imaginary-cross-spectrum / lead-lag part of our lagged covariance [BRIDGE]. So our
**model-free asymmetry is the empirical Lévy area**.

### 3.2 The exact identity for the linear Langevin [EST/BRIDGE — verified here]

With `dz = Az dt + √(2D) dW`, `A` stable, stationary covariance from the Lyapunov
equation `AΣ + ΣAᵀ + 2D = 0`, the expected Lévy-area production rate is the
Tomita–Tomita irreversible-circulation matrix `α* = ½(AΣ − ΣAᵀ)` (Strang 2024,
*Axioms* 13(12):820; Tomita–Tomita 1974; Risken 1989; Godrèche–Luck 2019). Using
`D = −½(AΣ + ΣAᵀ)`:

```
A_sol = A + D Σ⁻¹ = ½(A − Σ Aᵀ Σ⁻¹)   ⇒   A_sol · Σ = ½(AΣ − ΣAᵀ) = α*
EPR = tr(A_sol Σ A_solᵀ D⁻¹) = tr(α* Σ⁻¹ α*ᵀ D⁻¹)   (weighted Frobenius ‖α*‖²)
```

Both equalities verified numerically (random stable A, SPD D, r=6: `α*` is
antisymmetric, `α* == A_sol Σ`, and the two EPR expressions agree to machine
precision). **Consequences:**

- `wand_band_langevin.py` already produces the **band-resolved expected
  Lévy-area spectrum** (α* = A_sol·Σ); `f_sol = |Im λ(A_sol)|/2π` is the rotation
  *rate* of that circulation.
- The model-based EPR and the model-free `‖L−Lᵀ‖` are the **parametric and
  non-parametric estimators of one rough-path object** — which is why they agreed
  on the low-frequency-dominant gradient. This is a cross-validation, now with a
  name.
- ⚠ **Sign convention:** Strang writes drift as `−A`, giving `½(AΣ−ΣAᵀ)` with his
  sign flipped relative to our `dz=Az dt`. Match carefully when citing.

### 3.3 Parity under time reversal — with a correction [EST]

Path reversal is the tensor-algebra inverse / Hopf antipode: `S(X̄)=S(X)⁻¹=α(S(X))`,
`α = (−1)ⁿ ∘ ρ` (ρ = word reversal) (Chen 1957; Reutenauer 1993;
Chevyrev–Kormilitzin 2016). **Correction to the naïve guess:** the *log*-signature
negates **uniformly** — `log S(X̄) = −log S(X)`, factor `−1` at *every* level, **not
`(−1)ⁿ`** (a degree-n free-Lie element has ρ-parity `(−1)^{n−1}`, and
`(−1)ⁿ(−1)^{n−1}=−1`). So **every Lie coordinate of the log-signature is
time-odd**, and the leading nontrivial one (the level-1 increment aside) is the
Lévy area. A **band-resolved log-signature** is therefore a differentiable,
multichannel irreversibility feature *beyond* level 2.

⚠ **Do not overclaim** "nonzero Lévy area ⇔ irreversible": Gottwald–Melbourne
(2024, *Nonlinearity* 37:075018) show time-reversibility does not unconditionally
force the Lévy area to vanish; the equivalence is generic for the diffusion/current
setting but not an unconditional theorem.

### 3.4 Expected signature ↔ generator [EST]

The expected signature satisfies a generator-driven PDE: for `dX=μ dt+σ dW`, a
Feynman–Kac parabolic PDE in the generator `A=Σμⁱ∂ᵢ+½Σbʲᵏ∂²`, `b=σσᵀ` (Lyons–Ni
2015, *Ann. Probab.* 43:2729; Lyons–Ni–Tao 2024, arXiv:2401.02393; Ni 2012). The
degree-2 antisymmetric **signature cumulant** is the expected Lévy area
(Friz–Hager–Tapia 2022/2024; Bonnier–Oberhauser 2020). The driftless baseline
(Fawcett's formula) has symmetric level-2 ⇒ zero expected Lévy area; drift /
non-reversibility is what makes it nonzero — consistent with §3.2.

### 3.5 The signature-kernel MMD irreversibility test [GAP — citable contribution]

The signature kernel (Király–Oberhauser 2019; Salvi et al. 2021, Goursat PDE
`∂²k/∂s∂t = ⟨ẋ_s,ẏ_t⟩k`) gives a characteristic/universal kernel on path laws, so
the **signature-MMD** is zero **iff** two path laws coincide (Chevyrev–Oberhauser
2022; Gretton et al. 2012). Therefore:

> **sig-MMD( law(X), law(X̄) )**, calibrated against a phase-randomized / time-reversal
> surrogate null, is a complete, multichannel, differentiable irreversibility
> statistic — zero iff the process is statistically time-reversible, and strictly
> richer than the pairwise lagged-covariance (INSIDEOUT) or the Langevin α*
> (it captures all cross-channel signed-area / lead-lag asymmetries at all orders).

This assembly is **unpublished** (a thorough search found sig-MMD used for
two-sample and conditional-independence tests — Manten et al. 2025 — but never for
forward-vs-reversed irreversibility). It is well-motivated by existing machinery
and is the natural capstone test. It also coincides with where the **surrogate
brief** independently arrived: dissipation `= k_BT·D_KL(forward‖reversed)`
(Kawai–Parrondo–Van den Broeck 2007; Roldán–Parrondo 2010), and the Diks-1995
forward-vs-reversed delay-vector test is an unnamed MMD between a path and its
time-reversal. The signature kernel is the modern, multichannel realization.

⚠ Terminology: papers titled "irreversibility as a *signature* of consciousness"
(de la Fuente et al. 2023) use "signature" colloquially — **not** the Lyons object.

---

## 4. The correct null (surrogates) — the #1 confound fix

Our current time-shuffle null tests against i.i.d. noise: it rejects for *any*
spectral structure, not irreversibility specifically — far too weak. The fix:

- **Multivariate phase randomization (MVPR)** [EST] — add the **same** random phase
  `φ(f)` to every channel at each frequency (Prichard–Theiler 1994). This preserves
  the full cross-spectrum (all auto/cross-correlations and lag structure, which
  depend only on phase *differences*) and yields a **linear-Gaussian,
  time-reversible** null (Weiss 1975). The principled "is the irreversibility beyond
  a linear Gaussian process with the same cross-spectrum?" null.
- **Multivariate IAAFT** [EST] (Schreiber–Schmitz 2000) — additionally matches each
  channel's (non-Gaussian) marginal, ruling out the "linear but skewed-innovation"
  alternative (which is generically irreversible). The needed second control.
- ⚠ Reversibility holds only **in distribution** — rank the statistic against the
  surrogate **ensemble**; never assume a single surrogate ≈ 0. **Twin surrogates are
  the wrong null** here (they reproduce the irreversible dynamics).
- **Field note** [EST]: the landmark Lynn et al. 2021 (*PNAS*, broken detailed
  balance in the brain) used shuffle + bootstrap, **not** a Gaussian-linear null —
  so MVPR is an improvement over both our pipeline and that reference.

**JAX** [EST]: `X=rfft(x)`; draw `φ`; **zero DC and (even n) Nyquist** so they stay
real; `irfft(|X|·e^{iφ}, n=n)`; broadcast one `φ` across the channel axis;
`vmap` over **split** keys (vmapping one key silently reuses it). Differentiable in
`x` (only the φ draw is non-differentiable, `stop_gradient`-able). Enable x64. The
IAAFT rank step (`argsort`) has zero gradient a.e. — phase randomization does not.
No JAX/GPU surrogate library exists — a batched MVPR is genuinely novel.

---

## 5. The model-free cross-checks (and what each catches)

### 5.1 Recurrence / RQA — the determinism axis
- **Construction**: a single joint recurrence plot on the 18-dim envelope vector
  (not a Joint/cross RP — those are coupling tools for distinct subsystems). L∞ or
  L2 metric, **fixed recurrence rate** (not fixed ε) for cross-band/-subject
  comparability, and a **Theiler window** ≳ the autocorrelation time (critical:
  envelopes are heavily autocorrelated → DET inflates without it). [EST: Marwan et
  al. 2007; Marwan 2011; Schinkel et al. 2008]
- **What it adds**: DET / L_max are positive determinism measures — the test the
  DMD/SINDy/DYSCO≈0 only gestured at. Recurrence-network **transitivity dimension**
  (Donner et al. 2011) is the *nonlinear* effective dimension vs the *linear* svht
  rank — a noisy limit cycle near a 1–2D manifold inside the 18-D PCA box would show
  a low, near-integer transitivity dimension.
- **Memory confound** [EST]: a dense N×N RP at N=180 k is ~32 GB bool / ~130 GB
  float32 — infeasible/pointless. Use **windowed** RQA (a few k² per window — also
  what we want for non-stationarity), **fixed-RR sparse/kNN**, or tiled
  accumulation (PyRQA does 1 M pts in ~69 s).
- **JAX**: a *soft, windowed, differentiable* RP (sigmoid threshold) + a **soft-DET**
  surrogate (diagonal-correlation sums) is a buildable **novelty** — no published
  differentiable-RQA library exists; the nearest prior art is Soft-DTW
  (Cuturi–Blondel 2017). Keep exact line statistics on the CPU/OpenCL oracle
  (PyRQA/pyunicorn) and validate the soft version against it.

### 5.2 Visibility-graph irreversibility — a parameter-free arrow-of-time check
- **Construction**: directed horizontal visibility graph, irreversibility = KLD
  between in/out (retarded/advanced) degree distributions, including the
  degree–degree distribution to catch zero-net-current cases (Lacasa et al. 2012).
- **Multichannel**: **per-coordinate DHVG-KLD aggregated over the 18 channels** —
  every piece individually validated. ⚠ Multiplex-VG irreversibility is **not** a
  validated standard measure (the 2015 multiplex VG carries no arrow-of-time
  statistic) — do not rely on it.
- **JAX**: keep as CPU oracle (combinatorial, O(N log N), already cheap; ordinal
  statistics resist meaningful differentiation). The Lévy area is the JAX-native
  counterpart that measures the same rotational asymmetry differentiably.
- ⚠ Benchmark first: Zanin–Papo (2021) compares VG-KLD against the lagged-cov /
  INSIDEOUT estimators — read before committing to VG as the only check.

### 5.3 CCM / EDM — the strongest determinism test, and a Jacobian bridge
- **S-map θ test** [EST]: θ=0 ≡ a single global linear map (≡ DMD / linear SINDy);
  if forecast skill *improves* for θ>0, the dynamics are nonlinear/state-dependent —
  a **positive** detector of the deterministic skeleton the linear methods miss.
  Pair with the simplex horizon-decay test (chaos vs additive noise).
- **S-map local Jacobians = the local drift matrix** [EST: Deyle et al. 2016] —
  evaluated sequentially along the attractor; eigenvalues give time-varying
  stability (Ushio et al. 2018). This is a direct, nonlinear generalization of the
  Langevin A / DMD operator — a coupling route too.
- **Multichannel**: multiview embedding (Ye–Sugihara 2016) — designed to turn
  dimensionality into an asset on short noisy series.
- **JAX** (strong port candidate): simplex = soft-kNN (temperature-softmax over
  neighbor distances; Frosst et al. 2019) — already smooth; S-map = locally-weighted
  **ridge** regression via the normal equations `solve(AᵀWA+λI, AᵀWy)` (differentiable
  in θ; dodges `lstsq`'s missing JVP and SVD-gradient NaNs — aligns with the
  "Tikhonov, not truncated SVD" rule). DeepEDM (ICML 2025) is the existence proof
  (PyTorch). Keep CCM convergence sweeps + surrogate significance on a pyEDM oracle.

### 5.4 Transfer entropy / directed information — directed coupling (Leg B feed)
- **Bits** [EST]: greedy multivariate conditional TE with hierarchical testing
  (IDTxl; Novelli et al. 2019) — validated to ~100 nodes / 10⁴ samples, "fits MEG."
  ⚠ Data-hungry; envelope autocorrelation cuts *effective* samples — use
  block/cyclic-shift surrogates; feasible only on long records.
- **JAX**: classical KSG/Frenzel–Pompe are kNN-count estimators — CPU/OpenCL oracle.
  For a differentiable directed-information *in bits*, port the **DV-bound DINE/TREET**
  (Tsur et al. 2023; Luxembourg et al. 2024) — **not** InfoNCE: InfoNCE is
  upper-bounded by `log(batch size)` (Poole et al. 2019), so our CEBRA/InfoNCE
  machinery can only give an **ordinal** coupling score (fixed batch N), never bits.

### 5.5 Persistent homology — the rigorous ring test (CEBRA H¹)
- **Construction** [EST]: VR persistent homology, a single high-persistence H¹ bar =
  a genuine 1-cycle; certify with a **Fasy et al. 2014 bootstrap** or persistence
  landscapes (Bubenik 2015) — do not eyeball the longest bar.
- ⚠ **18-D curse of dimensionality** [EST]: Euclidean distances concentrate and the
  true hole can vanish from the diagram. Pre-reduce dimension or use a
  **diffusion/spectral distance**, and a **DTM filtration** (Anai et al. SoCG 2019)
  for outlier robustness.
- **JAX**: **no** JAX-native PH exists; differentiable PH is PyTorch/TF-only and only
  differentiates through the filtration (the combinatorial pairing is
  piecewise-constant). Keep as CPU/GPU oracle (ripser / giotto-ph — multicore often
  beats GPU on a 20-core box; Ripser++ for CUDA). The signature view offers a
  complementary, differentiable read: **H¹ = "is there a hole," Lévy area = "net
  circulation around it," f_sol = "at what rate"** — our weak-TINDA / ring-but-≈0-
  rotation result is precisely *nonzero H¹ with small Lévy area*.

---

## 6. JAX library status (signatures)

| library | backend | JAX-native | diff | GPU | computes |
|---|---|---|---|---|---|
| **signax** (have it, v0.2.1) | JAX | ✓ | ✓ | ✓ | signature + **log-signature** |
| **sigkerax** | JAX | ✓ | ✓ | ✓ | sig **kernel** (Goursat PDE); linear/RBF static only, immature |
| keras_sig (2025) | Keras3/JAX | ✓ | ✓ | ✓ | signatures (GPU-parallel) |
| sigkernel | PyTorch | ✗ | ✓ | ✓ | sig kernel (adjoint PDE) |
| KSig (2025) | CuPy | ✗ | ✗ | ✓ | sig kernel, RFSF, low-rank |
| signatory | PyTorch | ✗ | ✓ | ✓ | sig + log-sig (**discontinued**) |

`signax` already exports what we need (`signature`, `logsignature`,
`signature_to_logsignature`, `signature_combine`). A JAX sig-kernel is either
`sigkerax` or a short `lax.scan` anti-diagonal Goursat sweep. The surrounding
Kidger stack (diffrax/equinox/optax/lineax) is already ours; diffrax's Neural CDEs
are rough-path-adjacent but do not compute signatures — compose `signax`.

---

## 7. Prioritized roadmap

1. **Multivariate phase-randomization null** (replace the shuffle) — the minimal
   correct fix; directly hardens the band-irreversibility + EPR we just committed,
   and beats the Lynn-2021 reference null. *(confound, JAX, easy)*
2. **Band-resolved Lévy-area / log-signature spectrum** via `signax` — the
   differentiable, multichannel re-derivation of the irreversibility gradient;
   cross-checks α* = A_sol·Σ from the Langevin against the model-free path estimate.
   *(mechanism, JAX, have-the-lib)*
3. **RQA DET + transitivity dimension** (windowed, fixed-RR, Theiler) and the
   **S-map θ test** — the determinism axis we are missing; turns "DMD≈0" into a
   positive statement. *(confound+mechanism; RQA oracle + CCM JAX port)*
4. **sig-kernel MMD(X, X̄)** with the MVPR null — the complete multichannel
   irreversibility test (a [GAP] contribution). *(mechanism, JAX)*
5. Second wave: **persistent homology** of the CEBRA embedding (fix the 18-D
   distance first), **IDTxl** directed networks (feeds connectome harmonics, Leg B).

Items 1–2 (and the soft-RQA/CCM ports) are JAX-native and compose with the
existing `jax.grad` graph; items 3–5 lean on CPU/GPU oracles in `.venv-oracle`
alongside MNE/osl-dynamics/pyunicorn, validated against the JAX versions.

---

## 8. Caveats / things to verify before publishing

- Sign convention in α* (Strang writes drift `−A`); confirm against `dz=Az dt`.
- log-signature parity is **uniform `−1`**, not `(−1)ⁿ` — fix in any docstring/paper.
- "nonzero Lévy area ⇔ irreversible" is **not** unconditional (Gottwald–Melbourne 2024).
- sig-MMD-vs-reversal and the CEBRA-InfoNCE→TE surrogate are **[GAP]** (our
  constructions), not prior art.
- multiplex-VG irreversibility and a JAX-native PH do **not** exist as turnkey tools.
- Several end-page numbers were confirmed from preprints/secondary sources (APS HTML
  403s); verify against publisher PDFs for the manuscript.

## Key references (by section)

- **Signatures/Lévy area**: Lyons 1998; Friz–Victoir 2010; Chevyrev–Kormilitzin
  2016 (arXiv:1603.03788); Lyons–Ni 2015 (*Ann. Probab.* 43:2729); Chevyrev–Lyons
  2016; Chevyrev–Oberhauser 2022 (*JMLR* 23:176); Friz–Hager–Tapia 2022/2024.
- **OU circulation / EPR**: Tomita–Tomita 1974 (*PTP* 51:1731); Risken 1989;
  Godrèche–Luck 2019; **Strang 2024** (*Axioms* 13(12):820, arXiv:2411.07613);
  arXiv:2207.05197 (OU EPR ↔ consciousness).
- **Sig kernels / MMD**: Király–Oberhauser 2019 (*JMLR* 20:31); Salvi et al. 2021
  (*SIMODS* 3:873); Gretton et al. 2012 (*JMLR* 13:723); Manten et al. 2025 (ICLR).
- **Surrogates**: Theiler et al. 1992; Prichard–Theiler 1994 (*PRL* 73:951);
  Schreiber–Schmitz 1996/2000; Weiss 1975; Diks et al. 1995;
  Kawai–Parrondo–Van den Broeck 2007.
- **Recurrence/VG**: Marwan et al. 2007 (*Phys. Rep.* 438:237); Donner et al. 2011
  (*EPJB* 84:653); Lacasa et al. 2012 (*EPJB* 85:217); Donges et al. 2015 (pyunicorn,
  *Chaos* 25:113101); Rawald et al. 2017 (PyRQA); Zanin–Papo 2021.
- **CCM/EDM**: Sugihara–May 1990; Sugihara et al. 2012 (*Science* 338:496);
  Ye–Sugihara 2016; Deyle et al. 2016 (*Proc. R. Soc. B* 283:20152258); DeepEDM
  (ICML 2025, arXiv:2506.06454).
- **TE / directed info**: Schreiber 2000; Barnett et al. 2009; Novelli et al. 2019
  (*Net. Neurosci.* 3:827); IDTxl (Wollstadt et al. 2019); MINE 2018; Poole et al.
  2019; DINE (Tsur et al. 2023); TREET (Luxembourg et al. 2024).
- **PH**: Carlsson 2009; Fasy et al. 2014 (*Ann. Stat.* 42:2301); Bubenik 2015;
  Anai et al. SoCG 2019 (DTM); Carrière et al. 2021 (ICML); Bauer 2021 (Ripser).
- **Brain irreversibility**: Lynn et al. 2021 (*PNAS* 118:e2109889118); Lynn et al.
  2022 (*PRL* 129:118101); Deco et al. 2022 (INSIDEOUT, *Commun. Biol.* 5:572).
