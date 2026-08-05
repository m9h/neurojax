# `neurojax.thermo` — non-equilibrium brain dynamics (FDT violations) in JAX

Differentiable implementation of the framework in **Berjaga-Buisan et al. (2026),
*Cell Reports* 45:117782, "Thermodynamics of consciousness: Non-equilibrium brain dynamics track
conscious states"** — fit a multivariate Ornstein–Uhlenbeck model to *spontaneous* activity
("generative effective connectivity"), then measure departures from the fluctuation–dissipation
theorem as a stimulation-free correlate of conscious state.

## Result: the mouse arm replicates, with open EBRAINS data

Data: [10.25493/WKA8-Q4T](https://doi.org/10.25493/WKA8-Q4T) — 8 mice × 3 isoflurane levels,
32-ch ECoG, Spike2, CC BY-NC-SA. Delta band (0.5–4 Hz) per the paper, τ = 103 ms, 4 restarts.
**24/24 recordings analyzed.**

| anesthesia depth | n | FDT violation (mean ± sd) | entropy production |
|---|---|---|---|
| light | 8 | **108.53 ± 34.91** | 3949 |
| mid | 8 | **79.95 ± 14.80** | 1773 |
| deep | 8 | **60.07 ± 25.95** | 1563 |

- **Spearman(depth, FDT violation) = −0.538, p = 0.0067**
- **Spearman(depth, entropy production) = −0.472, p = 0.020** (independent measure, same direction)

The paper reports ρ = 0.885 for the same comparison. **The direction, ordering and significance
reproduce; the effect size is weaker** (|ρ| 0.538 vs 0.885), and only 4/8 mice are strictly
monotone individually.

### What is *not* claimed

- **This is not their exact statistic.** Their FDT violation values span ≈0.7242–0.7252 — a very
  narrow range — whereas ours span 1–156. We implement the relative mismatch between the true
  response `expm(Aτ)` and the equilibrium FDT prediction; their off-equilibrium extension
  (Cugliandolo–Kurchan) is normalized differently. So this reproduces *the phenomenon* with *a*
  non-equilibrium measure, not their number.
- Pipeline choices differ: τ, downsampling (→38.8 Hz), optimizer, and 4 restarts vs their 1000.
- **The PCI arm cannot be replicated from open data.** Every EBRAINS file is `stim-SPN`; the
  metadata defines this as "Spontanious stimulation" and the recordings' `Stim` event channel is
  present but empty. PCI needs evoked responses.

## Why a JAX implementation

- **True gradients.** GEC is fitted by differentiating *through* the Lyapunov solve and matrix
  exponential. The reference uses a heuristic pseudo-gradient
  (`B += α(FC_emp − FC_mod) + δ(FS_emp − FS_mod)`) that treats `dFC/dB` as the identity.
- **Guaranteed stability.** `A = −(LLᵀ + εI) + (M − Mᵀ)` makes `A + Aᵀ ≺ 0`, so every iterate is
  Hurwitz and the Lyapunov solve cannot blow up. The reference initializes stable and relies on the
  updates staying there. The parameterization also splits the physics: the antisymmetric block is
  exactly what breaks detailed balance.
- **Batching.** The 1000 random restarts the reference farms out to SLURM are one `vmap`.

## Correctness: an analytic oracle, not a reference implementation

The MATLAB reference needs R2022b+ (NSG ships 2020B), so no reference oracle was available. Linear
OU processes are exactly solvable instead, giving a stronger test — detailed balance *must* give
exactly zero:

| antisymmetric component | FDT violation | entropy production | time-reversal asym. |
|---|---|---|---|
| **0 (symmetric A)** | **0.000000** | **0.0000** | **0.000000** |
| 0.25 | 0.134 | 0.290 | 0.094 |
| 1.00 | 0.509 | 4.62 | 0.491 |
| 2.00 | 0.899 | 18.36 | 1.249 |

`tests/test_fdt.py` — 5/5 green.

## Red-green contract

The replication claim is an executable assertion, not a sentence in a README that can drift.
`tests/test_replication_contract.py` is **RED** until the pipeline has been run against the real
EBRAINS data and the published effect actually reproduces:

```bash
pytest src/neurojax/thermo/tests -q        # RED  -> "no results at ...; run the replication first"
python -m neurojax.thermo.replicate_ebrains --restarts 4 --steps 300
pytest src/neurojax/thermo/tests -q        # GREEN -> 11 passed, 1 xfailed
```

It asserts: all 24 recordings analyzed (nothing silently dropped), fits finite, the empirical
lagged covariance genuinely asymmetric, **FDT violations decrease with anesthesia depth
(rho < 0, p < 0.05)**, group means ordered light > mid > deep, and entropy production
independently agreeing.

The gap we have *not* closed is also a test — `test_effect_size_matches_paper` asserts
|rho| >= 0.8 and is marked `xfail(strict=True)`. It stays red until the effect size approaches the
published 0.885, and will fail loudly (XPASS) if it ever does, forcing the claim to be updated
rather than quietly overstated.

## Use

```bash
python -m neurojax.thermo.replicate_ebrains --restarts 4 --steps 300
```
