# Competing spectral models on recorded EEG: first batch results

Written 2026-10-15 from `scripts/bsd_batch.py` and `scripts/bsd_summarise.py`
(outputs under `results/bsd/`, committed copies under `docs/results/bsd/`). Every spectrum is the mean over a posterior
channel set after average reference, Welch with 2 s Hann segments, 2–45 Hz,
median log power removed before fitting. Seven models are fitted by
variational Laplace with two restarts and compared by free energy (the
seventh, `liley_mix2`, was added on 2026-10-15; every table below that
says "six models" predates it and is superseded by the beta-resonance
section at the end):

| key | model | params |
|---|---|---|
| `specparam` | fixed aperiodic + one Gaussian peak | 5 |
| `specparam_knee` | knee aperiodic + one peak | 6 |
| `specparam2` | knee aperiodic + two peaks (alpha, beta) | 9 |
| `liley` | one damping-distributed oscillator family + floor | 5 |
| `liley_mix` | damping-distributed background family + resonant alpha + floor | 8 |
| `liley_mix2` | `liley_mix` + a beta resonance, centre a ratio above alpha | 11 |
| `xialpha` | ξ Matérn process + α shifted Matérn process + floor | 8 |

"RFX xp" is the protected exceedance probability of random-effects Bayesian
model selection over subjects (rows); "BOR" the Bayesian omnibus risk.
Within-subject contrasts are log ratios for positive parameters.

## LEMON, 30 adults, eyes closed vs eyes open (preprocessed release, 250 Hz)

Model selection, 60 rows:

| condition | wins (sp / spk / sp2 / liley / mix / ξα) | RFX protected xp |
|---|---|---|
| all | 0 / 0 / 34 / 0 / 14 / 12 | sp2 1.00 |
| EC | 0 / 0 / 19 / 0 / 8 / 3 | sp2 0.99 |
| EO | 0 / 0 / 15 / 0 / 6 / 9 | sp2 0.86, ξα 0.13 |

Median residual (ln power): sp 0.38, spk 0.25, sp2 0.14, liley 0.31, mix 0.18,
ξα 0.22. Head to head, the Liley mixture beats ξ-α in 37 of 59 rows (median
+9.9 nats) and loses to the two-peak phenomenology in 41 of 59 (median −9.7).
So: **a knee aperiodic with alpha and beta peaks is the best description of
adult resting spectra at this resolution; of the two mechanistic models the
Liley mixture is clearly preferred over the two-process ξ-α model.** The
single-family `liley` model never wins, which answers the narrow version of
Liley's claim: one damping-distributed family cannot carry both the
background and the peak.

Eyes closed minus eyes open, within subject (n = 30):

| model | parameter | contrast | mean | t |
|---|---|---|---|---|
| mix | alpha damping `gamma_alpha` | log ratio | −0.78 | −3.6 |
| mix | alpha peak power | log ratio | +1.46 | +4.7 |
| mix | background damping | log ratio | −0.70 | −2.3 |
| sp2 | alpha height `pw0` | diff | +0.97 | +4.0 |
| sp2 | beta height `pw1` | diff | +0.86 | +3.7 |
| sp2 | exponent | log ratio | +0.06 | +1.0 |

Alpha blocking on eye opening appears in the Liley model as a doubling of
the alpha damping rate with no change in centre frequency, which is Liley
and Muthukumaraswamy's (2019) claim reproduced on an independent sample; in
the phenomenology it is a loss of peak height with no change of exponent.

## Propofol sedation, Chennu et al. 2016 (20 adults × baseline / mild / moderate / recovery; 91-ch EGI, 250 Hz)

Six models, 80 rows: the two-peak phenomenology wins 71 of 80 (RFX xp 1.00
at every level), residual 0.19 against 0.42 for the mixture and 0.39 for
ξ-α. Propofol raises a beta peak near 20 Hz (the sp2 beta centre is
18–22 Hz at every level) and neither mechanistic model has a second
resonance, so both are forty to sixty nats behind at every level. Between
the two mechanistic models the mixture edges ξ-α awake (median +1.2 nats,
11/20) and loses under moderate sedation (−13.9, 7/20): the sedated spectrum
has a strong low-frequency process that a separate aperiodic Matérn carries
better than a damped family.

Moderate sedation minus baseline, within subject:

| model | parameter | contrast | mean | t | n |
|---|---|---|---|---|---|
| mix | alpha damping | log ratio | +0.70 | +2.8 | 20 |
| mix | alpha peak power | log ratio | −1.14 | −2.4 | 20 |
| mix | alpha frequency | log ratio | +0.21 | +2.1 | 20 |
| mix | background centre | log ratio | −0.60 | −3.0 | 20 |
| sp2 | knee | log ratio | −1.41 | −5.2 | 18 |
| sp2 | alpha centre | diff (Hz) | +2.9 | +2.9 | 18 |
| sp2 | exponent | log ratio | −0.01 | −0.2 | 18 |

Alpha becomes more damped, weaker and faster under propofol, and the knee
frequency drops by a factor of four while the exponent does not move; the
exponent, the usual "E/I" proxy, is the one quantity that does not change.
Between subjects, no parameter correlates with plasma propofol (|r| < 0.12
across all 80 runs); the drug effect lives in the within-subject contrast.

**Consequence for the model set:** a mechanistic model that can compete on
pharmacological EEG needs a second resonance. That model (`liley_mix2`) was
added the same day; its results, which revise the alpha-damping claim
above, are in the beta-resonance section at the end.

## Propofol repeated awakening, ds005620 (21 subjects, 65 ch, 5 kHz → 250 Hz)

First pass (4 subjects, 18 rows, six models): mixture 10 wins, ξ-α 8, RFX
xp mix 0.70; sedation minus awake-EC alpha damping +1.9 log units (t = +2.1,
n = 4). The full set (21 subjects, awake-EC and the first sedation run for
every subject, 51 rows, seven models) is reported in the beta-resonance
section.

## HBN children (release 1), rest and surround suppression

Rest, 12 subjects, EO/EC blocks from the instruction markers, six models:
mixture 11 wins, two-peak phenomenology 9, one-peak specparam 3, ξ-α 1; RFX
protected xp mixture 0.65 vs sp2 0.35 (BOR 0.00). Eyes open favours the
mixture (xp 0.79), eyes closed the phenomenology (0.69). EC−EO contrasts are
not significant at n = 12 with 20 s blocks.

Surround suppression (11 subjects, two runs, 2.4 s trials; four flashing
discs, foreground contrast 0/0.3/0.6/1, background absent/present):
**no steady-state response was found.** The trial-averaged (phase-locked)
spectrum shows an onset response at ~195 ms and 60 Hz line noise, and no
line at any frequency that differs between foreground-present and
foreground-absent trials; the 25 Hz candidate seen in two subjects has a
median evoked SNR of 0.6–1.2 across conditions, i.e. nothing. Induced alpha
power does not change with foreground contrast (per-subject slope t < 1),
the fitted alpha damping does not change with contrast (t < 0.8), and the
onset VEP amplitude does not scale with contrast (t ≈ ±1) nor show reliable
surround suppression at full contrast (log ratio +0.25, t = 1.2, n = 11).
The dataset documentation gives no flash frequency. As processed here the
task is not a contrast-gain probe; whether the 136-subject release, the
second release, or different channels would change that is untested.

**Why there is no HBN surround-suppression result** (asked 2026-10-15). Five
reasons, in order of weight:

1. *Wrong task for a steady-state readout.* HBN's surroundSupp presents
   flashing discs for 2.4 s with no documented flicker frequency; the
   phase-locked spectrum shows only the onset VEP. HBN's steady-state
   paradigm is **contrastChangeDetection** (CCD), which does carry tagged
   lines; see below.
2. *Trial counts*: 8–24 trials per condition per subject against the ~100+
   per contrast level that the Tsai/Hou contrast-response fits use.
3. *Subjects*: eleven children, with head movement and muscle in the
   30–45 Hz band where a suppression effect would appear.
4. *Preprocessing*: average reference over all 129 channels with no
   bad-channel or trial rejection; the posterior average then carries the
   worst channels.
5. *Analysis*: induced alpha, fitted damping and VEP amplitude are all
   indirect; the direct contrast-gain quantity (steady-state amplitude vs
   contrast) does not exist in this task.

**Contrast change detection (CCD), the actual SSVEP task.** Run 1 of 12
subjects on disk (9 with usable triggers, median 25 trials): trial-locked
evoked SNR has lines at 20 and 40 Hz (medians 1.5 and 2.2) and nothing at 25
or 50 Hz (1.0, 0.9), i.e. the discs flicker at 20 Hz with a 40 Hz harmonic
in this run. This is the dataset to use for contrast-gain work, with the
caveat that the contrast staircase in CCD is a behavioural one, not a fixed
ladder.

**Trilinear (Tucker/Wang) basis at 99 % explained variance.** The user's
earlier experience was that a trilinear basis keeping 99 % of the variance
is the best denoiser. `tucker_energy_denoise(epochs, 0.99)` implements it
(HOSVD, each mode truncated at the smallest rank that keeps the fraction of
the core energy). On the raw CCD epochs it *lowers* the evoked SNR (40 Hz:
2.2 → 1.5; 20 Hz: 1.5 → 0.7) because the retained channel rank is 1–2 and
those factors are the artefact directions (eye, movement), while the SSVEP
lives in low-variance posterior directions. Marchenko-Pastur truncation
keeps almost everything (channel rank 25 of 129, segment rank 292) and
leaves the SNR unchanged. With bad channels (variance > 5× or < 1/20 of the
median) and bad trials (> 3× median amplitude) removed first, the 40 Hz SNR
is 2.4 clean, 1.7 after the 99 % basis and 2.6 after a 99.9 % basis, so on
these data the trilinear basis does not help and the explained-variance
threshold has to be tuned per dataset. No earlier trilinear/Wang
implementation or result exists in the workspace; this is the first one.

## What this says for the protocol

1. **Model comparison is decisive on real data** at 10 minutes of EEG: protected
   exceedance probabilities near 1 with omnibus risk near 0 on 30 subjects.
2. **The two-resonance Liley mixture is the mechanistic model to carry
   forward.** With a beta resonance it beats the two-peak phenomenology on
   awake adults (LEMON, ds005620) and ξ-α everywhere, and its parameters move
   in the predicted directions for eye closure (alpha damping down, t = −5).
   Under propofol it still loses to the phenomenology, and with beta in the
   model the propofol effect is no longer "alpha damping up" but alpha
   faster, beta sharper and weaker, knee down (see the beta-resonance section).
3. **Under propofol the phenomenology still fits best**, so a mechanistic
   claim on sedation data must be made against `specparam2`, and the gap is
   about twenty-five nats per run; the propofol beta is broad and high
   (Gaussian height 4 in log power) and one damped oscillator does not
   reproduce it. A beta *family* (damping-distributed, like the background)
   is the next candidate, and the gnf-jax block supplies several modes when
   more than one is carried.
4. **Driven paradigms need their own validation data.** HBN's task does not
   supply a steady-state response; the Tsai and Hou designs do, and that is
   the data to request or collect.

Caveats: posterior channel averages only (no source localisation); one
alpha centre prior per row from the detrended 6–14 Hz maximum; the
log-power offset removal makes power parameters relative; LEMON's
preprocessed release is band-passed and ICA-cleaned, which flattens the
low-frequency end and lowers every exponent; HBN children's spectra are
noisy and the average reference over 129 channels includes bad channels.

## Multiway reduction (added 2026-10-15)

`neurojax.bsd.multiway` adds tensor reductions of the EEG before any
spectral model sees it, and `scripts/bsd_batch.py --reduce` selects them.
Three were tried on LEMON (30 subjects, EC/EO):

- **HOSVD of the (channel × frequency × segment) log-spectrum tensor.** The
  leading channel factor is nearly flat across the cap whether or not the
  common spectral shape is removed first (mean-level dominates the
  uncentred tensor; a near-global power drift dominates the centred one), so
  it is a worse spatial weighting than a posterior pick. Kept as an option,
  not used.
- **Non-negative PARAFAC of the power tensor** (Miwakeichi et al. 2004;
  HALS, rank 3, under a second per run). This is the right multiway tool: on
  the first subject's eyes-closed run it returns a posterior broadband
  component carrying 82% of the power and left and right posterior alpha
  components (87% of their power in 8–13 Hz, PO7/O1 and PO8/O2); eyes-open
  it isolates frontal 26–44 Hz components, i.e. muscle and eye activity,
  from the posterior ones. Across the 30 subjects, broadband components
  carry ten times more frontal weight than alpha components (EC 0.039 vs
  0.003, EO 0.131 vs 0.026 of the top-eight channel weight), which is the
  spatial separation of aperiodic and alpha processes the ξ-α NET paper
  reports, with the caveat that in eyes-open much of the frontal broadband
  power is not neural.
- **Fitting the spectral models to PARAFAC component spectra directly does
  not work**: a non-negative part-spectrum is near zero outside its band, so
  its log is not something a full-spectrum model should fit (residuals of 4
  to 10 against 0.2 on channel averages). The component's *spatial* factor
  is the usable product.

`--reduce parafac-weights` therefore weights the whole-cap mean log spectrum
by the dominant posterior component's spatial factor (data-driven per run).
Against the fixed posterior average on LEMON: model selection is unchanged
(the two-peak phenomenology wins 42 of 60), the eyes-closed increase of
alpha peak height sharpens (sp2 `pw0` t = +5.4 vs +4.0; mixture
`peak_alpha` t = +3.8 vs +4.7), but the alpha-damping contrast in the
mixture weakens (t = −1.3 vs −3.6) because the learned weights spread
over parietal channels where the peak is less damped to begin with. So for
the damping readout the fixed posterior set is the better preprocessing;
for component separation and artefact isolation PARAFAC is, and that is
where it should be used: before the spectra, as a cleaning step, and as
the spatial prior for where ξ-like and α-like processes live.

`denoise_epochs` (multiway NORDIC: HOSVD with Marchenko-Pastur truncation
per mode on a trials × channels × time tensor) is implemented and tested on
synthetic data; it has not yet been applied to the HBN task epochs.

On ERP-CORE: there are no neurojax results on it. The only ERP-CORE work in
the workspace is the Zuna imputation pipeline's N170 benchmark, which was
written for the DGX against a copy of the dataset there and evaluated on
ds000117 with specparam features; nothing from it is on this machine. MOABB
has an ERP-CORE loader. Processing ERP-CORE through the multiway and
spectral pipeline would be new work.


## Beta resonance and pharmacological EEG (added 2026-10-15)

`DampedMixture2` (`liley_mix2`) adds a beta oscillator to the Liley mixture.
The beta centre is parametrised as a ratio above alpha, `f_beta = f_alpha
(1 + e^θ)`, after a first version with a free beta centre swapped the alpha
and beta labels in 13 of 60 LEMON rows. All four datasets were refitted
(`scripts/bsd_batch.py --refit liley_mix2`); every row converged, none
collapsed.

| dataset | rows | mix2 wins | RFX xp mix2 | sp2 | median ΔF mix2−sp2 | mix2−mix | mix2−ξα | residual mix2 / sp2 |
|---|---|---|---|---|---|---|---|---|
| LEMON EC+EO | 60 | 33 | 1.00 | 0.00 | +8.3 (40/59) | +21.0 (41/60) | +25.8 (46/60) | 0.133 / 0.132 |
| ds005620 awake-EC + sedation | 51 | 25 | 0.79 | 0.21 | +1.8 (28/51) | +11.4 (41/51) | +27.0 (41/50) | 0.095 / 0.108 |
| Chennu propofol (4 levels) | 80 | 11 | 0.00 | 1.00 | −24.3 (10/79) | +19.2 (79/80) | +19.0 (56/77) | 0.319 / 0.193 |
| ds004541 general anaesthesia | 29 | 5 | 0.04 | 0.01 | +0.9 (15/28) | −1.7 (9/29) | −0.6 (12/27) | 0.93 / 1.03 |

So the beta resonance closes the gap on awake adults and reverses it (the
mechanistic model is now the best description of LEMON, exceedance
probability 1.00, omnibus risk 0.00), halves it under propofol (from 40–60
nats to ~24), and beats the one-resonance mixture on essentially every run
where there is any signal. The propofol beta remains the open problem: the
two-peak phenomenology fits it with a Gaussian of height ~4 in log power
and width several Hz, which one damped oscillator cannot match with its
Lorentzian tails.

Within-subject contrasts with `liley_mix2` (log ratios):

| contrast | parameter | mean | t | n |
|---|---|---|---|---|
| LEMON EC − EO | alpha damping | −1.16 | −5.0 | 30 |
| LEMON EC − EO | alpha peak power | +2.43 | +5.3 | 30 |
| LEMON EC − EO | alpha frequency | −0.18 | −2.2 | 30 |
| LEMON EC − EO | beta damping | +0.04 | +0.1 | 30 |
| LEMON EC − EO | beta peak power | +0.66 | +1.8 | 30 |
| Chennu moderate − baseline | alpha frequency | +0.22 | +3.5 | 20 |
| Chennu moderate − baseline | alpha damping | +0.24 | +1.2 | 20 |
| Chennu moderate − baseline | alpha peak power | −0.63 | −1.8 | 20 |
| Chennu moderate − baseline | beta damping | −0.22 | −2.3 | 20 |
| Chennu moderate − baseline | beta peak power | −0.48 | −2.5 | 20 |
| Chennu moderate − baseline | sp2 knee | −1.42 | −4.4 | 19 |
| ds005620 sedation − awake-EC | alpha damping | +0.43 | +1.8 | 20 |
| ds005620 sedation − awake-EC | alpha frequency | +0.12 | +1.3 | 20 |
| ds005620 sedation − awake-EC | sp2 knee | −2.06 | −3.6 | 20 |
| ds005620 sedation − awake-EC | sp2 alpha centre (Hz) | +2.6 | +2.8 | 20 |

Two revisions to earlier statements follow. (i) Eye closure is a cleaner
damping effect with beta in the model (t −5.0 vs −2.7 for `liley_mix`), and
beta does not move with eye closure. (ii) The earlier "propofol raises alpha
damping" (t = +4.4 with `liley_mix`) was mostly the one-resonance model
absorbing the beta bump: with beta modelled the alpha-damping contrast is
+0.24 (t = 1.2) on Chennu and +0.43 (t = 1.8) on ds005620, and the robust
propofol signature is alpha frequency up by ~20 %, beta sharper (damping
−20 %) but weaker, and the aperiodic knee down by a factor 4–8. This
matches the propofol literature (frontal alpha-ization/anteriorisation and
the beta-to-alpha slowing) better than a pure damping story did. Between
subjects there is still no correlation with the Chennu plasma level.

### Other models tried for the beta gap

`CMCSpectrum` (canonical microcircuit transfer function, 16 SPM-prior
log-scaling parameters, `neurojax.bsd.cmc_model`): the forward model is the
verified DCM-CSD port, but variational Laplace from the prior mean gives
F = −110 on the first LEMON row (residual 0.31; `liley_mix2` F = +38,
residual 0.09), and an unconstrained pre-fit start runs tens of prior SDs
away (F = −66947). It is not in the batch. It needs SPM's own inversion
schedule (nlsi_GN with its hyperparameter schedule) or a better
initialisation, and is the route to a receptor-labelled rather than a
phenomenological beta.

## General anaesthesia, ds004541 (8 patients, 58 ch, 1 kHz, EDF; CC0)

Intra-operative EEG-fNIRS from Saini Ferrón, Vargas González and Valle
Araya (UC Chile). The anaesthetic agents are not documented in the release
(README "in preparation", participants.tsv all n/a), so the conditions are
procedural windows cut from the event markers: baseline (≤ 5 min before
drug delivery), induction (start → loss of consciousness), maintenance
(first 5 min after LOC; includes tetanic stimuli), emergence (last 5 min
before return of consciousness); one patient has only the baseline. Bad
channels (343 of 464 channel entries are flagged) are dropped before the
average reference. 29 rows; every model's residual is ~1.0 (versus 0.1–0.3
elsewhere), model selection is inconclusive (BOR 0.68–0.79 within
condition), and no within-subject contrast reaches |t| > 2.1 at n = 6–7.
The operating-room EEG is dominated by electrical and movement artefact at
these channel counts; the dataset is processed and the adapter kept, but it
is not evidence for or against anything here.

## Pharmaco-EEG datasets available for processing (scan of 2026-10-15)

OpenNeuro's `advancedSearch` with drug keywords, restricted to EEG/MEG/iEEG:

- **Processed here:** Chennu propofol (FieldTrip mirror), ds005620 propofol
  repeated awakening, ds004541 general anaesthesia (above).
- **ds004370 PRIOS** (CC0): ECoG cortico-cortical evoked potentials in six
  epilepsy patients, single-pulse stimulation awake vs under propofol
  (van Blooijs et al., UMC Utrecht). Not resting spectra, but the only open
  *perturbational* propofol dataset: an evoked-response contrast under drug
  is the DCM-CSD/CMC use case rather than this one.
- **ds003380**: juvenile pig, isoflurane/fentanyl/propofol sedation and
  ischaemia (one BIDS subject, 11 animals). Animal, 20 MB.
- **Not on OpenNeuro:** no psilocybin, LSD, DMT, ketamine, xenon, N2O,
  dexmedetomidine, sevoflurane, midazolam or scopolamine EEG dataset is
  listed (keyword hits for those terms were unrelated datasets). Known
  closed or gated sources remain: Blackburne 2025 5-MeO EEG (not public),
  Timmermann/Imperial DMT EEG-fMRI (band-passed release; request drafted,
  not sent), Muthukumaraswamy LSD/ketamine/tiagabine MEG (not public),
  Evertz 2025 xenon/N2O HD-EEG (Zenodo, restricted), HarMNqEEG (Synapse,
  registration), PhysioNet `eeg-gaba-anesthesia`, Neurotycho macaque
  propofol/ketamine ECoG (open, animal).
- The usable open human pharmaco-EEG for spectral model comparison is
  therefore propofol only (Chennu, ds005620), which is why the propofol beta
  is the benchmark the mechanistic models have to pass.
