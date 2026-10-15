# Competing spectral models on recorded EEG: first batch results

Written 2026-10-15 from `scripts/bsd_batch.py` and `scripts/bsd_summarise.py`
(outputs under `results/bsd/`, committed copies under `docs/results/bsd/`). Every spectrum is the mean over a posterior
channel set after average reference, Welch with 2 s Hann segments, 2–45 Hz,
median log power removed before fitting. Six models are fitted by
variational Laplace with two restarts and compared by free energy:

| key | model | params |
|---|---|---|
| `specparam` | fixed aperiodic + one Gaussian peak | 5 |
| `specparam_knee` | knee aperiodic + one peak | 6 |
| `specparam2` | knee aperiodic + two peaks (alpha, beta) | 9 |
| `liley` | one damping-distributed oscillator family + floor | 5 |
| `liley_mix` | damping-distributed background family + resonant alpha + floor | 8 |
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
pharmacological EEG needs a second resonance. A mixture with background
family plus alpha and beta oscillators (11 parameters) is the next model
to add, and the gnf-jax block supplies exactly that when more than one
mode is carried.

## Propofol repeated awakening, ds005620 (4 subjects, 65 ch, 5 kHz → 250 Hz)

18 rows (awake EC/EO, TMS-awake, sedation runs). Mixture 10 wins, ξ-α 8;
RFX xp mix 0.70. Sedation minus awake-EC: alpha damping +1.9 log units
(t = +2.1, n = 4). Too few subjects for more than a direction, and the
direction agrees with Chennu.

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

## What this says for the protocol

1. **Model comparison is decisive on real data** at 10 minutes of EEG: protected
   exceedance probabilities near 1 with omnibus risk near 0 on 30 subjects.
2. **The Liley mixture is the mechanistic model to carry forward.** It beats
   ξ-α on awake adults and children, loses to it under moderate propofol
   sedation, and its parameters move in the predicted directions for both
   eye closure (damping down) and sedation (damping up). The switch of winner
   under sedation is itself a result: a strong low-frequency process appears
   that a separate aperiodic Matérn process describes better than a damped
   family does, which is the kind of state change the E/I block must be able
   to represent.
3. **The phenomenology still fits best**, so a mechanistic claim must be made
   against `specparam2`, not against the one-peak specparam, and the gap is
   about ten nats per subject; a drug effect expressed in mixture parameters
   is interpretable where a change in two Gaussian heights is not.
4. **Driven paradigms need their own validation data.** HBN's task does not
   supply a steady-state response; the Tsai and Hou designs do, and that is
   the data to request or collect.

Caveats: posterior channel averages only (no source localisation); one
alpha centre prior per row from the detrended 6–14 Hz maximum; the
log-power offset removal makes power parameters relative; LEMON's
preprocessed release is band-passed and ICA-cleaned, which flattens the
low-frequency end and lowers every exponent; HBN children's spectra are
noisy and the average reference over 129 channels includes bad channels.
