#!/usr/bin/env python
"""Epoch-locked vs inter-trial-interval (ITI) split of the WH discrete-state flux
test -- does the "flux circulates along TINDA order" divergence from WAND (WH:
33%, below chance; WAND: 75%, see wh_langevin.py / wand_langevin.py) reflect
trial-locked task structure, or does it persist in ongoing (non-evoked) activity?

For every face-recognition trial (BIDS events.tsv, 8 subjects x 2 runs = 16
sessions), extracts two disjoint windows on the K=12 HMM state sequence
(oracle_gamma.npy, native ~100 Hz resolution, same states `wh_langevin.py` uses):
  epoch-locked : [onset, onset+1.0s)   -- evoked + early cognitive response
  ITI          : [onset+2.0s, onset+2.8s) -- late inter-trial interval, clear of
                 both this trial's evoked response and the next trial's onset
                 (min ISI across all runs is ~2.99s)
Bigram transition counts are accumulated separately per condition across all
trials/sessions (transition_flux/discrete_entropy_production are linear in the
raw count matrix, so summing per-window counts is exact, unlike concatenating
non-contiguous state subsequences which would fabricate spurious transitions at
window boundaries), then the same "flux along TINDA order" and discrete entropy
production metrics as `wh_langevin.py` are computed for each condition.

    WH_OUT=/data/datasets/wh_src8_full \\
      WH_BASE=/data/datasets/ds000117-download PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wh_epoch_vs_iti_langevin.py \\
      0 1 9 4 8 2 5 7 11 10 3 6
"""

import csv
import os
import sys

import numpy as np

from neurojax.dynamics import transition_flux

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src8_full")
BASE = os.environ.get("WH_BASE", "/data/datasets/ds000117-download")
SUBJECTS = os.environ.get(
    "WH_SUBJECTS", "sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07,sub-08"
).split(",")
RUNS = [int(r) for r in os.environ.get("WH_RUNS", "1,2").split(",")]
FS = 100.0                    # wh_source_parcels_prep.py resample rate
N_EMBED_TRIM = 7              # symmetric TDE(n_embeddings=15) head trim, empirically confirmed
SEQ_LEN = 200                 # osl-dynamics HMM sequence_length -> tail truncation per session
EPOCH_WIN = (0, 100)          # samples post-onset: [0, 1.0s)
ITI_WIN = (200, 280)          # samples post-onset: [2.0s, 2.8s)


def session_lengths():
    """Raw (pre-TDE) per-session sample counts, in the exact loop order
    wh_source_parcels_prep.py used (needed to locate each session's slice of
    oracle_gamma.npy). Recomputed here from each run's raw duration via events.tsv
    is NOT reliable (events don't cover the tail), so these are read back from the
    known, verified session lengths logged during the original prep run."""
    # sub-01..sub-08, runs 1,2 -- exact per-session parcel counts from the
    # wh_source_parcels_prep.py run that produced WH_OUT (sum = 790900, matches
    # parcels68.npy).
    return [49100, 49700, 49400, 49300, 50600, 49400, 49700, 48800,
            49100, 50400, 48900, 48700, 49300, 49200, 50000, 49300]


def trial_onsets(subj, run):
    f = f"{BASE}/{subj}/ses-meg/meg/{subj}_ses-meg_task-facerecognition_run-{run:02d}_events.tsv"
    onsets = []
    with open(f) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            onsets.append(float(row["onset"]))
    return onsets


def main():
    tinda_order = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(12))
    gamma = np.load(os.path.join(OUT, "oracle_gamma.npy")).astype(np.float32)
    states_full = gamma.argmax(1)
    K = gamma.shape[1]
    print(f"gamma {gamma.shape}, TINDA order {tinda_order}", flush=True)

    raw_lens = session_lengths()
    trimmed_lens = [n - 2 * N_EMBED_TRIM for n in raw_lens]
    gamma_lens = [(n // SEQ_LEN) * SEQ_LEN for n in trimmed_lens]
    assert sum(gamma_lens) == gamma.shape[0], (sum(gamma_lens), gamma.shape[0])
    gamma_offsets = np.concatenate([[0], np.cumsum(gamma_lens)])[:-1]

    sess_idx = 0
    n_epoch_win = n_iti_win = n_skipped = 0
    N_epoch = np.zeros((K, K), dtype=np.int64)
    N_iti = np.zeros((K, K), dtype=np.int64)

    for subj in SUBJECTS:
        for run in RUNS:
            onsets = trial_onsets(subj, run)
            g_off = gamma_offsets[sess_idx]
            g_len = gamma_lens[sess_idx]
            for onset_s in onsets:
                raw_local = int(round(onset_s * FS))
                g_local = raw_local - N_EMBED_TRIM

                e0, e1 = g_local + EPOCH_WIN[0], g_local + EPOCH_WIN[1]
                i0, i1 = g_local + ITI_WIN[0], g_local + ITI_WIN[1]
                if 0 <= e0 and e1 <= g_len:
                    w = states_full[g_off + e0: g_off + e1]
                    idx = w[:-1] * K + w[1:]
                    N_epoch += np.bincount(idx, minlength=K * K).reshape(K, K)
                    n_epoch_win += 1
                if 0 <= i0 and i1 <= g_len:
                    w = states_full[g_off + i0: g_off + i1]
                    idx = w[:-1] * K + w[1:]
                    N_iti += np.bincount(idx, minlength=K * K).reshape(K, K)
                    n_iti_win += 1
                else:
                    n_skipped += 1
            sess_idx += 1

    print(f"windows used: {n_epoch_win} epoch-locked, {n_iti_win} ITI "
          f"({n_skipped} trial windows dropped at session edges)", flush=True)

    def report(name, N):
        F = N - N.T
        P = N / N.sum()
        Pt = P.T
        mask = (P > 0) & (Pt > 0)
        eps = float(np.sum(np.where(mask, P * np.log(np.where(mask, P / np.where(Pt > 0, Pt, 1.0), 1.0)), 0.0)))
        cyc = np.array([F[tinda_order[i], tinda_order[(i + 1) % K]] for i in range(K)])
        frac_forward = float(np.mean(np.sign(cyc) == np.sign(np.median(cyc))))
        print(f"  [{name}] discrete entropy production = {eps:.5f} | "
              f"flux along TINDA order = {100 * frac_forward:.0f}% of edges same direction "
              f"| total transitions = {int(N.sum())}", flush=True)

    print("\n========  WH epoch-locked vs ITI discrete-state flux  ========")
    report("epoch-locked (0-1.0s post-stimulus)", N_epoch)
    report("ITI (2.0-2.8s post-stimulus)", N_iti)
    print("  cf. wh_langevin.py on the FULL sequence (no epoch split): 33% flux-along-order")
    print("  cf. wand_langevin.py on WAND resting MEG (no trial structure): 75% flux-along-order")


if __name__ == "__main__":
    main()
