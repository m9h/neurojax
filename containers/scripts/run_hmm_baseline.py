#!/usr/bin/env python
"""Run osl-dynamics HMM baseline and save results for comparison.

Expects /data to contain:
  - osl_sim_timeseries.npy   (T, C) simulated time series
  - osl_sim_states_true.npy  (T, S) true one-hot state assignments

Writes to /data/osl_baseline/:
  - gamma.npy          (T, S) inferred state probabilities
  - means.npy          (S, C) learned state means
  - covariances.npy    (S, C, C) learned state covariances
  - trans_prob.npy     (S, S) learned transition matrix
  - history.npy        (n_epochs,) training loss history
  - config.json        model configuration used
"""

import json
import os
import sys
import time

import numpy as np

# Stub out FSL (not needed for HMM fitting)
import types
fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model


def main():
    data_dir = os.environ.get("DATA_DIR", "/data")
    out_dir = os.path.join(data_dir, "osl_baseline")
    os.makedirs(out_dir, exist_ok=True)

    n_states = int(os.environ.get("N_STATES", "8"))
    n_epochs = int(os.environ.get("N_EPOCHS", "20"))
    sequence_length = int(os.environ.get("SEQ_LEN", "2000"))

    # Load data
    ts_path = os.path.join(data_dir, "osl_sim_timeseries.npy")
    ts = np.load(ts_path)
    n_channels = ts.shape[1]
    print(f"Loaded {ts_path}: {ts.shape}")

    # Prepare data via osl-dynamics Data class
    data = Data([ts])
    data.prepare({"standardize": {}})
    print(f"Prepared data shape: {data[0].shape}")

    # Configure HMM
    config = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=sequence_length,
        learn_means=True,
        learn_covariances=True,
        learn_trans_prob=True,
        batch_size=64,
        learning_rate=0.01,
        n_epochs=n_epochs,
    )

    print(f"\nConfig: n_states={n_states}, n_channels={n_channels}, "
          f"seq_len={sequence_length}, n_epochs={n_epochs}")

    # Build and train
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)

    print("\nTraining HMM...")
    t0 = time.time()
    history = model.fit(data)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # Extract results
    gamma = model.get_alpha(data)
    if isinstance(gamma, list):
        gamma = gamma[0]
    gamma = np.array(gamma)

    means, covs = model.get_means_covariances()
    trans_prob = model.get_trans_prob()

    # Save
    np.save(os.path.join(out_dir, "gamma.npy"), gamma)
    np.save(os.path.join(out_dir, "means.npy"), means)
    np.save(os.path.join(out_dir, "covariances.npy"), covs)
    np.save(os.path.join(out_dir, "trans_prob.npy"), trans_prob)

    loss_arr = np.array(history["loss"]) if isinstance(history, dict) else np.array(history)
    np.save(os.path.join(out_dir, "history.npy"), loss_arr)

    config_dict = {
        "n_states": n_states,
        "n_channels": n_channels,
        "n_epochs": n_epochs,
        "sequence_length": sequence_length,
        "elapsed_seconds": elapsed,
        "final_loss": float(loss_arr[-1]) if len(loss_arr) > 0 else None,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  gamma:       {gamma.shape}")
    print(f"  means:       {means.shape}")
    print(f"  covariances: {covs.shape}")
    print(f"  trans_prob:  {trans_prob.shape}")


if __name__ == "__main__":
    main()
