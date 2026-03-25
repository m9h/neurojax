#!/usr/bin/env python
"""Generate synthetic HMM data using osl-dynamics and save to /data.

Writes:
  - osl_sim_timeseries.npy   (T, C) observed time series
  - osl_sim_states_true.npy  (T, S) true one-hot state assignments
  - osl_sim_means_true.npy   (S, C) true state means
  - osl_sim_covs_true.npy    (S, C, C) true state covariances
"""

import os
import sys
import types

# Stub FSL
fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import numpy as np
from osl_dynamics.simulation.hmm import HMM_MVN


def main():
    data_dir = os.environ.get("DATA_DIR", "/data")
    os.makedirs(data_dir, exist_ok=True)

    n_samples = int(os.environ.get("N_SAMPLES", "25600"))
    n_states = int(os.environ.get("N_STATES", "8"))
    n_channels = int(os.environ.get("N_CHANNELS", "38"))
    stay_prob = float(os.environ.get("STAY_PROB", "0.95"))
    seed = int(os.environ.get("SEED", "42"))

    np.random.seed(seed)

    print(f"Generating HMM_MVN: {n_samples} samples, {n_states} states, "
          f"{n_channels} channels, stay_prob={stay_prob}")

    sim = HMM_MVN(
        n_samples=n_samples,
        trans_prob="sequence",
        means="random",
        covariances="random",
        n_states=n_states,
        n_channels=n_channels,
        stay_prob=stay_prob,
    )

    np.save(os.path.join(data_dir, "osl_sim_timeseries.npy"), sim.time_series)
    np.save(os.path.join(data_dir, "osl_sim_states_true.npy"), sim.state_time_course)
    np.save(os.path.join(data_dir, "osl_sim_means_true.npy"), sim.means)
    np.save(os.path.join(data_dir, "osl_sim_covs_true.npy"), sim.covariances)

    print(f"Saved to {data_dir}/:")
    print(f"  timeseries:  {sim.time_series.shape}")
    print(f"  states:      {sim.state_time_course.shape}")
    print(f"  means:       {sim.means.shape}")
    print(f"  covariances: {sim.covariances.shape}")


if __name__ == "__main__":
    main()
