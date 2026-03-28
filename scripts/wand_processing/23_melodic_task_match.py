#!/usr/bin/env python3
"""Match MELODIC components to task design — find task-related ICs.

Given MELODIC output + task events.tsv, identifies which ICA components
correlate most strongly with the task timing. This reveals:
  1. Which component(s) are task-related vs noise
  2. The spatial map of the task-related network
  3. Dimensionality of task-related signal vs structured noise

Usage:
    python 23_melodic_task_match.py \\
        --melodic-dir derivatives/fsl-fmri/sub-08033/ses-03/task-categorylocaliser_run-1_melodic \\
        --events sub-08033/ses-03/func/sub-08033_ses-03_task-categorylocaliser_run-1_events.tsv \\
        --tr 2.0
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_events(events_path):
    """Load BIDS events.tsv → dict of {condition: [(onset, duration), ...]}."""
    conditions = {}
    with open(events_path) as f:
        header = f.readline().strip().split('\t')
        onset_idx = header.index('onset')
        dur_idx = header.index('duration')
        type_idx = header.index('trial_type')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(onset_idx, dur_idx, type_idx):
                cond = parts[type_idx]
                onset = float(parts[onset_idx])
                dur = float(parts[dur_idx])
                conditions.setdefault(cond, []).append((onset, dur))
    return conditions


def make_design_matrix(conditions, n_timepoints, tr):
    """Create a design matrix by convolving events with canonical HRF."""
    times = np.arange(n_timepoints) * tr

    # Canonical double-gamma HRF (Glover 1999)
    def hrf(t):
        from scipy.stats import gamma as gamma_dist
        h = gamma_dist.pdf(t, 6) - 0.35 * gamma_dist.pdf(t, 16)
        return h / h.max()

    hrf_times = np.arange(0, 32, tr)
    hrf_kernel = hrf(hrf_times)

    design = {}
    for cond, events in conditions.items():
        if cond == 'baseline':
            continue  # skip baseline condition
        # Build boxcar
        boxcar = np.zeros(n_timepoints)
        for onset, dur in events:
            start = int(onset / tr)
            end = int((onset + dur) / tr)
            boxcar[start:min(end, n_timepoints)] = 1.0
        # Convolve with HRF
        convolved = np.convolve(boxcar, hrf_kernel)[:n_timepoints]
        # Demean
        convolved -= convolved.mean()
        design[cond] = convolved

    return design


def correlate_components(melodic_dir, design, n_timepoints):
    """Correlate each MELODIC component timecourse with task regressors."""
    # Load MELODIC mixing matrix (component timecourses)
    mix_file = os.path.join(melodic_dir, "melodic_mix")
    if not os.path.exists(mix_file):
        print(f"ERROR: {mix_file} not found")
        return None

    mixing = np.loadtxt(mix_file)  # (n_timepoints, n_components)
    n_components = mixing.shape[1]

    results = []
    for ic in range(n_components):
        tc = mixing[:, ic]
        # Demean
        tc = tc - tc.mean()
        tc_norm = tc / max(np.std(tc), 1e-10)

        ic_result = {"component": ic + 1}
        max_corr = 0

        for cond, regressor in design.items():
            reg_norm = regressor / max(np.std(regressor), 1e-10)
            corr = float(np.corrcoef(tc_norm, reg_norm[:len(tc_norm)])[0, 1])
            ic_result[f"corr_{cond}"] = round(corr, 4)
            max_corr = max(max_corr, abs(corr))

        ic_result["max_abs_corr"] = round(max_corr, 4)
        results.append(ic_result)

    # Sort by max correlation
    results.sort(key=lambda x: x["max_abs_corr"], reverse=True)
    return results


def plot_results(results, design, melodic_dir, output_path, tr):
    """Plot top components vs task design."""
    if not HAS_MPL:
        return

    mix_file = os.path.join(melodic_dir, "melodic_mix")
    mixing = np.loadtxt(mix_file)

    n_top = min(5, len(results))
    fig, axes = plt.subplots(n_top + 1, 1, figsize=(14, 3 * (n_top + 1)), sharex=True)

    # Plot task design
    times = np.arange(mixing.shape[0]) * tr
    ax = axes[0]
    for cond, reg in design.items():
        ax.plot(times[:len(reg)], reg / max(np.abs(reg).max(), 1e-10), label=cond, alpha=0.7)
    ax.set_title("Task Design (HRF-convolved)", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.set_ylabel("Amplitude")

    # Plot top components
    for i in range(n_top):
        ic = results[i]["component"] - 1
        ax = axes[i + 1]
        tc = mixing[:, ic]
        ax.plot(times, tc, color='steelblue', linewidth=0.8)
        corr_str = ", ".join([f"{k}={v:.3f}" for k, v in results[i].items()
                               if k.startswith("corr_")])
        ax.set_title(f"IC {results[i]['component']} (max |r| = {results[i]['max_abs_corr']:.3f}) — {corr_str}",
                      fontsize=10)
        ax.set_ylabel("AU")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Match MELODIC ICs to task design")
    p.add_argument("--melodic-dir", required=True)
    p.add_argument("--events", required=True, help="BIDS events.tsv")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    melodic_dir = args.melodic_dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = melodic_dir

    print(f"=== MELODIC-Task Component Matching ===")
    print(f"  MELODIC: {melodic_dir}")
    print(f"  Events:  {args.events}")
    print(f"  TR:      {args.tr}s")

    # Load events
    conditions = load_events(args.events)
    print(f"  Conditions: {list(conditions.keys())}")

    # Get number of timepoints from mixing matrix
    mix_file = os.path.join(melodic_dir, "melodic_mix")
    if not os.path.exists(mix_file):
        print(f"ERROR: MELODIC output not found at {melodic_dir}")
        print("  Run MELODIC first, then re-run this script.")
        sys.exit(1)

    mixing = np.loadtxt(mix_file)
    n_timepoints = mixing.shape[0]
    n_components = mixing.shape[1]
    print(f"  Timepoints: {n_timepoints}, Components: {n_components}")

    # Build design matrix
    design = make_design_matrix(conditions, n_timepoints, args.tr)

    # Correlate
    results = correlate_components(melodic_dir, design, n_timepoints)

    # Print top matches
    print(f"\n  Top task-correlated components:")
    for r in results[:10]:
        ic = r["component"]
        corr = r["max_abs_corr"]
        status = "★ TASK" if corr > 0.3 else ("~ weak" if corr > 0.15 else "  noise")
        print(f"    {status}  IC {ic:3d}: max |r| = {corr:.3f}")

    # Save results
    results_path = os.path.join(out_dir, "task_component_match.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Plot
    plot_results(results, design, melodic_dir,
                  os.path.join(out_dir, "task_component_match.png"), args.tr)

    # Summary
    task_components = [r for r in results if r["max_abs_corr"] > 0.3]
    noise_components = [r for r in results if r["max_abs_corr"] < 0.1]
    print(f"\n  Summary:")
    print(f"    Task-related (|r| > 0.3): {len(task_components)} components")
    print(f"    Noise (|r| < 0.1):        {len(noise_components)} components")
    print(f"    Total:                     {n_components} components")
    print(f"    Signal dimensionality:     ~{len(task_components)} task + {n_components - len(noise_components) - len(task_components)} ambiguous")


if __name__ == "__main__":
    main()
