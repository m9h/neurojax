"""Export release derivatives to vpjax's portable observation format."""

import argparse

from neurojax.analysis.autonomic import (
    standardize_observations,
    window_observations,
)
from neurojax.io.dmt_autonomic import load_dmt_derivatives
from neurojax.io.observations import save_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="extracted release with participants.tsv")
    parser.add_argument("output", help="output .npz")
    parser.add_argument("--subject", default="S04")
    parser.add_argument("--session", type=int, choices=[1, 2], default=1)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--features", nargs="+", default=["heart_rate", "smna", "rvt"])
    args = parser.parse_args()
    drug = load_dmt_derivatives(args.root, args.subject, args.session, "dmt")
    baseline = load_dmt_derivatives(args.root, args.subject, args.session, "rs")
    for name in args.features:
        if name not in drug or name not in baseline:
            raise ValueError(f"feature {name} missing from drug or baseline")
        if not drug[name].valid.any() or not baseline[name].valid.any():
            raise ValueError(f"feature {name} excluded or has no valid samples")
    drug = window_observations(
        {name: drug[name] for name in args.features}, window_s=args.window_s
    )
    baseline = window_observations(
        {name: baseline[name] for name in args.features}, window_s=args.window_s
    )
    standardized = standardize_observations(drug, baseline)
    save_observations(args.output, standardized)
    for name, stream in standardized.items():
        print(f"{name}: {stream.valid.sum()}/{len(stream.values)} valid windows")


if __name__ == "__main__":
    main()
