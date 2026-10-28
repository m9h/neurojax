#!/usr/bin/env python
"""How much of each NLTSA measure the standard spectral parameters explain.

    python scripts/consciousness_bench/spectral_explained.py <dataset> <cond_a> <cond_b>

Reads <CB_OUT>/<dataset>/scores.csv (NLTSA run) and scores_spectral.csv
(``run_bench.py --measures spectral --tag spectral``) and writes
spectral_explained_<a>_vs_<b>.csv.
"""
import os
import sys
from pathlib import Path

import pandas as pd

from neurojax.bench.consciousness_bench import spectral_explained

ds, a, b = sys.argv[1:4]
out = Path(os.environ.get("CB_OUT", "/data/datasets/consciousness-bench")) / ds
ex = spectral_explained(pd.read_csv(out / "scores.csv"), pd.read_csv(out / "scores_spectral.csv"), a, b)
ex.to_csv(out / f"spectral_explained_{a}_vs_{b}.csv", index=False)
print(f"[spectral_explained] {ds}: {b} - {a}")
print(ex[["measure", "n", "r2", "d_raw", "d_residual"]].to_string(index=False, float_format="%.3f"))
