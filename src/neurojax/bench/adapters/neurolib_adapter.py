"""NeurolibFitnessAdapter — subprocess oracle wrapping neurolib.

Runs neurolib models in a subprocess to avoid JAX/numba conflicts.
Uses neurolib's bundled HCP connectomes for validation. Serves as
the DEAP optimization baseline and BOLD/FC oracle.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from neurojax.bench.fitness import FitnessAdapter, FitnessResult, ObjectiveSpec


# Script template executed in subprocess (neurolib environment)
_NEUROLIB_SCRIPT = '''
import json
import sys
import numpy as np

def run():
    from neurolib.models.hopf import HopfModel
    from neurolib.utils.loadData import Dataset
    from neurolib.utils.functions import fc

    params = json.loads(sys.argv[1])
    dataset_name = sys.argv[2]
    duration = float(sys.argv[3])

    # Load dataset
    ds = Dataset(dataset_name)

    # Create model with empirical connectivity
    model = HopfModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = duration
    model.params["sigma_ou"] = 0.14
    model.params["K_gl"] = 2.0

    # Apply user parameters
    for k, v in params.items():
        model.params[k] = v

    model.run()

    # Get BOLD
    bold = model["BOLD"].T if "BOLD" in model.outputs else model.output.T

    # Compute FC
    fc_matrix = fc(bold)

    result = {
        "fc": fc_matrix.tolist(),
        "bold_shape": list(bold.shape),
    }
    print(json.dumps(result))

run()
'''


class NeurolibFitnessAdapter:
    """FitnessAdapter wrapping neurolib as a subprocess oracle.

    Runs neurolib in an isolated process to avoid dependency conflicts
    with JAX. Uses neurolib's bundled HCP connectome.
    """

    def __init__(
        self,
        empirical_fc: np.ndarray,
        dataset_name: str = "hcp",
        duration: float = 60_000.0,
        neurolib_python: Optional[str] = None,
        param_bounds: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.empirical_fc = np.asarray(empirical_fc)
        self.dataset_name = dataset_name
        self.duration = duration
        # Path to a Python interpreter with neurolib installed
        # If None, uses current interpreter (may fail if neurolib not installed)
        self._python = neurolib_python or sys.executable
        self._param_bounds = param_bounds or {
            "K_gl": (0.0, 10.0),
            "sigma_ou": (0.0, 0.5),
        }

    @property
    def parameter_space(self) -> dict[str, tuple[float, float]]:
        return dict(self._param_bounds)

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        return [
            ObjectiveSpec("fc_correlation", "maximize"),
            ObjectiveSpec("fcd_ks_distance", "minimize"),
        ]

    def evaluate(self, params: dict[str, float]) -> FitnessResult:
        """Run neurolib simulation in subprocess, return fitness."""
        t0 = time.perf_counter()

        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(_NEUROLIB_SCRIPT)
            script_path = f.name

        try:
            result = subprocess.run(
                [
                    self._python,
                    script_path,
                    json.dumps(params),
                    self.dataset_name,
                    str(self.duration),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                return FitnessResult(
                    fc_correlation=0.0,
                    fcd_ks_distance=1.0,
                    wall_time=time.perf_counter() - t0,
                    metadata={"error": result.stderr[:500]},
                )

            output = json.loads(result.stdout)
            sim_fc = np.array(output["fc"])

            # Compute matrix correlation
            r_fc = self._matrix_correlation(sim_fc, self.empirical_fc)

            return FitnessResult(
                fc_correlation=r_fc,
                fcd_ks_distance=0.0,  # FCD requires raw BOLD, not returned here
                simulated_fc=sim_fc,
                wall_time=time.perf_counter() - t0,
            )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError) as e:
            return FitnessResult(
                fc_correlation=0.0,
                fcd_ks_distance=1.0,
                wall_time=time.perf_counter() - t0,
                metadata={"error": str(e)},
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    def evaluate_batch(
        self, params_batch: list[dict[str, float]]
    ) -> list[FitnessResult]:
        return [self.evaluate(p) for p in params_batch]

    @staticmethod
    def _matrix_correlation(mat1: np.ndarray, mat2: np.ndarray) -> float:
        """Pearson correlation of upper-triangular elements (numpy)."""
        n = mat1.shape[0]
        idx = np.triu_indices(n, k=1)
        v1 = mat1[idx]
        v2 = mat2[idx]
        if np.std(v1) == 0 or np.std(v2) == 0:
            return 0.0
        return float(np.corrcoef(v1, v2)[0, 1])

    @staticmethod
    def load_hcp_empirical_fc(dataset_name: str = "hcp") -> np.ndarray:
        """Load empirical FC from neurolib's bundled dataset.

        Must be called in an environment where neurolib is installed.
        Returns the group-averaged FC matrix.
        """
        from neurolib.utils.loadData import Dataset

        ds = Dataset(dataset_name)
        return ds.FCs.mean(axis=0) if hasattr(ds, "FCs") else ds.FC
