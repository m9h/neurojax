"""FitnessAdapter protocol and FitnessResult for whole-brain model benchmarking.

The FitnessAdapter is the central abstraction that all optimizers target.
Two oracle implementations (SPM/DCM, neurolib) and one primary implementation
(vbjax) conform to this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ObjectiveSpec:
    """Specification for a single optimization objective."""

    name: str
    direction: str  # "maximize" or "minimize"
    weight: float = 1.0


@dataclass
class FitnessResult:
    """Result of evaluating a parameter set on a brain model.

    Carries both scalar fitness metrics and optional raw outputs
    for downstream analysis.
    """

    fc_correlation: float
    fcd_ks_distance: float
    raw_objectives: dict[str, float] = field(default_factory=dict)
    simulated_fc: np.ndarray | None = None
    simulated_bold: np.ndarray | None = None
    wall_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.raw_objectives:
            self.raw_objectives = {
                "fc_correlation": self.fc_correlation,
                "fcd_ks_distance": self.fcd_ks_distance,
            }


@runtime_checkable
class FitnessAdapter(Protocol):
    """Interface between optimizers and brain simulation backends.

    Implementations:
    - SPMAdapter: SPM/DCM JR dynamics oracle
    - NeurolibFitnessAdapter: neurolib BOLD/FC oracle + DEAP baseline
    - VbjaxFitnessAdapter: vbjax with JAX autodiff gradients
    """

    @property
    def parameter_space(self) -> dict[str, tuple[float, float]]:
        """Parameter names mapped to (lower_bound, upper_bound)."""
        ...

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        """List of objectives with names, directions, and weights."""
        ...

    def evaluate(self, params: dict[str, float]) -> FitnessResult:
        """Run simulation with given parameters, return fitness."""
        ...

    def evaluate_batch(
        self, params_batch: list[dict[str, float]]
    ) -> list[FitnessResult]:
        """Batch evaluation. Default: sequential loop over evaluate().

        VbjaxFitnessAdapter overrides with vmap for GPU parallelism.
        """
        ...
