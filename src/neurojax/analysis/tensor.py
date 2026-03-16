"""Tensor operations — re-exported from jaxctrl for backwards compatibility.

The core tensor algebra (unfold, fold, mode_dot, HOSVD, Tucker) now
lives in ``jaxctrl._tensor_ops``.  This module re-exports the same
names for existing neurojax code.
"""

from jaxctrl import (
    hosvd,
    mode_dot,
    tensor_fold as fold,
    tensor_unfold as unfold,
    tucker_to_tensor,
)

__all__ = ["unfold", "fold", "mode_dot", "hosvd", "tucker_to_tensor"]
