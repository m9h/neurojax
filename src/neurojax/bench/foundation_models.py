"""EEG/MEG foundation-model adapters + identity-trap diagnostics — from emeg-fm.

Re-exports the emeg-fm peer package (github.com/m9h/emeg-fm) so NeuroJAX gains
EEG/MEG foundation-model integration plus the FMScope identity-trap audit —
capabilities OSL / osl-dynamics has no equivalent of:

* **FM adapters** (:func:`make_hf_encoder`, :class:`HFModelAdapter`) — wrap a
  HuggingFace EEG-FM (REVE, LaBraM, …) as a frozen feature encoder.
* **Subject-axis LEACE erasure** (:mod:`erasure`,
  :func:`subject_axis_erasure`) — least-squares concept erasure of the
  subject-identity axis (Belrose 2023), the core of the identity-trap test:
  closed-form whitening (:func:`whiten`) + eraser (:func:`subject_eraser`).
* **Identity-trap audit** (:mod:`audit`, :func:`audit_cell`) — the pooled vs
  per-trial erasure comparison that exposes the pooling artifact.
* **MOABB cohort builder** (:func:`build_moabb_cohort`) and the linear SVM probe
  (:mod:`svm_probe`).

Torch is imported lazily inside emeg-fm, so importing this module is light; the
FM adapters pull torch/transformers only when actually called. Install via the
``[eeg-fm]`` extra (path source to ../emeg-fm); inert when the peer is absent.

Example::

    from neurojax.bench.foundation_models import make_hf_encoder, subject_axis_erasure
    enc = make_hf_encoder("brain-bzh/reve-base")        # frozen EEG-FM features
    result = subject_axis_erasure(features, subject_ids)  # identity-trap test
"""

from emeg_fm import adapters, moabb_cohort
from emeg_fm.adapters import HFEncoderParams, HFModelAdapter, make_hf_encoder
from emeg_fm.moabb_cohort import build_moabb_cohort
from fmscope.diagnostics import erasure
from fmscope.diagnostics.erasure import (
    ErasureResult,
    apply_eraser,
    subject_axis_erasure,
    subject_eraser,
    subject_probe,
    subspace_overlap,
    whiten,
)
from fmscope.training import svm_probe
from fmscope.verdict import audit
from fmscope.verdict.audit import AuditConfig, audit_cell

__all__ = [
    "adapters",
    "moabb_cohort",
    "erasure",
    "audit",
    "svm_probe",
    "make_hf_encoder",
    "HFModelAdapter",
    "HFEncoderParams",
    "build_moabb_cohort",
    "whiten",
    "subject_eraser",
    "apply_eraser",
    "subject_probe",
    "subspace_overlap",
    "subject_axis_erasure",
    "ErasureResult",
    "audit_cell",
    "AuditConfig",
]
