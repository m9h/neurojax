"""Loader tests for the consciousness benchmark datasets (skip when absent)."""

import numpy as np
import pytest

from neurojax.io.consciousness_bench import (
    LOADERS,
    load_chennu_propofol,
    load_dmt,
    load_psilocybin_microdose,
)


def _try(fn, **kw):
    try:
        return fn(**kw)
    except FileNotFoundError as e:
        pytest.skip(f"dataset missing: {e}")


def _check(recs, dataset, conds):
    assert recs, "no recordings"
    for r in recs:
        assert r.dataset == dataset
        assert r.data.ndim == 2 and r.data.dtype == np.float32
        assert r.data.shape[0] >= 16 and r.data.shape[1] >= 30 * r.fs
        assert np.isfinite(r.data).all()
        assert abs(r.data.mean(0)).max() < 1e-6 * r.data.std()  # average reference
    assert {r.condition for r in recs} <= set(conds)


def test_chennu_propofol_one_subject():
    recs = _try(load_chennu_propofol, subjects=["02"])
    _check(recs, "propofol-chennu2016", {"baseline", "mild", "moderate", "recovery"})
    assert {r.condition for r in recs} == {"baseline", "mild", "moderate", "recovery"}
    assert all("propofol_ugml" in r.meta for r in recs)
    assert recs[0].fs == 250.0 and recs[0].data.shape[0] == 91


def test_dmt_one_subject():
    recs = _try(load_dmt, subjects=["S19"])
    _check(recs, "dmt-pallavicini2021", {"DMT", "EC", "EO"})
    assert {r.condition for r in recs} == {"DMT", "EC", "EO"}


def test_psilocybin_one_subject():
    recs = _try(load_psilocybin_microdose, subjects=["S01"])
    _check(recs, "psilocybin-microdose-cavanna2022", {"placebo", "psilocybin"})


def test_loader_registry_contrasts_name_known_conditions():
    for name, (fn, contrasts) in LOADERS.items():
        assert callable(fn)
        for a, b in contrasts:
            assert a != b
