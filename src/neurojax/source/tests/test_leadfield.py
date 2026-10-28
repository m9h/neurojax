"""Tests for neurojax.source.leadfield — the leadfield as a Lineax operator with
differentiable CiftiStorm-style geometric corrections and leadfield QC."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from neurojax.source.leadfield import (
    Leadfield,
    leadfield_qc,
    point_spread_width,
)


@pytest.fixture
def free_problem():
    """Random free-orientation gain (M, N, 3) with unit cortical normals."""
    key = jax.random.PRNGKey(0)
    kg, kn, kr = jax.random.split(key, 3)
    n_sensors, n_sources = 24, 10
    gain_free = jax.random.normal(kg, (n_sensors, n_sources, 3))
    # give the sources a spread of depths: scale columns by 10x range
    scale = jnp.logspace(-1, 0, n_sources)
    gain_free = gain_free * scale[None, :, None]
    normals = jax.random.normal(kn, (n_sources, 3))
    normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
    positions = jax.random.uniform(kr, (n_sources, 3), minval=-0.07, maxval=0.07)
    return gain_free, normals, positions


def test_fixed_orientation_gain_with_no_correction(free_problem):
    gain_free, normals, positions = free_problem
    lf = Leadfield(gain_free, normals, positions, depth=0.0)
    expected = jnp.einsum("mnk,nk->mn", gain_free, normals)
    np.testing.assert_allclose(lf.gain(), expected, rtol=1e-5)
    assert lf.n_sensors == 24 and lf.n_sources == 10


def test_full_depth_compensation_equalises_column_norms(free_problem):
    gain_free, normals, positions = free_problem
    lf = Leadfield(gain_free, normals, positions, depth=1.0)
    norms = jnp.linalg.norm(lf.gain(), axis=0)
    np.testing.assert_allclose(norms, norms.mean(), rtol=1e-4)
    # partial compensation shrinks but does not remove the depth bias
    lf_half = Leadfield(gain_free, normals, positions, depth=0.5)
    spread_half = jnp.std(jnp.log(jnp.linalg.norm(lf_half.gain(), axis=0)))
    spread_none = jnp.std(jnp.log(jnp.linalg.norm(
        Leadfield(gain_free, normals, positions, depth=0.0).gain(), axis=0)))
    assert 0 < spread_half < spread_none


def test_orientations_stay_unit_length_under_tilt(free_problem):
    gain_free, normals, positions = free_problem
    lf = Leadfield(gain_free, normals, positions)
    lf = lf.with_tilt(jnp.full_like(normals, 0.3))
    np.testing.assert_allclose(jnp.linalg.norm(lf.orientations(), axis=1), 1.0, atol=1e-6)
    # the tilted orientations differ from the normals
    assert jnp.max(jnp.abs(lf.orientations() - normals)) > 0.05


def test_lineax_operator_solves_least_squares(free_problem):
    gain_free, normals, positions = free_problem
    lf = Leadfield(gain_free, normals, positions, depth=0.0)
    op = lf.operator()
    assert isinstance(op, lx.AbstractLinearOperator)
    assert op.out_size() == 24 and op.in_size() == 10
    G = lf.gain()
    x_true = jax.random.normal(jax.random.PRNGKey(1), (10,))
    y = G @ x_true
    sol = lx.linear_solve(op, y, solver=lx.AutoLinearSolver(well_posed=False))
    np.testing.assert_allclose(sol.value, x_true, rtol=1e-4, atol=1e-5)


def test_gradients_flow_through_depth_and_tilt(free_problem):
    gain_free, normals, positions = free_problem
    lf = Leadfield(gain_free, normals, positions, depth=0.5)
    x = jax.random.normal(jax.random.PRNGKey(2), (10, 50))
    target = jax.random.normal(jax.random.PRNGKey(3), (24, 50))

    def misfit(model):
        return jnp.mean((model.gain() @ x - target) ** 2)

    grads = jax.grad(misfit)(lf)
    assert jnp.isfinite(grads.depth) and jnp.abs(grads.depth) > 0
    assert jnp.all(jnp.isfinite(grads.tilt)) and jnp.linalg.norm(grads.tilt) > 0
    # the raw gain and the normals are fixed geometry, not trainable
    assert jnp.linalg.norm(grads.gain_free) == 0 and jnp.linalg.norm(grads.normals) == 0


def test_leadfield_qc_reports_conditioning_and_depth_outliers(free_problem):
    gain_free, normals, positions = free_problem
    G = Leadfield(gain_free, normals, positions, depth=0.0).gain()
    qc = leadfield_qc(G)
    s = np.linalg.svd(np.asarray(G), compute_uv=False)
    np.testing.assert_allclose(qc["condition_number"], s[0] / s[-1], rtol=1e-4)
    np.testing.assert_allclose(qc["singular_values"], s, rtol=1e-4)
    assert qc["effective_rank"] == 10
    assert "rank_tol" in qc  # the threshold is reported, never hidden
    assert qc["depth_outliers"].size == 0

    # a numerically dead source (e.g. inside the inner skull) is an outlier
    G_bad = G.at[:, 3].multiply(1e-4)
    qc_bad = leadfield_qc(G_bad)
    assert 3 in qc_bad["depth_outliers"]

    # a duplicated source column is rank deficient at the reported tolerance
    G_dup = G.at[:, 4].set(G[:, 5])
    qc_dup = leadfield_qc(G_dup)
    assert qc_dup["effective_rank"] == 9


def test_point_spread_width_zero_for_exact_inverse(free_problem):
    gain_free, normals, positions = free_problem
    G = Leadfield(gain_free, normals, positions, depth=0.0).gain()
    W_exact = jnp.linalg.pinv(G)
    width = point_spread_width(W_exact, G, positions)
    assert width.shape == (10,)
    np.testing.assert_allclose(width, 0.0, atol=1e-3)
    # a heavily regularised minimum-norm kernel smears sources: positive width
    GGt = G @ G.T
    W_smooth = G.T @ jnp.linalg.inv(GGt + 10.0 * jnp.trace(GGt) / 24 * jnp.eye(24))
    width_smooth = point_spread_width(W_smooth, G, positions)
    assert jnp.all(width_smooth > 1e-3)


def test_from_mne_forward_roundtrip():
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    # small EEG sphere forward on a discrete source space: no data download
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = ["Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "F3", "F4", "T7", "T8"]
    info = mne.create_info(ch_names, 100.0, "eeg")
    info.set_montage(montage)
    rng = np.random.RandomState(0)
    rr = rng.uniform(-0.03, 0.03, (6, 3)) + np.array([0.0, 0.0, 0.04])
    nn = rng.normal(size=(6, 3))
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=nn))
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    trans = mne.transforms.Transform("head", "mri", np.eye(4))
    fwd = mne.make_forward_solution(info, trans, src, sphere, eeg=True, meg=False)

    lf = Leadfield.from_mne_forward(fwd)
    assert lf.gain_free.shape == (len(ch_names), 6, 3)
    assert lf.normals.shape == (6, 3) and lf.positions.shape == (6, 3)
    np.testing.assert_allclose(np.asarray(lf.normals), nn, atol=1e-6)
    # a fixed-orientation MNE forward and our fixed-orientation gain agree
    fwd_fix = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False)
    np.testing.assert_allclose(np.asarray(lf.gain()), fwd_fix["sol"]["data"],
                               rtol=1e-5, atol=1e-12)
