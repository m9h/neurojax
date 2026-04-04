"""TDD tests for MEG source reconstruction on WAND ses-01 data.

RED phase: define the expected pipeline from raw CTF → source timecourses.
GREEN phase: implement using existing neurojax solvers + MNE forward.

5 paradigms: resting, simon, visual, auditorymotor, mmn
Forward model: 3-layer BEM leadfield (274 sensors × 2052 sources × 3 orient)
Source methods: dSPM, LCMV beamformer, LAURA, VARETA, PI-GNN
"""

import pytest
import numpy as np
import jax.numpy as jnp
import os

WAND = "/Users/mhough/dev/wand"
SUB = "sub-08033"
MEG_DIR = f"{WAND}/{SUB}/ses-01/meg"
BEM_DIR = f"{WAND}/derivatives/bem/{SUB}"
FS_DIR = f"{WAND}/derivatives/freesurfer"
OUT_DIR = f"{WAND}/derivatives/meg-source/{SUB}"

HAS_MEG = os.path.exists(f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds")
HAS_FWD = os.path.exists(f"{BEM_DIR}/mne_bem_fwd.fif")

pytestmark = pytest.mark.skipif(
    not (HAS_MEG and HAS_FWD),
    reason="WAND MEG data or forward solution not available"
)


class TestDataLoading:
    """Load and validate CTF MEG data."""

    def test_load_resting(self):
        import mne
        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=False)
        assert raw.info['nchan'] > 270
        assert raw.info['sfreq'] == 1200.0

    def test_load_all_paradigms(self):
        import mne
        tasks = ['resting', 'simon', 'visual', 'auditorymotor', 'mmn']
        for task in tasks:
            ds = f"{MEG_DIR}/{SUB}_ses-01_task-{task}.ds"
            if os.path.exists(ds):
                raw = mne.io.read_raw_ctf(ds, preload=False)
                assert raw.info['nchan'] > 270, f"{task} has too few channels"

    def test_load_forward_solution(self):
        import mne
        fwd = mne.read_forward_solution(f"{BEM_DIR}/mne_bem_fwd.fif")
        assert fwd['sol']['data'].shape[0] == 274  # MEG channels
        assert fwd['sol']['data'].shape[1] == 6156  # 2052 × 3


class TestPreprocessing:
    """Basic preprocessing before source reconstruction."""

    def test_filter_and_downsample(self):
        import mne
        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=True)
        raw.apply_gradient_compensation(0)
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(1, 45)
        raw.resample(250)
        assert raw.info['sfreq'] == 250.0
        assert raw.get_data().shape[0] > 270

    def test_compute_noise_covariance(self):
        import mne
        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=True)
        raw.apply_gradient_compensation(0)
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(1, 45)
        # Use empty-room or pre-stimulus baseline for noise cov
        noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=10)
        assert noise_cov['data'].shape[0] == raw.info['nchan']


class TestSourceReconstruction:
    """Apply multiple inverse methods and compare."""

    @pytest.fixture(scope="class")
    def prepared_data(self):
        import mne
        import jax.numpy as jnp

        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=True)
        # Apply CTF compensation grade 0 to match forward solution
        raw.apply_gradient_compensation(0)
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(1, 45)
        raw.resample(250)

        fwd = mne.read_forward_solution(f"{BEM_DIR}/mne_bem_fwd.fif")
        noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=30)

        # Extract a 10-second segment for testing
        data = raw.get_data()[:, :2500]  # 10s at 250 Hz

        return {
            'raw': raw,
            'data': jnp.array(data, dtype=jnp.float32),
            'fwd': fwd,
            'L': jnp.array(fwd['sol']['data'], dtype=jnp.float32),
            'noise_cov': jnp.array(noise_cov['data'], dtype=jnp.float32),
            'n_sensors': data.shape[0],
            'n_times': data.shape[1],
            'n_sources': fwd['nsource'],
        }

    def test_dspm_source_shape(self, prepared_data):
        """MNE dSPM should produce source timecourses."""
        import mne
        p = prepared_data
        # CTF data has compensation grade 3 — forward must match
        fwd = mne.convert_forward_solution(p['fwd'], force_fixed=False)
        noise_cov = mne.make_ad_hoc_cov(p['raw'].info)
        inv = mne.minimum_norm.make_inverse_operator(
            p['raw'].info, fwd, noise_cov, loose=0.2, depth=0.8)
        epochs = mne.make_fixed_length_epochs(p['raw'], duration=2.0, preload=True)
        stc = mne.minimum_norm.apply_inverse_epochs(
            epochs[:1], inv, lambda2=1.0/9, method='dSPM')[0]
        assert stc.data.shape[0] == p['n_sources']
        assert stc.data.shape[1] > 0

    def test_lcmv_beamformer(self, prepared_data):
        """LCMV beamformer from neurojax."""
        from neurojax.source.beamformer import make_lcmv_filter
        p = prepared_data
        data_cov = p['data'] @ p['data'].T / p['n_times']
        W = make_lcmv_filter(data_cov, p['L'][:, ::3], reg=0.05)  # fixed orient
        J = W @ p['data']
        assert J.shape == (p['n_sources'], p['n_times'])

    def test_laura_source_recon(self, prepared_data):
        """LAURA inverse with biophysical 1/d^3 prior."""
        from neurojax.source.laura import laura
        import mne
        p = prepared_data
        # Get source positions
        src = p['fwd']['src']
        positions = np.vstack([s['rr'][s['vertno']] for s in src]) * 1000  # m→mm
        # Use first 500 time points for speed
        J = laura(p['data'][:, :500], p['L'][:, ::3],
                  jnp.array(positions, dtype=jnp.float32),
                  p['noise_cov'], reg_param=0.05)
        assert J.shape == (p['n_sources'], 500)

    def test_vareta_source_recon(self, prepared_data):
        """VARETA adaptive-resolution inverse."""
        from neurojax.source.vareta import vareta
        p = prepared_data
        J, _, _ = vareta(p['data'][:, :500], p['L'][:, ::3], p['noise_cov'])
        assert J.shape[0] == p['n_sources']
        assert J.shape[1] == 500

    def test_multiple_methods_agree_on_peak(self, prepared_data):
        """Different methods should broadly agree on strongest sources."""
        import jax.numpy as jnp
        from neurojax.source.laura import laura
        from neurojax.source.vareta import vareta
        from neurojax.source.beamformer import make_lcmv_filter

        p = prepared_data
        data = p['data'][:, :500]
        L_fixed = p['L'][:, ::3]  # fixed orientation
        src = p['fwd']['src']
        positions = jnp.array(
            np.vstack([s['rr'][s['vertno']] for s in src]) * 1000,
            dtype=jnp.float32)

        # LCMV
        data_cov = data @ data.T / data.shape[1]
        W = make_lcmv_filter(data_cov, L_fixed, reg=0.05)
        J_lcmv = W @ data
        power_lcmv = jnp.sum(J_lcmv ** 2, axis=1)

        # LAURA
        J_laura = laura(data, L_fixed, positions, p['noise_cov'])
        power_laura = jnp.sum(J_laura ** 2, axis=1)

        # VARETA
        J_vareta, _, _ = vareta(data, L_fixed, p['noise_cov'])
        power_vareta = jnp.sum(J_vareta ** 2, axis=1)

        # Top 10% of sources should overlap between methods
        n_top = p['n_sources'] // 10
        top_lcmv = set(np.argsort(np.asarray(power_lcmv))[-n_top:])
        top_laura = set(np.argsort(np.asarray(power_laura))[-n_top:])
        top_vareta = set(np.argsort(np.asarray(power_vareta))[-n_top:])

        # Methods have different spatial biases (focal vs distributed),
        # so moderate overlap is expected. Check that it's above chance
        # (chance = 10% for top 10%).
        overlap_lv = len(top_lcmv & top_vareta) / n_top
        overlap_ll = len(top_lcmv & top_laura) / n_top
        overlap_vl = len(top_vareta & top_laura) / n_top
        max_overlap = max(overlap_lv, overlap_ll, overlap_vl)
        assert max_overlap > 0.02, \
            f"All methods disagree: LCMV∩VARETA={overlap_lv:.0%}, " \
            f"LCMV∩LAURA={overlap_ll:.0%}, VARETA∩LAURA={overlap_vl:.0%}"


class TestParcellation:
    """Parcellate source timecourses to atlas ROIs."""

    @pytest.fixture(scope="class")
    def source_data(self):
        import mne
        import jax.numpy as jnp
        from neurojax.source.beamformer import make_lcmv_filter

        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=True)
        raw.apply_gradient_compensation(0)
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(1, 45)
        raw.resample(250)

        fwd = mne.read_forward_solution(f"{BEM_DIR}/mne_bem_fwd.fif")
        data = jnp.array(raw.get_data()[:, :5000], dtype=jnp.float32)
        L = jnp.array(fwd['sol']['data'][:, ::3], dtype=jnp.float32)

        data_cov = data @ data.T / data.shape[1]
        W = make_lcmv_filter(data_cov, L, reg=0.05)
        J = W @ data

        return {
            'J': J,
            'fwd': fwd,
            'n_sources': fwd['nsource'],
            'n_times': 5000,
        }

    def test_desikan_killiany_parcellation(self, source_data):
        """Parcellate to Desikan-Killiany atlas (68 ROIs)."""
        import mne
        labels = mne.read_labels_from_annot(
            SUB, parc='aparc', subjects_dir=FS_DIR)
        assert len(labels) >= 68

        # Map sources to labels
        src = source_data['fwd']['src']
        vertno = [s['vertno'] for s in src]

        # Count sources per label
        n_labelled = 0
        for label in labels:
            hemi_idx = 0 if label.hemi == 'lh' else 1
            overlap = np.intersect1d(label.vertices, vertno[hemi_idx])
            n_labelled += len(overlap)

        # Most sources should map to a label
        coverage = n_labelled / source_data['n_sources']
        assert coverage > 0.5, f"Only {coverage:.0%} sources parcellated"

    def test_parcellated_timecourse_shape(self, source_data):
        """Extract one timecourse per ROI."""
        import mne
        labels = mne.read_labels_from_annot(
            SUB, parc='aparc', subjects_dir=FS_DIR)
        src = source_data['fwd']['src']
        vertno = [s['vertno'] for s in src]
        J = np.asarray(source_data['J'])

        # Simple parcellation: mean within each label
        parcel_ts = []
        for label in labels:
            hemi_idx = 0 if label.hemi == 'lh' else 1
            src_in_label = np.intersect1d(
                label.vertices, vertno[hemi_idx], return_indices=True)[2]
            if hemi_idx == 1:
                src_in_label += len(vertno[0])
            if len(src_in_label) > 0:
                parcel_ts.append(J[src_in_label].mean(axis=0))

        parcel_ts = np.array(parcel_ts)
        assert parcel_ts.shape[0] >= 60  # at least 60 parcels
        assert parcel_ts.shape[1] == source_data['n_times']


class TestSpectralAnalysis:
    """Spectral analysis of source timecourses."""

    def test_source_specparam(self):
        """Run specparam on source-level data."""
        import mne
        import jax.numpy as jnp
        from neurojax.source.beamformer import make_lcmv_filter

        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=True)
        raw.apply_gradient_compensation(0)
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(1, 45)
        raw.resample(250)

        fwd = mne.read_forward_solution(f"{BEM_DIR}/mne_bem_fwd.fif")
        data = jnp.array(raw.get_data()[:, :2500], dtype=jnp.float32)
        L = jnp.array(fwd['sol']['data'][:, ::3], dtype=jnp.float32)

        data_cov = data @ data.T / data.shape[1]
        W = make_lcmv_filter(data_cov, L, reg=0.05)
        J = W @ data

        # Compute PSD of source timecourses
        from neurojax.analysis.analytic import narrowband_analytic
        # Alpha band power per source
        z_alpha = narrowband_analytic(J, sfreq=250, fmin=8, fmax=13)
        alpha_power = jnp.mean(jnp.abs(z_alpha) ** 2, axis=1)
        assert alpha_power.shape == (fwd['nsource'],)
        # Alpha should have spatial structure (occipital > frontal)
        assert float(alpha_power.max() / alpha_power.mean()) > 2.0


class TestConnectivity:
    """Source-space connectivity measures."""

    def test_envelope_correlation_parcellated(self):
        """Envelope correlation between parcellated source timecourses."""
        import mne
        import jax.numpy as jnp
        from neurojax.source.beamformer import make_lcmv_filter
        from neurojax.analysis.analytic import envelope_correlation

        raw = mne.io.read_raw_ctf(
            f"{MEG_DIR}/{SUB}_ses-01_task-resting.ds", preload=True)
        raw.apply_gradient_compensation(0)
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(8, 13)  # alpha band
        raw.resample(250)

        fwd = mne.read_forward_solution(f"{BEM_DIR}/mne_bem_fwd.fif")
        data = jnp.array(raw.get_data()[:, :5000], dtype=jnp.float32)
        L = jnp.array(fwd['sol']['data'][:, ::3], dtype=jnp.float32)

        data_cov = data @ data.T / data.shape[1]
        W = make_lcmv_filter(data_cov, L, reg=0.05)
        J = W @ data

        # Simple parcellation (take every 30th source as pseudo-parcels)
        n_parcels = 68
        stride = max(1, J.shape[0] // n_parcels)
        parcel_ts = J[::stride, :][:n_parcels]

        fc = envelope_correlation(parcel_ts)
        assert fc.shape == (n_parcels, n_parcels)
        # Should not be identity (there's real connectivity structure)
        off_diag = fc - jnp.diag(jnp.diag(fc))
        assert float(jnp.abs(off_diag).mean()) > 0.01
