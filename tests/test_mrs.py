"""Tests for neurojax.qmri.mrs and neurojax.qmri.mrs_tensor — MRS processing.

TDD: tests define expected behaviour *before* the implementation exists.
All tests are expected to be RED (ImportError / failing) until the
corresponding modules are implemented.

Covers:
    - Phase correction (0th and 1st order)
    - Frequency alignment across transients
    - Water removal via HLSVD
    - Multi-coil combination (sensitivity-weighted, Tucker/PARAFAC)
    - Multiway tensor decompositions for MRS data
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


# =====================================================================
# Fixtures — reusable synthetic MRS data
# =====================================================================

@pytest.fixture
def rng():
    """Deterministic NumPy RNG for reproducible synthetic data."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sampling_params():
    """Standard MRS acquisition parameters.

    Returns a dict with dwell_time (s), num_points, bandwidth (Hz),
    and centre_freq (Hz, 3T proton Larmor).
    """
    n_pts = 2048
    bw = 2000.0  # Hz
    dwell = 1.0 / bw
    return {
        "n_pts": n_pts,
        "bandwidth": bw,
        "dwell_time": dwell,
        "centre_freq": 127.7e6,  # 3T 1H Larmor
    }


@pytest.fixture
def time_axis(sampling_params):
    """Time vector (seconds) for a synthetic FID."""
    n = sampling_params["n_pts"]
    dt = sampling_params["dwell_time"]
    return np.arange(n) * dt


@pytest.fixture
def freq_axis(sampling_params):
    """Frequency axis (Hz) centred on zero, for spectra of length n_pts."""
    n = sampling_params["n_pts"]
    dt = sampling_params["dwell_time"]
    return np.fft.fftshift(np.fft.fftfreq(n, d=dt))


@pytest.fixture
def synthetic_naa_fid(time_axis, sampling_params):
    """Single-metabolite FID: NAA singlet at 2.02 ppm.

    Returns complex128 FID array.
    The transmitter is assumed on-resonance with water (4.7 ppm), so
    the NAA offset is (4.7 - 2.02) * cf_mhz Hz.
    """
    cf_mhz = sampling_params["centre_freq"] / 1e6
    naa_offset_hz = (4.7 - 2.02) * cf_mhz  # Hz offset from transmitter
    t2_star = 0.15  # 150 ms T2*
    amplitude = 100.0
    t = time_axis
    fid = amplitude * np.exp(2j * np.pi * naa_offset_hz * t) * np.exp(-t / t2_star)
    return fid.astype(np.complex128)


@pytest.fixture
def synthetic_water_naa_fid(time_axis, sampling_params):
    """FID with dominant water peak at 4.7 ppm and small NAA at 2.02 ppm.

    Water amplitude is 1000x NAA to mimic in-vivo unsuppressed acquisition.
    Transmitter is on-resonance with water (4.7 ppm), so water is at 0 Hz
    and NAA is offset by (4.7 - 2.02) * cf_mhz Hz.
    Returns (fid, naa_only_fid) so tests can compare before/after water removal.
    """
    cf_mhz = sampling_params["centre_freq"] / 1e6
    t = time_axis

    water_hz = 0.0  # on-resonance with transmitter
    naa_hz = (4.7 - 2.02) * cf_mhz  # Hz offset from transmitter
    t2_water = 0.05
    t2_naa = 0.15

    water = 10000.0 * np.exp(2j * np.pi * water_hz * t) * np.exp(-t / t2_water)
    naa = 10.0 * np.exp(2j * np.pi * naa_hz * t) * np.exp(-t / t2_naa)
    return (water + naa).astype(np.complex128), naa.astype(np.complex128)


@pytest.fixture
def multicoil_fid(synthetic_naa_fid, rng):
    """Simulated 32-channel coil data: (n_pts, 32).

    Each coil sees the same FID scaled by a complex sensitivity and
    corrupted by Gaussian noise.
    """
    n_coils = 32
    n_pts = len(synthetic_naa_fid)
    sensitivities = rng.normal(size=(n_coils,)) + 1j * rng.normal(size=(n_coils,))
    sensitivities /= np.abs(sensitivities).max()  # normalise
    # shape: (n_pts, n_coils)
    data = synthetic_naa_fid[:, None] * sensitivities[None, :]
    noise_sigma = 0.5
    noise = noise_sigma * (rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape))
    return (data + noise).astype(np.complex128), sensitivities


@pytest.fixture
def multicoil_multitransient_tensor(synthetic_naa_fid, rng):
    """3-way MRS tensor: (time_points, coils, transients) = (2064, 32, 64).

    Suitable for Tucker/PARAFAC coil combination tests.
    Note: uses 2064 time-points (padded from 2048) to match the shape
    specification in the requirements.
    """
    n_pts_padded = 2064
    n_coils = 32
    n_transients = 64
    fid_padded = np.zeros(n_pts_padded, dtype=np.complex128)
    fid_padded[: len(synthetic_naa_fid)] = synthetic_naa_fid

    sensitivities = rng.normal(size=(n_coils,)) + 1j * rng.normal(size=(n_coils,))
    transient_weights = 1.0 + 0.05 * rng.normal(size=(n_transients,))

    tensor = np.einsum("t,c,r->tcr", fid_padded, sensitivities, transient_weights)
    noise = 0.3 * (rng.normal(size=tensor.shape) + 1j * rng.normal(size=tensor.shape))
    return (tensor + noise).astype(np.complex128)


# =====================================================================
# 1. Phase correction
# =====================================================================

class TestPhaseCorrection:
    """Verify automatic zero- and first-order phase correction."""

    def test_zero_order_corrects_inversion(self, synthetic_naa_fid, time_axis):
        """Multiply FID by -1 (180-deg phase shift); auto_phase_correct_0th
        should flip the spectrum back so the real part is predominantly
        positive."""
        from neurojax.qmri.mrs import auto_phase_correct_0th

        inverted_fid = -1.0 * synthetic_naa_fid
        corrected_fid, phi0 = auto_phase_correct_0th(inverted_fid)

        spectrum = np.fft.fftshift(np.fft.fft(corrected_fid))
        peak_idx = np.argmax(np.abs(spectrum))
        assert spectrum.real[peak_idx] > 0, (
            "After 0th-order correction of an inverted FID, the dominant "
            "spectral peak should have positive real part"
        )

    def test_first_order_removes_linear_phase(self, synthetic_naa_fid, freq_axis, sampling_params):
        """Apply a known linear phase gradient exp(i * alpha * freq) to a
        spectrum, then verify auto_phase_correct_1st removes it so that
        the corrected spectrum's real part matches the original."""
        from neurojax.qmri.mrs import auto_phase_correct_1st

        dwell = sampling_params["dwell_time"]
        cf_mhz = sampling_params["centre_freq"] / 1e6

        from neurojax.qmri.mrs import ppm_axis, fid_to_spectrum

        # Build reference spectrum via fid_to_spectrum (with line broadening)
        spectrum_orig = fid_to_spectrum(synthetic_naa_fid, dwell)
        ppm = ppm_axis(len(synthetic_naa_fid), dwell, cf_mhz=cf_mhz)

        alpha = 0.005  # rad / Hz — first-order phase slope
        phase_ramp = np.exp(1j * alpha * freq_axis)
        distorted_spectrum = spectrum_orig * phase_ramp

        # Convert back to FID for the correction function
        distorted_fid = np.fft.ifft(np.fft.ifftshift(distorted_spectrum))
        distorted_spec = fid_to_spectrum(distorted_fid, dwell)
        corrected_spec, phi0, phi1 = auto_phase_correct_1st(distorted_spec, ppm)

        # Compare real parts around the peak region
        peak = np.argmax(np.abs(spectrum_orig))
        window = slice(max(0, peak - 20), peak + 20)
        correlation = np.abs(np.corrcoef(
            spectrum_orig.real[window], corrected_spec.real[window]
        )[0, 1])
        assert correlation > 0.90, (
            f"Correlation between original and phase-corrected real spectrum "
            f"should exceed 0.90, got {correlation:.3f}"
        )

    def test_phase_output_finite(self, synthetic_naa_fid):
        """Phase values returned by auto_phase_correct_0th must be finite."""
        from neurojax.qmri.mrs import auto_phase_correct_0th

        _, phi0 = auto_phase_correct_0th(synthetic_naa_fid)
        assert np.isfinite(phi0), f"Phase value is not finite: {phi0}"
        assert isinstance(float(phi0), float), "Phase should be convertible to float"


# =====================================================================
# 2. Frequency alignment
# =====================================================================

class TestFrequencyAlignment:
    """Verify that frequency_align corrects inter-transient drift."""

    @pytest.fixture
    def shifted_fid_pair(self, time_axis, sampling_params):
        """Two FIDs with NAA peaks at slightly different frequencies.

        FID-A: NAA at 2.02 ppm (offset from water transmitter).
        FID-B: NAA at 2.02 ppm + 3 Hz drift.
        """
        cf_mhz = sampling_params["centre_freq"] / 1e6
        naa_hz = (4.7 - 2.02) * cf_mhz  # Hz offset from transmitter
        drift_hz = 3.0
        t2 = 0.15
        amp = 100.0
        t = time_axis

        fid_a = amp * np.exp(2j * np.pi * naa_hz * t) * np.exp(-t / t2)
        fid_b = amp * np.exp(2j * np.pi * (naa_hz + drift_hz) * t) * np.exp(-t / t2)
        return fid_a.astype(np.complex128), fid_b.astype(np.complex128)

    def test_align_shifts_to_reference(self, shifted_fid_pair, sampling_params):
        """After alignment, peak positions of the two FIDs should agree
        to within 0.5 Hz."""
        from neurojax.qmri.mrs import frequency_align

        fid_a, fid_b = shifted_fid_pair
        dt = sampling_params["dwell_time"]
        cf_mhz = sampling_params["centre_freq"] / 1e6
        # frequency_align expects (n_spectral, n_averages)
        fids = np.stack([fid_a, fid_b], axis=-1)  # (n_pts, 2)

        aligned = frequency_align(fids, dwell=dt, cf_mhz=cf_mhz)

        n = sampling_params["n_pts"]
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dt))

        spec_a = np.fft.fftshift(np.fft.fft(aligned[:, 0]))
        spec_b = np.fft.fftshift(np.fft.fft(aligned[:, 1]))
        peak_a = freqs[np.argmax(np.abs(spec_a))]
        peak_b = freqs[np.argmax(np.abs(spec_b))]

        assert abs(peak_a - peak_b) < 0.5, (
            f"Aligned peak positions should be within 0.5 Hz, "
            f"got delta = {abs(peak_a - peak_b):.2f} Hz"
        )

    def test_aligned_correlation_higher(self, shifted_fid_pair, sampling_params):
        """Cross-correlation magnitude between aligned FIDs should exceed
        that of the original (drifted) pair."""
        from neurojax.qmri.mrs import frequency_align

        fid_a, fid_b = shifted_fid_pair
        dt = sampling_params["dwell_time"]
        cf_mhz = sampling_params["centre_freq"] / 1e6
        # frequency_align expects (n_spectral, n_averages)
        fids = np.stack([fid_a, fid_b], axis=-1)  # (n_pts, 2)

        aligned = frequency_align(fids, dwell=dt, cf_mhz=cf_mhz)

        def _xcorr_peak(x, y):
            """Max abs cross-correlation between two signals."""
            cc = np.abs(np.correlate(x, y, mode="full"))
            return cc.max()

        corr_before = _xcorr_peak(fid_a, fid_b)
        corr_after = _xcorr_peak(aligned[:, 0], aligned[:, 1])

        assert corr_after >= corr_before * 0.99, (
            f"Aligned correlation ({corr_after:.2f}) should be at least as "
            f"high as pre-alignment ({corr_before:.2f})"
        )


# =====================================================================
# 3. Water removal
# =====================================================================

class TestWaterRemoval:
    """Verify HLSVD-based water suppression."""

    def test_hlsvd_removes_water(self, synthetic_water_naa_fid, sampling_params):
        """After HLSVD water removal, the water peak power should drop by
        more than 80 % relative to the original."""
        from neurojax.qmri.mrs import hlsvd_water_removal

        fid_with_water, _ = synthetic_water_naa_fid
        dt = sampling_params["dwell_time"]
        cf = sampling_params["centre_freq"]

        cleaned_fid = hlsvd_water_removal(
            fid_with_water, dwell=dt, cf_mhz=cf / 1e6,
            water_range_ppm=(4.5, 5.0),
        )

        # Measure water peak power before and after
        n = len(fid_with_water)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
        water_hz = 0.0  # water is on-resonance with transmitter
        water_mask = np.abs(freqs - water_hz) < 30.0  # +/- 30 Hz around water

        spec_before = np.fft.fftshift(np.fft.fft(fid_with_water))
        spec_after = np.fft.fftshift(np.fft.fft(cleaned_fid))

        power_before = np.sum(np.abs(spec_before[water_mask]) ** 2)
        power_after = np.sum(np.abs(spec_after[water_mask]) ** 2)

        suppression = 1.0 - (power_after / power_before)
        assert suppression > 0.80, (
            f"Water suppression should exceed 80 %, got {suppression * 100:.1f} %"
        )

    def test_metabolites_preserved(self, synthetic_water_naa_fid, sampling_params):
        """After water removal, NAA peak amplitude should remain within 20 %
        of the water-free reference."""
        from neurojax.qmri.mrs import hlsvd_water_removal

        fid_with_water, naa_only_fid = synthetic_water_naa_fid
        dt = sampling_params["dwell_time"]
        cf = sampling_params["centre_freq"]

        cleaned_fid = hlsvd_water_removal(
            fid_with_water, dwell=dt, cf_mhz=cf / 1e6,
            water_range_ppm=(4.5, 5.0),
        )

        cf_mhz = cf / 1e6
        n = len(naa_only_fid)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
        naa_hz = (4.7 - 2.02) * cf_mhz  # Hz offset from transmitter
        naa_mask = np.abs(freqs - naa_hz) < 20.0  # +/- 20 Hz around NAA

        spec_ref = np.fft.fftshift(np.fft.fft(naa_only_fid))
        spec_cleaned = np.fft.fftshift(np.fft.fft(cleaned_fid))

        amp_ref = np.max(np.abs(spec_ref[naa_mask]))
        amp_cleaned = np.max(np.abs(spec_cleaned[naa_mask]))

        relative_error = abs(amp_cleaned - amp_ref) / amp_ref
        assert relative_error < 0.20, (
            f"NAA amplitude after water removal should be within 20 % of "
            f"reference, got {relative_error * 100:.1f} % error"
        )


# =====================================================================
# 4. Coil combination
# =====================================================================

class TestCoilCombination:
    """Verify multi-coil combination methods."""

    def test_sensitivity_weighted_snr(self, multicoil_fid):
        """SNR of the sensitivity-weighted combined signal should exceed
        the SNR of the best individual coil."""
        from neurojax.qmri.mrs import sensitivity_weighted_combine

        data, sensitivities = multicoil_fid  # (n_pts, n_coils)
        # sensitivity_weighted_combine expects (n_spectral, n_channels, n_averages)
        data_3d = data[:, :, np.newaxis]  # (n_pts, n_coils, 1)
        combined = sensitivity_weighted_combine(data_3d)  # (n_pts, 1)
        combined = combined[:, 0]  # squeeze averages dim

        def _estimate_snr(signal):
            """SNR = max(|spectrum|) / std(noise tail)."""
            spec = np.abs(np.fft.fft(signal))
            # Use upper 25 % of spectrum as noise region
            noise_region = spec[int(0.75 * len(spec)):]
            return np.max(spec) / (np.std(noise_region) + 1e-12)

        combined_snr = _estimate_snr(combined)

        best_single = max(
            _estimate_snr(data[:, c]) for c in range(data.shape[1])
        )
        assert combined_snr > best_single, (
            f"Combined SNR ({combined_snr:.1f}) should exceed best single "
            f"coil SNR ({best_single:.1f})"
        )

    def test_parafac_shape(self, multicoil_multitransient_tensor):
        """tucker_coil_combine on (2064, 32, 64) should return (2064, 64),
        collapsing the coil dimension."""
        from neurojax.qmri.mrs import tucker_coil_combine

        tensor = multicoil_multitransient_tensor  # (2064, 32, 64)
        assert tensor.shape == (2064, 32, 64), (
            f"Input fixture shape mismatch: {tensor.shape}"
        )

        combined = tucker_coil_combine(tensor)
        assert combined.shape == (2064, 64), (
            f"Expected (2064, 64) after coil combination, got {combined.shape}"
        )

    def test_output_is_complex(self, multicoil_fid):
        """Combined coil data must remain complex-valued."""
        from neurojax.qmri.mrs import sensitivity_weighted_combine

        data, _ = multicoil_fid
        # sensitivity_weighted_combine expects (n_spectral, n_channels, n_averages)
        data_3d = data[:, :, np.newaxis]
        combined = sensitivity_weighted_combine(data_3d)
        assert np.iscomplexobj(combined), (
            f"Combined signal should be complex, got dtype {combined.dtype}"
        )


# =====================================================================
# 5. Multiway MRS tensor decompositions
# =====================================================================

class TestMultiwayMRS:
    """Verify Tucker and PARAFAC decompositions on MRS-shaped tensors."""

    def test_tucker_decomposition_shapes(self):
        """Decompose a (100, 8, 16) tensor with ranks (10, 4, 8).

        Verify that the core tensor and each factor matrix have the
        correct shapes.
        """
        from neurojax.qmri.mrs_tensor import mrs_tucker_decomposition

        key = jax.random.PRNGKey(7)
        tensor = jax.random.normal(key, (100, 8, 16)) + 0j  # complex

        ranks = (10, 4, 8)
        result = mrs_tucker_decomposition(np.array(tensor), ranks=ranks)
        core = result["core"]
        factors = result["factors"]

        assert core.shape == ranks, (
            f"Core shape should be {ranks}, got {core.shape}"
        )
        # Factor k has shape (original_dim_k, rank_k)
        expected_factor_shapes = [(100, 10), (8, 4), (16, 8)]
        for k, (factor, expected) in enumerate(zip(factors, expected_factor_shapes)):
            assert factor.shape == expected, (
                f"Factor {k} shape should be {expected}, got {factor.shape}"
            )

    def test_parafac_recovers_components(self, rng):
        """Create a synthetic rank-3 tensor and verify that PARAFAC
        identifies 3 components (i.e., the reconstruction error with
        rank=3 is substantially lower than with rank=1)."""
        from neurojax.qmri.mrs_tensor import mrs_parafac

        # Build a rank-3 tensor: T = sum_r a_r (x) b_r (x) c_r
        n_components = 3
        shape = (50, 10, 20)
        factors_true = [
            rng.normal(size=(shape[k], n_components))
            for k in range(3)
        ]
        tensor = np.zeros(shape, dtype=np.float64)
        for r in range(n_components):
            tensor += np.einsum(
                "i,j,k->ijk",
                factors_true[0][:, r],
                factors_true[1][:, r],
                factors_true[2][:, r],
            )
        # Add mild noise
        noise = 0.01 * rng.normal(size=shape)
        tensor += noise

        # Decompose with correct rank
        result_3 = mrs_parafac(
            tensor, n_components=n_components,
        )
        weights_3 = result_3["weights"]
        factors_3 = result_3["factors"]

        # Decompose with rank-1
        result_1 = mrs_parafac(
            tensor, n_components=1,
        )
        weights_1 = result_1["weights"]
        factors_1 = result_1["factors"]

        # Reconstruct both and compare error
        def _reconstruct(weights, factors):
            """Rebuild tensor from CP factors."""
            recon = np.zeros(shape, dtype=np.float64)
            for r in range(len(weights)):
                recon = recon + weights[r] * np.einsum(
                    "i,j,k->ijk",
                    factors[0][:, r],
                    factors[1][:, r],
                    factors[2][:, r],
                )
            return recon

        err_3 = np.linalg.norm(tensor - _reconstruct(weights_3, factors_3))
        err_1 = np.linalg.norm(tensor - _reconstruct(weights_1, factors_1))

        assert float(err_3) < float(err_1) * 0.5, (
            f"Rank-3 error ({float(err_3):.4f}) should be substantially "
            f"smaller than rank-1 error ({float(err_1):.4f})"
        )
        assert len(weights_3) == n_components, (
            f"Expected {n_components} components, got {len(weights_3)}"
        )
