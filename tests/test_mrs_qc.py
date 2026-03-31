"""Tests for MRS QC report generation.

Verifies that the HTML report contains required sections:
spectra plot, alignment metrics, rejection info, and metabolite table.
"""
import numpy as np
import pytest

from neurojax.analysis.mrs_qc import generate_qc_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_result():
    """Create a minimal dict that mimics a processed MEGA-PRESS result."""
    n = 2048
    dwell = 2.5e-4
    cf = 123.25e6

    rng = np.random.default_rng(42)
    diff = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    edit_on = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    edit_off = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    sum_spec = edit_on + edit_off

    n_transients = 64
    freq_shifts = rng.standard_normal(n_transients) * 2.0  # Hz
    phase_shifts = rng.standard_normal(n_transients) * 0.3  # rad
    rejected = np.zeros(n_transients, dtype=bool)
    rejected[5] = True
    rejected[12] = True
    rejected[40] = True  # 3 out of 64 rejected

    return {
        'diff': diff,
        'edit_on': edit_on,
        'edit_off': edit_off,
        'sum_spec': sum_spec,
        'freq_shifts': freq_shifts,
        'phase_shifts': phase_shifts,
        'rejected': rejected,
        'n_averages': n_transients - int(rejected.sum()),
        'dwell_time': dwell,
        'bandwidth': 1.0 / dwell,
        'centre_freq': cf,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQCReport:
    def test_qc_report_generates_html(self):
        """Output is valid HTML."""
        result = _make_dummy_result()
        html = generate_qc_report(result)
        assert isinstance(html, str)
        assert html.strip().startswith("<!DOCTYPE html>") or html.strip().startswith("<html")
        assert "</html>" in html

    def test_qc_report_contains_spectra_plot(self):
        """Report includes a spectrum figure as base64 inline image."""
        result = _make_dummy_result()
        html = generate_qc_report(result)
        # Should contain at least one base64-encoded PNG image
        assert "data:image/png;base64," in html
        # Should reference spectrum in some heading or alt text
        assert "spectr" in html.lower()  # "spectrum" or "spectra"

    def test_qc_report_contains_alignment_metrics(self):
        """Report includes frequency shift statistics."""
        result = _make_dummy_result()
        html = generate_qc_report(result)
        # Should contain freq shift info
        assert "frequency" in html.lower() or "freq" in html.lower()
        # Should have some numeric content showing shift stats
        assert "shift" in html.lower() or "drift" in html.lower() or "Hz" in html

    def test_qc_report_contains_rejection_info(self):
        """Report includes rejection count and percentage."""
        result = _make_dummy_result()
        html = generate_qc_report(result)
        # 3 out of 64 rejected = 4.69%
        assert "reject" in html.lower()
        # Should mention the count (3) somewhere
        assert "3" in html
        # Should show a percentage
        assert "%" in html

    def test_qc_report_contains_metabolite_table(self):
        """If fitting results provided, includes metabolite concentrations."""
        result = _make_dummy_result()
        fitting_results = {
            'GABA': {'concentration_mM': 1.8, 'crlb_percent': 12.0},
            'NAA': {'concentration_mM': 12.5, 'crlb_percent': 3.0},
            'Cr': {'concentration_mM': 8.1, 'crlb_percent': 5.0},
        }
        html = generate_qc_report(result, fitting_results=fitting_results)
        # Should contain a table with metabolite info
        assert "<table" in html.lower()
        assert "GABA" in html
        assert "NAA" in html
        assert "Cr" in html
        assert "1.8" in html  # GABA concentration
        assert "12.5" in html  # NAA concentration
