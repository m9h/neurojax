"""MRS quality control report generation.

Generates a self-contained HTML QC report with inline base64 plots
for a processed MEGA-PRESS dataset. No external CSS/JS dependencies.
"""

import base64
import io
import numpy as np


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    return encoded


def _make_spectra_plot(result: dict) -> str:
    """Create a spectra plot (diff, edit-on, edit-off) and return base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(result['diff'])
    dwell = result['dwell_time']
    cf = result.get('centre_freq', 123.25e6)

    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    ppm = freq / (cf / 1e6) + 4.65

    diff_spec = np.fft.fftshift(np.fft.fft(np.asarray(result['diff'])))
    on_spec = np.fft.fftshift(np.fft.fft(np.asarray(result['edit_on'])))
    off_spec = np.fft.fftshift(np.fft.fft(np.asarray(result['edit_off'])))

    # Plot in a reasonable ppm range
    mask = (ppm >= 0.5) & (ppm <= 4.5)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(ppm[mask], np.real(on_spec[mask]), 'b-', linewidth=0.8)
    axes[0].set_title('Edit-ON Spectrum')
    axes[0].set_ylabel('Intensity')

    axes[1].plot(ppm[mask], np.real(off_spec[mask]), 'g-', linewidth=0.8)
    axes[1].set_title('Edit-OFF Spectrum')
    axes[1].set_ylabel('Intensity')

    axes[2].plot(ppm[mask], np.real(diff_spec[mask]), 'r-', linewidth=0.8)
    axes[2].set_title('Difference Spectrum (GABA)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_xlabel('Chemical Shift (ppm)')

    for ax in axes:
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle('MEGA-PRESS Spectra', fontsize=14)
    fig.tight_layout()

    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return encoded


def _make_alignment_plot(result: dict) -> str:
    """Create an alignment metrics plot and return base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freq_shifts = np.asarray(result['freq_shifts'])
    phase_shifts = np.asarray(result['phase_shifts'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    ax1.plot(freq_shifts, 'o-', markersize=3, linewidth=0.8)
    ax1.set_ylabel('Frequency Shift (Hz)')
    ax1.set_title('Per-Transient Frequency Shifts')
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.degrees(phase_shifts), 'o-', markersize=3, linewidth=0.8, color='orange')
    ax2.set_ylabel('Phase Shift (degrees)')
    ax2.set_xlabel('Transient index')
    ax2.set_title('Per-Transient Phase Shifts')
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return encoded


def generate_qc_report(
    result: dict,
    fitting_results: dict | None = None,
    title: str = "MRS Quality Control Report",
) -> str:
    """Generate an HTML QC report for a processed MRS dataset.

    Parameters
    ----------
    result : dict
        Processed MEGA-PRESS result with keys:
        diff, edit_on, edit_off, sum_spec, freq_shifts, phase_shifts,
        rejected, n_averages, dwell_time, bandwidth, centre_freq.
    fitting_results : dict, optional
        Metabolite fitting results, keyed by metabolite name, each value
        a dict with 'concentration_mM' and 'crlb_percent'.
    title : str
        Report title.

    Returns
    -------
    html : str
        Self-contained HTML report string.
    """
    # Generate plots
    spectra_b64 = _make_spectra_plot(result)
    alignment_b64 = _make_alignment_plot(result)

    # Compute alignment statistics
    freq_shifts = np.asarray(result['freq_shifts'])
    phase_shifts = np.asarray(result['phase_shifts'])
    rejected = np.asarray(result['rejected'])

    n_total = len(rejected)
    n_rejected = int(rejected.sum())
    reject_pct = 100.0 * n_rejected / n_total if n_total > 0 else 0.0

    freq_mean = float(np.mean(freq_shifts))
    freq_std = float(np.std(freq_shifts))
    freq_max = float(np.max(np.abs(freq_shifts)))

    phase_mean_deg = float(np.degrees(np.mean(phase_shifts)))
    phase_std_deg = float(np.degrees(np.std(phase_shifts)))

    # Build HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        f'  <title>{title}</title>',
        '  <meta charset="utf-8">',
        '  <style>',
        '    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }',
        '    h1 { color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }',
        '    h2 { color: #4a90d9; margin-top: 30px; }',
        '    table { border-collapse: collapse; width: 100%; margin: 10px 0; }',
        '    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }',
        '    th { background: #4a90d9; color: white; }',
        '    tr:nth-child(even) { background: #f2f2f2; }',
        '    .metric { display: inline-block; margin: 10px 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '    .metric-value { font-size: 24px; font-weight: bold; color: #333; }',
        '    .metric-label { font-size: 12px; color: #666; }',
        '    img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }',
        '  </style>',
        '</head>',
        '<body>',
        f'  <h1>{title}</h1>',
        '',
        '  <h2>Spectra</h2>',
        f'  <img src="data:image/png;base64,{spectra_b64}" alt="MEGA-PRESS spectra plot">',
        '',
        '  <h2>Alignment Metrics</h2>',
        f'  <img src="data:image/png;base64,{alignment_b64}" alt="Alignment metrics plot">',
        '  <div>',
        f'    <div class="metric"><div class="metric-value">{freq_mean:.2f} Hz</div><div class="metric-label">Mean frequency shift</div></div>',
        f'    <div class="metric"><div class="metric-value">{freq_std:.2f} Hz</div><div class="metric-label">Std frequency shift</div></div>',
        f'    <div class="metric"><div class="metric-value">{freq_max:.2f} Hz</div><div class="metric-label">Max |frequency shift|</div></div>',
        f'    <div class="metric"><div class="metric-value">{phase_mean_deg:.1f} deg</div><div class="metric-label">Mean phase shift</div></div>',
        f'    <div class="metric"><div class="metric-value">{phase_std_deg:.1f} deg</div><div class="metric-label">Std phase shift</div></div>',
        '  </div>',
        '',
        '  <h2>Outlier Rejection</h2>',
        f'  <p>Rejected transients: <strong>{n_rejected}</strong> out of {n_total} ({reject_pct:.1f}%)</p>',
        f'  <p>Averages used: <strong>{result["n_averages"]}</strong></p>',
    ]

    if n_rejected > 0:
        rejected_indices = np.where(rejected)[0].tolist()
        html_parts.append(f'  <p>Rejected indices: {rejected_indices}</p>')

    # Metabolite table
    if fitting_results is not None and len(fitting_results) > 0:
        html_parts.extend([
            '',
            '  <h2>Metabolite Concentrations</h2>',
            '  <table>',
            '    <thead>',
            '      <tr><th>Metabolite</th><th>Concentration (mM)</th><th>CRLB (%)</th></tr>',
            '    </thead>',
            '    <tbody>',
        ])
        for metab, vals in fitting_results.items():
            conc = vals.get('concentration_mM', 'N/A')
            crlb = vals.get('crlb_percent', 'N/A')
            conc_str = f'{conc:.1f}' if isinstance(conc, (int, float)) else str(conc)
            crlb_str = f'{crlb:.1f}' if isinstance(crlb, (int, float)) else str(crlb)
            html_parts.append(f'      <tr><td>{metab}</td><td>{conc_str}</td><td>{crlb_str}</td></tr>')
        html_parts.extend([
            '    </tbody>',
            '  </table>',
        ])

    # Acquisition summary
    html_parts.extend([
        '',
        '  <h2>Acquisition Summary</h2>',
        '  <table>',
        f'    <tr><td>Dwell time</td><td>{result["dwell_time"]*1e6:.1f} us</td></tr>',
        f'    <tr><td>Bandwidth</td><td>{result["bandwidth"]:.0f} Hz</td></tr>',
        f'    <tr><td>Spectral points</td><td>{len(result["diff"])}</td></tr>',
        f'    <tr><td>Total transients</td><td>{n_total}</td></tr>',
        '  </table>',
        '',
        '</body>',
        '</html>',
    ])

    return '\n'.join(html_parts)
