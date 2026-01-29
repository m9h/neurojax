"""
HTML Reporting Module for NeuroJAX.
Generates FSL-style browser reports with Data Quality, Dimensionality, and Spectral insights.
"""
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.decomposition import PCA
import mne

from neurojax.spectral import fit_spectrum, PowerSpectrumModel

class HTMLReport:
    """
    Generates a static HTML report.
    """
    def __init__(self, title="NeuroJAX Analysis Report"):
        self.title = title
        self.sections = []

    def add_section(self, title, description, figures):
        """
        Add a section with figures.
        figures: list of matplotlib Figures.
        """
        self.sections.append({
            "title": title,
            "description": description,
            "figures": figures
        })

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def save(self, filename="report.html"):
        """
        Render and save HTML.
        """
        html = [
            f"<html><head><title>{self.title}</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 40px; background: #f0f0f0; }",
            "h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }",
            "h2 { color: #666; margin-top: 30px; }",
            ".section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".figures { display: flex; flex-wrap: wrap; gap: 20px; }",
            ".fig-container { text-align: center; }",
            "img { max-width: 100%; border: 1px solid #ddd; }",
            "</style>",
            "</head><body>",
            f"<h1>{self.title}</h1>"
        ]

        for section in self.sections:
            html.append(f"<div class='section'>")
            html.append(f"<h2>{section['title']}</h2>")
            html.append(f"<p>{section['description']}</p>")
            html.append("<div class='figures'>")
            
            for i, fig in enumerate(section['figures']):
                b64 = self._fig_to_base64(fig)
                html.append(f"<div class='fig-container'>")
                html.append(f"<img src='data:image/png;base64,{b64}'>")
                html.append(f"</div>")
                
            html.append("</div>") # figures
            html.append("</div>") # section

        html.append("</body></html>")
        
        with open(filename, 'w') as f:
            f.write("\n".join(html))
        print(f"Report saved to {filename}")

# --- Plotting Components ---

def plot_quality_metrics(epochs: mne.Epochs):
    """
    Generate Data Quality plots: Global Field Power (GFP) stats.
    """
    # 1. Subject-average Variance Map (Sensitivity to artifacts)
    # 2. GFP over time
    
    figs = []
    
    # GFP
    data = epochs.get_data() # (n_epochs, n_sensors, n_times)
    gfp = np.std(data, axis=1).mean(axis=0) # Mean GFP across epochs
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    times = epochs.times
    ax1.plot(times, gfp, color='k', linewidth=1.5)
    ax1.set_title("Global Field Power (GFP) - Grand Average")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GFP (T)")
    ax1.grid(True, alpha=0.3)
    figs.append(fig1)
    
    # Sensor Variance Map (Standard Deviation across time/epochs)
    # Identify noisy channels
    std_map = np.std(data, axis=(0, 2))
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    mne.viz.plot_topomap(std_map, epochs.info, axes=ax2, show=False, cmap='viridis')
    ax2.set_title("Sensor Noise Level (Std Dev)")
    figs.append(fig2)
    
    return figs

def plot_dimensionality(data: np.ndarray):
    """
    Scree Plot and Cumulative Explained Variance (PCA).
    data: (n_samples, n_features)
    """
    # PCA
    n_components = min(50, min(data.shape))
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Scree
    ax1.plot(range(1, 11), pca.explained_variance_ratio_[:10] * 100, 'o-', color='navy')
    ax1.set_title("Scree Plot (First 10)")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.grid(True)
    
    # Cumulative
    ax2.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_) * 100, 'r-')
    ax2.axhline(95, color='k', linestyle='--', label='95%')
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.legend()
    ax2.grid(True)
    
    return [fig]

def plot_spectral_analysis(freqs, psd_data):
    """
    NeuroJAX Spectral Analysis (FOOOF-style).
    Decomposes spectrum into Periodic (Oscillations) and Aperiodic (1/f).
    
    freqs: (n_freqs,)
    psd_data: (n_freqs,) - mean PSD
    """
    # 1. Fit Model using neurojax.spectral
    params = fit_spectrum(jnp.array(freqs), jnp.array(psd_data), n_peaks=2)
    
    # 2. Generate Components
    model = PowerSpectrumModel()
    
    # Aperiodic: offset, exponent
    offset, exponent = params[0], params[1]
    aperiodic_fit = offset - exponent * jnp.log10(freqs)
    
    # Full Model
    full_fit = model(freqs, params)
    
    # Periodic (Difference)
    periodic_fit = full_fit - aperiodic_fit
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    ax.loglog(freqs, 10**psd_data, label='Raw Data', color='black', alpha=0.5, linewidth=2)
    
    # Full Fit
    ax.loglog(freqs, 10**full_fit, label='Full Fit (NeuroJAX)', color='red', linestyle='--')
    
    # Aperiodic
    ax.loglog(freqs, 10**aperiodic_fit, label=f'Aperiodic (1/f^{exponent:.2f})', color='blue', linestyle=':')
    
    # Periodic (re-added to baseline for visualization if peaks exist)
    # Or just flatten data
    
    ax.set_title("Spectral Decomposition (Periodic + Aperiodic)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    return [fig]
