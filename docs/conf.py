# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------
project = 'neurojax'
copyright = '2026, NeuroJAX Contributors'
author = 'NeuroJAX Contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',          # Markdown support
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
]

# -- Mock imports for ReadTheDocs ------------------------------------------------
# These packages contain compiled C/CUDA extensions or heavy native dependencies
# that are not available in the ReadTheDocs build environment. Mocking them
# allows autodoc to introspect the Python source without actually importing them.
autodoc_mock_imports = [
    # JAX ecosystem
    "jax",
    "jaxlib",
    "equinox",
    "diffrax",
    "optax",
    "distrax",
    "lineax",
    "optimistix",
    "jaxtyping",
    "jraph",
    "jaxctrl",
    "chex",
    "signax",
    "jinns",
    "scico",
    "tensorstore",
    "vbjax",
    # Numerical / scientific
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "h5py",
    # Neuroimaging
    "mne",
    "nibabel",
    "nilearn",
    "pynwb",
    "specparam",
    "surfplot",
    "brainspace",
    # Plotting
    "matplotlib",
    "seaborn",
    # Data access
    "dandi",
    "fsspec",
    "s3fs",
    "openneuro",
    "pyxdf",
    # Misc
    "requests",
    "tqdm",
    "click",
    "anthropic",
    # Bench extras
    "cma",
    "llamea",
]

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = "NeuroJAX Documentation"

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "tasklist",
]
