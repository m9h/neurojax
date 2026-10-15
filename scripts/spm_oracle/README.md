# Live-SPM oracle for the DCM-CSD port

`generate_cmc_csd_oracle.py` runs SPM's `spm_dcm_neural_priors`, `spm_L_priors`,
`spm_ssr_priors`, `spm_dcm_x_neural` and `spm_csd_mtf` for the single- and
four-node LFP architectures of SPM 26's DCM M/EEG test 6, at the prior mean and
at a perturbed parameter set, and saves `y`, `P`, `Hz`, `X` and the eigenspectra
to a MAT file. The committed copy is `tests/data/dcm/cmc_csd_spm25_oracle.mat`
(SPM 25.01 via spm-python 25.1.2, MATLAB Runtime R2025b, 2026-10-15).

To regenerate you need spm-python with the MATLAB Runtime. On macOS compiled
SPM must run under the runtime's `mwpython`, and the runtime must be started
with `-mlstartup -nojvm,-nodisplay` or the process exits silently at the first
compiled call (with or without a system JDK). `mwpython_launch.py` does that;
edit its paths, then:

    python mwpython_launch.py generate_cmc_csd_oracle.py /path/to/oracle.mat

The compiled endpoint cannot return cells of complex N-D arrays, which is why
the generator runs the whole computation as evaluated MATLAB code and saves a
file rather than returning values to Python.
