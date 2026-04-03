# Head Modeling and Forward Solutions

This tutorial covers the full pipeline from anatomical MRI to leadfield
matrix, including multi-tissue segmentation, surface extraction,
conductivity assignment, and BEM/FEM forward solutions.

## Why the leadfield matters

Every source imaging method depends on the leadfield matrix **L** which
maps neural current density to sensor measurements: **Y = LJ + n**.
The accuracy of **L** determines the upper bound on source localisation
accuracy. We showed that for MEG:

| Factor | Impact on leadfield | Effort |
|--------|-------------------|--------|
| Inner skull surface geometry | **Dominant** | High |
| Skull conductivity (for EEG) | Moderate (3x sensitivity) | Medium |
| BEM mesh resolution (ico-4 vs 5) | <0.1% | Low |
| BEM solver (MNE vs OpenMEEG) | <0.01% | Low |

## Step 1: CHARM 60-tissue segmentation

The SAMSEG+CHARM atlas (FreeSurfer 8.2) segments the head into 60
tissue types including differentiated skull layers:

```python
from neurojax.geometry.charm import run_charm_segmentation, segmentation_summary

seg_path = run_charm_segmentation(
    t1_path="freesurfer/sub-001/mri/nu.mgz",
    output_dir="derivatives/samseg-charm/sub-001",
    threads=4
)

# Inspect tissue volumes
labels, affine = load_segmentation(seg_path)
summary = segmentation_summary(labels)
for lid, info in sorted(summary.items(), key=lambda x: -x[1]['voxels'])[:10]:
    print(f"  {info['name']:30s} {info['voxels']:>8,} voxels  σ={info['conductivity_Sm']:.3f} S/m")
```

Key tissues for head modeling:

| Label | Tissue | σ (S/m) | Role |
|-------|--------|---------|------|
| 915 | Cortical bone | 0.006 | High-resistance skull layer |
| 916 | Cancellous bone | 0.025 | 4x more conductive than cortical |
| 911 | Skin/scalp | 0.465 | Outer boundary |
| 24 | CSF | 1.654 | Low-resistance brain buffer |
| 3, 42 | Cerebral cortex | 0.276 | Source space |
| 2, 41 | White matter | 0.126 | Anisotropic (from DTI) |

## Step 2: Surface extraction

Extract nested tissue boundaries using marching cubes:

```python
from neurojax.geometry.charm import extract_tissue_surfaces

surfaces = extract_tissue_surfaces(labels, affine)
# Returns: inner_skull, outer_skull, outer_skin (+ optional CSF, cancellous)

for name, (verts, faces) in surfaces.items():
    print(f"  {name}: {len(verts):,} vertices, {len(faces):,} faces")
```

## Step 3: Conductivity from qMRI

Standard approaches use population-average conductivities. With
quantitative MRI data (T1, BPF from QMT), we can derive
subject-specific skull conductivity:

```python
from neurojax.geometry.charm import assign_conductivities

# Population-average conductivities
sigma_standard = assign_conductivities(labels)

# qMRI-informed: BPF modulates bone conductivity
# Higher BPF = more macromolecular = denser bone = lower σ
# Limitation: conventional MRI can't directly measure bone T1
# (T2* ~0.5ms, signal decays before readout). UTE/ZTE needed
# for direct measurement. BPF is a proxy via partial volume.
```

**Important limitation:** Conventional MRI cannot directly image cortical
bone (T2* < 1 ms). The BPF values in skull voxels are dominated by
partial volume with marrow and soft tissue. True bone conductivity
mapping requires ultrashort echo time (UTE) or zero echo time (ZTE)
sequences. Our BPF-informed model is a useful approximation, not a
direct measurement.

## Step 4: BEM forward solution

Three BEM solver options, all producing nearly identical MEG leadfields:

```python
import mne

# Option A: MNE linear collocation (fastest)
surfs = mne.make_bem_model(subject, subjects_dir=fs_dir,
                            conductivity=(0.3, 0.006, 0.3), ico=4)
bem = mne.make_bem_solution(surfs)

# Option B: OpenMEEG symmetric BEM (more accurate for EEG)
bem_om = mne.make_bem_solution(surfs, solver='openmeeg')

# Option C: qMRI-informed skull conductivity
sigma_skull_eff = 0.019  # from BPF model
surfs_qmri = mne.make_bem_model(subject, subjects_dir=fs_dir,
                                 conductivity=(0.3, sigma_skull_eff, 0.465))
bem_qmri = mne.make_bem_solution(surfs_qmri)

# Forward solution
src = mne.setup_source_space(subject, spacing='oct5', subjects_dir=fs_dir)
fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem)
L = fwd['sol']['data']  # (n_sensors, n_sources * 3)
```

## Step 5: Differentiable FEM forward (advanced)

For end-to-end optimisation of conductivity, use the pure-JAX FEM:

```python
from neurojax.geometry.fem_forward import (
    assemble_stiffness, solve_forward, dipole_rhs, sigma_from_qmri
)
import jax

# Differentiable conductivity from qMRI features
sigma = sigma_from_qmri(t1_values, bpf_values, tissue_labels, params)

# FEM stiffness matrix (differentiable w.r.t. sigma)
K = assemble_stiffness(vertices, elements, sigma)

# Solve for a dipole source
f = dipole_rhs(vertices, elements, dipole_pos, dipole_moment)
phi = solve_forward(vertices, elements, sigma, f)

# Gradients flow end-to-end
def loss(params):
    sigma = sigma_from_qmri(t1, bpf, labels, params)
    K = assemble_stiffness(verts, elems, sigma)
    phi = jnp.linalg.solve(K + 1e-10 * jnp.eye(K.shape[0]), f)
    return jnp.sum((phi[sensor_idx] - measured) ** 2)

grad_params = jax.grad(loss)(params)  # optimise conductivity model
```

**Memory warning:** Dense stiffness matrix needs N^2 x 8 bytes.
For 10K nodes = 800 MB, 50K nodes = 20 GB. Use the DGX for
production-scale head models. The code checks available memory
before allocation and raises `MemoryError` if insufficient.

## Comparing leadfield variants

We computed five leadfield variants for sub-08033 and found:

| Variant | Change | Mean column r |
|---------|--------|--------------|
| ico-4 to ico-5 | 4x mesh resolution | 0.9999 |
| σ_skull 0.006 to 0.01 | 67% conductivity | 0.9998 |
| MNE to OpenMEEG | Solver formulation | 0.9999 |
| σ_skull 0.006 to 0.019 (qMRI) | 3.2x conductivity | 0.9976 |

For MEG, all factors produce <0.3% difference except qMRI-derived
conductivity, which matters for deep/basal sources (min r = 0.50).
For EEG, skull conductivity is 10-30x more important.

## Next steps

- See `tests/test_phantom_benchmark.py` for cross-method validation
- See `src/neurojax/geometry/charm.py` for the full CHARM wrapper API
- See `src/neurojax/geometry/fem_forward.py` for the differentiable FEM
- For JAX-FEM + PETSc on large meshes, use the DGX
