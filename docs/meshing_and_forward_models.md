# Meshing Backends and Forward Model Ecosystem

A guide to the mesh generation, surface format, and forward solver
ecosystem that neurojax draws from and interoperates with. Understanding
this landscape is essential because the same head geometry must flow
through segmentation, meshing, forward modeling, and source imaging ---
often crossing tool boundaries.

## Surface and mesh formats

| Format | Type | Tools | Notes |
|--------|------|-------|-------|
| **FreeSurfer binary** | Triangular surface | FreeSurfer, MNE | Big-endian binary, `.pial`, `.white`, `.sphere` |
| **GIFTI** (`.gii`) | Surface + data | HCP, Connectome Workbench, FreeSurfer | XML+binary, per-vertex data overlays, standard for cortical surfaces |
| **CIFTI** (`.dtseries.nii`, `.dconn.nii`) | Combined surface + volume | HCP, Workbench | Grayordinates: cortical surface vertices + subcortical voxels in one file |
| **VTK** (`.vtk`) | Surface or volume mesh | FSL betsurf, ParaView, PyVista | Legacy ASCII or XML, triangles or tetrahedra |
| **STL** | Triangular surface | 3D printing, mesh tools | No connectivity metadata |
| **OFF** | Triangular surface | FSL BET, iso2mesh | Simple vertex + face list |
| **INRIA** (`.inr`, `.mesh`) | Volume image / tet mesh | CGAL, Fijee, INRIA tools | Used by Fijee for FEM mesh generation |
| **Gmsh** (`.msh`) | Tet/hex mesh | Gmsh, JAX-FEM | Scriptable, tagged physical groups for tissue labels |
| **meshio** | Any of the above | Python (meshio library) | Universal converter between formats |
| **jraph GraphsTuple** | Graph | neurojax, DeepMind | JAX-native graph for GNN message-passing |

**GIFTI/CIFTI** are the modern standard for cortical data. The HCP
pipelines produce CIFTI grayordinates combining ~32K surface vertices
per hemisphere with ~30K subcortical voxels, enabling unified
surface+volume analysis. FreeSurfer 7+ and MNE both read/write GIFTI.
neurojax should target CIFTI-compatible output for interoperability
with Connectome Workbench and HCP tools.

## Mesh generation engines

Before discussing the tools, it helps to understand the underlying mesh
generators they depend on. Most neuroimaging tools do not implement
meshing from scratch --- they wrap one of a handful of computational
geometry libraries:

| Engine | Type | License | Used by |
|--------|------|---------|---------|
| **CGAL** (Computational Geometry Algorithms Library) | Delaunay tet, surface meshing, mesh optimization | GPL/LGPL | Fijee, iso2mesh (optional), FreeSurfer (internal) |
| **Gmsh** | Structured/unstructured tet/hex, scripted | GPL | SimNIBS, JAX-FEM, DUNEuro (optional) |
| **TetGen** | Constrained Delaunay tetrahedralization | AGPLv3 | iso2mesh (primary), brain2mesh, EIDORS |
| **NETGEN** | Advancing-front tet meshing | LGPL | EIDORS, NGSolve, DUNEuro (optional) |
| **Triangle** (Shewchuk) | 2D constrained Delaunay | Free (non-commercial) | iso2mesh (surface), many tools |
| **FreeSurfer deformable mesh** | Active surface model (not Delaunay) | Custom | FreeSurfer recon-all (cortical surfaces) |
| **Marching cubes** | Isosurface from voxel grid | Public domain | scikit-image, VTK, neurojax (CHARM surface extraction) |
| **MNE icosahedron decimation** | Recursive subdivision of icosahedron | BSD | MNE (BEM surface decimation, source space) |
| **DUNE grid** | Multi-resolution adaptive grid | GPL | DUNEuro |

**Key distinction:** Surface meshing (triangles on boundaries) vs volume
meshing (tetrahedra filling the interior). BEM only needs surface meshes;
FEM needs volume meshes. Most volume meshers take surface meshes as input
(boundary-constrained tetrahedralization).

The practical consequence: your choice of forward solver constrains your
mesher. SimNIBS uses Gmsh because GetDP (its FEM solver) reads Gmsh
format natively. Fijee uses CGAL because FEniCS/dolfin imports CGAL
meshes via DOLFIN-XML. DUNEuro can use Gmsh, NETGEN, or structured
hex grids via DUNE's own grid manager. EIDORS defaults to NETGEN for
its built-in mesh generation.

## Tool-specific details

### FreeSurfer

FreeSurfer's `recon-all` produces the foundational cortical surfaces
(white, pial, inflated, sphere) at ~160K vertices per hemisphere via
a deformable mesh algorithm. The watershed BEM tool extracts nested
head surfaces (brain, inner skull, outer skull, scalp) for forward
modeling. SAMSEG with the CHARM atlas extends this to 60 tissue labels
including differentiated skull layers.

- **Mesher:** Custom deformable surface model (active contour on
  tessellated sphere) for cortical surfaces; watershed algorithm
  for BEM surfaces; icosahedron subdivision for decimation
- **Solver:** None (surface generation only; forward solving delegated to MNE)
- **Strengths:** Gold-standard cortical reconstruction, spherical
  registration, atlas parcellation, 30+ years of validation
- **Mesh type:** Triangular surfaces (no volume meshing)
- **neurojax integration:** `geometry/surface.py` (pure Python reader),
  `geometry/charm.py` (SAMSEG+CHARM wrapper)

### MNE-Python

MNE provides the BEM pipeline from FreeSurfer surfaces to leadfield
matrix. It handles decimation (ico-3 through ico-5), BEM matrix
assembly (linear collocation), and forward solution computation for
MEG/EEG. Also wraps OpenMEEG for symmetric BEM.

- **Mesher:** Icosahedron recursive subdivision (ico-3 to ico-5) for
  BEM surface decimation; octahedron subdivision for source spaces
- **Solver:** Custom linear collocation BEM (C, wrapped in Python);
  OpenMEEG symmetric BEM (optional backend)
- **Strengths:** Complete MEG/EEG forward pipeline, sensor handling,
  coordinate transforms, extensive validation
- **Mesh type:** Decimated triangular surfaces from FreeSurfer
- **neurojax integration:** Used for all current leadfield computation

### OpenMEEG

Symmetric BEM solver from the INRIA Rennes Athena/Odyssee team (Clerc,
Papadopoulo, Gramfort). More accurate than MNE's linear collocation
for EEG (accounts for skull conductivity jumps correctly). Supports
nested and non-nested geometries.

- **Mesher:** None (consumes triangular surfaces from any source)
- **Solver:** Symmetric BEM with Galerkin formulation (dense matrix,
  LU factorization); C++ with BLAS/LAPACK, Python bindings via SWIG
- **Strengths:** Mathematically rigorous symmetric formulation, EEG
  accuracy, same group that developed the subtraction method
- **Mesh type:** Triangular surfaces (from any source)
- **neurojax integration:** `openmeeg` Python bindings, also via
  MNE's `make_bem_solution(solver='openmeeg')`

### iso2mesh / brain2mesh

Qianqian Fang's MATLAB/Octave toolbox for tetrahedral mesh generation
from surface or volumetric data. `brain2mesh` specifically generates
multi-tissue head meshes from FreeSurfer segmentations. Uses CGAL or
TetGen for Delaunay tetrahedralization.

- **Mesher:** TetGen (primary) or CGAL for Delaunay tetrahedralization;
  Triangle (Shewchuk) for 2D surface remeshing; custom MATLAB/Octave
  wrappers. `brain2mesh` adds tissue-aware mesh generation from
  FreeSurfer segmentations with quality-controlled element sizes.
- **Solver:** None (mesh generation only; forward solving via MCX for
  photon transport, RedBird for DOT, or external FEM)
- **Strengths:** Robust tet meshing, multi-tissue support, integrates
  with MCX (Monte Carlo photon transport) and RedBird (DOT)
- **Mesh type:** Tetrahedral volume meshes
- **neurojax integration:** `brain2mesh_surfaces.mat` already computed
  for sub-08033; dot-jax uses iso2mesh for DOT forward models

### SimNIBS / CHARM-GEMS

SimNIBS provides the complete pipeline from T1w to electric field
simulation for TMS/tDCS. CHARM (via the GEMS engine shared with
FreeSurfer SAMSEG) segments 21+ tissue types. SimNIBS then generates
tetrahedral meshes and solves the Laplace equation using GetDP (a
general-purpose FEM solver).

- **Mesher:** Gmsh (Python API) for tetrahedral mesh generation from
  CHARM tissue surfaces; mesh quality optimization via Gmsh's built-in
  algorithms (Frontal-Delaunay, HXT)
- **Solver:** GetDP (General environment for the Treatment of Discrete
  Problems) --- a general-purpose FEM solver reading Gmsh format natively.
  Solves Laplace equation for tDCS/TMS electric field.
- **Strengths:** Complete TMS/tDCS pipeline, validated tissue
  segmentation, electric field optimization
- **Mesh type:** Tetrahedral (Gmsh-based)
- **neurojax integration:** `charm-gems` Python bindings at
  `/Users/mhough/dev/charm-gems/.venv`, SAMSEG+CHARM atlas in
  FreeSurfer 8.2

### DUNEuro

The Dune Neuroimaging project (Vorwerk, Wolters et al., Münster) is
the modern successor to CAUCHY/SimBio/NeuroFEM. It provides
high-order FEM (P1, P2, and higher) for EEG/MEG forward modeling
with full support for anisotropic conductivity from DTI. Uses the
DUNE (Distributed and Unified Numerics Environment) framework.

- **Mesher:** Flexible --- reads Gmsh, NETGEN, or uses DUNE's own
  `ALUGrid` / `YaspGrid` for structured hexahedral grids. The DUNE
  grid interface abstracts over multiple backends.
- **Solver:** DUNE ISTL (Iterative Solver Template Library) with
  algebraic multigrid (AMG) preconditioning; supports CG, BiCGSTAB,
  GMRES. Also interfaces with direct solvers (UMFPACK, SuperLU).
- **Strengths:** High-order elements (P1, P2, and higher --- better
  accuracy per DOF than P1), anisotropic conductivity, subtraction
  and partial integration approaches for dipole singularity, validated
  against analytical solutions. Actively maintained by the Wolters group.
- **Mesh type:** Tetrahedral or hexahedral, structured or unstructured
- **Continuity with CAUCHY:** DUNEuro is the direct descendant of the
  CAUCHY/SimBio/NeuroFEM lineage (1993+), now using modern DUNE
  numerics instead of custom C code
- **neurojax integration:** Not yet integrated; candidate for high-accuracy
  FEM validation benchmark

### David Holder's UCL Fast EIT tools

The Holder group at UCL developed fast EIT (Electrical Impedance
Tomography) software for real-time brain imaging during epilepsy and
stroke. Their tools include EIDORS (Electrical Impedance Tomography
and Diffuse Optical Tomography Reconstruction Software) and custom
fast EIT solvers optimised for the complete electrode model.

- **Mesher:** NETGEN (primary, for EIDORS built-in mesh generation);
  also supports Gmsh, Distmesh, and custom mesh import
- **Solver:** EIDORS uses its own MATLAB-based FEM solver with the
  complete electrode model (CEM); also interfaces with external
  solvers. Fast matrix assembly optimised for real-time reconstruction.
- **Strengths:** Real-time capable, complete electrode model, clinical
  validation (stroke detection via EIT), extensive forward model library
- **Mesh type:** Tetrahedral (EIDORS/NETGEN), hexahedral
- **Relevance:** The same Laplace equation (nabla . sigma nabla phi = -I)
  underlies both EIT and EEG forward modeling. EIT additionally recovers
  sigma from boundary measurements --- the inverse of what our qMRI
  conductivity module does (estimate sigma from tissue properties).
  The UCL stroke EIT dataset is loaded by sbi4dwi (`uclh_eit.py`).
- **neurojax integration:** EIT forward/inverse in sbi4dwi biophysics
  module; shared FEM core opportunity

### Fijee

The predecessor C++/FEniCS project (Cobigo, from Wendling's group at
LTSI/INSERM Rennes). Couples FEM forward modeling with Wendling/
Jansen-Rit neural mass dynamics for simulated EEG. Uses INRIA mesh
format, CGAL meshing, FEniCS/dolfin solver with UFL formulations.

- **Mesher:** CGAL 3D mesh generation from INRIA image format (`.inr`)
  volume labels; surface meshing via CGAL surface mesh. Mesh written
  in DOLFIN-XML for FEniCS import.
- **Solver:** FEniCS/dolfin with PETSc backend; UFL variational
  formulations compiled to C++ via FFC. Direct (MUMPS/LU) or
  iterative (CG + AMG) solvers.
- **Strengths:** Integrated forward + neural mass pipeline, tensor
  conductivity from DTI, tDCS complete electrode model, dipole
  subtraction method
- **Mesh type:** Tetrahedral (CGAL/INRIA format)
- **neurojax integration:** Architecture and physics ported to JAX;
  `fem_forward.py` implements the same Laplace equation

### JAX-FEM / neurojax differentiable FEM

The new addition: a differentiable FEM forward model in pure JAX
(`fem_forward.py`) and optionally backed by JAX-FEM + PETSc for
large-scale problems. The key innovation over all tools above is
**end-to-end differentiability** --- gradients flow from the loss
function back through source imaging, leadfield, FEM assembly,
and conductivity mapping into qMRI tissue parameters.

- **Mesher:** Consumes meshes from any source via meshio; Gmsh via
  JAX-FEM for generation; marching cubes (scikit-image) for surface
  extraction from CHARM segmentations
- **Solver:** Pure JAX conjugate gradient (small meshes, fully
  JIT-compiled) or PETSc via JAX-FEM (large meshes, AMG
  preconditioning, MPI parallel). Both are differentiable.
- **Strengths:** Fully differentiable, GPU-accelerated, end-to-end
  optimisation of conductivity, natural integration with neural
  networks (PI-GNN, PINNs)
- **Mesh type:** Tetrahedral P1 (pure JAX), or any element via JAX-FEM
- **Unique capability:** `sigma_from_qmri()` maps T1/BPF/tissue labels
  to conductivity with learnable parameters optimisable via `jax.grad`

## The meshing pipeline for multi-modal physics

The same head model geometry serves multiple physical simulations:

```
                        ┌─────────────────────┐
                        │   T1w + qMRI data    │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Segmentation                │
                    │  FreeSurfer / SAMSEG+CHARM   │
                    │  60 tissue labels            │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Surface meshes   │  │ Tetrahedral mesh │  │ Regular grid     │
    │ (marching cubes) │  │ (iso2mesh/Gmsh)  │  │ (rasterize)      │
    │                  │  │                  │  │                  │
    │ → BEM (MNE,      │  │ → FEM (JAX-FEM,  │  │ → k-space (jwave,│
    │   OpenMEEG)      │  │   DUNEuro, Fijee)│  │   acoustic sim)  │
    │                  │  │                  │  │                  │
    │ EEG/MEG leadfield│  │ EEG/MEG/EIT/tDCS │  │ TUS, DOT (MCX)  │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Source imaging / Inverse      │
                    │ 15 methods in neurojax        │
                    │ (all use the same leadfield L)│
                    └──────────────────────────────┘
```

**Key principle:** segment once, mesh for each physics, solve with the
appropriate method. The shared geometry and tissue properties (from
CHARM + qMRI) ensure consistency across modalities.

## Interoperability matrix

| From \ To | FreeSurfer | MNE | OpenMEEG | iso2mesh | SimNIBS | DUNEuro | Fijee | JAX-FEM | jwave |
|-----------|-----------|-----|----------|----------|---------|---------|-------|---------|-------|
| **FreeSurfer** | -- | read_surface | via MNE | surf2mesh | charm-gems | meshio | INRIA convert | meshio | rasterize |
| **GIFTI/CIFTI** | mris_convert | read | -- | -- | -- | -- | -- | nibabel | -- |
| **VTK** | -- | -- | file I/O | -- | -- | meshio | -- | meshio+PyVista | -- |
| **Gmsh** | -- | -- | -- | -- | native | meshio | -- | native | -- |
| **CHARM seg** | native | via FS | surfaces | label2mesh | native | meshio | -- | charm.py | mesh_rasterizer |

## Recommended reading

- Wolters et al. (2006) NeuroImage --- anisotropic FEM, CAUCHY/SimBio heritage
- Gramfort et al. (2010) BioMedical Engineering OnLine --- OpenMEEG
- Vorwerk et al. (2014) NeuroImage --- DUNEuro, FEM vs BEM comparison
- Hassan et al. (2014) PLOS ONE --- Fijee, EEG source connectivity
- Fang & Boas (2009) Biomedical Optics Express --- iso2mesh/brain2mesh
- Holder (2004) Electrical Impedance Tomography --- UCL EIT methods
- Saturnino et al. (2019) NeuroImage --- SimNIBS CHARM
- Warner et al. (2019) bioRxiv --- SCI head model
