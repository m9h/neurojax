"""SCI Head Model dataset loader (Warner et al. 2019).

Downloads and loads the high-resolution tetrahedral head model from the
Scientific Computing and Imaging (SCI) Institute, University of Utah.

The dataset provides:
- Multi-tissue tetrahedral mesh (8 tissue types, up to 60M elements)
- 128 and 256-channel EEG electrode positions
- T1/T2/DWI/DTI/fMRI imaging data
- Segmentation labels from Seg3D

Reference:
    Warner A, Tate J, Burton B, Johnson CR (2019).
    A high-resolution head and brain computer model for forward and
    inverse EEG simulation. bioRxiv 552190.

Dataset: https://www.sci.utah.edu/sci-head-model/
"""

import os
import logging
import zipfile
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np

logger = logging.getLogger(__name__)

# ── Dataset URLs ──────────────────────────────────────────────────────
BASE_URL = "https://sci.utah.edu/~datasets/SCI_headmodel"
COMPONENTS = {
    "mesh": f"{BASE_URL}/Mesh.zip",
    "segmentation": f"{BASE_URL}/Segmentation.zip",
    "eeg": f"{BASE_URL}/EEG.zip",
    "t1": f"{BASE_URL}/T1.zip",
    "t2": f"{BASE_URL}/T2.zip",
    "dwi": f"{BASE_URL}/DWI.zip",
    "dti": f"{BASE_URL}/DTI.zip",
    "fmri": f"{BASE_URL}/fMRI.zip",
    "pseudoct": f"{BASE_URL}/Pseudo-CT.zip",
    "simulations": f"{BASE_URL}/Simulations.zip",
}

# ── Tissue labels and conductivities (S/m) ────────────────────────────
# From Warner et al. 2019 and Gabriel et al. 1996 literature values
SCI_TISSUE = {
    0: ("Air", 0.0),
    1: ("Scalp", 0.43),
    2: ("Skull", 0.006),
    3: ("Sinus", 0.0),       # air-filled cavities
    4: ("CSF", 1.79),
    5: ("Gray_Matter", 0.33),
    6: ("White_Matter", 0.14),
    7: ("Eyes", 1.5),
}


class HeadMesh(NamedTuple):
    """Tetrahedral head mesh with tissue labels and electrode positions."""
    vertices: np.ndarray       # (n_vertices, 3) node coordinates in mm
    elements: np.ndarray       # (n_elements, 4) tetrahedral connectivity
    tissue_labels: np.ndarray  # (n_elements,) tissue label per element
    conductivity: np.ndarray   # (n_elements,) conductivity in S/m
    electrode_pos: np.ndarray  # (n_electrodes, 3) electrode positions in mm
    electrode_nodes: np.ndarray  # (n_electrodes,) nearest node index per electrode
    tissue_map: Dict[int, str]   # {label_id: tissue_name}


# ── Download ──────────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {pct:3d}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)


def download_sci_head_model(
    dest_dir: str,
    components: Sequence[str] = ("mesh", "segmentation", "eeg"),
    force: bool = False,
) -> Path:
    """Download SCI head model components.

    Args:
        dest_dir: directory to store downloaded data
        components: which components to download
            (default: mesh, segmentation, eeg)
        force: re-download even if files exist

    Returns:
        Path to the dataset directory
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for comp in components:
        if comp not in COMPONENTS:
            raise ValueError(
                f"Unknown component '{comp}'. "
                f"Available: {list(COMPONENTS.keys())}"
            )
        url = COMPONENTS[comp]
        zip_path = dest / f"{comp}.zip"
        extract_dir = dest / comp

        if extract_dir.exists() and not force:
            logger.info(f"  {comp}: already exists at {extract_dir}")
            continue

        # Download
        logger.info(f"  Downloading {comp} from {url}")
        print(f"Downloading {comp}...")
        urlretrieve(url, str(zip_path), reporthook=_progress_hook)
        print()  # newline after progress

        # Extract
        logger.info(f"  Extracting {comp}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(dest))

        # Clean up zip
        zip_path.unlink()
        logger.info(f"  {comp}: extracted to {dest}")

    return dest


# ── Mesh loading ──────────────────────────────────────────────────────

def _load_vtk_mesh(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load tetrahedral mesh from VTK file via meshio."""
    import meshio
    mesh = meshio.read(path)
    vertices = mesh.points.astype(np.float64)

    # Extract tetrahedral cells
    tets = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
            break

    if tets is None:
        raise ValueError(f"No tetrahedral cells found in {path}")

    # Extract tissue labels from cell data
    labels = np.zeros(len(tets), dtype=np.int32)
    for key in ("material", "MaterialID", "mat_id", "tissue",
                "gmsh:physical", "medit:ref"):
        if key in mesh.cell_data:
            for i, cell_block in enumerate(mesh.cells):
                if cell_block.type == "tetra":
                    labels = np.asarray(
                        mesh.cell_data[key][i], dtype=np.int32
                    )
                    break
            break

    return vertices, tets.astype(np.int64), labels


def _load_tetgen(prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TetGen .node/.ele files.

    TetGen .node format:
        n_points  dim  n_attr  n_boundary_markers
        idx  x  y  z  [attributes...]  [boundary_marker]

    TetGen .ele format:
        n_tets  nodes_per_tet  n_attributes
        idx  n0  n1  n2  n3  [attribute]
    """
    node_file = prefix + ".node"
    ele_file = prefix + ".ele"

    # Parse .node
    with open(node_file) as f:
        header = f.readline().split()
        n_pts = int(header[0])
        vertices = np.zeros((n_pts, 3), dtype=np.float64)
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            idx = int(parts[0])
            vertices[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]

    # Parse .ele
    with open(ele_file) as f:
        header = f.readline().split()
        n_tets = int(header[0])
        n_attr = int(header[2]) if len(header) > 2 else 0
        elements = np.zeros((n_tets, 4), dtype=np.int64)
        labels = np.zeros(n_tets, dtype=np.int32)
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            idx = int(parts[0])
            elements[idx] = [int(parts[1]), int(parts[2]),
                             int(parts[3]), int(parts[4])]
            if n_attr > 0 and len(parts) > 5:
                labels[idx] = int(float(parts[5]))

    return vertices, elements, labels


def _load_scirun_pts_elem(
    pts_path: str, elem_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SCIRun .pts/.elem format.

    .pts:  n_points\\n  x y z\\n  ...
    .elem: n_elems\\n  n0 n1 n2 n3 [label]\\n  ...
    """
    # Parse .pts
    with open(pts_path) as f:
        n_pts = int(f.readline().strip())
        vertices = np.zeros((n_pts, 3), dtype=np.float64)
        for i in range(n_pts):
            parts = f.readline().strip().split()
            vertices[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    # Parse .elem
    with open(elem_path) as f:
        n_elems = int(f.readline().strip())
        elements = np.zeros((n_elems, 4), dtype=np.int64)
        labels = np.zeros(n_elems, dtype=np.int32)
        for i in range(n_elems):
            parts = f.readline().strip().split()
            elements[i] = [int(parts[0]), int(parts[1]),
                           int(parts[2]), int(parts[3])]
            if len(parts) > 4:
                labels[i] = int(parts[4])

    return vertices, elements, labels


def _find_mesh_files(data_dir: Path) -> Tuple[str, str]:
    """Auto-detect mesh format in a directory.

    Returns:
        (format, path_or_prefix) tuple
    """
    data_dir = Path(data_dir)

    # Look recursively
    for root, dirs, files in os.walk(data_dir):
        root = Path(root)
        for f in files:
            fpath = root / f
            # VTK
            if f.endswith((".vtk", ".vtu", ".vtp")):
                return "vtk", str(fpath)
            # meshio-compatible
            if f.endswith(".msh"):
                return "vtk", str(fpath)  # meshio handles gmsh too

        # TetGen pair
        node_files = [f for f in files if f.endswith(".node")]
        for nf in node_files:
            prefix = str(root / nf[:-5])
            if os.path.exists(prefix + ".ele"):
                return "tetgen", prefix

        # SCIRun pair
        pts_files = [f for f in files if f.endswith(".pts")]
        for pf in pts_files:
            base = str(root / pf[:-4])
            for ext in (".elem", ".tet"):
                if os.path.exists(base + ext):
                    return "scirun", base

    raise FileNotFoundError(
        f"No mesh files found in {data_dir}. "
        "Expected .vtk, .node/.ele (TetGen), or .pts/.elem (SCIRun)."
    )


def load_mesh(
    data_dir: str,
    mesh_file: Optional[str] = None,
    conductivity_map: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a tetrahedral head mesh from the SCI dataset.

    Args:
        data_dir: path to the dataset (or mesh subdirectory)
        mesh_file: explicit path to mesh file (auto-detected if None)
        conductivity_map: {tissue_label: sigma_Sm} override

    Returns:
        (vertices, elements, tissue_labels, conductivity)
    """
    if mesh_file:
        ext = Path(mesh_file).suffix
        if ext in (".vtk", ".vtu", ".vtp", ".msh"):
            vertices, elements, labels = _load_vtk_mesh(mesh_file)
        elif ext == ".node":
            vertices, elements, labels = _load_tetgen(mesh_file[:-5])
        elif ext == ".pts":
            elem_path = mesh_file[:-4] + ".elem"
            vertices, elements, labels = _load_scirun_pts_elem(
                mesh_file, elem_path
            )
        else:
            # Try meshio as fallback
            vertices, elements, labels = _load_vtk_mesh(mesh_file)
    else:
        fmt, path = _find_mesh_files(Path(data_dir))
        if fmt == "vtk":
            vertices, elements, labels = _load_vtk_mesh(path)
        elif fmt == "tetgen":
            vertices, elements, labels = _load_tetgen(path)
        elif fmt == "scirun":
            vertices, elements, labels = _load_scirun_pts_elem(
                path + ".pts", path + ".elem"
            )

    # Assign conductivities
    cmap = conductivity_map or {k: v[1] for k, v in SCI_TISSUE.items()}
    conductivity = np.zeros(len(elements), dtype=np.float64)
    for label_id, sigma in cmap.items():
        conductivity[labels == label_id] = sigma

    logger.info(
        f"Loaded mesh: {len(vertices)} nodes, {len(elements)} tets, "
        f"{len(np.unique(labels))} tissues"
    )
    return vertices, elements, labels, conductivity


# ── Electrode loading ─────────────────────────────────────────────────

def load_electrodes(
    data_dir: str,
    n_channels: int = 128,
) -> np.ndarray:
    """Load electrode positions from the SCI dataset.

    Args:
        data_dir: path to the dataset or EEG subdirectory
        n_channels: 128 or 256

    Returns:
        (n_channels, 3) electrode positions in mm
    """
    data_dir = Path(data_dir)

    # Search for electrode files (various formats)
    patterns = [
        f"*{n_channels}*electrode*",
        f"*{n_channels}*sensor*",
        f"*electrode*{n_channels}*",
        "*electrode*.txt",
        "*electrode*.csv",
        "*sensor*.txt",
        "*.elc",
        "*.elp",
        "*.sfp",
    ]

    elec_file = None
    for pattern in patterns:
        matches = list(data_dir.rglob(pattern))
        if matches:
            elec_file = matches[0]
            break

    if elec_file is None:
        raise FileNotFoundError(
            f"No electrode file found for {n_channels} channels in {data_dir}"
        )

    # Try to parse as whitespace-delimited xyz coordinates
    positions = []
    with open(elec_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            # Find 3 consecutive floats (may have a label prefix)
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    floats = []
            if len(floats) >= 3:
                positions.append(floats[-3:])

    if not positions:
        raise ValueError(f"Could not parse electrode positions from {elec_file}")

    positions = np.array(positions, dtype=np.float64)
    logger.info(f"Loaded {len(positions)} electrodes from {elec_file}")
    return positions


def snap_electrodes_to_surface(
    electrode_pos: np.ndarray,
    vertices: np.ndarray,
    elements: np.ndarray,
    tissue_labels: np.ndarray,
    scalp_label: int = 1,
) -> np.ndarray:
    """Find nearest mesh node to each electrode on the scalp surface.

    Args:
        electrode_pos: (n_electrodes, 3) electrode positions
        vertices: (n_vertices, 3) mesh node coordinates
        elements: (n_elements, 4) tetrahedral connectivity
        tissue_labels: (n_elements,) tissue label per element
        scalp_label: label ID for scalp tissue (default: 1)

    Returns:
        (n_electrodes,) indices of nearest scalp nodes
    """
    # Find nodes belonging to scalp elements
    scalp_elems = elements[tissue_labels == scalp_label]
    scalp_nodes = np.unique(scalp_elems.ravel())

    if len(scalp_nodes) == 0:
        # Fallback: use boundary nodes (nodes on the convex hull surface)
        logger.warning("No scalp-labeled elements found, using all nodes")
        scalp_nodes = np.arange(len(vertices))

    scalp_verts = vertices[scalp_nodes]  # (n_scalp, 3)

    # Find nearest scalp node for each electrode
    electrode_nodes = np.zeros(len(electrode_pos), dtype=np.int64)
    for i, pos in enumerate(electrode_pos):
        dists = np.sum((scalp_verts - pos) ** 2, axis=1)
        electrode_nodes[i] = scalp_nodes[np.argmin(dists)]

    return electrode_nodes


# ── Full dataset loader ───────────────────────────────────────────────

def load_sci_head_model(
    data_dir: str,
    n_channels: int = 128,
    conductivity_map: Optional[Dict[int, float]] = None,
) -> HeadMesh:
    """Load the complete SCI head model: mesh + electrodes + conductivities.

    Args:
        data_dir: path to the downloaded SCI dataset
        n_channels: 128 or 256 electrode configuration
        conductivity_map: optional conductivity override

    Returns:
        HeadMesh named tuple with all components
    """
    data_dir = Path(data_dir)

    # Load mesh
    vertices, elements, labels, sigma = load_mesh(
        str(data_dir), conductivity_map=conductivity_map
    )

    # Load electrodes
    try:
        electrode_pos = load_electrodes(str(data_dir), n_channels)
    except FileNotFoundError:
        logger.warning(
            f"Electrode file not found. Generating {n_channels} "
            "electrodes on scalp surface."
        )
        electrode_pos = _generate_electrode_positions(
            vertices, elements, labels, n_channels
        )

    # Snap electrodes to mesh
    electrode_nodes = snap_electrodes_to_surface(
        electrode_pos, vertices, elements, labels
    )

    tissue_map = {k: v[0] for k, v in SCI_TISSUE.items()}

    return HeadMesh(
        vertices=vertices,
        elements=elements,
        tissue_labels=labels,
        conductivity=sigma,
        electrode_pos=electrode_pos,
        electrode_nodes=electrode_nodes,
        tissue_map=tissue_map,
    )


def _generate_electrode_positions(
    vertices: np.ndarray,
    elements: np.ndarray,
    tissue_labels: np.ndarray,
    n_electrodes: int,
    scalp_label: int = 1,
) -> np.ndarray:
    """Generate approximately uniform electrode positions on scalp.

    Used as fallback when electrode file is not available.
    Distributes electrodes on the upper hemisphere of the scalp surface.
    """
    # Get scalp nodes
    scalp_elems = elements[tissue_labels == scalp_label]
    if len(scalp_elems) == 0:
        scalp_nodes = np.arange(len(vertices))
    else:
        scalp_nodes = np.unique(scalp_elems.ravel())

    scalp_verts = vertices[scalp_nodes]

    # Centre on centroid
    centroid = scalp_verts.mean(axis=0)
    relative = scalp_verts - centroid

    # Keep upper hemisphere (z > median)
    z_median = np.median(relative[:, 2])
    upper = scalp_verts[relative[:, 2] > z_median]

    if len(upper) < n_electrodes:
        upper = scalp_verts

    # Farthest-point sampling for approximate uniformity
    selected = [0]
    dists = np.full(len(upper), np.inf)
    for _ in range(n_electrodes - 1):
        d = np.sum((upper - upper[selected[-1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected.append(np.argmax(dists))

    return upper[selected]


# ── Synthetic head model for testing ──────────────────────────────────

def make_spherical_head(
    radii: Tuple[float, ...] = (80.0, 86.0, 92.0, 95.0),
    conductivities: Tuple[float, ...] = (0.33, 1.79, 0.006, 0.43),
    tissue_names: Tuple[str, ...] = ("Brain", "CSF", "Skull", "Scalp"),
    n_electrodes: int = 32,
    mesh_density: float = 8.0,
) -> HeadMesh:
    """Create a concentric-sphere head model for testing.

    Generates a simple tetrahedral mesh of nested spheres with known
    conductivities. Useful for validating EIT algorithms against
    analytic solutions.

    Args:
        radii: shell radii in mm (innermost to outermost)
        conductivities: conductivity per shell in S/m
        tissue_names: name per shell
        n_electrodes: number of electrodes on outer surface
        mesh_density: approximate element edge length in mm

    Returns:
        HeadMesh with spherical geometry
    """
    from scipy.spatial import Delaunay

    # Generate points in the volume with small random perturbation
    # to avoid degenerate tetrahedra from regular grids
    r_max = radii[-1]
    n_per_axis = int(2 * r_max / mesh_density) + 1
    lin = np.linspace(-r_max, r_max, n_per_axis)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Add small random jitter to break grid degeneracy
    rng = np.random.RandomState(42)
    jitter = mesh_density * 0.05  # 5% of element size
    pts += rng.uniform(-jitter, jitter, pts.shape)

    # Keep only points inside outermost sphere
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    mask = r <= r_max * 1.01
    vertices = pts[mask]

    # Delaunay tetrahedralization
    tri = Delaunay(vertices)
    elements = tri.simplices

    # Assign tissue labels based on element centroid radius
    centroids = np.mean(vertices[elements], axis=1)  # (n_elem, 3)
    r_cent = np.sqrt(np.sum(centroids ** 2, axis=1))

    labels = np.zeros(len(elements), dtype=np.int32)
    sigma = np.zeros(len(elements), dtype=np.float64)
    for i, (r_in, r_out) in enumerate(
        zip([0.0] + list(radii[:-1]), radii)
    ):
        mask = (r_cent >= r_in) & (r_cent < r_out)
        labels[mask] = i
        sigma[mask] = conductivities[i]

    # Electrodes on outer surface (approximately uniform)
    # Use golden spiral for even distribution
    electrode_pos = np.zeros((n_electrodes, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_electrodes):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_electrodes)
        phi = 2 * np.pi * i / golden_ratio
        electrode_pos[i] = r_max * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

    # Snap to nearest mesh node
    electrode_nodes = np.zeros(n_electrodes, dtype=np.int64)
    for i, pos in enumerate(electrode_pos):
        dists = np.sum((vertices - pos) ** 2, axis=1)
        electrode_nodes[i] = np.argmin(dists)
        electrode_pos[i] = vertices[electrode_nodes[i]]

    tissue_map = {i: name for i, name in enumerate(tissue_names)}

    return HeadMesh(
        vertices=vertices,
        elements=elements.astype(np.int64),
        tissue_labels=labels,
        conductivity=sigma,
        electrode_pos=electrode_pos,
        electrode_nodes=electrode_nodes,
        tissue_map=tissue_map,
    )
