"""Source space analysis and Inverse Solvers.

Includes classical linear/iterative methods (MNE, VARETA, LAURA, beamformers,
CHAMPAGNE, HIGGS) and the PI-GNN deep learning source imaging module.
"""

from neurojax.source.graph_utils import (
    mesh_to_graph,
    adjacency_from_faces,
    graph_laplacian,
    compute_vertex_features,
    orientation_matrix,
)
from neurojax.source.source_gnn import (
    SourceGNN,
    GraphConvLayer,
    estimate_tikhonov_reg,
    tikhonov_inverse,
    truncated_svd_inverse,
    train_source_gnn,
)
