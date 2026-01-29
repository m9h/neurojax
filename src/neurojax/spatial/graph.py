"""
Graph Spatial Processing for EEG using Jraph.

Represents EEG Montage as a Graph (Nodes=Electrodes, Edges=Adjacency).
Implements Graph Laplacian Smoothing for spatial contiguity.
"""

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import mne
from scipy.spatial.distance import pdist, squareform

class EEGGraph:
    def __init__(self, montage_name="GSN-HydroCel-129", adjacency_dist=0.04):
        """
        Initialize EEG Graph from standard montage.
        
        Args:
            montage_name: MNE montage.
            adjacency_dist: Distance threshold (meters) for edges.
        """
        self.montage = mne.channels.make_standard_montage(montage_name)
        pos_dict = self.montage.get_positions()['ch_pos']
        # Convert to array, sorted by channel name or index?
        # MNE montages use channel names. We need a consistent ordering (e.g. E1, E2...).
        # Let's assume input data follows the montage channel list.
        
        self.ch_names = self.montage.ch_names
        self.points = np.array([pos_dict[ch] for ch in self.ch_names]) # (N, 3)
        
        # Build Adjacency
        self.dist_matrix = squareform(pdist(self.points))
        self.adj_matrix = (self.dist_matrix < adjacency_dist).astype(np.float32)
        np.fill_diagonal(self.adj_matrix, 0.0) # No self-loops for now
        
        # JAX conversion
        self.senders, self.receivers = np.where(self.adj_matrix)
        self.n_node = len(self.ch_names)
        self.n_edge = len(self.senders)
        
        # Graph Laplacian (Normalized)
        # D = Degree matrix
        degree = np.sum(self.adj_matrix, axis=1)
        # L = D - A
        self.laplacian = np.diag(degree) - self.adj_matrix
        # Normalized L: I - D^-0.5 A D^-0.5 ?
        # Let's use combinatorial L for simple diffusion.
        
        self.L_jax = jnp.array(self.laplacian)

    def get_graph(self, features):
        """
        Create jraph.GraphsTuple with node features.
        
        Args:
            features: (n_nodes, feature_dim)
        """
        return jraph.GraphsTuple(
            nodes=features,
            edges=None,
            senders=jnp.array(self.senders),
            receivers=jnp.array(self.receivers),
            n_node=jnp.array([self.n_node]),
            n_edge=jnp.array([self.n_edge]),
            globals=None
        )

    def smooth(self, data, alpha=0.5):
        """
        Apply Graph Laplacian Smoothing.
        X_new = X - alpha * L * X
        (Diffusion step)
        
        Args:
            data: (n_channels, n_times)
            alpha: Smoothing factor (0=None, small=smooth)
        """
        # X: (N, T)
        # L: (N, N)
        # L @ X -> (N, T)
        
        # Stability check: alpha < 1/max_eigenvalue(L)
        
        diff = self.L_jax @ data
        return data - alpha * diff

    def jraph_convolution(self, data):
        """
        Example GNN layer using jraph (Graph Convolution).
        f' = f + sum(neighbors)
        """
        graph = self.get_graph(data)
        
        def update_node_fn(node_features, sent_attributes, received_attributes, global_attributes):
            # received_attributes: sum of neighbor messages (if aggregate_nodes uses sum)
            return node_features + received_attributes
            
        def update_edge_fn(edge_features, sender_node_attributes, receiver_node_attributes, global_attributes):
            # Message is sender attributes
            return sender_node_attributes

        net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            aggregate_nodes_for_globals_fn=None
        )
        
        # jraph expects (N, F) features. Our data is (N, T).
        # We can treat T as features or batch?
        # T as features: smoothing across space, treating timepoints as vector.
        return net(graph).nodes
