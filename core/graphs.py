__all__ = ["BaseTimeGraph", "TimeGraph", "MultivariateTimeGraph"]

import networkx as nx
import hashlib
import numpy as np


class BaseTimeGraph:
    """
    Base class for all time graph types.
    """
    def __init__(self):
        self.embeddings = None

    def get_embedding(self):
        return self.embeddings

    def _hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.get_graph().nodes()) + str(self.get_graph().edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()


    def get_graph(self):
        raise NotImplementedError()




class TimeGraph(BaseTimeGraph):
    """
    Stores a single-variable graph.
    """
    def __init__(self, graph, graphs=None, is_implemented=True, histogram_frequencies=None,
                 histogram_bins=None, w=1, tau=1, quantiles=None, quantile_values=None):
        super().__init__()
        self.graph = graph
        self.orig_graph = None
        self.graphs = graphs
        self.attribute = 'value'
        self.sequences = None
        self.sequence_visitor = None
        self.is_implemented = is_implemented
        self.histogram_frequencies = histogram_frequencies
        self.histogram_bins = histogram_bins
        self.w = w
        self.tau = tau
        self.quantiles = quantiles
        self.quantile_values = quantile_values

    def get_is_implemented(self):
        return self.is_implemented

    def get_graph(self):
        return self.graph
    
    def neighbors(self, node):
        """Return neighbors of a node from the underlying graph."""
        return self.graph.neighbors(node)

    def _get_graphs(self):
        return self.graphs

    def _get_histogram(self):
        return self.histogram_frequencies, self.histogram_bins

    def _get_w_tau(self):
        return self.w, self.tau

    def _get_quantiles(self):
        return self.quantiles, self.quantile_values

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    @property
    def successors(self):
        return self.graph.successors

    def __getitem__(self, node):
        return self.graph[node]


class MultivariateTimeGraph(BaseTimeGraph):
    """
    Stores and fuses multiple graphs from multivariate timeseries.
    """
    def __init__(self, graphs, fusion='union'):
        super().__init__()
        self.fusion = fusion
        if fusion == 'union':
            self.graph = nx.disjoint_union_all(graphs)
        elif fusion == 'fused':
            self.graph = self._fuse_graphs(graphs)
        else:
            raise ValueError("Unsupported fusion mode")
        self.graphs = graphs

    def _fuse_graphs(self, graphs):
        # Attribute fusion: create a node for each index, features is a vector of each graph's node value
        G = nx.Graph()
        node_count = len(graphs[0].nodes)
        for i in range(node_count):
            features = [g.nodes[i].get('value', 0) for g in graphs]
            G.add_node(i, features=np.array(features))
        for g in graphs:
            G.add_edges_from(g.edges)
        return G

    def get_graph(self):
        return self.graph
