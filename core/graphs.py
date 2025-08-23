__all__ = [ "BaseTimeGraph","TimeGraph"]

import networkx as nx
import hashlib
import numpy as np
from abc import ABC, abstractmethod


class BaseTimeGraph(ABC):
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


    @abstractmethod
    def get_graph(self):
        """Return the underlying NetworkX graph for this time graph."""
        pass

class TimeGraph(BaseTimeGraph):
    """
    Stores already made graph, allows us to add edges and links between nodes.
    
    **Attributes:**

    - `graph`: object networkx.Graph
    
    """
    
    def __init__(self, graph, graphs = None, is_implemented = True, histogram_frequencies = None, histogram_bins = None, w = 1, tau = 1, quantiles = None, quantile_values = None):
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
        self.embeddings = None
        self.quantiles = quantiles
        self.quantile_values = quantile_values

    def get_is_implemented(self):
        return self.is_implemented

    def _get_graph(self):
        return self.graph

    def _get_graphs(self):
        return self.graphs
    
    def get_graph(self):
        return self.graph

    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], color = "#66ffff")
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight, color = "#66ffff")
        return self

    def link(self, link_strategy):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        self.graph = link_strategy.link(self)
        return self

    def _hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def combine_identical_subgraphs(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()
        for j in range(len(self.graphs)):
            for i, node_1 in enumerate(list(self.graphs[j].values())):
                if node_1 not in self.graph:
                    continue

                for key, node_2 in list(self.graphs[j].items())[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(set(list(node_1.edges)) == set(list(node_2.edges))):
                        self.graph = self._combine_subgraphs(self.graph, node_1, node_2, self.attribute)
                        del self.graphs[j][key]
                        

        return self

    def _combine_subgraphs(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])

        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor, graph_binding = "sliding window", color = "#00ff00")

        graph.remove_node(node_2)
        return graph

    def combine_identical_nodes(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()

        for i, node_1 in enumerate(list(self.graph.nodes(data=True))):
            if node_1 not in self.graph:
                continue

            for node_2 in list(self.graph.nodes(data=True))[i+1:]:
                if node_2 == None:
                    break
                if node_2 not in self.graph:
                    continue

                if(node_1[self.attribute] == node_2[self.attribute]):
                    self.graph = self._combine_nodes(self.graph, node_1, node_2, self.attribute)

        return self

    def _combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)

        graph.remove_node(node_2)
        return graph

    
    
    def to_sequence(self, sequence_visitor):
        if not self.is_implemented:
            raise ValueError(f"This function is not yet implemented for this type of graph.")
        self.sequence_visitor = sequence_visitor
        self.sequences = sequence_visitor.to_sequence(self)
        return self
    


    
    def _get_histogram(self):
        return self.histogram_frequencies, self.histogram_bins
    
    def _get_w_tau(self):
        return self.w, self.tau

    def to_embedding(self, embedding_visitor):
        self.embeddings = embedding_visitor.get_graph_embedding()
        return self
    
    def get_embedding(self):
        return self.embeddings

    def _get_quantiles(self):
        return self.quantiles, self.quantile_values
