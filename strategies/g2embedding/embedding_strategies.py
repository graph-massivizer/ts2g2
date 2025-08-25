__all__ = [
    "get_graph_embedding",
    "GraphEmbeddingStrategy",
    "EigenvectorEmbedding",
    "DegreeEmbedding",
    "PagerankEmbedding",
]

import numpy as np
import networkx as nx
from strategies.register import register_strategy, get_strategy
from .interface import GraphEmbeddingStrategy

# Helper to accept either a BaseTimeGraph-like wrapper or a raw networkx graph
def _as_nx(graph):
    try:
        return graph.get_graph()  # BaseTimeGraph-style wrapper
    except AttributeError:
        return graph  # assume it's already a networkx.Graph




@register_strategy("g2embedding", "eigenvector")
class EigenvectorEmbedding(GraphEmbeddingStrategy):
    def embed(self, graph) -> np.ndarray:
        G = _as_nx(graph)
        centrality = nx.eigenvector_centrality_numpy(G)
        return np.array(list(centrality.values()), dtype=float)


@register_strategy("g2embedding", "degree")
class DegreeEmbedding(GraphEmbeddingStrategy):
    def embed(self, graph) -> np.ndarray:
        G = _as_nx(graph)
        degree_dict = dict(G.degree())
        return np.array(list(degree_dict.values()), dtype=float)


@register_strategy("g2embedding", "pagerank")
class PagerankEmbedding(GraphEmbeddingStrategy):
    def embed(self, graph) -> np.ndarray:
        G = _as_nx(graph)
        pr = nx.pagerank(G)
        return np.array(list(pr.values()), dtype=float)


def get_graph_embedding(method: str = 'eigenvector'):
    """Return the registered embedding strategy *class* for the given method.

    Usage:
        EmbeddingCls = get_graph_embedding("degree")
        embedding = EmbeddingCls().embed(G)
    """
    return get_strategy("g2embedding", method)

if __name__ == "__main__":
    # Example usage
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle graph

    for method in ["eigenvector", "degree", "pagerank"]:
        EmbeddingCls = get_graph_embedding(method)
        embedding = EmbeddingCls().embed(G)
        print(f"{method.capitalize()} embedding: {embedding}")