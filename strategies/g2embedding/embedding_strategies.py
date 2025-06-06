__all__ = ["get_graph_embedding"]

import numpy as np
import networkx as nx
from strategies.register import register_strategy, get_strategy

@register_strategy("g2embedding", "eigenvector")
def eigenvector_embedding(graph):
    """Compute eigenvector centrality embedding of the graph."""
    centrality = nx.eigenvector_centrality_numpy(graph.get_graph())
    return np.array(list(centrality.values()))

@register_strategy("g2embedding", "degree")
def degree_embedding(graph):
    """Compute degree centrality embedding of the graph."""
    degree_dict = dict(graph.degree())
    return np.array(list(degree_dict.values()))

@register_strategy("g2embedding", "pagerank")
def pagerank_embedding(graph):
    """Compute pagerank embedding of the graph."""
    pagerank_dict = nx.pagerank(graph)
    return np.array(list(pagerank_dict.values()))

def get_graph_embedding(method='eigenvector'):
    """
    Compute embedding for a graph using the specified method.

    Parameters:
        graph (networkx.Graph): The graph to embed.
        method (str): The name of the embedding strategy.

    Returns:
        np.ndarray: The embedding vector.
    """
    return get_strategy("g2embedding", method)

if __name__ == "__main__":
    # Example usage
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle graph

    for method in ["eigenvector", "degree", "pagerank"]:
        embedding = get_graph_embedding(G, method=method)
        print(f"{method.capitalize()} embedding: {embedding}")