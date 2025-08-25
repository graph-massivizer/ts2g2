__all__ = [
    "get_combine_indentical_strategy",
    "MergeStrategy",
    "CombineIdenticalSubgraphs",
    "CombineIdenticalNodes",
]

from abc import ABC, abstractmethod
from typing import Any, Dict

from strategies.register import register_strategy, get_strategy


# ---- Interface ----
class MergeStrategy(ABC):
    """Abstract base class for merge strategies. Implement combine(...) -> graph."""

    @abstractmethod
    def apply(self, graph, *args, **kwargs):
        raise NotImplementedError


# ---- Helpers ----

def _combine_subgraphs_impl(graph, node_1, node_2, att: str):
    """Combines nodes node_1 and node_2 (which are graphs) inside the main graph."""
    # Merge attributes att of corresponding nodes in the two subgraphs (assuming att is a list)
    nodes1 = list(node_1.nodes(data=True))
    nodes2 = list(node_2.nodes(data=True))
    for i in range(len(nodes1)):
        data_1 = nodes1[i][1][att]
        data_2 = nodes2[i][1][att]
        data_1.extend(data_2)

    # Transfer external neighbors of node_2 to node_1
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor, graph_binding="sliding window", color="#00ff00")

    graph.remove_node(node_2)
    return graph


def _combine_nodes_impl(graph, node_1, node_2, att: str):
    """Combines node_2 into node_1 by merging attributes and neighbors."""
    graph.nodes[node_1][att].extend(graph.nodes[node_2][att])
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor)
    graph.remove_node(node_2)
    return graph


# ---- Strategies ----
@register_strategy("merge", "combine_identical_subgraphs")
class CombineIdenticalSubgraphs(MergeStrategy):
    def __init__(self, attribute: str = "value"):
        self.attribute = attribute

    def apply(self, graph, graphs):
        """Combines nodes that are identical subgraphs in sliding window mechanism.

        Parameters
        ----------
        graph : nx.Graph
            Main graph, where nodes are subgraph objects.
        graphs : list[dict]
            Sliding window list; each element is a {key: subgraph} dictionary.
        """
        for j in range(len(graphs)):
            # Traverse subgraphs in the window (by value only)
            values_list = list(graphs[j].values())
            for i, node_1 in enumerate(values_list):
                if node_1 not in graph:
                    continue
                # Compare pairwise starting from i+1 to avoid duplicate comparisons
                items_tail = list(graphs[j].items())[i + 1 :]
                for key, node_2 in items_tail:
                    if node_2 is None or node_2 not in graph:
                        continue
                    # Compare if the edge sets of two subgraphs are identical
                    if set(node_1.edges) == set(node_2.edges):
                        graph = _combine_subgraphs_impl(graph, node_1, node_2, self.attribute)
                        del graphs[j][key]
        return graph


@register_strategy("merge", "combine_identical_nodes")
class CombineIdenticalNodes(MergeStrategy):
    def __init__(self, attribute: str = "value"):
        self.attribute = attribute

    def apply(self, graph):
        """Combines nodes in a single graph that have the same value of the specified attribute."""
        nodes_with_data = list(graph.nodes(data=True))
        for i, (n1, d1) in enumerate(nodes_with_data):
            if n1 not in graph:
                continue
            for n2, d2 in nodes_with_data[i + 1 :]:
                if n2 is None or n2 not in graph:
                    continue
                if d1[self.attribute] == d2[self.attribute]:
                    graph = _combine_nodes_impl(graph, n1, n2, self.attribute)
        return graph


# ---- Factory ----

def get_combine_indentical_strategy(method: str):
    return get_strategy("merge", method)
