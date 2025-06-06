__all__ = ["get_merge_strategy"]

from strategies.register import register_strategy, get_strategy

@register_strategy("merge", "combine_identical_subgraphs")
def combine_identical_subgraphs(graph, graphs, attribute="features"):
    """Combines nodes that are identical subgraphs in sliding window mechanism."""
    # orig_graph = graph.copy()
    for j in range(len(graphs)):
        for i, node_1 in enumerate(list(graphs[j].values())):
            if node_1 not in graph:
                continue
            for key, node_2 in list(graphs[j].items())[i+1:]:
                if node_2 is None or node_2 not in graph:
                    continue
                if set(list(node_1.edges)) == set(list(node_2.edges)):
                    graph = _combine_subgraphs(graph, node_1, node_2, attribute)
                    del graphs[j][key]
    return graph

def _combine_subgraphs(graph, node_1, node_2, att):
    """Combines nodes node_1 and node_2, that are graphs."""
    for i in range(len(list(node_1.nodes(data=True)))):
        data_1 = list(node_1.nodes(data=True))[i][1][att]
        data_2 = list(node_2.nodes(data=True))[i][1][att]
        data_1.extend(data_2)
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor, graph_binding="sliding window", color="#00ff00")
    graph.remove_node(node_2)
    return graph

@register_strategy("merge", "combine_identical_nodes")
def combine_identical_nodes(graph, attribute="features"):
    """Combines nodes that have the same value of the specified attribute."""
    orig_graph = graph.copy()
    for i, node_1 in enumerate(list(graph.nodes(data=True))):
        if node_1 not in graph:
            continue
        for node_2 in list(graph.nodes(data=True))[i+1:]:
            if node_2 is None or node_2 not in graph:
                continue
            if node_1[1][attribute] == node_2[1][attribute]:
                graph = _combine_nodes(graph, node_1[0], node_2[0], attribute)
    return graph

def _combine_nodes(graph, node_1, node_2, att):
    """Combines node_2 into node_1 by merging attributes and neighbors."""
    graph.nodes[node_1][att].extend(graph.nodes[node_2][att])
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor)
    graph.remove_node(node_2)
    return graph

def get_merge_strategy(name):
    return get_strategy("merge", name)