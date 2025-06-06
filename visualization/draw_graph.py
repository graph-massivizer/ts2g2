__all__ = ["draw_g"]
import matplotlib.pyplot as plt
import networkx as nx
#TODO: add node weights and names to draw function


def draw_g(timegraph):
    """Draws the created graph"""
    
    pos=nx.spring_layout(timegraph.get_graph(), seed=1)
    
    nodes_missing_color = [(u, d) for u, d in timegraph.graph.nodes(data=True) if 'color' not in d]

    edge_colors = [timegraph.graph[u][v]['color'] for u, v in timegraph.graph.edges()]
    nodes_colors = [d.get("color", "gray") for u, d in timegraph.graph.nodes(data=True)]
    
    # Get edge weights to adjust edge thickness
    edges = timegraph.graph.edges(data=True)
    weights = []
    for _, _, data in edges:
        weights.append(data.get('weight', 1))
    #weights = [data['weight'] for _, _, data in edges]  # Extract weights

    # Normalize weights for better visual scaling (optional, depending on your range of weights)
    max_weight = max(weights) if weights else 1  # Avoid division by zero
    min_weight = min(weights) if weights else 0
    if max_weight == min_weight:
        normalized_weights = [1 for weight in weights]
    else:
        normalized_weights = [(1 + 4 * (weight - min_weight) / (max_weight - min_weight)) for weight in
                            weights]

    nx.draw(timegraph.graph, pos, node_size=100, node_color=nodes_colors, with_labels=False, edge_color=edge_colors, width=normalized_weights)
    plt.savefig('example.png')


# #detached from ProximityGraphStrategy
# def _draw_network(self):
#     """
#     Draw the generated network.
#     """
#     pos = nx.spring_layout(self.network,k=2, iterations=50)

#     # Get edge weights to adjust edge thickness
#     edges = self.network.edges(data=True)
#     weights = [data['weight'] for _, _, data in edges]  # Extract weights

#     # Normalize weights for better visual scaling (optional, depending on your range of weights)
#     max_weight = max(weights) if weights else 1  # Avoid division by zero
#     min_weight = min(weights) if weights else 0
#     normalized_weights = [(1 + 4 * (weight - min_weight) / (max_weight - min_weight)) for weight in
#                             weights]  # Normalize between 1 and 5

#     # Draw the network with thicker lines based on the edge weights
#     nx.draw(self.network, pos, with_labels=True, edge_color='black', width=normalized_weights)
