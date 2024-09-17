import numpy as np
import networkx as nx
import matplotlib as plt
from scipy.spatial.distance import pdist, squareform

class ProximityNetwork:
    def __init__(self, time_series, method="cycle", segment_length=10, threshold=0.5, k=5, epsilon=0.5):
        """
        Initialize the ProximityNetwork with given parameters.

        Parameters:
        - time_series: Input time series data.
        - method: Type of network ("cycle", "correlation", "recurrence").
        - segment_length: Segment length for cycle and correlation networks.
        - threshold: Threshold for correlation or recurrence networks.
        - k: Number of nearest neighbors for k-NNN and ANNN.
        - epsilon: Distance threshold for ε-Recurrence Networks.
        """
        self.time_series = time_series
        self.method = method
        self.segment_length = segment_length
        self.threshold = threshold
        self.k = k
        self.epsilon = epsilon
        self.network = None

    def create_network(self, recurrence_type="epsilon"):
        """
        Create the appropriate network based on the method and recurrence type.

        Parameters:
        - recurrence_type: "epsilon" (for ε-Recurrence), "k-nnn", or "annn" for adaptive nearest neighbor network.
        """
        if self.method == "cycle":
            self.network = self.CycleNetwork(self.time_series, self.segment_length, self.threshold).create()
        elif self.method == "correlation":
            self.network = self.CorrelationNetwork(self.time_series, self.segment_length, self.threshold).create()
        elif self.method == "recurrence":
            self.network = self.RecurrenceNetwork(self.time_series, self.k, self.epsilon).create(recurrence_type)
        else:
            raise ValueError("Invalid method selected. Choose 'cycle', 'correlation', or 'recurrence'.")

        self._draw_network()  # Draw the network

    def _draw_network(self):
        """
        Draw the generated network.
        """
        pos = nx.spring_layout(self.network,k=2, iterations=50)

        # Get edge weights to adjust edge thickness
        edges = self.network.edges(data=True)
        weights = [data['weight'] for _, _, data in edges]  # Extract weights

        # Normalize weights for better visual scaling (optional, depending on your range of weights)
        max_weight = max(weights) if weights else 1  # Avoid division by zero
        min_weight = min(weights) if weights else 0
        normalized_weights = [(1 + 4 * (weight - min_weight) / (max_weight - min_weight)) for weight in
                              weights]  # Normalize between 1 and 5

        # Draw the network with thicker lines based on the edge weights
        nx.draw(self.network, pos, with_labels=True, edge_color='black', width=normalized_weights)


    class CycleNetwork:
        def __init__(self, time_series, segment_length, threshold):
            self.time_series = time_series
            self.segment_length = segment_length
            self.threshold = threshold

        def create(self) -> object:
            """
            Create a Cycle Network.
            Nodes represent cycles of the time series.
            Edges are created based on the correlation between cycles.
            """
            G = nx.Graph()
            cycles = [self.time_series[i:i + self.segment_length] for i in
                      range(0, len(self.time_series) - self.segment_length + 1)]

            for i, cycle in enumerate(cycles):
                G.add_node(i, cycle=cycle)

            # Connect cycles based on correlation
            for i in range(len(cycles)):
                for j in range(i + 1, len(cycles)):
                    # Ensure cycles are of equal length
                    if len(cycles[i]) == len(cycles[j]):
                        corr = np.corrcoef(cycles[i], cycles[j])[0, 1]
                        if corr > self.threshold:
                            G.add_edge(i, j, weight=corr)
                    else:
                        print(f"Skipping correlation between segments of different lengths: {len(cycles[i])} and {len(cycles[j])}")

            return G

    class CorrelationNetwork:
        def __init__(self, time_series, segment_length, threshold):
            self.time_series = time_series
            self.segment_length = segment_length
            self.threshold = threshold

        def create(self):
            """
            Create a Correlation Network.
            Nodes represent segments of the time series.
            Edges are created based on the correlation between segments.
            """
            G = nx.Graph()
            segments = [self.time_series[i:i + self.segment_length] for i in
                        range(0, len(self.time_series) - self.segment_length + 1)]

            for i, segment in enumerate(segments):
                G.add_node(i, segment=segment)

            # Connect nodes based on correlation
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    corr = np.corrcoef(segments[i], segments[j])[0, 1]
                    if corr > self.threshold:
                        G.add_edge(i, j, weight=corr)

            return G

    class RecurrenceNetwork:
        def __init__(self, time_series, k, epsilon):
            self.time_series = time_series
            self.k = k
            self.epsilon = epsilon

        def create(self, recurrence_type):
            """
            Create a Recurrence Network.
            Depending on the type (ε-Recurrence, k-NNN, ANNN), nodes are connected differently.

            Parameters:
            - recurrence_type: "epsilon" (for ε-Recurrence), "k-nnn" for k-nearest neighbor, or "annn" for adaptive nearest neighbor network.
            """
            if recurrence_type == "epsilon":
                return self._create_epsilon_recurrence_network()
            elif recurrence_type == "k-nnn":
                return self._create_knn_network()
            elif recurrence_type == "annn":
                return self._create_adaptive_knn_network()
            else:
                raise ValueError("Invalid recurrence type. Choose 'epsilon', 'k-nnn', or 'annn'.")

        def _create_epsilon_recurrence_network(self):
            """
            Create an ε-Recurrence Network.
            Nodes represent individual time points, and edges are created if the distance between nodes is less than ε.
            """
            G = nx.Graph()
            for i in range(len(self.time_series)):
                G.add_node(i, value=self.time_series[i])

            # Connect nodes based on epsilon threshold
            for i in range(len(self.time_series)):
                for j in range(i + 1, len(self.time_series)):
                    dist = abs(self.time_series[i] - self.time_series[j])
                    if dist <= self.epsilon:
                        print(f"Checking edge ({i}, {j}): distance = {dist}, epsilon = {self.epsilon}")
                        G.add_edge(i, j, weight=dist)

            return G
        def _create_knn_network(self):
            """
            Create a k-Nearest Neighbor Network (k-NNN).
            Each node is connected to its k nearest neighbors based on the distance between time points.
            """
            G = nx.Graph()
            for i in range(len(self.time_series)):
                G.add_node(i, value=self.time_series[i])

            # Compute pairwise distances between all nodes
            distances = squareform(pdist(self.time_series.reshape(-1, 1)))

            # Connect each node to its k nearest neighbors
            for i in range(len(self.time_series)):
                nearest_neighbors = np.argsort(distances[i])[1:self.k]
                for j in nearest_neighbors:
                    G.add_edge(i, j, weight=distances[i][j])
                print(nearest_neighbors)

            return G

        def _create_adaptive_knn_network(self):
            """
            Create an Adaptive Nearest Neighbor Network (ANNN).
            Similar to k-NNN, but the number of neighbors is adapted based on local density.
            """
            G = nx.Graph()
            for i in range(len(self.time_series)):
                G.add_node(i, value=self.time_series[i])

            # Compute pairwise distances between all nodes
            distances = squareform(pdist(self.time_series.reshape(-1, 1)))

            # For each node, dynamically adjust the number of neighbors based on local density
            for i in range(len(self.time_series)):
                sorted_distances = np.sort(distances[i])
                local_density = np.mean(sorted_distances[1:self.k + 1])  # Mean distance to k nearest neighbors
                adaptive_threshold = local_density * 1.2  # Example: Adjust threshold based on local density

                # Connect neighbors within the adaptive threshold
                for j in range(len(self.time_series)):
                    if distances[i][j] < adaptive_threshold and i != j:
                        G.add_edge(i, j, weight=distances[i][j])

            return G