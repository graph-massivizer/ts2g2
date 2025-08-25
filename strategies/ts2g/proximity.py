__all__ = ["ProximityGraphStrategy"]
from core import TimeGraph
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import TYPE_CHECKING
from strategies.ts2g.interface import TimeseriesToGraphStrategy

if TYPE_CHECKING:
    from core.time_series import Timeseries
    from core.graphs import BaseTimeGraph

# --- Extracted strategies for proximity graphs ---

class CycleNetworkStrategy:
    def __init__(self, time_series, segment_length, threshold):
        self.time_series = time_series
        self.segment_length = segment_length
        self.threshold = threshold

    def create(self) -> nx.Graph:
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

class CorrelationNetworkStrategy:
    def __init__(self, time_series, segment_length, threshold):
        self.time_series = time_series
        self.segment_length = segment_length
        self.threshold = threshold

    def create(self) -> nx.Graph:
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

class RecurrenceNetworkStrategy:
    def __init__(self, time_series, k, epsilon, recurrence_type="epsilon"):
        self.time_series = time_series
        self.k = k
        self.epsilon = epsilon
        self.recurrence_type = recurrence_type

    def create(self) -> nx.Graph:
        """
        Create a Recurrence Network.
        Depending on the type (ε-Recurrence, k-NNN, ANNN), nodes are connected differently.
        """
        if self.recurrence_type == "epsilon":
            return self._create_epsilon_recurrence_network()
        elif self.recurrence_type == "k-nnn":
            return self._create_knn_network()
        elif self.recurrence_type == "annn":
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


# --- ProximityGraphStrategy using extracted strategies ---
class ProximityGraphStrategy(TimeseriesToGraphStrategy):
    def __init__(self, method="cycle", segment_length=10, threshold=0.5, k=5, epsilon=0.5, recurrence_type="epsilon"):
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
        self.time_series = None
        self.method = method
        self.segment_length = segment_length
        self.threshold = threshold
        self.k = k
        self.epsilon = epsilon
        self.network = None
        self.recurrence_type = recurrence_type

    def to_graph(self, ts: "Timeseries") -> "BaseTimeGraph":
        """Create the appropriate proximity network based on the configured method.

        Parameters
        ----------
        ts : Timeseries
            Input univariate time series container.
        """
        self.time_series = ts.get_data()
        # Choose strategy based on method
        if self.method == "cycle":
            strategy = CycleNetworkStrategy(self.time_series, self.segment_length, self.threshold)
        elif self.method == "correlation":
            strategy = CorrelationNetworkStrategy(self.time_series, self.segment_length, self.threshold)
        elif self.method == "recurrence":
            strategy = RecurrenceNetworkStrategy(self.time_series, self.k, self.epsilon, self.recurrence_type)
        else:
            raise ValueError("Invalid method selected. Choose 'cycle', 'correlation', or 'recurrence'.")
        self.network = strategy.create()
        nx.set_edge_attributes(self.network, "#660066", "color")
        return TimeGraph(self.network)

    def _get_name(self):
        return "Proximity network graph."
    
    def _has_value(self):
        return False
    
    def _has_implemented_to_ts(self):
        return False
    
    def _get_w_tau(self):
        return None, None
    
    def _get_bins(self):
        return None, None
