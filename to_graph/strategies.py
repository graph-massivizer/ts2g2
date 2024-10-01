"""
Time series graphs
"""
import networkx as nx
import itertools
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform
import hashlib



class EdgeWeightingStrategy:
    def weight (self, G, x1, x2, y1, y2):
        return None


class EdgeWeightingStrategyAngle(EdgeWeightingStrategy):
    absolute_value = True

    def __init__(self, absolute_value):
        self.absolute_value = absolute_value

    def weight_edge(self, G, x1, x2, y1, y2):
        slope = (y2 - y1) / (x2 - x1)
        angle = math.atan(slope)
        if self.absolute_value:
            return abs(angle)
        return angle


class EdgeWeightingStrategyNull(EdgeWeightingStrategy):
    def weight_edge(self, G, x1, x2, y1, y2):
        return None


class TimeseriesGraph:
    def __init__(self, graph):
        self.graph = graph

    def to_sequence(self, graph_to_timeseries_strategy, sequence_length):
        return graph_to_timeseries_strategy.to_sequence(self.graph, sequence_length)


class TimeseriesToOrdinalPatternGraph:
    def __init__(self, w, tau, use_quantiles=False, Q=4):
        self.w = w
        self.tau = tau
        self.use_quantiles = use_quantiles
        self.Q = Q

    def embeddings(self, time_series):
        n = len(time_series)
        embedded_series = [time_series[i:i + self.w * self.tau:self.tau] for i in range(n - self.w * self.tau + 1)]
        return np.array(embedded_series)

    def ordinal_pattern(self, vector):
        if self.use_quantiles:
            quantiles = np.linspace(0, 1, self.Q + 1)[1:-1]
            thresholds = np.quantile(vector, quantiles)
            ranks = np.zeros(len(vector), dtype=int)
            for i, value in enumerate(vector):
                ranks[i] = np.sum(value > thresholds)
        else:
            indexed_vector = [(value, index) for index, value in enumerate(vector)]
            sorted_indexed_vector = sorted(indexed_vector, key=lambda x: x[0])
            ranks = [0] * len(vector)
            for rank, (value, index) in enumerate(sorted_indexed_vector):
                ranks[index] = rank
        return tuple(ranks)

    def multivariate_embeddings(self, multivariate_time_series):
        m = len(multivariate_time_series)
        n = min(len(series) for series in multivariate_time_series)
        embedded_series = []
        for i in range(n - self.w * self.tau + 1):
            window = []
            for series in multivariate_time_series:
                window.append(series[i:i + self.w * self.tau:self.tau])
            embedded_series.append(np.array(window))
        return np.array(embedded_series)

    def multivariate_ordinal_pattern(self, vectors):
        # vectors is a 2D array of shape (m, w) where m is the number of variables
        m, w = vectors.shape
        diffs = np.diff(vectors, axis=1)
        patterns = []
        for i in range(m):
            # Determine the trend for each variable
            pattern = tuple(1 if diff > 0 else 0 for diff in diffs[i])
            patterns.append(pattern)
        # Flatten the patterns to create a combined pattern
        combined_pattern = tuple([p[i] for p in patterns for i in range(len(p))])
        return combined_pattern

    def to_graph(self, timeseries_stream):
        timeseries = timeseries_stream.read()
        if isinstance(timeseries, list) and isinstance(timeseries[0], np.ndarray):
            # Multivariate case
            embedded_series = self.multivariate_embeddings(timeseries)
            ordinal_patterns = [self.multivariate_ordinal_pattern(vec) for vec in embedded_series]
        else:
            # Univariate case
            embedded_series = self.embeddings(timeseries)
            ordinal_patterns = [self.ordinal_pattern(vec) for vec in embedded_series]

        G = nx.DiGraph()
        transitions = {}

        for i in range(len(ordinal_patterns) - 1):
            pattern = ordinal_patterns[i]
            next_pattern = ordinal_patterns[i + 1]
            if pattern not in G:
                G.add_node(pattern, ordinal_pattern=pattern)
            if next_pattern not in G:
                G.add_node(next_pattern, ordinal_pattern=next_pattern)
            if (pattern, next_pattern) not in transitions:
                transitions[(pattern, next_pattern)] = 0
            transitions[(pattern, next_pattern)] += 1

        for (start, end), weight in transitions.items():
            G.add_edge(start, end, weight=weight / len(ordinal_patterns))
        
        nx.set_edge_attributes(G, "#ffff00", "color")

        return TimeseriesGraph(G)
    
    def _get_name(self):
        return "Ordinal partition strategy"
    
    def _has_value(self):
        return False
    
    def _has_implemented_to_ts(self):
        return True
    
    def _get_w_tau(self):
        return self.w, self.tau
    
    def _get_bins(self):
        return None, None

class TimeseriesToCorrelationGraph:
    def __init__(self):
        pass
    
    def to_graph(self, timeseries):
        timeseries = timeseries.read()

        G = nx.Graph()
        G.add_node(self._hash(timeseries), timeseries = timeseries)

        return TimeseriesGraph(G)
    
    def _hash(self, timeseries):
        str_to_hash = ','.join(map(str, timeseries))
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def _get_name(self):
        return "Pearson correlation graph"
    
    def _has_value(self):
        return False
    
    def _has_implemented_to_ts(self):
        return False
    
    def _get_w_tau(self):
        return None, None
    
    def _get_bins(self):
        return None, None


class TimeseriesToProximityNetworkGraph:
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

    def to_graph(self, timeseries):
        self.time_series = timeseries.read()
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
            self.network = self.RecurrenceNetwork(self.time_series, self.k, self.epsilon).create(self.recurrence_type)
        else:
            raise ValueError("Invalid method selected. Choose 'cycle', 'correlation', or 'recurrence'.")
        nx.set_edge_attributes(self.network, "#660066", "color")
        return TimeseriesGraph(self.network)
        #self._draw_network()  # Draw the network

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


class TimeseriesToQuantileGraph:
    def __init__(self, Q, phi = 1):
        self.Q = Q
        self.phi = phi
        self.bins = []
        self.values = []

    def discretize_to_quantiles(self, time_series):
        time_series = time_series.read()
        quantiles = np.linspace(0, 1, self.Q + 1)
        quantile_bins = np.quantile(time_series, quantiles)
        quantile_bins[0] -= 1e-9
        quantile_indices = np.digitize(time_series, quantile_bins, right=True) - 1
        return quantile_bins, quantile_indices

    def mean_jump_length(self, time_series):
        mean_jumps = []
        for phi in range(1, self.phi + 1):
            G = self.to_graph(time_series, phi)
            jumps = []
            for (i, j) in G.edges:
                weight = G[i][j]['weight']
                jumps.append(abs(i - j) * weight)
            mean_jump = np.mean(jumps)
            mean_jumps.append(mean_jump)
        return np.array(mean_jumps)
    
    def get_quantile_values(self, timeseries, bins, indices):
        timeseries = timeseries.read()
        values = [[] for i in range(len(bins)-1)]
        
        for i in range(len(indices)):
            values[indices[i]].append(timeseries.iloc[i])
        
        self.values.append(values)
        return

    def to_graph(self, time_series, phi=1):
        quantile_bins, quantile_indices = self.discretize_to_quantiles(time_series)
        self.bins.append(quantile_bins) 
        self.get_quantile_values(time_series, quantile_bins, quantile_indices)

        G = nx.DiGraph()

        for i in range(self.Q):
            G.add_node(i, label=f'Q{i + 1}')

        for t in range(len(quantile_indices) - phi):
            q1, q2 = quantile_indices[t], quantile_indices[t + phi]
            if G.has_edge(q1, q2):
                G[q1][q2]['weight'] += 1
            else:
                G.add_edge(q1, q2, weight=1)

        # Normalize the edge weights to represent transition probabilities
        for i in range(self.Q):
            total_weight = sum([G[i][j]['weight'] for j in G.successors(i)])
            if total_weight > 0:
                for j in G.successors(i):
                    G[i][j]['weight'] /= total_weight
        nx.set_edge_attributes(G, "#b266ff", "color")
        return TimeseriesGraph(G)
    
    def _get_name(self):
        return "Graph generating strategy using quantiles."
    
    def _has_value(self):
        return False
    
    def _has_implemented_to_ts(self):
        return True
    
    def _get_w_tau(self):
        return None, None
    
    def _get_bins(self):
        return self.bins, self.values


class TimeseriesToGraphStrategy:
    visibility_constraints = []
    graph_type = "undirected"
    edge_weighting_strategy = EdgeWeightingStrategyNull()

    def __init__(self, visibility_constraints, graph_type="undirected",
                 edge_weighting_strategy=EdgeWeightingStrategyNull()):
        self.visibility_constraints = visibility_constraints
        self.graph_type = graph_type
        self.edge_weighting_strategy = edge_weighting_strategy

    def to_graph(self, timeseries_stream):
        """
            Return a Visibility Graph encoding the particular time series.

            A visibility graph converts a time series into a graph. The constructed graph
            uses integer nodes to indicate which event in the series the node represents.
            Edges are formed as follows: consider a bar plot of the series and view that
            as a side view of a landscape with a node at the top of each bar. An edge
            means that the nodes can be connected somehow by a "line-of-sight" without
            being obscured by any bars between the nodes and respecting all the
            specified visibility constrains.

            Parameters
            ----------
            timeseries_stream : Sequence[Number]
                   A Time Series sequence (iterable and sliceable) of numeric values
                   representing values at regular points in time.

                Returns
                -------
                NetworkX Graph
                    The Natural Visibility Graph of the timeseries time series

            Examples
            --------
            >>> stream = TimeseriesArrayStream([2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
            >>> timeseries = Timeseries(stream)
            >>> ts2g = TimeseriesToGraphStrategy([TimeseriesEdgeVisibilityConstraintsNatural()], "undirected", EdgeWeightingStrategyNull())
            >>> g = ts2g.to_graph(stream)
            >>> print(g)
        """
        timeseries = timeseries_stream.read()

        G = nx.path_graph(len(timeseries), create_using=self.initialize_graph(self.graph_type))
        nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

        # Check all combinations of nodes n series
        for (x1, y1), (x2, y2) in itertools.combinations(enumerate(timeseries), 2):
            is_visible = True
            for visibility_constraint in self.visibility_constraints:
                is_obstructed = visibility_constraint.is_obstructed(timeseries, x1, x2, y1, y2)
                is_visible = is_visible and not is_obstructed

                if not is_visible:
                    break
            if is_visible:
                weight = self.edge_weighting_strategy.weight_edge(G, x1, x2, y1, y2)
                if weight is not None:
                    G.add_edge(x1, x2, weight=weight)
                else:
                    G.add_edge(x1, x2)
        nx.set_edge_attributes(G, "#ff8000", "color")
        return TimeseriesGraph(G)
    def initialize_graph(self, graph_type):
        if (graph_type == "undirected"):
            return nx.Graph()
        return nx.DiGraph()
    
    def _get_name(self):
        name = ""
        for i in range(len(self.visibility_constraints)):
            name += self.visibility_constraints[i]._get_name()
        
        return name
    
    def _has_value(self):
        return True
    
    def _has_implemented_to_ts(self):
        return True
    
    def _get_w_tau(self):
        return None, None
    
    def _get_bins(self):
        return None, None


class TimeseriesEdgeVisibilityConstraints:
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        return None
    
    def _get_name(self):
        pass


class EdgeAdditionStrategy:
    def add_edge(self, G, x1, x2, weight=None):
        return None


class TimeseriesEdgeVisibilityConstraintsNatural(TimeseriesEdgeVisibilityConstraints):
    """
    Return a Natural Visibility Graph of a time series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a straight "line-of-sight" without
    being obscured by any bars between the nodes. The limit parameter introduces
    a limit of visibility and the nodes are connected only if, in addition to
    the natural visibility constraints, limit<x2-x1. Such a limit aims to reduce
    the effect of noise intrinsic to the data.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    limit : integer
       A limit established to the visibility between two bars of a time series.

    Returns
    -------
    Boolean
        Whether visibility is obstructed between the two bars of a time series.

    References
    ----------
    .. [1] Lacasa, Lucas, Bartolo Luque, Fernando Ballesteros, Jordi Luque, and Juan Carlos Nuno.
           "From time series to complex networks: The visibility graph." Proceedings of the
           National Academy of Sciences 105, no. 13 (2008): 4972-4975.
           https://www.pnas.org/doi/10.1073/pnas.0709247105
    .. [2] Zhou, Ting-Ting, Ningde Jin, Zhongke Gao and Yue-Bin Luo. “Limited penetrable visibility
           graph for establishing complex network from time series.” (2012).
           https://doi.org/10.7498/APS.61.030506
    """
    limit = 0
    name = "Natural visibility strategy"

    def __init__(self, limit=0):
        self.limit = limit
        if limit > 0:
            self.name += (f" with limit({limit})")

    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        slope = (y2 - y1) / (x2 - x1)
        offset = y2 - slope * x2

        return any(
            y > slope * x + offset
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )
    
    def _get_name(self):
        return self.name
        


class TimeseriesEdgeVisibilityConstraintsHorizontal(TimeseriesEdgeVisibilityConstraints):
    """
    Return a Horizontal Visibility Graph of a time series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a horizontal "line-of-sight" without
    being obscured by any bars between the nodes [1]. The limit parameter introduces
    a limit of visibility and the nodes are connected only if, in addition to
    the natural visibility constraints, limit<x2-x1. Such a limit aims to reduce
    the effect of noise intrinsic to the data [2].

    Parameters
    ----------
    limit : integer
       A limit established to the visibility between two bars of a time series.

    Returns
    -------
    Boolean
        Whether visibility is obstructed between the two bars of a time series.

    References
    ----------
    .. [1] Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
           "Horizontal visibility graphs: Exact results for random time series."
           Physical Review E, 80(4), 046103.
           http://dx.doi.org/10.1103/PhysRevE.80.046103
    .. [2] Zhou, Ting-Ting, Ningde Jin, Zhongke Gao and Yue-Bin Luo. “Limited
           penetrable visibility graph for establishing complex network from
           time series.” (2012). https://doi.org/10.7498/APS.61.030506
    .. [3] Wang, M., Vilela, A.L.M., Du, R. et al. Exact results of the limited
           penetrable horizontal visibility graph associated to random time series
           and its application. Sci Rep 8, 5130 (2018).
           https://doi.org/10.1038/s41598-018-23388-1
    """
    limit = 0
    name = "Horizontal visibility strategy"

    def __init__(self, limit=0):
        self.limit = limit
        self.name += (f" with limit({limit})")

    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        return any(
            y > max(y1, y2)
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )
    
    def _get_name(self):
        return self.name


class TimeseriesEdgeVisibilityConstraintsVisibilityAngle(TimeseriesEdgeVisibilityConstraints):
    """
    Return a Parametric Visibility Graph of a time series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Regardless the strategy used to define how edges are constructed, this class
    introduces additional constraints, ensuring that the angle between two
    observations meets a threshold parameter (visiblity_angle) [1]. Furthermore,
    additional constraints can be placed, ensuring that the absolute values of the
    angle between two observations and the threshold are considered [2].

    Parameters
    ----------
    visibility_angle : float
        A limit established to the visibility angle between two observations of a time series.
    consider_visibility_angle_absolute_value: bool, optional
        If True, the absolute values of the angle and the threshold are considered.

    Returns
    -------
    Boolean
        Whether visibility is obstructed between the two observations of a time series.

    References
    ----------
    .. [1] Bezsudnov, I. V., and A. A. Snarskii. "From the time series to the complex networks:
           The parametric natural visibility graph." Physica A: Statistical Mechanics and its
           Applications 414 (2014): 53-60.
           https://doi.org/10.1016/j.physa.2014.07.002
    .. [2] Supriya, S., Siuly, S., Wang, H., Cao, J., & Zhang, Y. (2016). Weighted visibility
           graph with complex network features in the detection of epilepsy. IEEE access, 4,
           6554-6566. https://doi.org/10.1109/ACCESS.2016.2612242
    """
    visibility_angle = 0
    consider_visibility_angle_absolute_value = True

    def __init__(self, visibility_angle=0, consider_visibility_angle_absolute_value=True):
        self.visibility_angle = visibility_angle
        self.consider_visibility_angle_absolute_value = consider_visibility_angle_absolute_value

    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        slope = (y2 - y1) / (x2 - x1)

        angle = math.atan(slope)
        visibility_angle = self.visibility_angle
        if (self.consider_visibility_angle_absolute_value):
            angle = abs(angle)
            visibility_angle = abs(self.visibility_angle)

        return angle < visibility_angle
    
    def _get_name(self):
        return (f" with angle({self.visibility_angle})")


class EdgeAdditionStrategyUnweighted(EdgeAdditionStrategy):
    def add_edge(self, G, x1, x2, weight=None):
        G.add_edge(x1, x2)


class EdgeAdditionStrategyWeighted(EdgeAdditionStrategy):
    def add_edge(self, G, x1, x2, weight=None):
        G.add_edge(x1, x2, weight=weight)



