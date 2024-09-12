"""
Time series graphs
"""
import networkx as nx
import itertools
import math
import numpy as np




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

        return TimeseriesGraph(G)
    
    def _get_name(self):
        return "Ordinal partition strategy"
    
    def _has_value(self):
        return False
    
    def _has_implemented_to_ts(self):
        return True
    
    def _get_w_tau(self):
        return self.w, self.tau


class TimeseriesToQuantileGraph:
    def __init__(self, Q, phi = 1):
        self.Q = Q
        self.phi = phi

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

    def to_graph(self, time_series, phi=1):
        quantile_bins, quantile_indices = self.discretize_to_quantiles(time_series)

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

        return TimeseriesGraph(G)
    
    def _get_name(self):
        return "Graph generating strategy using quantiles."
    
    def _has_value(self):
        return False
    
    def _has_implemented_to_ts(self):
        return False
    
    def _get_w_tau(self):
        return None, None


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



