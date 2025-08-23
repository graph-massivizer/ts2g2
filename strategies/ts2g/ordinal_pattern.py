__all__ = ["OrdinalPatternGraphStrategy"]
from core import TimeGraph
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.time_series import Timeseries
    from core.graphs import BaseTimeGraph
import networkx as nx
import numpy as np
from strategies.ts2g.interface import TimeseriesToGraphStrategy
 #Ordinal Pattern Transition Network，channel/channels
class OrdinalPatternGraphStrategy(TimeseriesToGraphStrategy):
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

    def to_graph(self, ts: "Timeseries") -> "BaseTimeGraph":
        """Build an ordinal pattern transition network.

        Parameters
        ----------
        ts : Timeseries | np.ndarray | list[np.ndarray]
            Preferred: a Timeseries instance (univariate). For backward
            compatibility, a raw 1D numpy array is also accepted. If a list of
            numpy arrays is provided, it's treated as multivariate.
        Returns
        -------
        TimeGraph
            A TimeGraph wrapping a directed NetworkX graph whose nodes are
            ordinal patterns and edges are transition probabilities.
        """
        # Extract raw data from Timeseries if provided
        if hasattr(ts, "get_data"):
            data = ts.get_data()
        else:
            data = ts

        # Determine univariate vs multivariate
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
            # Multivariate case: list of 1D arrays
            embedded_series = self.multivariate_embeddings(data)
            ordinal_patterns = [self.multivariate_ordinal_pattern(vec) for vec in embedded_series]
        else:
            # Univariate case: 1D array-like
            arr = np.asarray(data)
            embedded_series = self.embeddings(arr)
            ordinal_patterns = [self.ordinal_pattern(vec) for vec in embedded_series]

        G = nx.DiGraph()
        transitions = {}

        if len(ordinal_patterns) == 0:
            return TimeGraph(G)

        for i in range(len(ordinal_patterns) - 1):
            pattern = ordinal_patterns[i]
            next_pattern = ordinal_patterns[i + 1]
            if pattern not in G:
                G.add_node(pattern, ordinal_pattern=pattern)
            if next_pattern not in G:
                G.add_node(next_pattern, ordinal_pattern=next_pattern)
            transitions[(pattern, next_pattern)] = transitions.get((pattern, next_pattern), 0) + 1

        total = max(1, len(ordinal_patterns))
        for (start, end), weight in transitions.items():
            G.add_edge(start, end, weight=weight / total)

        nx.set_edge_attributes(G, "#ffff00", "color")
        return TimeGraph(G)
    
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
