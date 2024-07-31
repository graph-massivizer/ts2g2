import numpy as np
import networkx as nx

class MultivariateTimeseriesToOrdinalPatternGraph:
    def __init__(self, w, tau, use_quantiles=False, Q=4):
        self.w = w
        self.tau = tau
        self.use_quantiles = use_quantiles
        self.Q = Q

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

    def multivariate_ordinal_pattern(self, vectors):
        m, w = vectors.shape
        patterns = []
        for i in range(m):
            pattern = self.ordinal_pattern(vectors[i])
            patterns.append(pattern)
        combined_pattern = tuple([p[i] for p in patterns for i in range(len(p))])
        return combined_pattern

    def to_graph(self, multivariate_time_series):
        embedded_series = self.multivariate_embeddings(multivariate_time_series)
        ordinal_patterns = [self.multivariate_ordinal_pattern(vec) for vec in embedded_series]

        # Initialize a directed graph
        G = nx.DiGraph()
        transitions = {}

        # Record transitions between consecutive ordinal patterns
        for i in range(len(ordinal_patterns) - 1):
            pattern = ordinal_patterns[i]
            next_pattern = ordinal_patterns[i + 1]
            if pattern not in G:
                G.add_node(pattern)
            if next_pattern not in G:
                G.add_node(next_pattern)
            if (pattern, next_pattern) not in transitions:
                transitions[(pattern, next_pattern)] = 0
            transitions[(pattern, next_pattern)] += 1

        # Add edges to the graph with weights representing transition probabilities
        for (start, end), weight in transitions.items():
            G.add_edge(start, end, weight=weight / len(ordinal_patterns))

        return G
