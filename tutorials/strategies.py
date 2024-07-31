from core.model import TimeseriesGraph, TimeseriesArrayStream
import networkx as nx
import itertools
import math
import numpy as np
import pandas as pd
import os
from core import model


# %%
class EdgeWeightingStrategy:
    def weight(self, G, x1, x2, y1, y2):
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


class EdgeWeightingStrategyWeight(EdgeWeightingStrategy):
    def weight_edge(self, G, x1, x2, y1, y2):
        return G[x1][x2].get('weight', 1)


class TimeseriesToGraphStrategy:
    tt = []
    strategies = []
    def __init__(self, strategies, tt):
        self.strategies = strategies
        self.timeseries = tt

    def initialize_graph(self, graph_type):
        if (graph_type == "undirected"):
            return nx.Graph()
        return nx.DiGraph()

    def apply_strategies(self):
        for strategy in self.strategies:
            graph = strategy.to_graph(timeseries_stream=self.timeseries, graph_type=strategy.graph_type)
        return graph


class VisibilityGraphStrategy(TimeseriesToGraphStrategy):
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        return None

    def to_graph(self, timeseries_stream, graph_type):

        timeseries = model.TimeseriesArrayStream(timeseries_stream).read()

        G = nx.path_graph(len(timeseries), create_using=self.initialize_graph(graph_type))
        nx.set_node_attributes(G, dict(enumerate(timeseries)), "value")

        # Check all combinations of nodes n series
        for (x1, y1), (x2, y2) in itertools.combinations(enumerate(timeseries), 2):
            is_visible = True
            for visibility_constraint in self.strategies:
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


class EdgeAdditionStrategy:
    def add_edge(self, G, x1, x2, weight=None):
        return None



class TimeseriesEdgeVisibilityConstraintsNatural(VisibilityGraphStrategy):
    """
    Return a Natural Visibility Graph of a time series.
    """
    def __init__(self, limit=0, graph_type="undirected", edge_weighting_strategy=EdgeWeightingStrategyNull()):
        self.limit = limit
        self.graph_type = graph_type
        self.edge_weighting_strategy = edge_weighting_strategy

    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        slope = (y2 - y1) / (x2 - x1)
        offset = y2 - slope * x2

        return any(
            y > slope * x + offset
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )


class TimeseriesEdgeVisibilityConstraintsHorizontal(VisibilityGraphStrategy):
    """
    Return a Horizontal Visibility Graph of a time series.
    """

    def __init__(self, limit=0, graph_type="undirected", edge_weighting_strategy=EdgeWeightingStrategyNull()):
        self.limit = limit
        self.graph_type = graph_type
        self.edge_weighting_strategy = edge_weighting_strategy
    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        # check if any value between obstructs line of sight
        return any(
            y > max(y1, y2)
            for x, y in enumerate(timeseries[x1 + self.limit + 1: x2], start=x1 + self.limit + 1)
        )


class TimeseriesEdgeVisibilityConstraintsVisibilityAngle(VisibilityGraphStrategy):
    """
    Return a Parametric Visibility Graph of a time series.
    """
    def __init__(self, visibility_angle=0, graph_type="undirected", edge_weighting_strategy=EdgeWeightingStrategyNull(), consider_visibility_angle_absolute_value=True):
        self.visibility_angle = visibility_angle
        self.graph_type = graph_type
        self.edge_weighting_strategy = edge_weighting_strategy
        self.consider_visibility_angle_absolute_value = consider_visibility_angle_absolute_value

    def is_obstructed(self, timeseries, x1, x2, y1, y2):
        slope = (y2 - y1) / (x2 - x1)

        angle = math.atan(slope)
        visibility_angle = self.visibility_angle
        if self.consider_visibility_angle_absolute_value:
            angle = abs(angle)
            visibility_angle = abs(self.visibility_angle)

        return angle < visibility_angle


class TimeseriesToQuantileGraph(TimeseriesToGraphStrategy):
    def __init__(self, Q, phi=1, graph_type="directed"):
        self.Q = Q
        self.phi = phi
        self.graph_type=graph_type

    def discretize_to_quantiles(self, time_series):
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

    def to_graph(self, timeseries_stream, graph_type, phi=1):
        quantile_bins, quantile_indices = self.discretize_to_quantiles(timeseries_stream)

        G = self.initialize_graph(self.graph_type)

        for i in range(self.Q):
            G.add_node(i, label=f'Q{i + 1}')

        for t in range(len(quantile_indices) - phi):
            q1, q2 = quantile_indices[t], quantile_indi ces[t + phi]
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


class TimeseriesToOrdinalPatternGraph(TimeseriesToGraphStrategy):
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
            quantiles = np.linspace(0, 1, self.Q + 1)[1:-1]  # Compute quantile thresholds based on Q
            thresholds = np.quantile(vector, quantiles)  # Get the quantile values
            ranks = np.zeros(len(vector), dtype=int)

            for i, value in enumerate(vector):
                ranks[i] = np.sum(value > thresholds)  # Rank based on quantiles

        else:
            indexed_vector = [(value, index) for index, value in enumerate(vector)]
            sorted_indexed_vector = sorted(indexed_vector, key=lambda x: x[0])
            ranks = [0] * len(vector)
            for rank, (value, index) in enumerate(sorted_indexed_vector):
                ranks[index] = rank

        return tuple(ranks)

    def to_graph(self, time_series):
        embedded_series = self.embeddings(time_series)
        ordinal_patterns = [self.ordinal_pattern(vec) for vec in embedded_series]

        G = self.initialize_graph(self.graph_type)

        transitions = {}
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

        # Add edges with weights
        for (start, end), weight in transitions.items():
            G.add_edge(start, end, weight=weight / len(ordinal_patterns))

        return TimeseriesGraph(G)

# class EdgeAdditionStrategyUnweighted(EdgeAdditionStrategy):
#    def add_edge(self, G, x1, x2, weight=None):
#        G.add_edge(x1, x2)

# class EdgeAdditionStrategyWeighted(EdgeAdditionStrategy):
#    def add_edge(self, G, x1, x2, weight=None):
#        G.add_edge(x1, x2, weight=weight)