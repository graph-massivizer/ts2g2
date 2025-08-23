__all__ = ["QuantileTransitionGraphStrategy"]
from core import TimeGraph
import networkx as nx
import numpy as np
from typing import TYPE_CHECKING
from strategies.ts2g.interface import TimeseriesToGraphStrategy

if TYPE_CHECKING:
    from core.time_series import Timeseries
    from core.graphs import BaseTimeGraph

# Quantile discretization + transition graph strategy
class QuantileTransitionGraphStrategy(TimeseriesToGraphStrategy):
    def __init__(self, Q, phi = 1):
        self.Q = Q
        self.phi = phi
        self.bins = []
        self.values = []

    def discretize_to_quantiles(self, time_series):
        # Convert time series into quantile bins
        quantiles = np.linspace(0, 1, self.Q + 1)
        quantile_bins = np.quantile(time_series, quantiles)
        quantile_bins[0] -= 1e-9
        quantile_indices = np.digitize(time_series, quantile_bins, right=True) - 1
        return quantile_bins, quantile_indices

    def mean_jump_length(self, time_series):
        mean_jumps = []
        for phi in range(1, self.phi + 1):
            G_nx = self._build_graph(time_series, phi)
            jumps = []
            for (i, j) in G_nx.edges:
                weight = G_nx[i][j]['weight']
                jumps.append(abs(i - j) * weight)
            mean_jump = np.mean(jumps)
            mean_jumps.append(mean_jump)
        return np.array(mean_jumps)
    
    def get_quantile_values(self, timeseries, bins, indices):
        # Collect values for each quantile bin
        values = [[] for i in range(len(bins)-1)]
        
        for i in range(len(indices)):
            values[indices[i]].append(timeseries[i])
        
        self.values.append(values)
        return

    def _build_graph(self, time_series, phi=1):
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
        return G

    def to_graph(self, ts: "Timeseries") -> "BaseTimeGraph":
        data = ts.get_data()
        G_nx = self._build_graph(data, self.phi)
        nx.set_edge_attributes(G_nx, "#b266ff", "color")
        return TimeGraph(G_nx)
    
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
