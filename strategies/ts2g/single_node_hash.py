__all__ = ["SingleNodeHashGraphStrategy"]
import networkx as nx
import hashlib
from core import TimeGraph
from typing import TYPE_CHECKING
from strategies.ts2g.interface import TimeseriesToGraphStrategy

if TYPE_CHECKING:
    from core.time_series import Timeseries
    from core.graphs import BaseTimeGraph



# TimeseriesToCorrelationGraph, this is just a demonstration and still lacks functionality
class SingleNodeHashGraphStrategy(TimeseriesToGraphStrategy):
    def __init__(self):
        pass
    
    def to_graph(self, ts: "Timeseries") -> "BaseTimeGraph":
        data = ts.get_data()

        G = nx.Graph()
        G.add_node(self._hash(data), timeseries = data)

        return TimeGraph(G)
    
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



