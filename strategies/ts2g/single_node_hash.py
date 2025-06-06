__all__ = ["SingleNodeHashGraphStrategy"]
import networkx as nx
import hashlib
from core import TimeGraph



#TimeseriesToCorrelationGraph，这个只是演示，还缺乏功能
class SingleNodeHashGraphStrategy:
    def __init__(self):
        pass
    
    def to_graph(self, timeseries):
        timeseries = timeseries.read()

        G = nx.Graph()
        G.add_node(self._hash(timeseries), timeseries = timeseries)

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



