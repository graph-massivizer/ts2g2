import math
import networkx as nx
from core import TimeGraph, Timeseries
from strategies import ts2g

#to be continued
#for potential extend of multivariates
def to_graph(self, strategy: ts2g.strategy_builder):
    #
    pool = ["#000000", "#101010", "#202020", "#303030", "#404040", "#505050", "#606060", "#707070", "#808080", "#909090", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#d0d0d0", "#e0e0e0", "#f0f0f0"]
    num = math.floor(16/len(self.ts))
    if len(self.ts) > 16:
        num = 1
    color_counter = 0
    self.w, self.tau = strategy._get_w_tau()
    for ts in self.ts:
        graph_dict = {}
        order=[]

        counter = 0
        for timeseries in ts: 
            g =  strategy.to_graph(Timeseries(timeseries))
            g = g.graph
            
            if strategy._has_value():
                for i in range(len(g.nodes)):
                    old_value = g.nodes[i][self.attribute]
                    new_value = [old_value]
                    g.nodes[i][self.attribute] = new_value

            hash = self._hash(g)
            mapping = {node: f"{hash}_{node}" for node in g.nodes}
            g = nx.relabel_nodes(g, mapping)

            nx.set_node_attributes(g, pool[color_counter], "color")
            color_counter += num
            if color_counter > 15:
                color_counter = 0
            

            nx.set_edge_attributes(g, strategy._get_name(), "strategy")
            graph_dict[self._hash(g) + f"_{counter}"] = g
            order.append(self._hash(g) + f"_{counter}")

            if self.graph == None:
                self.graph = g

            counter+=1

        self.graphs.append(graph_dict)
        self.graph_order.append(order)
    
    self.quantiles, self.quantile_values = strategy._get_bins()
    
    if (len(self.ts) == 1 and len(self.ts[0]) == 1):
        return TimeGraph(self.graph, graphs = self.graphs[0], is_implemented=strategy._has_implemented_to_ts(), histogram_frequencies = self.histogram_frequencies, histogram_bins = self.histogram_bins, w = self.w, tau = self.tau, quantiles = self.quantiles, quantile_values=self.quantile_values)
    else:
        self.is_implemented = strategy._has_implemented_to_ts()
        return self