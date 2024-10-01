import hashlib
import networkx as nx
import functools
import pandas as pd
from scipy.interpolate import interp1d
from dtaidistance import dtw
from scipy.stats import pearsonr
import math

def compare(x, y):
    return x.get_strategy_precedence() - y.get_strategy_precedence()

class StrategyLinkingMultipleGraphs:
    """
    Links multiple graphs together.
    
    **Attributes:**

    - `graph`: networkx.Graph object
    - `graphs`: dictionary of networkx.Graph objects
    - `strategy_precedence`: tells in which order should the strategies be excetuted
    
    """
    
    def __init__(self, graphs, strategy_precedence):
        self.graphs = graphs
        self.graph = None
        self.strategy_precedence = strategy_precedence
        self.timeseries = None

    def get_strategy_precedence(self):
        return self.strategy_precedence

    def set_graphs(self, graphs, order, timeseries):
        pass

    def apply(self):
        pass


class StrategyLinkingGraphsByCorrelation(StrategyLinkingMultipleGraphs):
    def __init__(self, graphs, correlation):
        super().__init__(graphs, 2)
        self.correlation = correlation
    
    def set_graphs(self, graphs, order, timeseries):
        self.graphs = graphs
        return self

    def apply(self):
        g = nx.Graph()
        if isinstance(self.graphs, list):
            graphs = {}
            for i in range(len(self.graphs)):
                graphs[list(self.graphs[i].items())[0]] = list(self.graphs[i].values())[0]
            self.graphs = graphs

        for graph in self.graphs.values():
                g = nx.compose(g, graph)
        
        for (node_11, node_12) in zip(list(g.nodes(data = True)), list(g.nodes)):
            for (node_21, node_22) in zip(list(g.nodes(data = True)), list(g.nodes)):
                if node_11 == node_21:
                    continue
                ts1 = node_11[1]["timeseries"].reset_index(drop=True)
                ts2 = node_21[1]["timeseries"].reset_index(drop=True)
                corr = self.correlation.get_correlation(ts1, ts2)
                if pd.isna(corr):
                    corr = 0
                g.add_edge(node_12, node_22, weight = corr, intergraph_binding = 'Correlation', color = "#994c00")
        self.graph = g

        return self.graph, self.graphs

    def _hash(self, timeseries):
        str_to_hash = ','.join(map(str, timeseries))
        return hashlib.md5(str_to_hash.encode()).hexdigest()


class StrategyLinkingGraphsByCorrelationSlidingWindow(StrategyLinkingMultipleGraphs):
    def __init__(self, graphs, correlation):
        super().__init__(graphs, 1)
        self.correlation = correlation
    
    def set_graphs(self, graphs, order, timeseries):
        self.graphs = graphs
        return self
    
    def apply(self):
        g = nx.Graph()
        graphs = {}

        for j in range(len(self.graphs)):
            h = nx.Graph()
            for graph in self.graphs[j].values():
                h = nx.compose(h, graph)
            
            for (node_11, node_12) in zip(list(h.nodes(data = True)), list(h.nodes)):
                for (node_21, node_22) in zip(list(h.nodes(data = True)), list(h.nodes)):
                    if node_11 == node_21:
                        continue
                    ts1 = node_11[1]["timeseries"].reset_index(drop=True)
                    ts2 = node_21[1]["timeseries"].reset_index(drop=True)
                    corr = self.correlation.get_correlation(ts1, ts2)
                    if pd.isna(corr):
                        corr = 0
                    h.add_edge(node_12, node_22, weight = corr, intergraph_binding = 'Correlation aliding window', color = "#666600")
            graphs[self._hash(h)] = h
            g = nx.compose(g, h)
        
        self.graph = g
        self.graphs = graphs

        return self.graph, self.graphs
    
    def _hash(self, timeseries):
        str_to_hash = ','.join(map(str, timeseries))
        return hashlib.md5(str_to_hash.encode()).hexdigest()


class StrategyLinkingMultipleGraphsByTimeCooccurrence(StrategyLinkingMultipleGraphs):
    """Links nodes from multiple graphs based on their sequential order."""
    def __init__(self, graphs):
        super().__init__(graphs, 3)

    def set_graphs(self, graphs, order, timeseries):
        self.graphs = graphs
        return self

    def apply(self):

        min_size = None

        if isinstance(self.graphs, list):
            graphs = {}
            for i in range(len(self.graphs)):
                graphs[list(self.graphs[i].items())[0]] = list(self.graphs[i].values())[0]
            self.graphs = graphs

        for graph in self.graphs.values():
            if min_size == None or len(graph.nodes) < min_size:
                min_size = len(graph.nodes)

        for hash, graph in self.graphs.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1

        g = None
        if isinstance(list(self.graphs.values())[0], nx.DiGraph):
            g = nx.DiGraph()
        else:
            g = nx.Graph()

        for graph in self.graphs.values():
            g = nx.compose(g, graph)

        i = 0
        j = 0
        for (node_11, node_12) in zip(list(g.nodes(data = True)), list(g.nodes)):

            i = 0
            for (node_21, node_22) in zip(list(g.nodes(data = True)), list(g.nodes)):
                if i == j:
                    i+=1
                    continue

                if node_11[1]['order'] == node_21[1]['order']:
                    g.add_edge(node_12, node_22, intergraph_binding = 'positional', color = "#ff007f")
                i+=1
            j+=1

        self.graph = g

        return self.graph, self.graphs


class StrategyLinkingMultipleGraphsByPositionalCorrelationSlidingWindow(StrategyLinkingMultipleGraphs):
    def __init__(self, graphs, correlation):
        super().__init__(graphs, 4)
        self.correlation = correlation
    
    def set_graphs(self, graphs, order, timeseries):
        self.graphs = graphs
        return self
    
    def apply(self):
        min_size = None

        if isinstance(self.graphs, list):
            graphs = {}
            for i in range(len(self.graphs)):
                graphs[list(self.graphs[i].items())[0]] = list(self.graphs[i].values())[0]
            self.graphs = graphs

        for graph in self.graphs.values():
            if min_size == None or len(graph.nodes) < min_size:
                min_size = len(graph.nodes)

        for hash, graph in self.graphs.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1
        
        g = None
        if isinstance(list(self.graphs.values())[0], nx.DiGraph):
            g = nx.DiGraph()
        else:
            g = nx.Graph()

        for graph in self.graphs.values():
            g = nx.compose(g, graph)

        i = 0
        j = 0
        for (node_11, node_12) in zip(list(g.nodes(data = True)), list(g.nodes)):

            i = 0
            for (node_21, node_22) in zip(list(g.nodes(data = True)), list(g.nodes)):
                if i == j:
                    i+=1
                    continue

                if node_11[1]['order'] == node_21[1]['order']:
                    ts1 = node_11[1]["timeseries"].reset_index(drop=True)
                    ts2 = node_21[1]["timeseries"].reset_index(drop=True)
                    corr = self.correlation.get_correlation(ts1, ts2)
                    if pd.isna(corr):
                        corr = 0
                    g.add_edge(node_12, node_22, weight = corr, intergraph_binding = 'positional', color = "#0000cc")
                i+=1
            j+=1

        self.graph = g

        return self.graph, self.graphs

class StrategyLinkingMultipleGraphsSlidingWindow(StrategyLinkingMultipleGraphs):
    """Sequentially links graphs made by sliding window mechanism."""
    def __init__(self, graphs, graph_order):
        super().__init__(graphs, 0)
        self.graph_order = graph_order

    def set_graphs(self, graphs, order, timeseries):
        self.graphs = graphs
        self.graph_order = order
        return self
    
    def _hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def apply(self):
        g = nx.Graph()
        graphs = {}
        pool = ["#000000", "#101010", "#202020", "#303030", "#404040", "#505050", "#606060", "#707070", "#808080", "#909090", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#d0d0d0", "#e0e0e0", "#f0f0f0"]
        num = math.floor(16/len(self.graphs))
        if len(self.graphs) > 16:
            num = 1
        color_counter = 0

        for j in range(len(self.graphs)):
            h = nx.Graph()

            for i in range(len(self.graph_order[j])-1):
                h.add_edge(self.graphs[j][self.graph_order[j][i]], self.graphs[j][self.graph_order[j][i+1]])
            nx.set_edge_attributes(h, "#00ff00", "color")
            nx.set_edge_attributes(h, "sliding window", "graph_binding")
            nx.set_node_attributes(h, pool[color_counter], "color")
            color_counter += num
            if color_counter > 15:
                color_counter = 0
            g = nx.compose(g, h)
            graphs[self._hash(h)] = h

        
        self.graph = g
        self.graphs = graphs

        return self.graph, self.graphs
    

class StrategyLinkingMultipleGraphsDynamicTimeWarping(StrategyLinkingMultipleGraphs):
    """Connects visibility graphs based on Dynamic time warping."""
    def __init__(self, graphs):
        super().__init__(graphs, 4)
    
    def set_graphs(self, graphs, order, timeseries):
        self.graphs = graphs
        self.timeseries  = timeseries
        return self
    
    def apply(self):
        g = nx.Graph()
        graphs = {}
        if isinstance(self.graphs, list):
            for i in range(len(self.graphs)):
                graphs[list(self.graphs[i].items())[0]] = list(self.graphs[i].values())[0]
        
        for hash, graph in graphs.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1

        for graph in graphs.values():
            g = nx.compose(g, graph)

        for i in range(len(self.graphs)):
            ts1 = self.timeseries[i][0].to_numpy()
            g1 = list(self.graphs[i].values())[0]

            for j in range(i+1, len(self.graphs)):
                ts2 = self.timeseries[j][0].to_numpy()
                g2 = list(self.graphs[j].values())[0]

                distance, paths = dtw.warping_paths(ts1, ts2, use_c=False)
                best_path = dtw.best_path(paths)

                for (a, b) in best_path:
                    node_1 = [node for node, attr in g1.nodes(data=True) if attr.get('order') == a][0]
                    node_2 = [node for node, attr in g2.nodes(data=True) if attr.get('order') == b][0]
                    g.add_edge(node_1, node_2, intergraph_binding = 'dynamical time warping', color = "#ff0000")

        self.graph = g
        self.graphs = graphs
        return self.graph, self.graphs


class LinkGraphs:
    """
    Builder class for linking multiple graphs, through which we can access linking strategies.
    
    **Attributes:**

    - `graph`: networkx.Graph object
    - `graphs`: dictionary of networkx.Graph objects
    - `command_array`: an array that stores linking strategies
    
    """
    def __init__(self):
        self.graphs = None
        self.graph = None
        self.graph_order = None
        self.command_array = []

    def time_cooccurrence(self):
        """Notes that we want to connect graphs in a multivariate graph based on time co-ocurrance."""
        self.command_array.append(StrategyLinkingMultipleGraphsByTimeCooccurrence(self.graphs))
        return self

    def sliding_window(self):
        """Notes that we want to connect graphs in a multivariate graph to create sliding window graph."""
        self.command_array.append(StrategyLinkingMultipleGraphsSlidingWindow(self.graphs, self.graph_order))
        return self
    
    def correlation(self, corr):
        """Notes that we want to connect graphs in a multivariate graph based on correlation."""
        self.command_array.append(StrategyLinkingGraphsByCorrelation(self.graphs, corr))
        return self
    
    def correlation_sliding_window(self, corr):
        """Notes that we want to connect graphs in a multivariate graph based on correlation for sliding window graph."""
        self.command_array.append(StrategyLinkingGraphsByCorrelationSlidingWindow(self.graphs, corr))
        return self
    
    def positional_correlation_sliding_window(self, corr):
        """Notes that we want to connect graphs in a multivariate graph based on correlation for same positioned windows of series."""
        self.command_array.append(StrategyLinkingMultipleGraphsByPositionalCorrelationSlidingWindow(self.graphs, corr))
        return self
    
    def dynamic_timewarping(self):
        """Notes that we want to connect graphs in a multivariate graph based on dynamic time warping."""
        self.command_array.append(StrategyLinkingMultipleGraphsDynamicTimeWarping(self.graphs))
        return self

    def link(self, graphs, graph_order, timeseries):
        self.graphs = graphs
        self.graph_order = graph_order

        self.command_array.sort(key=functools.cmp_to_key(compare))

        for strat in self.command_array:
            strat.set_graphs(self.graphs, graph_order, timeseries)
            self.graph, self.graphs = strat.apply()

        return self.graph


class Correlation:
    
    def get_correlation(self, ts1, ts2):
        pass
class PearsonCorrelation(Correlation):
    
    def get_correlation(self, ts1, ts2):
        return ts1.corr(ts2)